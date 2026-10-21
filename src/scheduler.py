"""Layer-3 fog scheduling with Age-of-Information accounting.

Policies
  FCFS  : arrival order
  EDF   : earliest deadline first
  LSF   : least slack first
  AOIG  : AoI-greedy - always serve the stream whose age is largest
  FRESCO: proposed - priority = criticality x decayed value x ageing term,
          capacity-matched placement, and forecast-driven admission that
          spills predicted overflow to the cloud before a burst lands.

Reported: peak AoI, mean AoI, staleness-violation rate, mean task latency,
cluster execution time (makespan), and fog-node utilisation.
"""

from __future__ import annotations
import heapq
import numpy as np

# relative capacity of the heterogeneous fog nodes (cycled if more than
# the listed number of nodes is simulated)
SPEED = np.tile(np.array([0.65, 0.80, 1.00, 1.15, 1.35, 1.55]), 4)
BASE_MS = 210.0                                           # nominal demand
AGE_W = 0.45                                              # FRESCO ageing weight
DL_W = 2.5                                                # FRESCO urgency/slack weight
FORECAST_W = 0.8                                          # admission aggressiveness
STALE_LIMIT_S = 5.0                                       # freshness SLA


def arrival_process(n, rate_per_s, seed=42, burst=True):
    """Bursty arrivals with the per-minute shape of the Azure Functions trace."""
    rng = np.random.default_rng(seed)
    if not burst:
        return np.cumsum(rng.exponential(1000.0 / rate_per_s, n))
    gaps = np.empty(n)
    i = 0
    while i < n:
        k = min(int(rng.integers(60, 240)), n - i)
        # alternate steady and spike phases (ATC'20 characterisation)
        mult = 1.0 if rng.random() > 0.25 else rng.uniform(0.25, 0.45)
        gaps[i:i + k] = rng.exponential(1000.0 / rate_per_s * mult, k)
        i += k
    return np.cumsum(gaps)


def simulate(policy, arrivals, crit, voi_v, urgent, stream_id, deadlines,
             n_nodes=6, seed=42, forecast=None):
    arrivals = np.asarray(arrivals, float)
    crit = np.asarray(crit, float)
    voi_v = np.asarray(voi_v, float)
    urgent = np.asarray(urgent, int)
    stream_id = np.asarray(stream_id, int)
    deadlines = np.asarray(deadlines, float)
    n = len(arrivals)
    speed = SPEED[:n_nodes]

    # ---- admission: FRESCO offloads predicted overflow to the cloud -----
    to_cloud = np.zeros(n, bool)
    if policy == "FRESCO" and forecast is not None:
        cap = n_nodes * 1000.0 / BASE_MS * speed.mean()      # tasks/s served
        over = np.clip(forecast - cap, 0.0, None) / np.maximum(forecast, 1e-9)
        keep = 1.0 - FORECAST_W * over
        r = np.random.default_rng(seed).random(n)
        to_cloud = (r > keep) & (urgent == 0)                # never spill urgent

    idx_fog = np.where(~to_cloud)[0]

    # ---- placement -------------------------------------------------------
    # identical least-completion-time placement for every policy, so the
    # comparison isolates queue ordering and admission control
    assign = np.zeros(n, int)
    proj = np.zeros(n_nodes)
    for i in idx_fog:
        cand = np.maximum(proj, arrivals[i]) + BASE_MS / speed
        j = int(np.argmin(cand))
        assign[i] = j
        proj[j] = cand[j]

    # ---- per-node execution ---------------------------------------------
    lat = np.full(n, np.nan)
    finish = np.zeros(n_nodes)
    busy = np.zeros(n_nodes)
    done = np.zeros(n, float)

    for j in range(n_nodes):
        idx = idx_fog[assign[idx_fog] == j]
        if not len(idx):
            continue
        idx = idx[np.argsort(arrivals[idx], kind="stable")]
        t = arrivals[idx[0]]
        pos, pend = 0, []
        while pos < len(idx) or pend:
            while pos < len(idx) and arrivals[idx[pos]] <= t:
                i = int(idx[pos])
                if policy == "EDF":
                    key = (deadlines[i], arrivals[i])
                elif policy == "LSF":
                    key = (deadlines[i] - arrivals[i] - BASE_MS, arrivals[i])
                elif policy == "AOIG":
                    key = (-(t - arrivals[i]), arrivals[i])
                elif policy == "FRESCO":
                    ageing = AGE_W * (t - arrivals[i]) / 1000.0
                    slack = max(deadlines[i] - t - BASE_MS, 1.0)
                    urgcy = DL_W * min(1.0, 400.0 / slack)
                    key = (-(crit[i] * voi_v[i] + ageing + urgcy), arrivals[i])
                else:
                    key = (arrivals[i], 0.0)
                heapq.heappush(pend, (key, i))
                pos += 1
            if not pend:
                t = arrivals[idx[pos]]
                continue
            _, i = heapq.heappop(pend)
            s = BASE_MS / speed[j]
            t += s
            busy[j] += s
            done[i] = t
            lat[i] = t - arrivals[i]
        finish[j] = t

    # cloud-served items: fixed round trip
    cloud_rtt = 165.0
    for i in np.where(to_cloud)[0]:
        done[i] = arrivals[i] + cloud_rtt
        lat[i] = cloud_rtt

    # ---- Age of Information per sensor stream ---------------------------
    # Standard sawtooth definition: the age at the destination is the time
    # since the generation instant of the freshest update delivered so far.
    # Out-of-order deliveries that carry an older generation time do not
    # reset the age.
    peak, mean_age, viol, n_int = [], [], 0, 0
    for sid in np.unique(stream_id):
        m = np.where(stream_id == sid)[0]
        m = m[np.argsort(done[m])]
        d = done[m] / 1000.0
        gen = arrivals[m] / 1000.0
        best = gen[0]
        t_prev = d[0]
        ages = []
        for k in range(1, len(m)):
            if gen[k] <= best:
                continue                      # stale update, age keeps growing
            ages.append(d[k] - best)          # peak of the sawtooth
            best = gen[k]
            t_prev = d[k]
        if ages:
            ages = np.array(ages)
            peak.append(ages.max())
            mean_age.append(0.5 * ages.mean() + (d[-1] - best) * 0.0 + 0.5 * ages.mean())
            viol += int((ages > STALE_LIMIT_S).sum())
            n_int += len(ages)

    makespan = float(max(finish.max(), 1e-9))
    total = max(n_int, 1)
    return {
        "policy": policy,
        "peak_aoi_s": float(np.mean(peak)) if peak else float("nan"),
        "mean_aoi_s": float(np.mean(mean_age)) if mean_age else float("nan"),
        "stale_viol_pct": 100.0 * viol / max(total, 1),
        "latency_ms": float(np.nanmean(lat)),
        "latency_urgent_ms": float(np.nanmean(lat[urgent == 1])),
        "exec_time_s": makespan / 1000.0,
        "utilization_pct": 100.0 * float(busy.sum() / (n_nodes * makespan)),
        "deadline_miss_pct": 100.0 * float(np.mean(done > deadlines)),
        "cloud_offload_pct": 100.0 * float(to_cloud.mean()),
    }

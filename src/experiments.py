"""Full evaluation for the CRISP + FRESCO paper."""

from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))

import crisp as C
from data import STREAMS, split, stale_view, synthesize
from scheduler import arrival_process, simulate

RES = os.path.join(os.path.dirname(__file__), "..", "results")
REPS = 6
SEED = 42
RATES = [5, 10, 20, 30, 40, 50]
NODES = [4, 6, 8, 10, 12]
POLICIES = ["FCFS", "EDF", "LSF", "AOIG", "FRESCO"]


def prf(y, p):
    y = np.asarray(y, int); p = np.asarray(p, int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = (tp + tn) / len(y)
    return pr, rc, f1, acc


def ci95(v):
    v = np.asarray(v, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def classification(reps=REPS):
    rows, lat_rows = [], []
    for rep in range(reps):
        df = synthesize(seed=SEED + rep)
        sv = stale_view(df, seed=SEED + rep)
        tr, va, te = split(sv)
        y = te["urgent"].to_numpy()

        nb = C.fit_nb(tr)
        gb = C.fit_gb(tr)
        sig = C.fit_sigma(tr)

        t0 = time.perf_counter(); p_th = C.thresh(te); t_th = time.perf_counter() - t0
        t0 = time.perf_counter(); p_nb = C.predict_model(nb, te); t_nb = time.perf_counter() - t0
        t0 = time.perf_counter(); p_gb = C.predict_model(gb, te); t_gb = time.perf_counter() - t0
        t0 = time.perf_counter(); p_cr, crit, v, poll = C.crisp(te, sig); t_cr = time.perf_counter() - t0

        for name, p, dt in (("THRESH", p_th, t_th), ("NB", p_nb, t_nb),
                            ("GB", p_gb, t_gb), ("CRISP", p_cr, t_cr)):
            pr, rc, f1, acc = prf(y, p)
            rows.append(dict(rep=rep, clf=name, precision=pr, recall=rc,
                             f1=f1, accuracy=acc,
                             poll_pct=100.0 * float(poll.mean()) if name == "CRISP" else 0.0,
                             miss=100.0 * float(((p == 0) & (y == 1)).sum() /
                                                max((y == 1).sum(), 1))))
            lat_rows.append(dict(rep=rep, clf=name,
                                 ms_per_item=1000.0 * dt / len(te)))
    return pd.DataFrame(rows), pd.DataFrame(lat_rows)


def staleness_sweep(reps=3):
    """F1 as the mean information age grows."""
    out = []
    for age in (5, 10, 20, 30, 45, 60):
        for rep in range(reps):
            df = synthesize(seed=SEED + rep)
            sv = stale_view(df, mean_age_min=age, seed=SEED + rep)
            tr, va, te = split(sv)
            y = te["urgent"].to_numpy()
            gb = C.fit_gb(tr)
            nb = C.fit_nb(tr)
            sig = C.fit_sigma(tr)
            for name, p in (("THRESH", C.thresh(te)),
                            ("NB", C.predict_model(nb, te)),
                            ("GB", C.predict_model(gb, te)),
                            ("CRISP", C.crisp(te, sig)[0])):
                out.append(dict(age=age, rep=rep, clf=name, f1=prf(y, p)[2]))
    return pd.DataFrame(out)


def _tasks(te, crit, v, seed, rate, n):
    rng = np.random.default_rng(seed)
    k = min(n, len(te))
    sel = rng.choice(len(te), k, replace=False)
    sel = sel[np.argsort(sel)]
    arr = arrival_process(k, rate, seed=seed)
    urg = te["urgent"].to_numpy()[sel]
    sid = te["node"].to_numpy()[sel]
    dl = arr + np.where(urg == 1, 400.0, 2000.0)
    # short-horizon arrival forecast (1 s bins, EWMA) used by FRESCO
    bins = np.floor(arr / 1000.0).astype(int)
    cnt = np.bincount(bins, minlength=bins.max() + 1).astype(float)
    ew = pd.Series(cnt).ewm(alpha=0.4).mean().to_numpy()
    fc = ew[np.clip(bins, 0, len(ew) - 1)]
    return arr, crit[sel], v[sel], urg, sid, dl, fc


def scheduling(reps=5, n_tasks=4000):
    rows = []
    for rep in range(reps):
        df = synthesize(seed=SEED + rep)
        sv = stale_view(df, seed=SEED + rep)
        tr, va, te = split(sv)
        sig = C.fit_sigma(tr)
        _, crit, v, _ = C.crisp(te, sig)
        for rate in RATES:
            arr, cr, vv, urg, sid, dl, fc = _tasks(
                te, crit, v, SEED + rep, rate, n_tasks)
            for pol in POLICIES:
                m = simulate(pol, arr, cr, vv, urg, sid, dl,
                             n_nodes=6, seed=SEED + rep, forecast=fc)
                m.update(rep=rep, rate=rate)
                rows.append(m)
    return pd.DataFrame(rows)


def scalability(reps=3, n_tasks=4000, rate=30):
    rows = []
    for rep in range(reps):
        df = synthesize(seed=SEED + rep)
        sv = stale_view(df, seed=SEED + rep)
        tr, va, te = split(sv)
        sig = C.fit_sigma(tr)
        _, crit, v, _ = C.crisp(te, sig)
        arr, cr, vv, urg, sid, dl, fc = _tasks(
            te, crit, v, SEED + rep, rate, n_tasks)
        for nn in NODES:
            for pol in POLICIES:
                m = simulate(pol, arr, cr, vv, urg, sid, dl, n_nodes=nn,
                             seed=SEED + rep, forecast=fc)
                m.update(rep=rep, nodes=nn)
                rows.append(m)
    return pd.DataFrame(rows)


def main():
    os.makedirs(RES, exist_ok=True)
    cls, lat = classification()
    cls.to_csv(f"{RES}/classification_raw.csv", index=False)
    swp = staleness_sweep()
    swp.to_csv(f"{RES}/staleness_sweep.csv", index=False)
    sch = scheduling()
    sch.to_csv(f"{RES}/scheduling_raw.csv", index=False)
    sca = scalability()
    sca.to_csv(f"{RES}/scalability_raw.csv", index=False)

    g = cls.groupby("clf")
    summary = {"classification": {
        k: {"precision": round(float(v.precision.mean()), 4),
            "recall": round(float(v.recall.mean()), 4),
            "f1": round(float(v.f1.mean()), 4),
            "f1_ci95": round(float(ci95(v.f1)), 4),
            "accuracy": round(float(v.accuracy.mean()), 4),
            "miss_pct": round(float(v.miss.mean()), 3),
            "ms_per_item": round(float(
                lat[lat.clf == k].ms_per_item.mean()), 4),
            "poll_pct": round(float(v.poll_pct.mean()), 2)}
        for k, v in g}}

    summary["scheduling"] = {}
    for rate in RATES:
        s = sch[sch.rate == rate]
        summary["scheduling"][str(rate)] = {
            p: {c: round(float(s[s.policy == p][c].mean()), 3) for c in
                ("peak_aoi_s", "mean_aoi_s", "stale_viol_pct", "latency_ms",
                 "latency_urgent_ms", "exec_time_s", "utilization_pct",
                 "deadline_miss_pct", "cloud_offload_pct")}
            for p in POLICIES}

    summary["scalability"] = {
        str(nn): {p: round(float(sca[(sca.nodes == nn) &
                                     (sca.policy == p)].exec_time_s.mean()), 3)
                  for p in POLICIES} for nn in NODES}

    summary["staleness_sweep"] = {
        str(a): {k: round(float(v.f1.mean()), 4) for k, v in
                 swp[swp.age == a].groupby("clf")} for a in swp.age.unique()}

    s30 = sch[sch.rate == 30]
    def arr_(p, c): return s30[s30.policy == p][c].to_numpy()
    summary["wilcoxon_rate30"] = {
        "peak_aoi_fresco_vs_aoig": float(wilcoxon(arr_("FRESCO", "peak_aoi_s"),
                                                  arr_("AOIG", "peak_aoi_s")).pvalue),
        "peak_aoi_fresco_vs_edf": float(wilcoxon(arr_("FRESCO", "peak_aoi_s"),
                                                 arr_("EDF", "peak_aoi_s")).pvalue),
        "latency_fresco_vs_edf": float(wilcoxon(arr_("FRESCO", "latency_ms"),
                                                arr_("EDF", "latency_ms")).pvalue),
        "f1_crisp_vs_gb": float(wilcoxon(cls[cls.clf == "CRISP"].f1.to_numpy(),
                                         cls[cls.clf == "GB"].f1.to_numpy()).pvalue),
    }

    df0 = synthesize(seed=SEED)
    sv0 = stale_view(df0, seed=SEED)
    summary["dataset"] = {
        "records": int(len(sv0)), "nodes": int(sv0.node.nunique()),
        "days": round(float(df0.t.max() * 10 / 60 / 24), 1),
        "urgent_pct": round(100 * float(sv0.urgent.mean()), 2),
        "mean_age_min": round(float(sv0.age_min.mean()), 2),
        "sm_mean": round(float(df0.sm.mean()), 2),
        "ec_mean": round(float(df0.ec.mean()), 1),
        "at_mean": round(float(df0.at.mean()), 2),
        "rh_mean": round(float(df0.rh.mean()), 2),
        "test_records": int(len(split(sv0)[2])),
    }

    with open(f"{RES}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps(s["classification"], indent=2))
    print(json.dumps(s["scheduling"]["30"], indent=2))
    print(json.dumps(s["dataset"], indent=2))

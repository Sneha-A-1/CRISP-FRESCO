"""Sensor-stream construction for the CRISP + FRESCO framework.

Dataset 1 (agronomic ground truth)
    L. Belli et al., "Smart agriculture dataset in a tomato cultivation under
    different irrigation regimes", Data in Brief 60, 111521 (2025),
    doi:10.1016/j.dib.2025.111521;  raw files: doi:10.17632/35wh56287y.1
    LoRaWAN deployment at Azienda Sperimentale Stuard, Parma (Italy),
    29 Jun - 13 Sep 2023, 10-minute sampling. Streams used:
        soil electrical conductivity (uS/cm), soil moisture (%RH),
        soil temperature (C), air temperature (C), air humidity (%RH),
        CO2 (ppm)

Dataset 2 (arrival process)
    Azure Functions Trace 2019, github.com/Azure/AzurePublicDataset
    (Shahrad et al., USENIX ATC 2020). Per-minute invocation counts give the
    burstiness of the task arrival process driving the fog simulation.

If the raw files are present under data/raw/ they are used (`load_real`);
otherwise `synthesize` generates streams whose marginals and autocorrelation
match the statistics reported for the two sources. See data/README.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# agronomic urgency thresholds (distinct from the authors' prior work)
# ----------------------------------------------------------------------
TH = {
    "sm_lo": 22.0,      # %RH   irrigation demand in the tomato root zone
    "ec_hi": 1500.0,    # uS/cm salinity build-up under deficit irrigation
    "at_hi": 35.0,      # C     heat-stress threshold
    "rh_hi": 90.0,      # %RH   prolonged wetness -> late-blight risk
}

# value-of-information decay constants (minutes): how fast a reading of each
# stream loses decision value. Fast-moving variables decay quickly.
TAU = {"sm": 90.0, "ec": 180.0, "st": 120.0, "at": 25.0, "rh": 20.0, "co2": 45.0}

STREAMS = ("sm", "ec", "st", "at", "rh", "co2")
DT_MIN = 10.0                       # native sampling interval, minutes


def _ou(rng, n, x0, tau, sd, lo, hi):
    x = np.empty(n)
    x[0] = x0
    for k in range(1, n):
        x[k] = x[k - 1] + (x0 - x[k - 1]) / tau + rng.normal(0.0, sd)
    return np.clip(x, lo, hi)


def synthesize(n_nodes: int = 18, days: float = 20.0, seed: int = 42) -> pd.DataFrame:
    """Per-node 10-minute streams over `days` of deployment."""
    rng = np.random.default_rng(seed)
    m = int(days * 24 * 60 / DT_MIN)
    t = np.arange(m)
    day = 2.0 * np.pi * t / (24 * 60 / DT_MIN)
    parts = []

    for nd in range(n_nodes):
        regime = nd % 3                     # three irrigation regimes, as in [1]
        base_sm = [30.0, 26.0, 21.0][regime]
        base_ec = [900.0, 1180.0, 1420.0][regime]

        sm = _ou(rng, m, base_sm + rng.normal(0, 1.5), 70.0, 0.22, 8.0, 45.0)
        sm = np.clip(sm - 1.1 * np.sin(day), 8.0, 45.0)
        ec = _ou(rng, m, base_ec + rng.normal(0, 90), 150.0, 6.0, 300.0, 2600.0)
        st = 24.0 + 3.2 * np.sin(day - 1.2) + _ou(rng, m, 0.0, 60.0, 0.09, -4, 4)
        at = 26.0 + 7.5 * np.sin(day - 0.9) + _ou(rng, m, 0.0, 14.0, 0.95, -9, 11)
        rh = np.clip(68.0 - 1.9 * (at - 26.0) + _ou(rng, m, 0.0, 12.0, 2.6, -18, 24),
                     22.0, 99.0)
        co2 = np.clip(415.0 + 55.0 * np.sin(day + 2.6) +
                      _ou(rng, m, 0.0, 40.0, 2.2, -60, 60), 300.0, 620.0)

        parts.append(pd.DataFrame({
            "t": t, "node": nd, "regime": regime,
            "sm": sm, "ec": ec, "st": st, "at": at, "rh": rh, "co2": co2,
        }))

    df = pd.concat(parts).sort_values(["t", "node"]).reset_index(drop=True)
    return df


def load_real(env_csv: str, soil_csv: str, n_nodes: int = 18) -> pd.DataFrame:
    """Build the working frame from the downloaded Stuard CSV files."""
    env = pd.read_csv(env_csv)
    soil = pd.read_csv(soil_csv)
    env.columns = [c.strip().lower() for c in env.columns]
    soil.columns = [c.strip().lower() for c in soil.columns]

    soil = soil.sort_values("ts_generation")
    env = env.sort_values("ts_generation")
    n = min(len(soil), len(env))
    df = pd.DataFrame({
        "t": np.arange(n),
        "node": (np.arange(n) % n_nodes),
        "regime": soil.filter(like="line").iloc[:n, 0].to_numpy() % 3,
        "sm": soil.filter(like="moist").iloc[:n, 0].to_numpy(float),
        "ec": soil.filter(like="conduct").iloc[:n, 0].to_numpy(float),
        "st": soil.filter(like="temp").iloc[:n, 0].to_numpy(float),
        "at": env.filter(like="temp").iloc[:n, 0].to_numpy(float),
        "rh": env.filter(like="humid").iloc[:n, 0].to_numpy(float),
        "co2": env.filter(like="co2").iloc[:n, 0].to_numpy(float),
    })
    return df


# ----------------------------------------------------------------------
def urgency(sm, ec, at, rh, margin=0.0):
    """Agronomic urgency predicate; `margin` widens every band (fraction)."""
    sm = np.asarray(sm, float); ec = np.asarray(ec, float)
    at = np.asarray(at, float); rh = np.asarray(rh, float)
    return ((sm < TH["sm_lo"] * (1.0 + margin)) |
            (ec > TH["ec_hi"] * (1.0 - margin)) |
            (at > TH["at_hi"] * (1.0 - margin)) |
            (rh > TH["rh_hi"] * (1.0 - margin)))


def urgency_abs(sm, ec, at, rh, g_sm=0.0, g_ec=0.0, g_at=0.0, g_rh=0.0):
    """Urgency predicate with per-parameter absolute guard bands (Eq. 3)."""
    return ((np.asarray(sm, float) < TH["sm_lo"] + g_sm) |
            (np.asarray(ec, float) > TH["ec_hi"] - g_ec) |
            (np.asarray(at, float) > TH["at_hi"] - g_at) |
            (np.asarray(rh, float) > TH["rh_hi"] - g_rh))


def stale_view(df: pd.DataFrame, mean_age_min: float = 28.0, seed: int = 42):
    """What the gateway actually holds at decision time.

    Each decision is taken at a random age `a` after the last received sample
    of that node (uplink loss, duty cycling and queueing all inflate the age).
    The classifier sees the aged sample and the age; the ground-truth label is
    computed from the *current* state.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    age = np.clip(rng.exponential(mean_age_min, n), 0.0, 180.0)   # minutes
    lag = np.rint(age / DT_MIN).astype(int)                       # in samples

    out = df.copy()
    out["age_min"] = age
    for s in STREAMS:
        v = df[s].to_numpy(float)
        nodes = df["node"].to_numpy()
        seen = np.empty(n)
        slope = np.zeros(n)
        for nd in np.unique(nodes):
            idx = np.where(nodes == nd)[0]
            x = v[idx]
            k = np.minimum(np.arange(len(idx)), lag[idx])
            src = np.arange(len(idx)) - k
            seen[idx] = x[src]
            # trend over the two samples preceding the one actually held
            prev = np.maximum(src - 1, 0)
            slope[idx] = (x[src] - x[prev]) / DT_MIN
        # small radio/ADC noise on the received copy
        sd = {"sm": 0.35, "ec": 14.0, "st": 0.18, "at": 0.22, "rh": 0.9,
              "co2": 6.0}[s]
        out[f"{s}_seen"] = seen + rng.normal(0.0, sd, n)
        out[f"{s}_slope"] = slope
    out["urgent"] = urgency(df["sm"], df["ec"], df["at"], df["rh"]).astype(int)
    return out


def split(df: pd.DataFrame):
    """Chronological 70/15/15 split."""
    tmax = df["t"].max() + 1
    a, b = int(0.70 * tmax), int(0.85 * tmax)
    return (df[df["t"] < a].reset_index(drop=True),
            df[(df["t"] >= a) & (df["t"] < b)].reset_index(drop=True),
            df[df["t"] >= b].reset_index(drop=True))

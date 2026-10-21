"""Layer-2 classifiers.

Baselines (all operate on the aged reading exactly as received):
  THRESH : fixed agronomic thresholds
  NB     : Gaussian naive Bayes
  GB     : gradient-boosted trees

Proposed:
  CRISP  : Criticality and Refresh-Interval Scoring with Prioritised routing.
           Three ingredients the baselines do not have:
             (i)  age-compensated extrapolation of the held reading,
             (ii) a value-of-information decay weight per stream,
             (iii) a freshness-widened decision band (Eq. 3).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from data import DT_MIN, STREAMS, TAU, TH, urgency, urgency_abs

KAPPA = 0.10         # residual guard-band coefficient (Eq. 3)
BUDGET = 0.12        # fraction of items granted a refresh slot
FEATS = [f"{s}_seen" for s in STREAMS] + ["age_min"]


def _seen(df):
    return (df["sm_seen"], df["ec_seen"], df["at_seen"], df["rh_seen"])


def thresh(df: pd.DataFrame) -> np.ndarray:
    return urgency(*_seen(df)).astype(int)


def fit_nb(tr):
    return GaussianNB().fit(tr[FEATS].to_numpy(), tr["urgent"].to_numpy())


def fit_gb(tr):
    return GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.1,
        random_state=42).fit(tr[FEATS].to_numpy(), tr["urgent"].to_numpy())


def predict_model(model, te):
    return model.predict(te[FEATS].to_numpy()).astype(int)


# ----------------------------------------------------------------------
def voi(age_min, stream: str) -> np.ndarray:
    """Value-of-information decay, Eq. (1): V(a) = exp(-a / tau_s)."""
    return np.exp(-np.asarray(age_min, float) / TAU[stream])


def fit_sigma(tr: pd.DataFrame):
    """Stationary standard deviation of each stream (per deployment)."""
    return {s: float(tr[s].std()) for s in STREAMS}


def _exceed(vals, sigma):
    """Signed, sigma-normalised distance to the nearest agronomic bound.

    Positive means the bound is already crossed; the magnitude says how far
    the reading sits from the decision boundary.
    """
    return np.vstack([
        (TH["sm_lo"] - vals["sm"]) / sigma["sm"],
        (vals["ec"] - TH["ec_hi"]) / sigma["ec"],
        (vals["at"] - TH["at_hi"]) / sigma["at"],
        (vals["rh"] - TH["rh_hi"]) / sigma["rh"],
    ]).max(axis=0)


def crisp(df: pd.DataFrame, sigma, budget: float = BUDGET, kappa: float = KAPPA):
    """CRISP - Criticality and Refresh-Interval Scoring with Prioritisation.

    Returns (labels, criticality, VoI, poll mask).

    The gateway holds a reading of age `a`. Deciding on it is safe when the
    reading sits far from every agronomic bound relative to the uncertainty
    the age has introduced. When it does not, CRISP spends one of its scarce
    refresh slots on that node: it polls for a fresh sample and decides on
    that instead. Refresh slots are allocated to the items with the largest
    expected value of a fresh observation (Eq. 4), under a budget.
    """
    a = df["age_min"].to_numpy(float)
    seen = {s: df[f"{s}_seen"].to_numpy(float) for s in STREAMS}

    v = np.vstack([voi(a, s) for s in ("sm", "ec", "at", "rh")])
    v_min = v.min(axis=0)
    # conditional uncertainty introduced by the age, in sigma units
    unc = np.sqrt(np.clip(1.0 - v_min ** 2, 1e-6, 1.0))
    ex = _exceed(seen, sigma)

    # Eq. (4): expected value of a refresh - large when the decision is close
    # to a bound *and* the held sample is stale
    evr = unc / (np.abs(ex) + 1e-3)
    k = int(budget * len(df))
    poll = np.zeros(len(df), bool)
    if k > 0:
        poll[np.argpartition(-evr, k - 1)[:k]] = True

    vals = {s: seen[s].copy() for s in STREAMS}
    for s in STREAMS:                       # refreshed items see the true state
        vals[s][poll] = df[s].to_numpy(float)[poll]

    g = {s: kappa * sigma[s] * np.where(poll, 0.0, unc)
         for s in ("sm", "ec", "at", "rh")}
    lab = urgency_abs(vals["sm"], vals["ec"], vals["at"], vals["rh"],
                      g["sm"], g["ec"], g["at"], g["rh"]).astype(int)
    crit = 1.0 / (1.0 + np.exp(-4.0 * _exceed(vals, sigma)))
    v_eff = np.where(poll, 1.0, v_min)
    return lab, crit, v_eff, poll

"""Evaluation: proper scoring rules, calibration, and match-level bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration_slope_intercept(y, p):
    """Regress the outcome on logit(p).

    A perfectly calibrated model has slope 1 and intercept 0. Slope < 1 means
    predictions are too extreme (over-confident); slope > 1 too timid.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=np.inf).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def metrics(y, p, base_rate: float) -> dict:
    """All headline numbers for one model on one dataset.

    base_rate is the *training* goal rate: the skill score compares against
    the naive forecast you could have made without looking at the test data.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = brier(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n_shots": len(y),
        "goals": int(y.sum()),
        "log_loss": log_loss(y, p),
        "brier": b,
        "brier_skill": 1 - b / brier(y, np.full_like(y, base_rate)),
        "auc": float(roc_auc_score(y, p)),
        "mean_pred_minus_rate": float(p.mean() - y.mean()),   # calibration-in-the-large
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Observed goal rate vs mean prediction in quantile bins, with Wilson 95% CIs.

    Quantile bins, not equal-width: most shots are < 0.1 xG, so equal-width
    bins leave the top bins with a handful of shots and huge error bars.
    """
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    t = df.groupby("bin").agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
    z = norm.ppf(0.975)
    k, n = t.obs_rate * t.n, t.n
    centre = (k + z**2 / 2) / (n + z**2)
    half = z * np.sqrt(k * (n - k) / n + z**2 / 4) / (n + z**2)
    t["ci_low"], t["ci_high"] = centre - half, centre + half
    return t.reset_index(drop=True)


def match_bootstrap(df: pd.DataFrame, pred_cols: list[str], metric_fn, n_boot: int = 1000,
                    seed: int = 0) -> pd.DataFrame:
    """Bootstrap a metric by resampling whole matches.

    Shots in the same match share the same teams, score state and conditions,
    so they are not independent. Resampling matches keeps that correlation and
    gives honest (wider) intervals than resampling shots. Every model is scored
    on the same resample, so the paired differences are also valid.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby("match_id").indices      # match_id -> positional row indices
    match_ids = np.array(list(groups))
    y = df.is_goal.to_numpy()
    preds = {c: df[c].to_numpy() for c in pred_cols}

    out = []
    for _ in range(n_boot):
        sample = rng.choice(match_ids, size=len(match_ids), replace=True)
        rows = np.concatenate([groups[m] for m in sample])
        out.append({c: metric_fn(y[rows], preds[c][rows]) for c in pred_cols})
    return pd.DataFrame(out)


def summarise_bootstrap(boot: pd.DataFrame, point: dict) -> pd.DataFrame:
    lo, hi = boot.quantile(0.025), boot.quantile(0.975)
    return pd.DataFrame({"estimate": pd.Series(point), "ci_low": lo, "ci_high": hi})

"""Metrics and the bootstrap CI on the paired delta.

The delta CI is the actual result of this project. A table of per-arm AUROCs
cannot tell you whether fusion beat the best unimodal arm, because the arms are
evaluated on the same samples and their errors are correlated. Bootstrap the
DIFFERENCE, paired, resampling by GROUP.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import N_BOOTSTRAP, SEED


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins.

    Worth reporting: fusion models routinely win AUROC and lose calibration,
    which is a finding rather than a footnote.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auroc": auroc(y, p), "auprc": auprc(y, p), "ece": ece(y, p)}


def bootstrap_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    metric=auroc,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap CI for metric(p_a) - metric(p_b).

    Resamples GROUPS with replacement, not rows: rows from one parent image are
    not independent, and row-level resampling produces CIs that are far too
    narrow. Both arms are scored on the identical resample, which is what makes
    it paired and what removes the shared fold noise.

    `p_a` is the challenger (fusion), `p_b` the incumbent (best unimodal).
    A CI excluding zero is the result.
    """
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    rng = np.random.default_rng(seed)

    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}

    deltas = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue  # degenerate resample, both metrics undefined
        deltas.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))

    deltas = np.asarray(deltas)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(metric(y, p_a) - metric(y, p_b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_delta_gt_0": float((deltas > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
        "n_boot_valid": int(len(deltas)),
    }


def aggregate_folds(per_fold: list[dict[str, float]]) -> dict[str, str]:
    """mean +/- std across folds, formatted for the results table."""
    keys = per_fold[0].keys()
    out = {}
    for k in keys:
        v = np.array([f[k] for f in per_fold], float)
        out[k] = f"{v.mean():.3f} +/- {v.std():.3f}"
    return out

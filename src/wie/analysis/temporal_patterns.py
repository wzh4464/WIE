"""RQ2: temporal influence-pattern classification (paper Sec 5.2 / App. patterns).

Given a per-epoch window-level influence matrix ``M[i, e]`` (one row per training
sample ``i``, one column per epoch ``e`` -- exactly the output of the
``wie_all_epochs`` calculator), classify each sample's *trajectory* into one of
four qualitative temporal roles the paper reports (Figure 3 / Table 5):

- **Stable** -- contributes consistently throughout; no significant trend and few
  sign flips (the "generalization backbone", dominant class).
- **Early Influencer** -- shapes decision boundaries early then fades: a
  significant *decreasing* trend.
- **Late Bloomer** -- contributes mainly near convergence: a significant
  *increasing* trend.
- **Highly Fluctuating** -- oscillates in sign across epochs; often indicates
  unstable/outlying or mislabeled data (the actionable class motivating RQ3).

Procedure (Sec 5.2): *"compute each example's window-level influence at every
epoch ..., standardize within each epoch to remove global loss decay, and
classify trajectories by OLS trend slope, p-value, and sign-flip count."* The
paper reports the resulting distribution but does not pin exact numeric
thresholds, so the decision cut-offs here are parameters with documented
defaults; tune ``p_threshold`` / ``flip_ratio_threshold`` to reproduce a
particular table.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

STABLE = "Stable"
EARLY = "Early Influencer"
LATE = "Late Bloomer"
FLUCTUATING = "Highly Fluctuating"

# Stable-dominant order (matches the paper's Table 5 column order).
PATTERN_LABELS: List[str] = [STABLE, EARLY, LATE, FLUCTUATING]


def load_influence_matrix(csv_path: str) -> np.ndarray:
    """Load a ``wie_all_epochs`` per-epoch influence CSV into ``M[i, e]``.

    The CSV (written by ``wie.infl.save_results`` for a per-epoch list) has a
    ``sample_idx`` column plus one ``influence_epoch_{e}`` column per epoch.
    Returns a ``(n_samples, n_epochs)`` array with columns ordered by epoch.
    """
    df = pd.read_csv(csv_path)
    epoch_cols = [c for c in df.columns if c.startswith("influence_epoch_")]
    if not epoch_cols:
        raise ValueError(
            f"{csv_path} has no 'influence_epoch_*' columns; expected a "
            "wie_all_epochs per-epoch influence CSV."
        )
    epoch_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    return np.stack([df[c].to_numpy(dtype=np.float64) for c in epoch_cols], axis=1)


def standardize_per_epoch(M: np.ndarray) -> np.ndarray:
    """Z-score each epoch (column) across samples to remove global loss decay.

    Columns with zero variance map to all-zeros (every sample is at the epoch
    mean, so the standardized contribution is zero).
    """
    M = np.asarray(M, dtype=np.float64)
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    safe = np.where(std > 0, std, 1.0)
    Z = (M - mean) / safe
    Z[:, (std.ravel() == 0)] = 0.0
    return Z


def _ols_trend(y: np.ndarray) -> Tuple[float, float, float]:
    """OLS trend of ``y`` vs epoch index: ``(slope, p-value, R^2)``.

    ``R^2`` (the squared Pearson r of the linear fit) is the *effect size* of the
    trend -- the fraction of the trajectory's variance the straight line
    explains. On long trajectories (many epochs) the slope p-value alone is a
    poor trend test: almost any faint monotone drift is "significant", so a
    pure-p classifier labels nearly every sample Early/Late and starves the
    Stable class. Gating additionally on ``R^2`` (see ``min_r2`` in
    :func:`classify_patterns`) restores the paper's Stable-dominant taxonomy.

    Returns ``(0.0, 1.0, 0.0)`` (no trend) when there are fewer than 3 points or
    ``y`` is constant, where the slope test is undefined.
    """
    n = len(y)
    if n < 3 or np.allclose(y, y[0]):
        return 0.0, 1.0, 0.0
    from scipy import stats

    x = np.arange(n, dtype=np.float64)
    res = stats.linregress(x, y)
    slope = float(res.slope)
    pval = float(res.pvalue)
    if not np.isfinite(pval):
        pval = 1.0
    r2 = float(res.rvalue) ** 2 if np.isfinite(res.rvalue) else 0.0
    return slope, pval, r2


def _ols_slope_pvalue(y: np.ndarray) -> Tuple[float, float]:
    """OLS slope and two-sided p-value of ``y`` vs epoch index.

    Backward-compatible thin wrapper over :func:`_ols_trend` (drops ``R^2``).
    """
    slope, pval, _ = _ols_trend(y)
    return slope, pval


def _sign_flip_count(y: np.ndarray) -> int:
    """Number of sign changes along ``y`` (zeros are ignored, not counted)."""
    signs = np.sign(y)
    nz = signs[signs != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def classify_patterns(
    M: np.ndarray,
    p_threshold: float = 0.05,
    flip_ratio_threshold: float = 0.5,
    slope_eps: float = 0.0,
    standardize: bool = True,
    min_r2: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Classify every sample's temporal influence trajectory.

    Parameters
    ----------
    M : array ``(n_samples, n_epochs)``
        Per-epoch window-level influence (e.g. from ``wie_all_epochs``).
    p_threshold : float
        OLS slope p-value below which a monotone trend is deemed significant
        (Early/Late); otherwise Stable.
    flip_ratio_threshold : float
        Fraction of adjacent-epoch sign flips at or above which a trajectory is
        classed Highly Fluctuating (checked first, as it overrides a trend).
    slope_eps : float
        Minimum |slope| (on the standardized trajectory) required alongside
        significance to call a trend; ``0`` relies on the p-value alone.
    standardize : bool
        Z-score within each epoch before classifying (paper's step to remove
        global loss decay). Set ``False`` to classify raw influence.
    min_r2 : float
        Minimum trend *effect size* (``R^2`` of the OLS fit) required alongside
        significance to call an Early/Late trend; weakly-explained (small-``R^2``)
        drifts fall back to Stable. ``0`` (default) disables the gate and relies
        on the p-value alone. On long (many-epoch) trajectories a pure-p test is
        degenerate -- nearly every sample is a "significant" faint trend, which
        starves the Stable class; a strong-trend cut (``R^2`` ~ 0.7) restores the
        paper's Stable-dominant, Late>Early Table-5 structure.

    Returns
    -------
    (labels, stats) : (np.ndarray[str], pandas.DataFrame)
        ``labels[i]`` is one of :data:`PATTERN_LABELS`; ``stats`` has one row per
        sample with columns ``slope, pvalue, sign_flips, flip_ratio, label``.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D (n_samples, n_epochs), got shape {M.shape}")
    n, E = M.shape
    Z = standardize_per_epoch(M) if standardize else M

    denom = max(E - 1, 1)
    labels = np.empty(n, dtype=object)
    slopes = np.zeros(n)
    pvals = np.ones(n)
    r2s = np.zeros(n)
    flips = np.zeros(n, dtype=int)

    for i in range(n):
        row = Z[i]
        f = _sign_flip_count(row)
        flips[i] = f
        slope, pval, r2 = _ols_trend(row)
        slopes[i] = slope
        pvals[i] = pval
        r2s[i] = r2
        flip_ratio = f / denom
        if E >= 3 and flip_ratio >= flip_ratio_threshold:
            labels[i] = FLUCTUATING
        elif pval < p_threshold and abs(slope) > slope_eps and r2 >= min_r2:
            labels[i] = EARLY if slope < 0 else LATE
        else:
            labels[i] = STABLE

    stats_df = pd.DataFrame(
        {
            "sample_idx": np.arange(n),
            "slope": slopes,
            "pvalue": pvals,
            "r_squared": r2s,
            "sign_flips": flips,
            "flip_ratio": flips / denom,
            "label": labels.astype(str),
        }
    )
    return labels.astype(str), stats_df


def pattern_distribution(labels: np.ndarray, as_percent: bool = True) -> dict:
    """Distribution of labels over :data:`PATTERN_LABELS` (Table-5 style).

    Always includes all four labels (zero for absent ones). Percentages when
    ``as_percent`` (default), else raw counts.
    """
    labels = np.asarray(labels).astype(str)
    total = len(labels)
    out = {}
    for lab in PATTERN_LABELS:
        c = int(np.sum(labels == lab))
        out[lab] = (100.0 * c / total if total else 0.0) if as_percent else c
    return out


def mean_trajectories(
    M: np.ndarray, labels: np.ndarray, standardize: bool = True
) -> Optional[pd.DataFrame]:
    """Mean (standardized) trajectory per pattern, for a Figure-3-style plot.

    Returns a DataFrame indexed by epoch with one column per present pattern, or
    ``None`` if there are no samples.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return None
    Z = standardize_per_epoch(M) if standardize else M
    labels = np.asarray(labels).astype(str)
    cols = {}
    for lab in PATTERN_LABELS:
        mask = labels == lab
        if mask.any():
            cols[lab] = Z[mask].mean(axis=0)
    if not cols:
        return None
    return pd.DataFrame(cols, index=np.arange(Z.shape[1]))

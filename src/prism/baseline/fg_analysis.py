from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from prism.model import GridDistribution
from prism.model._typing import EPS


@dataclass(frozen=True, slots=True)
class FgAnalysisSummary:
    entropy_mean: float
    entropy_median: float
    entropy_p95: float
    peak_count_mean: float
    sharpness_mean: float
    entropy_expression_spearman: float
    low_expression_entropy_mean: float
    high_expression_entropy_mean: float
    entropy_sharpness_spearman: float
    entropy_peak_spearman: float
    hvg_spearman: float
    overlap_at_100: float
    overlap_at_500: float


def fg_entropy(prior: GridDistribution) -> np.ndarray:
    weights = np.asarray(prior.weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    weights = np.clip(weights, EPS, None)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return -np.sum(weights * np.log(weights), axis=-1)


def fg_peak_count(prior: GridDistribution, threshold_rel: float = 0.05) -> np.ndarray:
    weights = np.asarray(prior.weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    peak_counts = np.zeros(weights.shape[0], dtype=np.int64)
    for i, row in enumerate(weights):
        cutoff = float(np.max(row)) * float(threshold_rel)
        count = 0
        for j in range(1, row.shape[0] - 1):
            if row[j] >= cutoff and row[j] >= row[j - 1] and row[j] >= row[j + 1]:
                if row[j] > row[j - 1] or row[j] > row[j + 1]:
                    count += 1
        if row.shape[0] > 0 and np.max(row) > 0 and count == 0:
            count = 1
        peak_counts[i] = count
    return peak_counts.astype(float)


def fg_sharpness(prior: GridDistribution) -> np.ndarray:
    weights = np.asarray(prior.weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    log_w = np.log(np.clip(weights, EPS, None))
    second = np.diff(log_w, n=2, axis=-1)
    if second.shape[-1] == 0:
        return np.zeros(weights.shape[0], dtype=float)
    return np.mean(np.abs(second), axis=-1)


def hvg_rank_consistency(
    fg_entropy_values: np.ndarray,
    hvg_scores: np.ndarray,
) -> dict[str, float]:
    entropy = np.asarray(fg_entropy_values, dtype=float).reshape(-1)
    hvg = np.asarray(hvg_scores, dtype=float).reshape(-1)
    if entropy.size != hvg.size:
        raise ValueError("fg_entropy and hvg_scores must have the same length")
    rho = float(np.asarray(stats.spearmanr(entropy, hvg))[0])
    return {
        "spearman_rho": 0.0 if not np.isfinite(rho) else rho,
        "overlap_at_100": _rank_overlap(entropy, hvg, 100),
        "overlap_at_500": _rank_overlap(entropy, hvg, 500),
    }


def fg_entropy_vs_expression(
    fg_entropy_values: np.ndarray,
    mean_expression: np.ndarray,
) -> dict[str, float]:
    entropy = np.asarray(fg_entropy_values, dtype=float).reshape(-1)
    expr = np.asarray(mean_expression, dtype=float).reshape(-1)
    if entropy.size != expr.size:
        raise ValueError("fg_entropy and mean_expression must have the same length")
    rho = float(np.asarray(stats.spearmanr(entropy, expr))[0])
    order = np.argsort(expr)
    low_idx = order[: max(1, entropy.size // 5)]
    high_idx = order[-max(1, entropy.size // 5) :]
    return {
        "spearman_rho": 0.0 if not np.isfinite(rho) else rho,
        "low_expression_entropy_mean": float(np.mean(entropy[low_idx])),
        "high_expression_entropy_mean": float(np.mean(entropy[high_idx])),
    }


def summarize_fg_analysis(
    priors: GridDistribution,
    mean_expression: np.ndarray,
    hvg_scores: np.ndarray,
) -> FgAnalysisSummary:
    entropy = fg_entropy(priors)
    peaks = fg_peak_count(priors)
    sharpness = fg_sharpness(priors)
    expr_summary = fg_entropy_vs_expression(entropy, mean_expression)
    rank_summary = hvg_rank_consistency(entropy, hvg_scores)
    peak_rho = float(np.asarray(stats.spearmanr(entropy, peaks))[0])
    sharpness_rho = float(np.asarray(stats.spearmanr(entropy, sharpness))[0])
    return FgAnalysisSummary(
        entropy_mean=float(np.mean(entropy)),
        entropy_median=float(np.median(entropy)),
        entropy_p95=float(np.quantile(entropy, 0.95)),
        peak_count_mean=float(np.mean(peaks)),
        sharpness_mean=float(np.mean(sharpness)),
        entropy_expression_spearman=float(expr_summary["spearman_rho"]),
        low_expression_entropy_mean=float(expr_summary["low_expression_entropy_mean"]),
        high_expression_entropy_mean=float(
            expr_summary["high_expression_entropy_mean"]
        ),
        entropy_sharpness_spearman=0.0
        if not np.isfinite(sharpness_rho)
        else sharpness_rho,
        entropy_peak_spearman=0.0 if not np.isfinite(peak_rho) else peak_rho,
        hvg_spearman=float(rank_summary["spearman_rho"]),
        overlap_at_100=float(rank_summary["overlap_at_100"]),
        overlap_at_500=float(rank_summary["overlap_at_500"]),
    )


def _rank_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(int(k), a.size, b.size)
    if k <= 0:
        return 0.0
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return float(len(top_a & top_b) / k)

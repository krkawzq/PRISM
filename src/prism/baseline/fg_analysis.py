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
    inflection_count_mean: float
    sharpness_mean: float
    entropy_expression_spearman: float
    low_expression_entropy_mean: float
    high_expression_entropy_mean: float
    entropy_sharpness_spearman: float
    entropy_peak_spearman: float
    entropy_inflection_spearman: float
    hvg_spearman: float
    structure_hvg_spearman: float
    overlap_at_100: float
    overlap_at_500: float
    structure_overlap_at_100: float
    structure_overlap_at_500: float


@dataclass(frozen=True, slots=True)
class HvgConsistencyResult:
    traditional_scores: np.ndarray
    entropy_scores: np.ndarray
    structure_scores: np.ndarray
    traditional_rank: np.ndarray
    entropy_rank: np.ndarray
    structure_rank: np.ndarray
    spearman_trad_vs_entropy: float
    spearman_trad_vs_structure: float
    overlap: dict[int, dict[str, float]]
    traditional_only: list[str]
    entropy_only: list[str]
    structure_only: list[str]
    peak_counts: np.ndarray
    inflection_counts: np.ndarray


def fg_entropy(prior: GridDistribution) -> np.ndarray:
    weights = _weights_2d(prior)
    weights = np.clip(weights, EPS, None)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return -np.sum(weights * np.log(weights), axis=-1)


def fg_peak_count(prior: GridDistribution) -> np.ndarray:
    weights = _weights_2d(prior)
    if weights.shape[1] < 3:
        return np.zeros(weights.shape[0], dtype=float)
    left = weights[:, :-2]
    center = weights[:, 1:-1]
    right = weights[:, 2:]
    peaks = (center > left) & (center > right)
    return np.sum(peaks, axis=-1).astype(float)


def fg_inflection_count(prior: GridDistribution) -> np.ndarray:
    weights = _weights_2d(prior)
    if weights.shape[1] < 4:
        return np.zeros(weights.shape[0], dtype=float)
    log_w = np.log(np.clip(weights, EPS, None))
    d2 = log_w[:, :-2] - 2.0 * log_w[:, 1:-1] + log_w[:, 2:]
    signs = np.sign(d2)
    sign_changes = (signs[:, 1:] * signs[:, :-1]) < 0
    return np.sum(sign_changes, axis=-1).astype(float)


def fg_sharpness(prior: GridDistribution) -> np.ndarray:
    weights = _weights_2d(prior)
    log_w = np.log(np.clip(weights, EPS, None))
    second = np.diff(log_w, n=2, axis=-1)
    if second.shape[-1] == 0:
        return np.zeros(weights.shape[0], dtype=float)
    return np.mean(np.abs(second), axis=-1)


def binned_zscore_variance(
    counts: np.ndarray,
    totals: np.ndarray,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    counts_np = np.asarray(counts, dtype=float)
    totals_np = np.asarray(totals, dtype=float).reshape(-1)
    normed = counts_np / np.maximum(totals_np[:, None], EPS) * np.median(totals_np)
    mean = np.mean(normed, axis=0)
    var = np.var(normed, axis=0)
    order = np.argsort(np.argsort(mean))
    bin_idx = np.minimum((order * n_bins) // max(len(mean), 1), n_bins - 1)
    scores = np.zeros_like(mean, dtype=float)
    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) < 2:
            continue
        bin_var = var[mask]
        mu = float(np.mean(bin_var))
        sigma = float(np.std(bin_var))
        if sigma < EPS:
            continue
        scores[mask] = (bin_var - mu) / sigma
    return scores, mean


def hvg_consistency_analysis(
    counts: np.ndarray,
    totals: np.ndarray,
    prior: GridDistribution,
    gene_names: list[str],
    top_k_list: list[int] | None = None,
    *,
    alpha: float = 0.5,
    beta: float = 0.2,
) -> HvgConsistencyResult:
    if top_k_list is None:
        top_k_list = [100, 200, 500, 1000]
    traditional_scores, _ = binned_zscore_variance(counts, totals)
    entropy_scores = fg_entropy(prior)
    peak_counts = fg_peak_count(prior)
    inflection_counts = fg_inflection_count(prior)
    structure_scores = entropy_scores * (
        1.0 + alpha * peak_counts + beta * inflection_counts
    )

    trad_rank = np.argsort(traditional_scores)[::-1]
    entropy_rank = np.argsort(entropy_scores)[::-1]
    structure_rank = np.argsort(structure_scores)[::-1]

    rho_te = _safe_spearman(traditional_scores, entropy_scores)
    rho_ts = _safe_spearman(traditional_scores, structure_scores)
    overlap: dict[int, dict[str, float]] = {}
    for k in top_k_list:
        kk = min(max(int(k), 1), len(gene_names))
        overlap[kk] = {
            "trad_entropy": _rank_overlap(trad_rank, entropy_rank, kk),
            "trad_structure": _rank_overlap(trad_rank, structure_rank, kk),
        }

    base_k = min(500, len(gene_names))
    top_trad = set(trad_rank[:base_k].tolist())
    top_entropy = set(entropy_rank[:base_k].tolist())
    top_structure = set(structure_rank[:base_k].tolist())
    traditional_only = [
        gene_names[idx] for idx in trad_rank if idx in (top_trad - top_entropy)
    ][:30]
    entropy_only = [
        gene_names[idx] for idx in entropy_rank if idx in (top_entropy - top_trad)
    ][:30]
    structure_only = [
        gene_names[idx] for idx in structure_rank if idx in (top_structure - top_trad)
    ][:30]

    return HvgConsistencyResult(
        traditional_scores=traditional_scores,
        entropy_scores=entropy_scores,
        structure_scores=structure_scores,
        traditional_rank=trad_rank,
        entropy_rank=entropy_rank,
        structure_rank=structure_rank,
        spearman_trad_vs_entropy=rho_te,
        spearman_trad_vs_structure=rho_ts,
        overlap=overlap,
        traditional_only=traditional_only,
        entropy_only=entropy_only,
        structure_only=structure_only,
        peak_counts=peak_counts,
        inflection_counts=inflection_counts,
    )


def fg_entropy_vs_expression(
    fg_entropy_values: np.ndarray,
    mean_expression: np.ndarray,
) -> dict[str, float]:
    entropy = np.asarray(fg_entropy_values, dtype=float).reshape(-1)
    expr = np.asarray(mean_expression, dtype=float).reshape(-1)
    rho = _safe_spearman(entropy, expr)
    order = np.argsort(expr)
    low_idx = order[: max(1, entropy.size // 5)]
    high_idx = order[-max(1, entropy.size // 5) :]
    return {
        "spearman_rho": rho,
        "low_expression_entropy_mean": float(np.mean(entropy[low_idx])),
        "high_expression_entropy_mean": float(np.mean(entropy[high_idx])),
    }


def summarize_fg_analysis(
    priors: GridDistribution,
    mean_expression: np.ndarray,
    hvg_scores: np.ndarray,
    *,
    structure_scores: np.ndarray | None = None,
) -> FgAnalysisSummary:
    entropy = fg_entropy(priors)
    peaks = fg_peak_count(priors)
    inflections = fg_inflection_count(priors)
    sharpness = fg_sharpness(priors)
    expr_summary = fg_entropy_vs_expression(entropy, mean_expression)
    hvg_rho = _safe_spearman(entropy, hvg_scores)
    peak_rho = _safe_spearman(entropy, peaks)
    sharpness_rho = _safe_spearman(entropy, sharpness)
    inflection_rho = _safe_spearman(entropy, inflections)
    if structure_scores is None:
        structure_scores = entropy * (1.0 + 0.5 * peaks + 0.2 * inflections)
    structure_hvg_rho = _safe_spearman(structure_scores, hvg_scores)
    return FgAnalysisSummary(
        entropy_mean=float(np.mean(entropy)),
        entropy_median=float(np.median(entropy)),
        entropy_p95=float(np.quantile(entropy, 0.95)),
        peak_count_mean=float(np.mean(peaks)),
        inflection_count_mean=float(np.mean(inflections)),
        sharpness_mean=float(np.mean(sharpness)),
        entropy_expression_spearman=float(expr_summary["spearman_rho"]),
        low_expression_entropy_mean=float(expr_summary["low_expression_entropy_mean"]),
        high_expression_entropy_mean=float(
            expr_summary["high_expression_entropy_mean"]
        ),
        entropy_sharpness_spearman=sharpness_rho,
        entropy_peak_spearman=peak_rho,
        entropy_inflection_spearman=inflection_rho,
        hvg_spearman=hvg_rho,
        structure_hvg_spearman=structure_hvg_rho,
        overlap_at_100=_score_overlap(entropy, hvg_scores, 100),
        overlap_at_500=_score_overlap(entropy, hvg_scores, 500),
        structure_overlap_at_100=_score_overlap(structure_scores, hvg_scores, 100),
        structure_overlap_at_500=_score_overlap(structure_scores, hvg_scores, 500),
    )


def _weights_2d(prior: GridDistribution) -> np.ndarray:
    weights = np.asarray(prior.weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    return weights


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = float(np.asarray(stats.spearmanr(a, b))[0])
    return 0.0 if not np.isfinite(rho) else rho


def _rank_overlap(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> float:
    top_a = set(rank_a[:k].tolist())
    top_b = set(rank_b[:k].tolist())
    return float(len(top_a & top_b) / max(k, 1))


def _score_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    kk = min(int(k), len(a), len(b))
    if kk <= 0:
        return 0.0
    return _rank_overlap(np.argsort(a)[::-1], np.argsort(b)[::-1], kk)

from __future__ import annotations

import numpy as np
from scipy import stats

from ..model import DTYPE_NP, EPS

__all__ = [
    "auroc_one_vs_rest",
    "depth_mutual_information",
    "depth_correlation",
    "dropout_recovery_rate",
    "evaluate_representations",
    "fisher_ratio",
    "gene_pair_correlation",
    "kruskal_wallis_stat",
    "log1p_normalize_total",
    "normalize_total",
    "raw_umi",
    "sparsity_signal_correlation",
    "treatment_conditional_cv",
    "zero_imputation_rank_preservation",
    "zero_group_consistency",
]


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须全部为有限值")
    return array


def _as_labels(labels: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} 不能为空")
    return array


def _check_same_length(**arrays: np.ndarray) -> None:
    lengths = {name: int(np.asarray(value).shape[0]) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        parts = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"输入长度不一致: {parts}")


def _as_matrix(counts: np.ndarray, *, name: str = "counts") -> np.ndarray:
    array = np.asarray(counts, dtype=DTYPE_NP)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须为二维，收到 shape={array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} 不能为空矩阵，收到 shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须全部为有限值")
    return array


def fisher_ratio(values: np.ndarray, labels: np.ndarray) -> float:
    """计算单基因在已知分组下的 Fisher ratio。"""
    values_np = _as_vector(values, name="values")
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, labels=labels_np)

    unique_labels = np.unique(labels_np)
    if unique_labels.size < 2:
        return 0.0

    grand_mean = float(np.mean(values_np))
    group_means: list[float] = []
    group_vars: list[float] = []
    group_sizes: list[int] = []

    for label in unique_labels:
        group = values_np[labels_np == label]
        if group.size < 2:
            continue
        group_means.append(float(np.mean(group)))
        group_vars.append(float(np.var(group, ddof=1)))
        group_sizes.append(int(group.size))

    if len(group_means) < 2:
        return 0.0

    group_means_np = np.asarray(group_means, dtype=DTYPE_NP)
    group_vars_np = np.asarray(group_vars, dtype=DTYPE_NP)
    group_sizes_np = np.asarray(group_sizes, dtype=DTYPE_NP)

    between = np.sum(group_sizes_np * (group_means_np - grand_mean) ** 2) / (
        len(group_means) - 1
    )
    within = np.average(group_vars_np, weights=group_sizes_np)

    if within < EPS:
        return float("inf") if between > EPS else 0.0
    return float(between / within)


def depth_correlation(values: np.ndarray, totals: np.ndarray) -> float:
    """计算单基因表示与测序深度的 Spearman 相关。"""
    values_np = _as_vector(values, name="values")
    totals_np = _as_vector(totals, name="totals")
    _check_same_length(values=values_np, totals=totals_np)

    if np.std(values_np) < EPS or np.std(totals_np) < EPS:
        return 0.0

    rho = float(np.asarray(stats.spearmanr(values_np, totals_np))[0])
    if not np.isfinite(rho):
        return 0.0
    return rho


def depth_mutual_information(
    values: np.ndarray, totals: np.ndarray, *, bins: int = 16
) -> float:
    values_np = _as_vector(values, name="values")
    totals_np = _as_vector(totals, name="totals")
    _check_same_length(values=values_np, totals=totals_np)
    if np.std(values_np) < EPS or np.std(totals_np) < EPS:
        return 0.0
    x_bin = _quantile_bins(values_np, bins)
    y_bin = _quantile_bins(totals_np, bins)
    return _mutual_information_from_bins(x_bin, y_bin)


def zero_group_consistency(
    values: np.ndarray,
    is_zero: np.ndarray,
    labels: np.ndarray,
) -> float:
    """评估原始零观测细胞在组内的去噪一致性。"""
    values_np = _as_vector(values, name="values")
    zero_mask = np.asarray(is_zero, dtype=bool).reshape(-1)
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, is_zero=zero_mask, labels=labels_np)

    if not np.any(zero_mask):
        return float("nan")

    cvs: list[float] = []
    for label in np.unique(labels_np):
        group = values_np[zero_mask & (labels_np == label)]
        if group.size < 2:
            continue

        mean_value = float(np.mean(group))
        if abs(mean_value) < EPS:
            continue

        cvs.append(float(np.std(group, ddof=1) / abs(mean_value)))

    if not cvs:
        return float("nan")

    return float(np.clip(1.0 - float(np.mean(cvs)), 0.0, 1.0))


def kruskal_wallis_stat(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    values_np = _as_vector(values, name="values")
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, labels=labels_np)
    groups = [values_np[labels_np == label] for label in np.unique(labels_np)]
    groups = [group for group in groups if group.size > 0]
    if len(groups) < 2:
        return 0.0, 1.0
    stat, pvalue = stats.kruskal(*groups)
    stat = float(stat) if np.isfinite(stat) else 0.0
    pvalue = float(pvalue) if np.isfinite(pvalue) else 1.0
    return stat, pvalue


def auroc_one_vs_rest(values: np.ndarray, labels: np.ndarray) -> float:
    values_np = _as_vector(values, name="values")
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, labels=labels_np)
    scores: list[float] = []
    for label in np.unique(labels_np):
        pos = values_np[labels_np == label]
        neg = values_np[labels_np != label]
        if pos.size == 0 or neg.size == 0:
            continue
        ranks = stats.rankdata(np.concatenate([pos, neg]))
        pos_ranks = ranks[: pos.size]
        u_stat = float(np.sum(pos_ranks) - pos.size * (pos.size + 1) / 2.0)
        denom = float(pos.size * neg.size)
        if denom <= 0:
            continue
        scores.append(u_stat / denom)
    if not scores:
        return 0.5
    return float(np.mean(scores))


def sparsity_signal_correlation(values: np.ndarray, zero_fraction: np.ndarray) -> float:
    values_np = _as_vector(values, name="values")
    zero_fraction_np = _as_vector(zero_fraction, name="zero_fraction")
    _check_same_length(values=values_np, zero_fraction=zero_fraction_np)
    if np.std(values_np) < EPS or np.std(zero_fraction_np) < EPS:
        return 0.0
    rho = float(np.asarray(stats.spearmanr(values_np, zero_fraction_np))[0])
    return 0.0 if not np.isfinite(rho) else rho


def treatment_conditional_cv(values: np.ndarray, labels: np.ndarray) -> float:
    values_np = _as_vector(values, name="values")
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, labels=labels_np)
    cvs: list[float] = []
    for label in np.unique(labels_np):
        group = values_np[labels_np == label]
        if group.size < 2:
            continue
        mean_value = float(np.mean(group))
        if abs(mean_value) < EPS:
            continue
        cvs.append(float(np.std(group, ddof=1) / abs(mean_value)))
    if not cvs:
        return 0.0
    return float(np.mean(cvs))


def dropout_recovery_rate(
    values: np.ndarray,
    raw_counts: np.ndarray,
    labels: np.ndarray,
    *,
    detected_threshold: float = 0.9,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
) -> float | None:
    values_np = _as_vector(values, name="values")
    raw_np = _as_vector(raw_counts, name="raw_counts")
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, raw_counts=raw_np, labels=labels_np)
    recoveries: list[float] = []
    for label in np.unique(labels_np):
        mask = labels_np == label
        group_raw = raw_np[mask]
        if group_raw.size == 0:
            continue
        detected_frac = float(np.mean(group_raw > 0))
        if detected_frac < detected_threshold:
            continue
        group_values = values_np[mask]
        zero_values = group_values[group_raw <= 0]
        nonzero_values = group_values[group_raw > 0]
        if zero_values.size == 0 or nonzero_values.size < 3:
            continue
        lo = float(np.quantile(nonzero_values, lower_q))
        hi = float(np.quantile(nonzero_values, upper_q))
        recoveries.append(float(np.mean((zero_values >= lo) & (zero_values <= hi))))
    if not recoveries:
        return None
    return float(np.mean(recoveries))


def zero_imputation_rank_preservation(
    values: np.ndarray,
    raw_counts: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    values_np = _as_vector(values, name="values")
    raw_np = _as_vector(raw_counts, name="raw_counts")
    labels_np = _as_labels(labels, name="labels")
    _check_same_length(values=values_np, raw_counts=raw_np, labels=labels_np)
    taus: list[float] = []
    for label in np.unique(labels_np):
        mask = labels_np == label
        group_raw = raw_np[mask]
        group_values = values_np[mask]
        if (
            group_raw.size < 3
            or not np.any(group_raw <= 0)
            or not np.any(group_raw > 0)
        ):
            continue
        tau = float(np.asarray(stats.kendalltau(group_values, group_raw))[0])
        if np.isfinite(tau):
            taus.append(tau)
    if not taus:
        return None
    return float(np.mean(taus))


def evaluate_representations(
    representations: dict[str, np.ndarray],
    *,
    totals: np.ndarray,
    raw_counts: np.ndarray,
    labels: np.ndarray | None = None,
    zero_fraction: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    totals_np = _as_vector(totals, name="totals")
    raw_np = _as_vector(raw_counts, name="raw_counts")
    zero_fraction_np = (
        None
        if zero_fraction is None
        else _as_vector(zero_fraction, name="zero_fraction")
    )
    label_np = None if labels is None else _as_labels(labels, name="labels")
    results: dict[str, dict[str, float | None]] = {}
    for name, values in representations.items():
        values_np = _as_vector(values, name=name)
        _check_same_length(values=values_np, totals=totals_np, raw_counts=raw_np)
        item: dict[str, float | None] = {
            "mean": float(np.mean(values_np)),
            "median": float(np.median(values_np)),
            "std": float(np.std(values_np)),
            "var": float(np.var(values_np)),
            "p95": float(np.quantile(values_np, 0.95)),
            "nonzero_frac": float(np.mean(values_np > 0)),
            "depth_corr": float(depth_correlation(values_np, totals_np)),
            "depth_mi": float(depth_mutual_information(values_np, totals_np)),
            "sparsity_corr": None,
            "fisher_ratio": None,
            "kruskal_h": None,
            "kruskal_p": None,
            "auroc_ovr": None,
            "zero_consistency": None,
            "zero_rank_tau": None,
            "dropout_recovery": None,
            "treatment_cv": None,
        }
        if zero_fraction_np is not None:
            item["sparsity_corr"] = float(
                sparsity_signal_correlation(values_np, zero_fraction_np)
            )
        if label_np is not None and len(np.unique(label_np)) >= 2:
            item["fisher_ratio"] = float(fisher_ratio(values_np, label_np))
            kw_h, kw_p = kruskal_wallis_stat(values_np, label_np)
            item["kruskal_h"] = kw_h
            item["kruskal_p"] = kw_p
            item["auroc_ovr"] = float(auroc_one_vs_rest(values_np, label_np))
            zero_value = zero_group_consistency(values_np, raw_np <= 0, label_np)
            item["zero_consistency"] = (
                None if not np.isfinite(zero_value) else float(zero_value)
            )
            item["zero_rank_tau"] = zero_imputation_rank_preservation(
                values_np, raw_np, label_np
            )
            item["dropout_recovery"] = dropout_recovery_rate(
                values_np, raw_np, label_np
            )
            item["treatment_cv"] = float(treatment_conditional_cv(values_np, label_np))
        results[name] = item
    return results


def gene_pair_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """计算两个基因表示之间的 Pearson 相关。"""
    values_a_np = _as_vector(values_a, name="values_a")
    values_b_np = _as_vector(values_b, name="values_b")
    _check_same_length(values_a=values_a_np, values_b=values_b_np)

    if np.std(values_a_np) < EPS or np.std(values_b_np) < EPS:
        return 0.0

    corr = float(np.asarray(stats.pearsonr(values_a_np, values_b_np))[0])
    if not np.isfinite(corr):
        return 0.0
    return corr


def raw_umi(counts: np.ndarray) -> np.ndarray:
    """返回原始 UMI 计数矩阵。"""
    return _as_matrix(counts)


def normalize_total(
    counts: np.ndarray,
    totals: np.ndarray,
    target: float | None = None,
) -> np.ndarray:
    """执行 total-count normalization。"""
    counts_np = _as_matrix(counts)
    totals_np = _as_vector(totals, name="totals")
    _check_same_length(counts=counts_np, totals=totals_np)

    if np.any(totals_np < 0):
        raise ValueError("totals 不能包含负数")

    resolved_target = float(np.median(totals_np)) if target is None else float(target)
    if not np.isfinite(resolved_target) or resolved_target < 0:
        raise ValueError(f"target 必须为非负有限值，收到 {resolved_target}")

    scale = resolved_target / np.maximum(totals_np, EPS)
    return counts_np * scale[:, None]


def log1p_normalize_total(
    counts: np.ndarray,
    totals: np.ndarray,
    target: float | None = None,
) -> np.ndarray:
    """执行 `log1p(normalize_total(counts, totals))`。"""
    return np.log1p(normalize_total(counts, totals, target))


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, qs)
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def _mutual_information_from_bins(x_bin: np.ndarray, y_bin: np.ndarray) -> float:
    joint = np.stack([x_bin, y_bin], axis=1)
    joint_unique, joint_counts = np.unique(joint, axis=0, return_counts=True)
    n = float(x_bin.size)
    mi = 0.0
    x_prob = {
        int(v): float(c / n)
        for v, c in zip(*np.unique(x_bin, return_counts=True), strict=False)
    }
    y_prob = {
        int(v): float(c / n)
        for v, c in zip(*np.unique(y_bin, return_counts=True), strict=False)
    }
    for (xv, yv), count in zip(joint_unique, joint_counts, strict=False):
        p_xy = float(count / n)
        mi += p_xy * np.log(
            max(p_xy, EPS) / max(x_prob[int(xv)] * y_prob[int(yv)], EPS)
        )
    return float(max(mi, 0.0))

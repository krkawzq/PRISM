from __future__ import annotations

import numpy as np
from scipy import stats

from ..model._typing import DTYPE_NP, EPS

__all__ = [
    "depth_correlation",
    "fisher_ratio",
    "gene_pair_correlation",
    "log1p_normalize_total",
    "normalize_total",
    "raw_umi",
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

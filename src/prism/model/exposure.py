from __future__ import annotations

import numpy as np

from .constants import DTYPE_NP, EPS


def mean_reference_count(reference_counts: np.ndarray) -> float:
    values = np.asarray(reference_counts, dtype=DTYPE_NP).reshape(-1)
    if values.size == 0:
        raise ValueError("reference_counts cannot be empty")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("reference_counts must be finite and positive")
    return float(np.mean(values))


def effective_exposure(reference_counts: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    reference = np.asarray(reference_counts, dtype=DTYPE_NP).reshape(-1)
    reference_mean = mean_reference_count(reference)
    return (reference / reference_mean) * float(scale)


def scaled_observation_mean(
    counts: np.ndarray,
    reference_counts: np.ndarray,
    scale: float,
) -> np.ndarray:
    counts_np = np.asarray(counts, dtype=DTYPE_NP)
    reference_np = np.asarray(reference_counts, dtype=DTYPE_NP).reshape(-1)
    return counts_np * float(scale) / max(mean_reference_count(reference_np), EPS)


def validate_binomial_observations(
    counts: np.ndarray,
    effective_exposure_values: np.ndarray,
) -> None:
    counts_np = np.asarray(counts, dtype=DTYPE_NP)
    exposure_np = np.asarray(effective_exposure_values, dtype=DTYPE_NP).reshape(-1)
    if counts_np.ndim != 2:
        raise ValueError(f"counts must be 2D, got shape={counts_np.shape}")
    if counts_np.shape[0] != exposure_np.shape[0]:
        raise ValueError(
            "counts and effective_exposure must agree on the observation axis, "
            f"got {counts_np.shape[0]} != {exposure_np.shape[0]}"
        )
    max_counts = np.max(counts_np, axis=1)
    invalid_rows = np.flatnonzero(max_counts > (exposure_np + 1e-9))
    if invalid_rows.size == 0:
        return
    row_idx = int(invalid_rows[0])
    raise ValueError(
        "binomial likelihood requires every observation to satisfy "
        "counts <= effective_exposure; "
        f"found row={row_idx} with max_count={max_counts[row_idx]:.6g} "
        f"and effective_exposure={exposure_np[row_idx]:.6g}. "
        "Increase scale or use negative_binomial/poisson."
    )


__all__ = [
    "effective_exposure",
    "mean_reference_count",
    "scaled_observation_mean",
    "validate_binomial_observations",
]

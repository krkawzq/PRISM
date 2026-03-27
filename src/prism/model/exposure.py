from __future__ import annotations

import numpy as np

from .constants import DTYPE_NP


def mean_reference_count(reference_counts: np.ndarray) -> float:
    values = np.asarray(reference_counts, dtype=DTYPE_NP).reshape(-1)
    if values.size == 0:
        raise ValueError("reference_counts cannot be empty")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("reference_counts must be finite and positive")
    return float(np.mean(values))


def effective_exposure(reference_counts: np.ndarray, S: float) -> np.ndarray:
    if not np.isfinite(S) or S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    reference = np.asarray(reference_counts, dtype=DTYPE_NP).reshape(-1)
    ref_mean = mean_reference_count(reference)
    return (reference / ref_mean) * float(S)


def ratio_observation_mean(
    counts: np.ndarray, reference_counts: np.ndarray, S: float
) -> np.ndarray:
    counts_np = np.asarray(counts, dtype=DTYPE_NP)
    reference_np = np.asarray(reference_counts, dtype=DTYPE_NP).reshape(-1)
    return counts_np * float(S) / max(mean_reference_count(reference_np), 1e-12)

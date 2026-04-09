from __future__ import annotations

import numpy as np

from .types import PoolEstimate, ScaleDiagnostic


def summarize_reference_scale(reference_counts: np.ndarray) -> ScaleDiagnostic:
    values = np.asarray(reference_counts, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("reference_counts cannot be empty")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("reference_counts must be finite and positive")
    return ScaleDiagnostic(
        mean_reference_count=float(np.mean(values)),
        suggested_scale=float(np.mean(values)),
        upper_quantile_scale=float(np.quantile(values, 0.9)),
    )


def fit_pool_scale(reference_counts: np.ndarray, **_: object) -> PoolEstimate:
    diagnostic = summarize_reference_scale(reference_counts)
    return PoolEstimate(point_scale=float(diagnostic.suggested_scale))


__all__ = [
    "fit_pool_scale",
    "summarize_reference_scale",
]

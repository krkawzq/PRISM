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
        median_reference_count=float(np.median(values)),
        suggested_S=float(np.mean(values)),
        lower_quantile_S=float(np.quantile(values, 0.1)),
        upper_quantile_S=float(np.quantile(values, 0.9)),
    )


class PoolFitReport(PoolEstimate):
    pass


def fit_pool_scale(reference_counts: np.ndarray, **_: object) -> PoolEstimate:
    diag = summarize_reference_scale(reference_counts)
    point_mu = float(np.log(max(diag.suggested_S, 1e-12)))
    return PoolEstimate(
        mu=point_mu,
        sigma=float(
            np.std(
                np.log(
                    np.clip(np.asarray(reference_counts, dtype=np.float64), 1e-12, None)
                )
            )
        ),
        point_mu=point_mu,
        point_eta=float(diag.suggested_S),
        used_posterior_softargmax=False,
    )


def fit_pool_scale_report(
    reference_counts: np.ndarray, **kwargs: object
) -> PoolFitReport:
    estimate = fit_pool_scale(reference_counts, **kwargs)
    return PoolFitReport(
        mu=float(estimate.mu),
        sigma=float(estimate.sigma),
        point_mu=float(estimate.point_mu),
        point_eta=float(estimate.point_eta),
        used_posterior_softargmax=bool(estimate.used_posterior_softargmax),
    )

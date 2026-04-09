from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .prior_curves import PriorCurve


def resolve_x_values(curve: "PriorCurve", *, x_axis: str) -> np.ndarray:
    if x_axis == "support":
        return np.asarray(curve.support, dtype=np.float64)
    if x_axis == "scaled":
        return np.asarray(curve.scaled_support, dtype=np.float64)
    if x_axis == "rate":
        if curve.support_domain != "rate":
            raise ValueError("x_axis='rate' is only valid for rate-domain priors")
        return np.asarray(curve.support, dtype=np.float64)
    raise ValueError(f"unsupported x_axis: {x_axis}")


def resolve_curve_y_values(curve: "PriorCurve", *, curve_mode: str) -> np.ndarray:
    if curve_mode == "density":
        return np.asarray(curve.prior_probabilities, dtype=np.float64)
    if curve_mode == "cdf":
        probabilities = np.asarray(curve.prior_probabilities, dtype=np.float64)
        total = max(float(np.sum(probabilities)), 1e-12)
        return np.cumsum(probabilities / total)
    raise ValueError(f"unsupported curve_mode: {curve_mode}")


def summarize_prior_curve(curve: "PriorCurve") -> dict[str, float]:
    support = np.asarray(curve.support, dtype=np.float64)
    scaled_support = np.asarray(curve.scaled_support, dtype=np.float64)
    probabilities = np.asarray(curve.prior_probabilities, dtype=np.float64)
    total = max(float(np.sum(probabilities)), 1e-12)
    normalized = probabilities / total
    map_idx = int(np.argmax(normalized))
    entropy = float(-np.sum(normalized * np.log(np.clip(normalized, 1e-12, None))))
    return {
        "mean_support": float(np.sum(support * normalized)),
        "mean_scaled_support": float(np.sum(scaled_support * normalized)),
        "map_support": float(support[map_idx]),
        "map_scaled_support": float(scaled_support[map_idx]),
        "entropy": entropy,
        "scale": float(curve.scale),
    }


def format_curve_stats(curve: "PriorCurve", *, fields: tuple[str, ...]) -> str:
    if not fields:
        return ""
    summary = summarize_prior_curve(curve)
    return ", ".join(f"{field}={summary[field]:.3g}" for field in fields)


def display_cutoff(
    x_values: np.ndarray, probabilities: np.ndarray, quantile: float
) -> float:
    x_np = np.asarray(x_values, dtype=np.float64).reshape(-1)
    p_np = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if x_np.size == 0:
        return 1.0
    mass = max(float(np.sum(p_np)), 1e-12)
    cdf = np.cumsum(p_np / mass)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), x_np.size - 1)
    return float(x_np[idx])


__all__ = [
    "display_cutoff",
    "format_curve_stats",
    "resolve_curve_y_values",
    "resolve_x_values",
    "summarize_prior_curve",
]

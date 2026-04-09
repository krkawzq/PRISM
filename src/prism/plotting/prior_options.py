from __future__ import annotations

from ._shared import dedupe_names

SUPPORTED_CURVE_MODES = ("density", "cdf")
SUPPORTED_Y_SCALES = ("linear", "log")
SUPPORTED_STAT_FIELDS = (
    "mean_support",
    "mean_scaled_support",
    "map_support",
    "map_scaled_support",
    "entropy",
    "scale",
)


def resolve_x_axis(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"scaled", "support", "rate"}:
        raise ValueError("x_axis must be one of: scaled, support, rate")
    return resolved


def resolve_curve_mode(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in SUPPORTED_CURVE_MODES:
        raise ValueError(
            "curve_mode must be one of: " + ", ".join(SUPPORTED_CURVE_MODES)
        )
    return resolved


def resolve_y_scale(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in SUPPORTED_Y_SCALES:
        raise ValueError("y_scale must be one of: " + ", ".join(SUPPORTED_Y_SCALES))
    return resolved


def resolve_stat_fields(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    resolved = dedupe_names(values)
    unknown = [value for value in resolved if value not in SUPPORTED_STAT_FIELDS]
    if unknown:
        raise ValueError(
            "unknown stats fields: "
            + ", ".join(unknown)
            + "; supported: "
            + ", ".join(SUPPORTED_STAT_FIELDS)
        )
    return tuple(resolved)


__all__ = [
    "SUPPORTED_CURVE_MODES",
    "SUPPORTED_STAT_FIELDS",
    "SUPPORTED_Y_SCALES",
    "resolve_curve_mode",
    "resolve_stat_fields",
    "resolve_x_axis",
    "resolve_y_scale",
]

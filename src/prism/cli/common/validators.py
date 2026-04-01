from __future__ import annotations

import numpy as np


def normalize_choice(
    value: str,
    *,
    supported: tuple[str, ...],
    option_name: str,
) -> str:
    resolved = value.strip().lower()
    if resolved not in supported:
        raise ValueError(f"{option_name} must be one of: {', '.join(supported)}")
    return resolved


def resolve_numpy_dtype(value: str, *, option_name: str = "--dtype") -> np.dtype:
    resolved = normalize_choice(
        value,
        supported=("float32", "float64"),
        option_name=option_name,
    )
    return np.dtype(np.float32 if resolved == "float32" else np.float64)


def resolve_prior_source(value: str, *, option_name: str = "--prior-source") -> str:
    return normalize_choice(
        value,
        supported=("global", "label"),
        option_name=option_name,
    )


__all__ = ["normalize_choice", "resolve_numpy_dtype", "resolve_prior_source"]

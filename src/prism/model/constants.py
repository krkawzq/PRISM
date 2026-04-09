from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np

EPS = 1e-12
NEG_INF = float("-inf")
DTYPE_NP = np.float64

NpArray: TypeAlias = np.ndarray
DistributionName: TypeAlias = Literal["binomial", "negative_binomial", "poisson"]
SupportDomain: TypeAlias = Literal["probability", "rate"]


__all__ = [
    "DTYPE_NP",
    "DistributionName",
    "EPS",
    "NEG_INF",
    "NpArray",
    "SupportDomain",
]

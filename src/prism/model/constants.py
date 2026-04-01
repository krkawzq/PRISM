from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import torch

EPS = 1e-12
NEG_INF = -1e30
DTYPE_NP = np.float64
DTYPE_TORCH = torch.float64

NpArray: TypeAlias = np.ndarray
TorchTensor: TypeAlias = torch.Tensor
TorchDtypeName: TypeAlias = Literal["float32", "float64"]
OptimizerName: TypeAlias = Literal["adam", "adamw", "sgd", "rmsprop"]
SchedulerName: TypeAlias = Literal["cosine", "linear", "constant", "step"]

OPTIMIZERS: frozenset[str] = frozenset({"adam", "adamw", "sgd", "rmsprop"})
SCHEDULERS: frozenset[str] = frozenset({"cosine", "linear", "constant", "step"})


def resolve_torch_dtype(name: TorchDtypeName | str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch dtype: {name}")

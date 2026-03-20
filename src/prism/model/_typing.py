from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch

# ---- 数值常量 ----

EPS = 1e-12
NEG_INF = -1e30
DTYPE_NP = np.float64
DTYPE_TORCH = torch.float64

# ---- 通用类型 ----

NpArray: TypeAlias = np.ndarray
TorchTensor: TypeAlias = torch.Tensor
ArrayLike: TypeAlias = NpArray | TorchTensor
ScalarLike: TypeAlias = float | int | np.floating | np.integer
GridBound: TypeAlias = ScalarLike | ArrayLike

OptimizerName: TypeAlias = Literal["adam", "adamw", "sgd", "rmsprop"]
SchedulerName: TypeAlias = Literal["cosine", "linear", "constant", "step"]

OPTIMIZERS: set[str] = {"adam", "adamw", "sgd", "rmsprop"}
SCHEDULERS: set[str] = {"cosine", "linear", "constant", "step"}


def _atleast_1d(x: GridBound) -> ArrayLike:
    if isinstance(x, torch.Tensor):
        return torch.atleast_1d(x)
    return np.atleast_1d(x)


def _expand_leading_axis(x: ArrayLike) -> ArrayLike:
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0)
    return x[np.newaxis, ...]


def _scalar_at(x: GridBound, idx: int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x[idx].item())
    return float(np.asarray(x)[idx])


def _shape_tuple(x: GridBound) -> tuple[int, ...]:
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return tuple(np.shape(x))


def _allclose_bound(lhs: GridBound, rhs: GridBound, *, atol: float = 1e-8) -> bool:
    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        lhs_t = lhs if isinstance(lhs, torch.Tensor) else torch.as_tensor(lhs)
        rhs_t = rhs if isinstance(rhs, torch.Tensor) else torch.as_tensor(rhs)
        return bool(torch.allclose(lhs_t, rhs_t, atol=atol, rtol=0.0))
    return bool(np.allclose(lhs, rhs, atol=atol, rtol=0.0))


def _check_grid_bounds(grid_min: GridBound, grid_max: GridBound) -> None:
    if isinstance(grid_min, torch.Tensor) or isinstance(grid_max, torch.Tensor):
        grid_min_t = (
            grid_min
            if isinstance(grid_min, torch.Tensor)
            else torch.as_tensor(grid_min)
        )
        grid_max_t = (
            grid_max
            if isinstance(grid_max, torch.Tensor)
            else torch.as_tensor(grid_max)
        )
        if bool(torch.any(grid_max_t <= grid_min_t)):
            raise ValueError("要求逐元素满足 grid_max > grid_min")
        return

    grid_min_np = np.asarray(grid_min)
    grid_max_np = np.asarray(grid_max)
    if np.any(grid_max_np <= grid_min_np):
        raise ValueError("要求逐元素满足 grid_max > grid_min")


@dataclass(frozen=True, slots=True)
class GridDistribution:
    """离散等距网格上的概率分布。"""

    grid_min: GridBound
    grid_max: GridBound
    weights: ArrayLike

    @property
    def M(self) -> int:
        return self.weights.shape[-1]

    @property
    def is_batched(self) -> bool:
        return self.weights.ndim == 2

    @property
    def B(self) -> int:
        return self.weights.shape[0] if self.is_batched else 1

    @property
    def step(self) -> Any:
        grid_max = cast(Any, self.grid_max)
        grid_min = cast(Any, self.grid_min)
        return (grid_max - grid_min) / (self.M - 1)

    def check_shape(self) -> None:
        weights_shape = tuple(self.weights.shape)
        if self.weights.ndim not in (1, 2):
            raise ValueError(
                f"weights 必须为 (M,) 或 (B, M)，收到 shape={weights_shape}"
            )

        expected_bound_shape = (self.B,) if self.is_batched else ()
        grid_min_shape = _shape_tuple(self.grid_min)
        grid_max_shape = _shape_tuple(self.grid_max)

        if grid_min_shape != expected_bound_shape:
            raise ValueError(
                f"grid_min shape 不匹配：期望 {expected_bound_shape}，收到 {grid_min_shape}"
            )
        if grid_max_shape != expected_bound_shape:
            raise ValueError(
                f"grid_max shape 不匹配：期望 {expected_bound_shape}，收到 {grid_max_shape}"
            )
        if self.M < 2:
            raise ValueError(f"M 必须 >= 2，收到 M={self.M}")

        _check_grid_bounds(self.grid_min, self.grid_max)

    def check_grid(self, other: GridDistribution, *, atol: float = 1e-8) -> None:
        self.check_shape()
        other.check_shape()

        if self.M != other.M:
            raise ValueError(f"M 不匹配：{self.M} != {other.M}")
        if self.is_batched != other.is_batched:
            raise ValueError("grid batch 模式不一致")
        if self.B != other.B:
            raise ValueError(f"B 不匹配：{self.B} != {other.B}")
        if not _allclose_bound(self.grid_min, other.grid_min, atol=atol):
            raise ValueError("grid_min 不匹配")
        if not _allclose_bound(self.grid_max, other.grid_max, atol=atol):
            raise ValueError("grid_max 不匹配")

    def batched(self) -> GridDistribution:
        if self.is_batched:
            return self
        return GridDistribution(
            grid_min=_atleast_1d(self.grid_min),
            grid_max=_atleast_1d(self.grid_max),
            weights=_expand_leading_axis(self.weights),
        )

    def __getitem__(self, idx: int) -> GridDistribution:
        if not self.is_batched:
            if idx != 0:
                raise IndexError(f"标量模式仅支持 idx=0, 收到 {idx}")
            return self
        return GridDistribution(
            grid_min=_scalar_at(self.grid_min, idx),
            grid_max=_scalar_at(self.grid_max, idx),
            weights=self.weights[idx],
        )

    def physical_idx(self, idx: int | ArrayLike) -> Any:
        grid_min = cast(Any, self.grid_min)
        return grid_min + idx * self.step


@dataclass(frozen=True, slots=True)
class PoolEstimate:
    """采样池标尺估计结果。"""

    mu: float
    sigma: float
    point_mu: float
    point_eta: float
    used_posterior_softargmax: bool = False


@dataclass(frozen=True, slots=True)
class GeneBatch:
    """一批基因的观测数据。"""

    gene_names: list[str]
    counts: np.ndarray  # (C, B)
    totals: np.ndarray  # (C,)

    @property
    def B(self) -> int:
        return len(self.gene_names)

    @property
    def C(self) -> int:
        return self.totals.shape[0]

    def check_shape(self) -> None:
        if self.counts.shape != (self.C, self.B):
            raise ValueError(
                f"counts shape 期望 ({self.C}, {self.B}), 收到 {self.counts.shape}"
            )


@dataclass(frozen=True, slots=True)
class LikelihoodCache:
    """预计算的对数似然矩阵。"""

    log_lik: ArrayLike
    s_hat: float
    grid_min: GridBound
    grid_max: GridBound

    @property
    def is_batched(self) -> bool:
        return self.log_lik.ndim == 3

    @property
    def C(self) -> int:
        return self.log_lik.shape[-2]

    @property
    def M(self) -> int:
        return self.log_lik.shape[-1]

    @property
    def B(self) -> int:
        return self.log_lik.shape[0] if self.is_batched else 1

    def check_shape(self) -> None:
        log_lik_shape = tuple(self.log_lik.shape)
        if self.log_lik.ndim not in (2, 3):
            raise ValueError(
                f"log_lik 必须为 (C, M) 或 (B, C, M)，收到 shape={log_lik_shape}"
            )

        expected_bound_shape = (self.B,) if self.is_batched else ()
        grid_min_shape = _shape_tuple(self.grid_min)
        grid_max_shape = _shape_tuple(self.grid_max)

        if grid_min_shape != expected_bound_shape:
            raise ValueError(
                f"grid_min shape 不匹配：期望 {expected_bound_shape}，收到 {grid_min_shape}"
            )
        if grid_max_shape != expected_bound_shape:
            raise ValueError(
                f"grid_max shape 不匹配：期望 {expected_bound_shape}，收到 {grid_max_shape}"
            )
        if self.M < 2:
            raise ValueError(f"M 必须 >= 2，收到 M={self.M}")

        _check_grid_bounds(self.grid_min, self.grid_max)

    def check_grid(
        self, other: GridDistribution | LikelihoodCache, *, atol: float = 1e-8
    ) -> None:
        self.check_shape()
        other.check_shape()

        if self.M != other.M:
            raise ValueError(f"M 不匹配：{self.M} != {other.M}")
        if self.is_batched != other.is_batched:
            raise ValueError("grid batch 模式不一致")
        if self.B != other.B:
            raise ValueError(f"B 不匹配：{self.B} != {other.B}")
        if not _allclose_bound(self.grid_min, other.grid_min, atol=atol):
            raise ValueError("grid_min 不匹配")
        if not _allclose_bound(self.grid_max, other.grid_max, atol=atol):
            raise ValueError("grid_max 不匹配")

    def batched(self) -> LikelihoodCache:
        if self.is_batched:
            return self
        return LikelihoodCache(
            log_lik=_expand_leading_axis(self.log_lik),
            s_hat=self.s_hat,
            grid_min=_atleast_1d(self.grid_min),
            grid_max=_atleast_1d(self.grid_max),
        )

    def __getitem__(self, idx: int) -> LikelihoodCache:
        if not self.is_batched:
            if idx != 0:
                raise IndexError(f"标量模式仅支持 idx=0, 收到 {idx}")
            return self
        return LikelihoodCache(
            log_lik=self.log_lik[idx],
            s_hat=self.s_hat,
            grid_min=_scalar_at(self.grid_min, idx),
            grid_max=_scalar_at(self.grid_max, idx),
        )

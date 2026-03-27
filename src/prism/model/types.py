from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .constants import DTYPE_NP, OPTIMIZERS, SCHEDULERS, OptimizerName, SchedulerName


def _as_1d_float(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _as_2d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    gene_names: list[str]
    counts: np.ndarray  # (C, G)
    reference_counts: np.ndarray  # (C,)

    @property
    def n_cells(self) -> int:
        return int(self.reference_counts.shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        if not self.gene_names:
            raise ValueError("gene_names cannot be empty")
        if len(self.gene_names) != len(set(self.gene_names)):
            raise ValueError("gene_names must be unique")
        counts = _as_2d_float(self.counts, name="counts")
        reference_counts = _as_1d_float(self.reference_counts, name="reference_counts")
        if counts.shape != (reference_counts.shape[0], len(self.gene_names)):
            raise ValueError(
                "counts shape must equal (n_cells, n_genes), "
                f"got {counts.shape} vs {(reference_counts.shape[0], len(self.gene_names))}"
            )
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative")
        if np.any(reference_counts <= 0):
            raise ValueError("reference_counts must be positive")

    @property
    def totals(self) -> np.ndarray:
        return self.reference_counts


@dataclass(frozen=True, slots=True)
class PriorGrid:
    gene_names: list[str]
    p_grid: np.ndarray  # (G, M) or (M,)
    weights: np.ndarray  # (G, M) or (M,)
    S: float
    grid_domain: Literal["p"] = "p"

    @property
    def M(self) -> int:
        return int(self.weights.shape[-1])

    @property
    def is_batched(self) -> bool:
        return self.weights.ndim == 2

    @property
    def G(self) -> int:
        return int(self.weights.shape[0]) if self.is_batched else 1

    @property
    def mu_grid(self) -> np.ndarray:
        return np.asarray(self.p_grid, dtype=DTYPE_NP) * float(self.S)

    def check_shape(self) -> None:
        if not self.gene_names:
            raise ValueError("gene_names cannot be empty")
        if len(self.gene_names) != len(set(self.gene_names)):
            raise ValueError("gene_names must be unique")
        if not np.isfinite(self.S) or self.S <= 0:
            raise ValueError(f"S must be positive, got {self.S}")
        p_grid = np.asarray(self.p_grid, dtype=DTYPE_NP)
        weights = np.asarray(self.weights, dtype=DTYPE_NP)
        if p_grid.ndim not in (1, 2):
            raise ValueError(f"p_grid must be 1D or 2D, got shape={p_grid.shape}")
        if weights.ndim not in (1, 2):
            raise ValueError(f"weights must be 1D or 2D, got shape={weights.shape}")
        if p_grid.shape != weights.shape:
            raise ValueError(
                f"p_grid and weights must have identical shape, got {p_grid.shape} != {weights.shape}"
            )
        if self.is_batched and weights.shape[0] != len(self.gene_names):
            raise ValueError(
                f"weights first dimension must match gene_names, got {weights.shape[0]} != {len(self.gene_names)}"
            )
        if (not self.is_batched) and len(self.gene_names) != 1:
            raise ValueError("unbatched PriorGrid requires exactly one gene")
        if np.any(~np.isfinite(p_grid)) or np.any(~np.isfinite(weights)):
            raise ValueError("p_grid and weights must be finite")
        if np.any(p_grid < 0) or np.any(p_grid > 1):
            raise ValueError("p_grid must lie in [0, 1]")
        sums = weights.sum(axis=-1)
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.any(np.abs(sums - 1.0) > 1e-6):
            raise ValueError("weights must sum to 1 along the grid axis")

    def batched(self) -> PriorGrid:
        self.check_shape()
        if self.is_batched:
            return self
        return PriorGrid(
            gene_names=list(self.gene_names),
            p_grid=np.asarray(self.p_grid, dtype=DTYPE_NP)[None, :],
            weights=np.asarray(self.weights, dtype=DTYPE_NP)[None, :],
            S=float(self.S),
            grid_domain=self.grid_domain,
        )

    def subset(self, gene_names: str | list[str]) -> PriorGrid:
        batched = self.batched()
        lookup = {name: idx for idx, name in enumerate(batched.gene_names)}
        names = [gene_names] if isinstance(gene_names, str) else list(gene_names)
        indices = np.asarray([lookup[name] for name in names], dtype=np.int64)
        p_grid = np.asarray(batched.p_grid, dtype=DTYPE_NP)[indices]
        weights = np.asarray(batched.weights, dtype=DTYPE_NP)[indices]
        if len(names) == 1:
            return PriorGrid(
                gene_names=names,
                p_grid=p_grid[0],
                weights=weights[0],
                S=float(batched.S),
                grid_domain=batched.grid_domain,
            )
        return PriorGrid(
            gene_names=names,
            p_grid=p_grid,
            weights=weights,
            S=float(batched.S),
            grid_domain=batched.grid_domain,
        )


@dataclass(frozen=True, slots=True)
class ScaleMetadata:
    S: float
    mean_reference_count: float


@dataclass(frozen=True, slots=True)
class PriorFitConfig:
    grid_size: int = 512
    sigma_bins: float = 1.0
    align_loss_weight: float = 1.0
    lr: float = 0.05
    n_iter: int = 100
    lr_min_ratio: float = 0.1
    grad_clip: float | None = None
    init_temperature: float = 1.0
    cell_chunk_size: int = 512
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"
    torch_dtype: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if self.sigma_bins < 0:
            raise ValueError("sigma_bins must be >= 0")
        if self.align_loss_weight < 0:
            raise ValueError("align_loss_weight must be >= 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self.lr_min_ratio < 0:
            raise ValueError("lr_min_ratio must be >= 0")
        if self.grad_clip is not None and self.grad_clip < 0:
            raise ValueError("grad_clip must be >= 0")
        if self.init_temperature <= 0:
            raise ValueError("init_temperature must be > 0")
        if self.cell_chunk_size < 1:
            raise ValueError("cell_chunk_size must be >= 1")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"unsupported scheduler: {self.scheduler}")
        if self.torch_dtype not in {"float64", "float32"}:
            raise ValueError(f"unsupported torch_dtype: {self.torch_dtype}")


@dataclass(frozen=True, slots=True)
class PriorFitResult:
    gene_names: list[str]
    priors: PriorGrid
    posterior_average: np.ndarray
    initial_posterior_average: np.ndarray
    initial_prior_weights: np.ndarray
    loss_history: list[float]
    nll_history: list[float]
    align_history: list[float]
    final_loss: float
    best_loss: float
    config: dict[str, Any]
    scale: ScaleMetadata


@dataclass(frozen=True, slots=True)
class InferenceResult:
    gene_names: list[str]
    p_grid: np.ndarray
    mu_grid: np.ndarray
    prior_weights: np.ndarray
    map_p: np.ndarray
    map_mu: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
    posterior: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ScaleDiagnostic:
    mean_reference_count: float
    median_reference_count: float
    suggested_S: float
    lower_quantile_S: float
    upper_quantile_S: float


@dataclass(frozen=True, slots=True)
class PoolEstimate:
    mu: float
    sigma: float
    point_mu: float
    point_eta: float
    used_posterior_softargmax: bool = False


@dataclass(frozen=True, slots=True)
class GeneBatch:
    gene_names: list[str]
    counts: np.ndarray
    totals: np.ndarray

    @property
    def reference_counts(self) -> np.ndarray:
        return self.totals

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.totals).reshape(-1).shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        ObservationBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(self.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(self.totals, dtype=DTYPE_NP),
        ).check_shape()

    def to_observation_batch(self) -> ObservationBatch:
        return ObservationBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(self.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(self.totals, dtype=DTYPE_NP),
        )


class GridDistribution(PriorGrid):
    def __init__(
        self,
        *,
        gene_names: list[str] | None = None,
        p_grid: np.ndarray | None = None,
        weights: np.ndarray,
        S: float = 1.0,
        grid_domain: Literal["p"] = "p",
        grid_min: float | np.ndarray | None = None,
        grid_max: float | np.ndarray | None = None,
    ) -> None:
        weights_np = np.asarray(weights, dtype=DTYPE_NP)
        if gene_names is None:
            gene_names = [
                f"gene_{idx}"
                for idx in range(weights_np.shape[0] if weights_np.ndim == 2 else 1)
            ]
        if p_grid is None:
            if grid_min is None or grid_max is None:
                raise ValueError(
                    "either p_grid or (grid_min, grid_max) must be provided"
                )
            if weights_np.ndim == 1:
                p_grid = np.linspace(
                    float(grid_min),
                    float(grid_max),
                    weights_np.shape[-1],
                    dtype=DTYPE_NP,
                )
            else:
                grid_min_np = np.asarray(grid_min, dtype=DTYPE_NP).reshape(-1)
                grid_max_np = np.asarray(grid_max, dtype=DTYPE_NP).reshape(-1)
                p_grid = np.stack(
                    [
                        np.linspace(
                            grid_min_np[idx],
                            grid_max_np[idx],
                            weights_np.shape[-1],
                            dtype=DTYPE_NP,
                        )
                        for idx in range(weights_np.shape[0])
                    ],
                    axis=0,
                )
        super().__init__(
            gene_names=list(gene_names),
            p_grid=np.asarray(p_grid, dtype=DTYPE_NP),
            weights=weights_np,
            S=float(S),
            grid_domain=grid_domain,
        )

    @property
    def grid_min(self) -> float | np.ndarray:
        p_grid = np.asarray(self.p_grid, dtype=DTYPE_NP)
        return float(p_grid[0]) if p_grid.ndim == 1 else p_grid[:, 0]

    @property
    def grid_max(self) -> float | np.ndarray:
        p_grid = np.asarray(self.p_grid, dtype=DTYPE_NP)
        return float(p_grid[-1]) if p_grid.ndim == 1 else p_grid[:, -1]

    @property
    def support(self) -> np.ndarray:
        return np.asarray(self.mu_grid, dtype=DTYPE_NP)

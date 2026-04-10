from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .constants import DTYPE_NP, DistributionName, SupportDomain


def _require_unique_names(names: list[str], *, field_name: str) -> list[str]:
    resolved = [str(name) for name in names]
    if not resolved:
        raise ValueError(f"{field_name} cannot be empty")
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{field_name} must be unique")
    return resolved


def _as_vector(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _as_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _as_support(
    values: np.ndarray | list[float] | list[list[float]], *, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D, got shape={array.shape}")
    if array.shape[-1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim == 2 and array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _as_probabilities(
    values: np.ndarray | list[float] | list[list[float]], *, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D, got shape={array.shape}")
    if array.shape[-1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim == 2 and array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    if np.any(np.abs(array.sum(axis=-1) - 1.0) > 1e-6):
        raise ValueError(f"{name} must sum to 1 along the support axis")
    return array


def _validate_support_domain(support: np.ndarray, *, domain: SupportDomain) -> None:
    if domain == "probability":
        if np.any(support < 0) or np.any(support > 1):
            raise ValueError("probability support must lie in [0, 1]")
        return
    if domain == "rate":
        if np.any(support < 0):
            raise ValueError("rate support must be >= 0")
        return
    raise ValueError(f"unsupported support domain: {domain}")


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    gene_names: list[str]
    counts: np.ndarray
    reference_counts: np.ndarray

    def __post_init__(self) -> None:
        names = _require_unique_names(self.gene_names, field_name="gene_names")
        counts = _as_matrix(self.counts, name="counts")
        reference_counts = _as_vector(
            self.reference_counts,
            name="reference_counts",
        )
        if counts.shape != (reference_counts.shape[0], len(names)):
            raise ValueError(
                "counts shape must equal (n_cells, n_genes), "
                f"got {counts.shape} vs {(reference_counts.shape[0], len(names))}"
            )
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative")
        if np.any(reference_counts <= 0):
            raise ValueError("reference_counts must be positive")
        object.__setattr__(self, "gene_names", names)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "reference_counts", reference_counts)

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.reference_counts).reshape(-1).shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        ObservationBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(self.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(self.reference_counts, dtype=DTYPE_NP),
        )


@dataclass(frozen=True, slots=True)
class GeneBatch:
    gene_names: list[str]
    counts: np.ndarray
    reference_counts: np.ndarray

    def __post_init__(self) -> None:
        normalized = ObservationBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(self.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(self.reference_counts, dtype=DTYPE_NP),
        )
        object.__setattr__(self, "gene_names", list(normalized.gene_names))
        object.__setattr__(
            self, "counts", np.asarray(normalized.counts, dtype=DTYPE_NP)
        )
        object.__setattr__(
            self,
            "reference_counts",
            np.asarray(normalized.reference_counts, dtype=DTYPE_NP),
        )

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.reference_counts).reshape(-1).shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        ObservationBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(self.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(self.reference_counts, dtype=DTYPE_NP),
        )

    def to_observation_batch(self) -> ObservationBatch:
        return ObservationBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(self.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(self.reference_counts, dtype=DTYPE_NP),
        )


@dataclass(frozen=True, slots=True, init=False)
class DistributionGrid(ABC):
    support: np.ndarray
    probabilities: np.ndarray

    def __init__(
        self,
        *,
        support: np.ndarray | list[float] | list[list[float]],
        probabilities: np.ndarray | list[float] | list[list[float]],
    ) -> None:
        object.__setattr__(self, "support", _as_support(support, name="support"))
        object.__setattr__(
            self,
            "probabilities",
            _as_probabilities(probabilities, name="probabilities"),
        )
        self.check_shape()

    @property
    @abstractmethod
    def distribution(self) -> DistributionName:
        raise NotImplementedError

    @property
    @abstractmethod
    def support_domain(self) -> SupportDomain:
        raise NotImplementedError

    @property
    def is_gene_specific(self) -> bool:
        return self.probabilities.ndim == 2

    @property
    def n_genes(self) -> int:
        return int(self.probabilities.shape[0]) if self.is_gene_specific else 1

    @property
    def n_support_points(self) -> int:
        return int(self.probabilities.shape[-1])

    def check_shape(self) -> None:
        if self.support.shape != self.probabilities.shape:
            raise ValueError(
                "support and probabilities must have identical shape, got "
                f"{self.support.shape} != {self.probabilities.shape}"
            )
        _validate_support_domain(self.support, domain=self.support_domain)

    def as_gene_specific(self) -> DistributionGrid:
        if self.is_gene_specific:
            return self
        return self.__class__(
            support=np.asarray(self.support, dtype=DTYPE_NP)[None, :],
            probabilities=np.asarray(self.probabilities, dtype=DTYPE_NP)[None, :],
        )

    def select_genes(self, indices: list[int] | np.ndarray) -> DistributionGrid:
        gene_specific = self.as_gene_specific()
        resolved = np.asarray(indices, dtype=np.int64).reshape(-1)
        support = np.asarray(gene_specific.support, dtype=DTYPE_NP)[resolved]
        probabilities = np.asarray(gene_specific.probabilities, dtype=DTYPE_NP)[
            resolved
        ]
        if resolved.size == 1:
            return self.__class__(
                support=support[0],
                probabilities=probabilities[0],
            )
        return self.__class__(support=support, probabilities=probabilities)


@dataclass(frozen=True, slots=True, init=False)
class BinomialDistributionGrid(DistributionGrid):
    @property
    def distribution(self) -> DistributionName:
        return "binomial"

    @property
    def support_domain(self) -> SupportDomain:
        return "probability"


@dataclass(frozen=True, slots=True, init=False)
class NegativeBinomialDistributionGrid(DistributionGrid):
    @property
    def distribution(self) -> DistributionName:
        return "negative_binomial"

    @property
    def support_domain(self) -> SupportDomain:
        return "probability"


@dataclass(frozen=True, slots=True, init=False)
class PoissonDistributionGrid(DistributionGrid):
    @property
    def distribution(self) -> DistributionName:
        return "poisson"

    @property
    def support_domain(self) -> SupportDomain:
        return "rate"


def make_distribution_grid(
    distribution: DistributionName,
    *,
    support: np.ndarray | list[float] | list[list[float]],
    probabilities: np.ndarray | list[float] | list[list[float]],
) -> DistributionGrid:
    if distribution == "binomial":
        return BinomialDistributionGrid(
            support=support,
            probabilities=probabilities,
        )
    if distribution == "negative_binomial":
        return NegativeBinomialDistributionGrid(
            support=support,
            probabilities=probabilities,
        )
    if distribution == "poisson":
        return PoissonDistributionGrid(
            support=support,
            probabilities=probabilities,
        )
    raise ValueError(f"unsupported distribution: {distribution}")


@dataclass(frozen=True, slots=True, init=False)
class PriorGrid:
    gene_names: list[str]
    distribution: DistributionGrid
    scale: float

    def __init__(
        self,
        *,
        gene_names: list[str],
        distribution: DistributionGrid,
        scale: float,
    ) -> None:
        object.__setattr__(
            self,
            "gene_names",
            _require_unique_names(gene_names, field_name="gene_names"),
        )
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "scale", float(scale))
        self.check_shape()

    @property
    def support(self) -> np.ndarray:
        return np.asarray(self.distribution.support, dtype=DTYPE_NP)

    @property
    def prior_probabilities(self) -> np.ndarray:
        return np.asarray(self.distribution.probabilities, dtype=DTYPE_NP)

    @property
    def support_domain(self) -> SupportDomain:
        return self.distribution.support_domain

    @property
    def distribution_name(self) -> DistributionName:
        return self.distribution.distribution

    @property
    def is_gene_specific(self) -> bool:
        return self.distribution.is_gene_specific

    @property
    def n_genes(self) -> int:
        return self.distribution.n_genes

    @property
    def n_support_points(self) -> int:
        return self.distribution.n_support_points

    @property
    def scaled_support(self) -> np.ndarray:
        if self.support_domain == "rate":
            return np.asarray(self.support, dtype=DTYPE_NP)
        return np.asarray(self.support, dtype=DTYPE_NP) * float(self.scale)

    def check_shape(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.is_gene_specific and self.n_genes != len(self.gene_names):
            raise ValueError(
                "distribution first dimension must match gene_names, "
                f"got {self.n_genes} != {len(self.gene_names)}"
            )
        if (not self.is_gene_specific) and len(self.gene_names) != 1:
            raise ValueError("non gene-specific priors require exactly one gene")

    def as_gene_specific(self) -> PriorGrid:
        if self.is_gene_specific:
            return self
        return PriorGrid(
            gene_names=list(self.gene_names),
            distribution=self.distribution.as_gene_specific(),
            scale=float(self.scale),
        )

    def select_genes(self, gene_names: str | list[str]) -> PriorGrid:
        gene_specific = self.as_gene_specific()
        names = [gene_names] if isinstance(gene_names, str) else list(gene_names)
        lookup = {name: idx for idx, name in enumerate(gene_specific.gene_names)}
        indices = [lookup[name] for name in names]
        return PriorGrid(
            gene_names=names,
            distribution=gene_specific.distribution.select_genes(indices),
            scale=float(gene_specific.scale),
        )


@dataclass(frozen=True, slots=True)
class ScaleMetadata:
    scale: float
    mean_reference_count: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not np.isfinite(self.mean_reference_count) or self.mean_reference_count <= 0:
            raise ValueError("mean_reference_count must be finite and positive")


@dataclass(frozen=True, slots=True)
class PriorFitConfig:
    n_support_points: int = 512
    max_em_iterations: int | None = 200
    convergence_tolerance: float = 1e-6
    cell_chunk_size: int = 512
    support_max_from: Literal["observed_max", "quantile"] = "observed_max"
    support_spacing: Literal["linear", "sqrt"] = "linear"
    support_scale: float = 1.5
    use_adaptive_support: bool = False
    adaptive_support_scale: float = 1.5
    adaptive_support_quantile_hi: float = 0.99
    likelihood: DistributionName = "binomial"
    nb_overdispersion: float = 0.01

    def __post_init__(self) -> None:
        if self.n_support_points < 2:
            raise ValueError("n_support_points must be >= 2")
        if self.max_em_iterations is not None and self.max_em_iterations < 1:
            raise ValueError("max_em_iterations must be >= 1")
        if self.convergence_tolerance < 0:
            raise ValueError("convergence_tolerance must be >= 0")
        if self.cell_chunk_size < 1:
            raise ValueError("cell_chunk_size must be >= 1")
        if self.support_max_from not in {"observed_max", "quantile"}:
            raise ValueError(f"unsupported support_max_from: {self.support_max_from}")
        if self.support_spacing not in {"linear", "sqrt"}:
            raise ValueError(f"unsupported support_spacing: {self.support_spacing}")
        if self.support_scale < 1.0:
            raise ValueError("support_scale must be >= 1")
        if self.adaptive_support_scale < 1.0:
            raise ValueError("adaptive_support_scale must be >= 1")
        if not (0.0 < self.adaptive_support_quantile_hi <= 1.0):
            raise ValueError("adaptive_support_quantile_hi must be in (0, 1]")
        if self.likelihood not in {"binomial", "negative_binomial", "poisson"}:
            raise ValueError(f"unsupported likelihood: {self.likelihood}")
        if self.nb_overdispersion <= 0:
            raise ValueError("nb_overdispersion must be > 0")


@dataclass(frozen=True, slots=True)
class PriorFitResult:
    gene_names: list[str]
    prior: PriorGrid
    posterior_mean_probabilities: np.ndarray
    objective_history: list[float]
    final_objective: float
    config: dict[str, Any]

    def __post_init__(self) -> None:
        names = _require_unique_names(self.gene_names, field_name="gene_names")
        posterior_mean_probabilities = _as_probabilities(
            self.posterior_mean_probabilities,
            name="posterior_mean_probabilities",
        )
        objective_history = [float(value) for value in self.objective_history]
        if not objective_history:
            raise ValueError("objective_history cannot be empty")
        if not np.all(np.isfinite(np.asarray(objective_history, dtype=DTYPE_NP))):
            raise ValueError("objective_history must contain only finite values")
        final_objective = float(self.final_objective)
        if not np.isfinite(final_objective):
            raise ValueError("final_objective must be finite")
        if abs(final_objective - objective_history[-1]) > 1e-9:
            raise ValueError("final_objective must equal objective_history[-1]")
        if names != list(self.prior.gene_names):
            raise ValueError("gene_names must match prior.gene_names")
        if posterior_mean_probabilities.shape != self.prior.prior_probabilities.shape:
            raise ValueError(
                "posterior_mean_probabilities must match prior probability shape, "
                f"got {posterior_mean_probabilities.shape} != {self.prior.prior_probabilities.shape}"
            )
        object.__setattr__(self, "gene_names", names)
        object.__setattr__(
            self,
            "posterior_mean_probabilities",
            posterior_mean_probabilities,
        )
        object.__setattr__(self, "objective_history", objective_history)
        object.__setattr__(self, "final_objective", final_objective)
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True, slots=True)
class InferenceResult:
    gene_names: list[str]
    support_domain: SupportDomain
    support: np.ndarray
    prior_probabilities: np.ndarray
    map_support: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
    posterior_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        names = _require_unique_names(self.gene_names, field_name="gene_names")
        support = _as_support(self.support, name="support")
        _validate_support_domain(support, domain=self.support_domain)
        prior_probabilities = _as_probabilities(
            self.prior_probabilities,
            name="prior_probabilities",
        )
        if support.shape != prior_probabilities.shape:
            raise ValueError(
                f"support and prior_probabilities must have identical shape, got {support.shape} != {prior_probabilities.shape}"
            )
        map_support = _as_matrix(self.map_support, name="map_support")
        posterior_entropy = _as_matrix(
            self.posterior_entropy,
            name="posterior_entropy",
        )
        prior_entropy = _as_matrix(self.prior_entropy, name="prior_entropy")
        mutual_information = _as_matrix(
            self.mutual_information,
            name="mutual_information",
        )
        if map_support.shape != posterior_entropy.shape:
            raise ValueError("map_support and posterior_entropy must match")
        if map_support.shape != prior_entropy.shape:
            raise ValueError("map_support and prior_entropy must match")
        if map_support.shape != mutual_information.shape:
            raise ValueError("map_support and mutual_information must match")
        if map_support.shape[1] != len(names):
            raise ValueError(
                "inference result second dimension must match gene_names, "
                f"got {map_support.shape[1]} != {len(names)}"
            )
        if self.posterior_probabilities is not None:
            posterior_probabilities = np.asarray(
                self.posterior_probabilities,
                dtype=DTYPE_NP,
            )
            if posterior_probabilities.ndim != 3:
                raise ValueError(
                    "posterior_probabilities must be 3D when present, "
                    f"got shape={posterior_probabilities.shape}"
                )
            if posterior_probabilities.shape[:2] != map_support.shape:
                raise ValueError(
                    "posterior_probabilities leading dimensions must match map_support, "
                    f"got {posterior_probabilities.shape[:2]} != {map_support.shape}"
                )
            if posterior_probabilities.shape[2] != support.shape[-1]:
                raise ValueError(
                    "posterior_probabilities support dimension must match support, "
                    f"got {posterior_probabilities.shape[2]} != {support.shape[-1]}"
                )
        else:
            posterior_probabilities = None
        object.__setattr__(self, "gene_names", names)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "prior_probabilities", prior_probabilities)
        object.__setattr__(self, "map_support", map_support)
        object.__setattr__(self, "posterior_entropy", posterior_entropy)
        object.__setattr__(self, "prior_entropy", prior_entropy)
        object.__setattr__(self, "mutual_information", mutual_information)
        object.__setattr__(self, "posterior_probabilities", posterior_probabilities)


@dataclass(frozen=True, slots=True)
class ScaleDiagnostic:
    mean_reference_count: float
    suggested_scale: float
    upper_quantile_scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_reference_count) or self.mean_reference_count <= 0:
            raise ValueError("mean_reference_count must be finite and positive")
        if not np.isfinite(self.suggested_scale) or self.suggested_scale <= 0:
            raise ValueError("suggested_scale must be finite and positive")
        if not np.isfinite(self.upper_quantile_scale) or self.upper_quantile_scale <= 0:
            raise ValueError("upper_quantile_scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class PoolEstimate:
    point_scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.point_scale) or self.point_scale <= 0:
            raise ValueError("point_scale must be finite and positive")


__all__ = [
    "BinomialDistributionGrid",
    "DistributionGrid",
    "GeneBatch",
    "InferenceResult",
    "make_distribution_grid",
    "NegativeBinomialDistributionGrid",
    "ObservationBatch",
    "PoissonDistributionGrid",
    "PoolEstimate",
    "PriorFitConfig",
    "PriorFitResult",
    "PriorGrid",
    "ScaleDiagnostic",
    "ScaleMetadata",
]

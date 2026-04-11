from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .exposure import mean_reference_count
from .fit import fit_gene_priors
from .types import ObservationBatch, PriorFitConfig, PriorGrid


@dataclass(frozen=True, slots=True)
class PriorEngineSetting:
    support_max_from: Literal["observed_max", "quantile"] = "observed_max"
    support_spacing: Literal["linear", "sqrt"] = "linear"
    support_scale: float = 1.5
    use_adaptive_support: bool = False
    adaptive_support_scale: float = 1.5
    adaptive_support_quantile: float = 0.99
    likelihood: Literal["binomial", "negative_binomial", "poisson"] = "binomial"
    nb_overdispersion: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "use_adaptive_support", bool(self.use_adaptive_support)
        )
        PriorFitConfig(
            support_max_from=self.support_max_from,
            support_spacing=self.support_spacing,
            support_scale=self.support_scale,
            use_adaptive_support=self.use_adaptive_support,
            adaptive_support_scale=self.adaptive_support_scale,
            adaptive_support_quantile=self.adaptive_support_quantile,
            likelihood=self.likelihood,
            nb_overdispersion=self.nb_overdispersion,
        )


@dataclass(frozen=True, slots=True)
class PriorEngineTrainingConfig:
    n_support_points: int = 512
    max_em_iterations: int | None = 200
    convergence_tolerance: float = 1e-6
    cell_chunk_size: int = 4096
    torch_dtype: Literal["float64", "float32"] = "float64"
    compile_model: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "compile_model", bool(self.compile_model))
        PriorFitConfig(
            n_support_points=self.n_support_points,
            max_em_iterations=self.max_em_iterations,
            convergence_tolerance=self.convergence_tolerance,
            cell_chunk_size=self.cell_chunk_size,
        )
        if self.torch_dtype not in {"float64", "float32"}:
            raise ValueError(f"unsupported torch_dtype: {self.torch_dtype}")


@dataclass(frozen=True, slots=True)
class FitSummary:
    gene_names: list[str]
    final_objective: float
    requested_max_em_iterations: int | None


@dataclass(frozen=True, slots=True)
class PriorFitReport:
    gene_names: list[str]
    support: np.ndarray
    scaled_support: np.ndarray
    prior_probabilities: np.ndarray
    posterior_mean_probabilities: np.ndarray
    final_objective: float
    config: dict[str, object]
    scale: float
    mean_reference_count: float


def _merge_config(
    setting: PriorEngineSetting,
    training_cfg: PriorEngineTrainingConfig,
) -> PriorFitConfig:
    return PriorFitConfig(
        n_support_points=training_cfg.n_support_points,
        max_em_iterations=training_cfg.max_em_iterations,
        convergence_tolerance=training_cfg.convergence_tolerance,
        cell_chunk_size=training_cfg.cell_chunk_size,
        support_max_from=setting.support_max_from,
        support_spacing=setting.support_spacing,
        support_scale=setting.support_scale,
        use_adaptive_support=setting.use_adaptive_support,
        adaptive_support_scale=setting.adaptive_support_scale,
        adaptive_support_quantile=setting.adaptive_support_quantile,
        likelihood=setting.likelihood,
        nb_overdispersion=setting.nb_overdispersion,
    )


class PriorEngine:
    def __init__(
        self,
        gene_names: list[str],
        setting: PriorEngineSetting = PriorEngineSetting(),
        device: str = "cpu",
    ) -> None:
        if not gene_names:
            raise ValueError("gene_names cannot be empty")
        if len(gene_names) != len(set(gene_names)):
            raise ValueError("gene_names must be unique")
        self.gene_names = list(gene_names)
        self.setting = setting
        self.device = device
        self._prior: PriorGrid | None = None

    def fit(
        self,
        batch: ObservationBatch,
        *,
        scale: float,
        training_cfg: PriorEngineTrainingConfig = PriorEngineTrainingConfig(),
        progress_callback=None,
        support_max: np.ndarray | None = None,
    ) -> FitSummary:
        report = self.fit_report(
            batch,
            scale=scale,
            training_cfg=training_cfg,
            progress_callback=progress_callback,
            support_max=support_max,
        )
        return FitSummary(
            gene_names=list(report.gene_names),
            final_objective=float(report.final_objective),
            requested_max_em_iterations=training_cfg.max_em_iterations,
        )

    def fit_report(
        self,
        batch: ObservationBatch,
        *,
        scale: float,
        training_cfg: PriorEngineTrainingConfig = PriorEngineTrainingConfig(),
        progress_callback=None,
        support_max: np.ndarray | None = None,
    ) -> PriorFitReport:
        result = fit_gene_priors(
            batch,
            scale=float(scale),
            config=_merge_config(self.setting, training_cfg),
            device=self.device,
            torch_dtype=training_cfg.torch_dtype,
            support_max=support_max,
            compile_model=training_cfg.compile_model,
            progress_callback=progress_callback,
        )
        self._prior = result.prior
        return PriorFitReport(
            gene_names=list(result.gene_names),
            support=np.asarray(result.prior.support),
            scaled_support=np.asarray(result.prior.scaled_support),
            prior_probabilities=np.asarray(result.prior.prior_probabilities),
            posterior_mean_probabilities=np.asarray(result.posterior_mean_probabilities),
            final_objective=float(result.final_objective),
            config=dict(result.config),
            scale=float(result.prior.scale),
            mean_reference_count=float(mean_reference_count(batch.reference_counts)),
        )

    def get_prior(self, gene_names: str | list[str]) -> PriorGrid | None:
        if self._prior is None:
            return None
        return self._prior.select_genes(gene_names)

    def is_fitted(self, gene_name: str) -> bool:
        return self._prior is not None and gene_name in self._prior.gene_names

    def is_all_fitted(self) -> bool:
        return self._prior is not None and set(self._prior.gene_names) == set(
            self.gene_names
        )


__all__ = [
    "FitSummary",
    "PriorEngine",
    "PriorEngineSetting",
    "PriorEngineTrainingConfig",
    "PriorFitReport",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fit import fit_gene_priors
from .constants import OptimizerName, SchedulerName
from .types import (
    GridDistribution,
    ObservationBatch,
    PriorFitConfig,
    PriorFitResult,
    PriorGrid,
)


@dataclass(frozen=True, slots=True)
class PriorEngineSetting:
    grid_size: int = 512
    sigma_bins: float = 1.0
    align_loss_weight: float = 1.0
    align_every: int = 1
    torch_dtype: Literal["float64", "float32"] = "float64"
    grid_max_method: Literal["observed_max", "quantile"] = "observed_max"
    grid_strategy: Literal["linear", "sqrt"] = "linear"
    sigma_anneal_start: float | None = None
    sigma_anneal_end: float | None = None
    adaptive_grid: bool = False
    adaptive_grid_fraction: float = 0.3
    align_mode: Literal["jsd", "kl", "weighted_jsd"] = "jsd"
    shrinkage_weight: float = 0.0
    likelihood: Literal["binomial", "negative_binomial", "poisson"] = "binomial"
    nb_overdispersion: float = 0.01


@dataclass(frozen=True, slots=True)
class PriorEngineTrainingConfig:
    lr: float = 0.05
    n_iter: int = 100
    lr_min_ratio: float = 0.1
    grad_clip: float | None = None
    early_stop_tol: float | None = None
    early_stop_patience: int | None = None
    init_method: Literal["posterior_mean", "uniform", "random"] = "posterior_mean"
    init_seed: int = 0
    init_temperature: float = 1.0
    cell_chunk_size: int = 512
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"
    cell_sample_fraction: float = 1.0
    cell_sample_seed: int = 0
    ensemble_restarts: int = 1


@dataclass(frozen=True, slots=True)
class FitSummary:
    gene_names: list[str]
    final_loss: float
    best_loss: float
    n_iter: int


@dataclass(frozen=True, slots=True)
class PriorFitReport:
    gene_names: list[str]
    p_grid: np.ndarray
    mu_grid: np.ndarray
    prior_weights: np.ndarray
    posterior_average: np.ndarray
    initial_posterior_average: np.ndarray
    initial_prior_weights: np.ndarray
    final_loss: float
    best_loss: float
    loss_history: list[float]
    nll_history: list[float]
    align_history: list[float]
    config: dict[str, object]
    S: float
    mean_reference_count: float

    @property
    def support(self) -> np.ndarray:
        return self.mu_grid

    @property
    def grid_min(self) -> np.ndarray:
        return self.p_grid[:, 0]

    @property
    def grid_max(self) -> np.ndarray:
        return self.p_grid[:, -1]


def _merge_config(
    setting: PriorEngineSetting,
    training_cfg: PriorEngineTrainingConfig,
) -> PriorFitConfig:
    return PriorFitConfig(
        grid_size=setting.grid_size,
        sigma_bins=setting.sigma_bins,
        align_loss_weight=setting.align_loss_weight,
        align_every=setting.align_every,
        lr=training_cfg.lr,
        n_iter=training_cfg.n_iter,
        lr_min_ratio=training_cfg.lr_min_ratio,
        grad_clip=training_cfg.grad_clip,
        early_stop_tol=training_cfg.early_stop_tol,
        early_stop_patience=training_cfg.early_stop_patience,
        init_method=training_cfg.init_method,
        init_seed=training_cfg.init_seed,
        init_temperature=training_cfg.init_temperature,
        cell_chunk_size=training_cfg.cell_chunk_size,
        optimizer=training_cfg.optimizer,
        scheduler=training_cfg.scheduler,
        torch_dtype=setting.torch_dtype,
        grid_max_method=setting.grid_max_method,
        grid_strategy=setting.grid_strategy,
        sigma_anneal_start=setting.sigma_anneal_start,
        sigma_anneal_end=setting.sigma_anneal_end,
        adaptive_grid=setting.adaptive_grid,
        adaptive_grid_fraction=setting.adaptive_grid_fraction,
        cell_sample_fraction=training_cfg.cell_sample_fraction,
        cell_sample_seed=training_cfg.cell_sample_seed,
        align_mode=setting.align_mode,
        shrinkage_weight=setting.shrinkage_weight,
        ensemble_restarts=training_cfg.ensemble_restarts,
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
        self._priors: PriorGrid | None = None

    def fit(
        self,
        batch: ObservationBatch,
        S: float | None = None,
        s_hat: float | None = None,
        training_cfg: PriorEngineTrainingConfig = PriorEngineTrainingConfig(),
        progress_callback=None,
        p_grid_max: np.ndarray | None = None,
    ) -> FitSummary:
        raw_S = S if S is not None else s_hat
        if raw_S is None:
            raise ValueError("S must be provided")
        resolved_S = float(raw_S)
        report = self.fit_report(
            batch,
            S=resolved_S,
            training_cfg=training_cfg,
            progress_callback=progress_callback,
            p_grid_max=p_grid_max,
        )
        return FitSummary(
            gene_names=list(report.gene_names),
            final_loss=float(report.final_loss),
            best_loss=float(report.best_loss),
            n_iter=int(training_cfg.n_iter),
        )

    def fit_report(
        self,
        batch: ObservationBatch,
        S: float | None = None,
        s_hat: float | None = None,
        training_cfg: PriorEngineTrainingConfig = PriorEngineTrainingConfig(),
        progress_callback=None,
        p_grid_max: np.ndarray | None = None,
    ) -> PriorFitReport:
        raw_S = S if S is not None else s_hat
        if raw_S is None:
            raise ValueError("S must be provided")
        resolved_S = float(raw_S)
        if hasattr(batch, "to_observation_batch"):
            batch = batch.to_observation_batch()  # type: ignore[assignment]
        result = fit_gene_priors(
            batch,
            S=resolved_S,
            config=_merge_config(self.setting, training_cfg),
            device=self.device,
            p_grid_max=p_grid_max,
            progress_callback=progress_callback,
        )
        self._priors = result.priors
        return PriorFitReport(
            gene_names=list(result.gene_names),
            p_grid=np.asarray(result.priors.p_grid, dtype=np.float64),
            mu_grid=np.asarray(result.priors.mu_grid, dtype=np.float64),
            prior_weights=np.asarray(result.priors.weights, dtype=np.float64),
            posterior_average=np.asarray(result.posterior_average, dtype=np.float64),
            initial_posterior_average=np.asarray(
                result.initial_posterior_average, dtype=np.float64
            ),
            initial_prior_weights=np.asarray(
                result.initial_prior_weights, dtype=np.float64
            ),
            final_loss=float(result.final_loss),
            best_loss=float(result.best_loss),
            loss_history=list(result.loss_history),
            nll_history=list(result.nll_history),
            align_history=list(result.align_history),
            config=dict(result.config),
            S=float(result.scale.S),
            mean_reference_count=float(result.scale.mean_reference_count),
        )

    def get_priors(self, gene_names: str | list[str]) -> PriorGrid | None:
        if self._priors is None:
            return None
        subset = self._priors.subset(gene_names)
        return GridDistribution(
            gene_names=list(subset.gene_names),
            p_grid=np.asarray(subset.p_grid, dtype=np.float64),
            weights=np.asarray(subset.weights, dtype=np.float64),
            S=float(subset.S),
        )

    def is_fitted(self, gene_name: str) -> bool:
        return self._priors is not None and gene_name in self._priors.gene_names

    def is_all_fitted(self) -> bool:
        return self._priors is not None and set(self._priors.gene_names) == set(
            self.gene_names
        )


PriorFitter = PriorEngine

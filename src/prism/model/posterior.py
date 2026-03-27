from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .infer import ALL_CHANNELS, CORE_CHANNELS, SignalChannel, infer_posteriors
from .types import InferenceResult, ObservationBatch, PriorGrid


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    gene_names: list[str]
    p_grid: np.ndarray
    mu_grid: np.ndarray
    prior_weights: np.ndarray
    posterior: np.ndarray
    map_p: np.ndarray
    map_mu: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray


@dataclass(frozen=True, slots=True)
class PosteriorBatchReport:
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


class Posterior:
    def __init__(
        self,
        gene_names: list[str],
        priors: PriorGrid,
        *,
        device: str = "cpu",
    ) -> None:
        self.gene_names = list(gene_names)
        self.priors = priors.subset(gene_names)
        self.device = device

    def query(
        self,
        gene_name: str,
        x_vals: np.ndarray,
        reference_counts: np.ndarray,
        s_hat: float | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        del s_hat
        batch = ObservationBatch(
            gene_names=[gene_name],
            counts=np.asarray(x_vals, dtype=np.float64).reshape(-1, 1),
            reference_counts=np.asarray(reference_counts, dtype=np.float64).reshape(-1),
        )
        result = self.extract(batch, channels=set(ALL_CHANNELS))
        return {
            key: value[:, 0] if value.ndim == 2 else value[0]
            for key, value in result.items()
        }

    def summarize(self, batch: ObservationBatch) -> PosteriorSummary:
        if hasattr(batch, "to_observation_batch"):
            batch = batch.to_observation_batch()  # type: ignore[assignment]
        result = infer_posteriors(
            batch, self.priors, device=self.device, include_posterior=True
        )
        if result.posterior is None:
            raise RuntimeError("posterior payload is unexpectedly missing")
        return PosteriorSummary(
            gene_names=list(result.gene_names),
            p_grid=result.p_grid,
            mu_grid=result.mu_grid,
            prior_weights=result.prior_weights,
            posterior=result.posterior,
            map_p=result.map_p,
            map_mu=result.map_mu,
            posterior_entropy=result.posterior_entropy,
            prior_entropy=result.prior_entropy,
            mutual_information=result.mutual_information,
        )

    def summarize_batch(
        self, batch: ObservationBatch, *, include_posterior: bool = False
    ) -> PosteriorBatchReport:
        if hasattr(batch, "to_observation_batch"):
            batch = batch.to_observation_batch()  # type: ignore[assignment]
        result = infer_posteriors(
            batch, self.priors, device=self.device, include_posterior=include_posterior
        )
        return PosteriorBatchReport(
            gene_names=list(result.gene_names),
            p_grid=result.p_grid,
            mu_grid=result.mu_grid,
            prior_weights=result.prior_weights,
            map_p=result.map_p,
            map_mu=result.map_mu,
            posterior_entropy=result.posterior_entropy,
            prior_entropy=result.prior_entropy,
            mutual_information=result.mutual_information,
            posterior=result.posterior,
        )

    def extract(
        self,
        batch: ObservationBatch,
        s_hat: float | np.ndarray | None = None,
        channels: set[SignalChannel] | None = None,
    ) -> dict[str, np.ndarray]:
        del s_hat
        if hasattr(batch, "to_observation_batch"):
            batch = batch.to_observation_batch()  # type: ignore[assignment]
        requested_raw = CORE_CHANNELS if channels is None else frozenset(channels)
        alias_map = {"signal": "map_mu", "support": "mu_grid"}
        requested = frozenset(
            alias_map.get(channel, channel) for channel in requested_raw
        )
        result = infer_posteriors(
            batch,
            self.priors,
            device=self.device,
            include_posterior="posterior" in requested,
        )
        payload: dict[str, np.ndarray] = {}
        if "signal" in requested_raw or "map_mu" in requested:
            payload["signal"] = result.map_mu
        if "map_p" in requested:
            payload["map_p"] = result.map_p
        if "map_mu" in requested:
            payload["map_mu"] = result.map_mu
        if "posterior_entropy" in requested:
            payload["posterior_entropy"] = result.posterior_entropy
        if "prior_entropy" in requested:
            payload["prior_entropy"] = result.prior_entropy
        if "mutual_information" in requested:
            payload["mutual_information"] = result.mutual_information
        if "posterior" in requested and result.posterior is not None:
            payload["posterior"] = result.posterior
            payload["p_grid"] = result.p_grid
            payload["mu_grid"] = result.mu_grid
            payload["support"] = result.mu_grid
            payload["prior_weights"] = result.prior_weights
        return payload


SignalExtractor = Posterior

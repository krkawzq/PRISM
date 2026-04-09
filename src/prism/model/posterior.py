from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from .infer import infer_posteriors
from .types import GeneBatch, InferenceResult, ObservationBatch, PriorGrid

SignalChannel = Literal[
    "signal",
    "map_probability",
    "map_support",
    "map_scaled_support",
    "map_rate",
    "posterior_entropy",
    "prior_entropy",
    "mutual_information",
    "posterior",
    "support",
    "scaled_support",
    "prior_probabilities",
]

CORE_CHANNELS = cast(
    frozenset[SignalChannel],
    frozenset({"signal", "posterior_entropy", "prior_entropy", "mutual_information"}),
)
ALL_CHANNELS = cast(
    frozenset[SignalChannel],
    frozenset(
        set(CORE_CHANNELS)
        | {
            "map_probability",
            "map_support",
            "map_scaled_support",
            "map_rate",
            "posterior",
            "support",
            "scaled_support",
            "prior_probabilities",
        }
    ),
)


def _scaled_values(prior: PriorGrid, values: np.ndarray) -> np.ndarray:
    if prior.support_domain == "rate":
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) * float(prior.scale)


def _map_probability(prior: PriorGrid, map_support: np.ndarray) -> np.ndarray:
    if prior.support_domain == "rate":
        return np.full_like(np.asarray(map_support, dtype=np.float64), np.nan)
    return np.asarray(map_support, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    gene_names: list[str]
    support_domain: str
    support: np.ndarray
    scaled_support: np.ndarray
    prior_probabilities: np.ndarray
    posterior_probabilities: np.ndarray
    map_probability: np.ndarray
    map_support: np.ndarray
    map_scaled_support: np.ndarray
    map_rate: np.ndarray | None
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray


@dataclass(frozen=True, slots=True)
class PosteriorBatchReport:
    gene_names: list[str]
    support_domain: str
    support: np.ndarray
    scaled_support: np.ndarray
    prior_probabilities: np.ndarray
    map_probability: np.ndarray
    map_support: np.ndarray
    map_scaled_support: np.ndarray
    map_rate: np.ndarray | None
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
    posterior_probabilities: np.ndarray | None = None


class Posterior:
    def __init__(
        self,
        gene_names: list[str],
        prior: PriorGrid,
        *,
        device: str = "cpu",
        torch_dtype: str = "float64",
        posterior_distribution: str = "auto",
        nb_overdispersion: float = 0.01,
        compile_model: bool = True,
    ) -> None:
        self.gene_names = list(gene_names)
        self.prior = prior.select_genes(gene_names)
        self.device = device
        self.torch_dtype = torch_dtype
        self.posterior_distribution = posterior_distribution
        self.nb_overdispersion = nb_overdispersion
        self.compile_model = bool(compile_model)

    def _infer(
        self,
        batch: ObservationBatch | GeneBatch,
        *,
        include_posterior: bool,
    ) -> InferenceResult:
        if isinstance(batch, GeneBatch):
            batch = batch.to_observation_batch()
        return infer_posteriors(
            batch,
            self.prior,
            device=self.device,
            include_posterior=include_posterior,
            torch_dtype=self.torch_dtype,
            posterior_distribution=cast(Any, self.posterior_distribution),
            nb_overdispersion=self.nb_overdispersion,
            compile_model=self.compile_model,
        )

    def summarize(self, batch: ObservationBatch | GeneBatch) -> PosteriorSummary:
        result = self._infer(batch, include_posterior=True)
        if result.posterior_probabilities is None:
            raise RuntimeError("posterior probabilities are unexpectedly missing")
        map_probability = _map_probability(self.prior, result.map_support)
        map_scaled_support = _scaled_values(self.prior, result.map_support)
        map_rate = map_scaled_support if result.support_domain == "rate" else None
        return PosteriorSummary(
            gene_names=list(result.gene_names),
            support_domain=result.support_domain,
            support=np.asarray(result.support, dtype=np.float64),
            scaled_support=np.asarray(self.prior.scaled_support, dtype=np.float64),
            prior_probabilities=np.asarray(
                result.prior_probabilities, dtype=np.float64
            ),
            posterior_probabilities=np.asarray(
                result.posterior_probabilities, dtype=np.float64
            ),
            map_probability=map_probability,
            map_support=np.asarray(result.map_support, dtype=np.float64),
            map_scaled_support=map_scaled_support,
            map_rate=None
            if map_rate is None
            else np.asarray(map_rate, dtype=np.float64),
            posterior_entropy=np.asarray(result.posterior_entropy, dtype=np.float64),
            prior_entropy=np.asarray(result.prior_entropy, dtype=np.float64),
            mutual_information=np.asarray(result.mutual_information, dtype=np.float64),
        )

    def summarize_batch(
        self,
        batch: ObservationBatch | GeneBatch,
        *,
        include_posterior: bool = False,
    ) -> PosteriorBatchReport:
        result = self._infer(batch, include_posterior=include_posterior)
        map_probability = _map_probability(self.prior, result.map_support)
        map_scaled_support = _scaled_values(self.prior, result.map_support)
        map_rate = map_scaled_support if result.support_domain == "rate" else None
        return PosteriorBatchReport(
            gene_names=list(result.gene_names),
            support_domain=result.support_domain,
            support=np.asarray(result.support, dtype=np.float64),
            scaled_support=np.asarray(self.prior.scaled_support, dtype=np.float64),
            prior_probabilities=np.asarray(
                result.prior_probabilities, dtype=np.float64
            ),
            map_probability=map_probability,
            map_support=np.asarray(result.map_support, dtype=np.float64),
            map_scaled_support=map_scaled_support,
            map_rate=None
            if map_rate is None
            else np.asarray(map_rate, dtype=np.float64),
            posterior_entropy=np.asarray(result.posterior_entropy, dtype=np.float64),
            prior_entropy=np.asarray(result.prior_entropy, dtype=np.float64),
            mutual_information=np.asarray(result.mutual_information, dtype=np.float64),
            posterior_probabilities=(
                None
                if result.posterior_probabilities is None
                else np.asarray(result.posterior_probabilities, dtype=np.float64)
            ),
        )

    def extract(
        self,
        batch: ObservationBatch | GeneBatch,
        channels: set[SignalChannel] | None = None,
    ) -> dict[str, np.ndarray]:
        requested = CORE_CHANNELS if channels is None else frozenset(channels)
        result = self._infer(batch, include_posterior="posterior" in requested)
        payload: dict[str, np.ndarray] = {}
        if "signal" in requested:
            payload["signal"] = _scaled_values(self.prior, result.map_support)
        if "map_probability" in requested:
            payload["map_probability"] = _map_probability(
                self.prior, result.map_support
            )
        if "map_support" in requested:
            payload["map_support"] = np.asarray(result.map_support, dtype=np.float64)
        if "map_scaled_support" in requested:
            payload["map_scaled_support"] = _scaled_values(
                self.prior, result.map_support
            )
        if "map_rate" in requested and result.support_domain == "rate":
            payload["map_rate"] = np.asarray(result.map_support, dtype=np.float64)
        if "posterior_entropy" in requested:
            payload["posterior_entropy"] = np.asarray(
                result.posterior_entropy, dtype=np.float64
            )
        if "prior_entropy" in requested:
            payload["prior_entropy"] = np.asarray(
                result.prior_entropy, dtype=np.float64
            )
        if "mutual_information" in requested:
            payload["mutual_information"] = np.asarray(
                result.mutual_information, dtype=np.float64
            )
        if "support" in requested:
            payload["support"] = np.asarray(result.support, dtype=np.float64)
        if "scaled_support" in requested:
            payload["scaled_support"] = np.asarray(
                self.prior.scaled_support, dtype=np.float64
            )
        if "prior_probabilities" in requested:
            payload["prior_probabilities"] = np.asarray(
                result.prior_probabilities, dtype=np.float64
            )
        if "posterior" in requested and result.posterior_probabilities is not None:
            payload["posterior"] = np.asarray(
                result.posterior_probabilities, dtype=np.float64
            )
        return payload

__all__ = [
    "ALL_CHANNELS",
    "CORE_CHANNELS",
    "Posterior",
    "PosteriorBatchReport",
    "PosteriorSummary",
    "SignalChannel",
]

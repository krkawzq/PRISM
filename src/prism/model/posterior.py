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


def _resolve_numpy_dtype(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported result_dtype: {name}")


def _scaled_values(prior: PriorGrid, values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    if prior.support_domain == "rate":
        return np.asarray(values, dtype=dtype)
    return np.asarray(values, dtype=dtype) * np.asarray(prior.scale, dtype=dtype)


def _map_probability(
    prior: PriorGrid,
    map_support: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    if prior.support_domain == "rate":
        return np.full_like(np.asarray(map_support, dtype=dtype), np.nan)
    return np.asarray(map_support, dtype=dtype)


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
        result_dtype: str | None = None,
        posterior_distribution: str = "auto",
        nb_overdispersion: float = 0.01,
        compile_model: bool = True,
        observation_chunk_size: int | None = None,
    ) -> None:
        self.gene_names = list(gene_names)
        self.prior = prior.select_genes(gene_names)
        self.device = device
        self.torch_dtype = torch_dtype
        self.result_dtype = torch_dtype if result_dtype is None else result_dtype
        self.posterior_distribution = posterior_distribution
        self.nb_overdispersion = nb_overdispersion
        self.compile_model = bool(compile_model)
        self.observation_chunk_size = observation_chunk_size
        self._output_dtype = _resolve_numpy_dtype(self.result_dtype)
        self._support_output = np.asarray(self.prior.support, dtype=self._output_dtype)
        self._scaled_support_output = np.asarray(
            self.prior.scaled_support,
            dtype=self._output_dtype,
        )
        self._prior_probabilities_output = np.asarray(
            self.prior.prior_probabilities,
            dtype=self._output_dtype,
        )

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
            result_dtype=self.result_dtype,
            posterior_distribution=cast(Any, self.posterior_distribution),
            nb_overdispersion=self.nb_overdispersion,
            compile_model=self.compile_model,
            observation_chunk_size=self.observation_chunk_size,
        )

    def summarize(self, batch: ObservationBatch | GeneBatch) -> PosteriorSummary:
        result = self._infer(batch, include_posterior=True)
        if result.posterior_probabilities is None:
            raise RuntimeError("posterior probabilities are unexpectedly missing")
        map_probability = _map_probability(
            self.prior,
            result.map_support,
            dtype=self._output_dtype,
        )
        map_scaled_support = _scaled_values(
            self.prior,
            result.map_support,
            dtype=self._output_dtype,
        )
        map_rate = map_scaled_support if result.support_domain == "rate" else None
        return PosteriorSummary(
            gene_names=list(result.gene_names),
            support_domain=result.support_domain,
            support=self._support_output,
            scaled_support=self._scaled_support_output,
            prior_probabilities=self._prior_probabilities_output,
            posterior_probabilities=np.asarray(
                result.posterior_probabilities,
                dtype=self._output_dtype,
            ),
            map_probability=map_probability,
            map_support=np.asarray(result.map_support, dtype=self._output_dtype),
            map_scaled_support=map_scaled_support,
            map_rate=(
                None
                if map_rate is None
                else np.asarray(map_rate, dtype=self._output_dtype)
            ),
            posterior_entropy=np.asarray(
                result.posterior_entropy,
                dtype=self._output_dtype,
            ),
            prior_entropy=np.asarray(result.prior_entropy, dtype=self._output_dtype),
            mutual_information=np.asarray(
                result.mutual_information,
                dtype=self._output_dtype,
            ),
        )

    def summarize_batch(
        self,
        batch: ObservationBatch | GeneBatch,
        *,
        include_posterior: bool = False,
    ) -> PosteriorBatchReport:
        result = self._infer(batch, include_posterior=include_posterior)
        map_probability = _map_probability(
            self.prior,
            result.map_support,
            dtype=self._output_dtype,
        )
        map_scaled_support = _scaled_values(
            self.prior,
            result.map_support,
            dtype=self._output_dtype,
        )
        map_rate = map_scaled_support if result.support_domain == "rate" else None
        return PosteriorBatchReport(
            gene_names=list(result.gene_names),
            support_domain=result.support_domain,
            support=self._support_output,
            scaled_support=self._scaled_support_output,
            prior_probabilities=self._prior_probabilities_output,
            map_probability=map_probability,
            map_support=np.asarray(result.map_support, dtype=self._output_dtype),
            map_scaled_support=map_scaled_support,
            map_rate=(
                None
                if map_rate is None
                else np.asarray(map_rate, dtype=self._output_dtype)
            ),
            posterior_entropy=np.asarray(
                result.posterior_entropy,
                dtype=self._output_dtype,
            ),
            prior_entropy=np.asarray(result.prior_entropy, dtype=self._output_dtype),
            mutual_information=np.asarray(
                result.mutual_information,
                dtype=self._output_dtype,
            ),
            posterior_probabilities=(
                None
                if result.posterior_probabilities is None
                else np.asarray(
                    result.posterior_probabilities,
                    dtype=self._output_dtype,
                )
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
            payload["signal"] = _scaled_values(
                self.prior,
                result.map_support,
                dtype=self._output_dtype,
            )
        if "map_probability" in requested:
            payload["map_probability"] = _map_probability(
                self.prior,
                result.map_support,
                dtype=self._output_dtype,
            )
        if "map_support" in requested:
            payload["map_support"] = np.asarray(
                result.map_support,
                dtype=self._output_dtype,
            )
        if "map_scaled_support" in requested:
            payload["map_scaled_support"] = _scaled_values(
                self.prior,
                result.map_support,
                dtype=self._output_dtype,
            )
        if "map_rate" in requested and result.support_domain == "rate":
            payload["map_rate"] = np.asarray(
                result.map_support,
                dtype=self._output_dtype,
            )
        if "posterior_entropy" in requested:
            payload["posterior_entropy"] = np.asarray(
                result.posterior_entropy,
                dtype=self._output_dtype,
            )
        if "prior_entropy" in requested:
            payload["prior_entropy"] = np.asarray(
                result.prior_entropy,
                dtype=self._output_dtype,
            )
        if "mutual_information" in requested:
            payload["mutual_information"] = np.asarray(
                result.mutual_information,
                dtype=self._output_dtype,
            )
        if "support" in requested:
            payload["support"] = self._support_output
        if "scaled_support" in requested:
            payload["scaled_support"] = self._scaled_support_output
        if "prior_probabilities" in requested:
            payload["prior_probabilities"] = self._prior_probabilities_output
        if "posterior" in requested and result.posterior_probabilities is not None:
            payload["posterior"] = np.asarray(
                result.posterior_probabilities,
                dtype=self._output_dtype,
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

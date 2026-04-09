from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import torch

from .constants import DTYPE_NP
from .exposure import effective_exposure, validate_binomial_observations
from .numeric import (
    entropy,
    log_binomial_likelihood_support,
    log_negative_binomial_likelihood_support,
    log_poisson_likelihood_support,
    posterior_from_log_likelihood,
)
from .types import GeneBatch, InferenceResult, ObservationBatch, PriorGrid

PosteriorDistribution = Literal["auto", "binomial", "negative_binomial", "poisson"]


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch_dtype: {name}")


def _resolve_distribution(
    prior: PriorGrid,
    posterior_distribution: PosteriorDistribution,
) -> Literal["binomial", "negative_binomial", "poisson"]:
    resolved = (
        prior.distribution_name
        if posterior_distribution == "auto"
        else posterior_distribution
    )
    if resolved != prior.distribution_name:
        raise ValueError(
            "posterior distribution mismatch with prior distribution; "
            f"prior={prior.distribution_name!r}, posterior={resolved!r}"
        )
    if prior.support_domain == "rate" and resolved != "poisson":
        raise ValueError(
            "rate support requires poisson posterior inference; "
            f"got posterior_distribution={resolved!r}"
        )
    if prior.support_domain == "probability" and resolved == "poisson":
        raise ValueError("poisson posterior inference requires rate support")
    return cast(Literal["binomial", "negative_binomial", "poisson"], resolved)


class _BasePosteriorInferencer(torch.nn.Module):
    def _summarize(
        self,
        posterior_probabilities: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_index = torch.argmax(posterior_probabilities, dim=-1)
        map_support = torch.gather(
            support[:, None, :].expand(-1, posterior_probabilities.shape[1], -1),
            2,
            map_index.unsqueeze(-1),
        ).squeeze(-1)
        posterior_entropy = cast(torch.Tensor, entropy(posterior_probabilities))
        prior_entropy = cast(torch.Tensor, entropy(prior_probabilities))[
            :, None
        ].expand(
            -1,
            posterior_probabilities.shape[1],
        )
        mutual_information = torch.clamp(prior_entropy - posterior_entropy, min=0.0)
        return map_support, posterior_entropy, prior_entropy, mutual_information


class _BinomialPosteriorInferencer(_BasePosteriorInferencer):
    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
        effective_exposure_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_likelihood = log_binomial_likelihood_support(
            counts,
            effective_exposure_values,
            support,
        )
        posterior_probabilities = posterior_from_log_likelihood(
            log_likelihood,
            prior_probabilities,
        )
        posterior_probabilities = cast(torch.Tensor, posterior_probabilities)
        map_support, posterior_entropy, prior_entropy, mutual_information = (
            self._summarize(
                posterior_probabilities,
                support,
                prior_probabilities,
            )
        )
        return (
            posterior_probabilities,
            map_support,
            posterior_entropy,
            prior_entropy,
            mutual_information,
        )


class _NegativeBinomialPosteriorInferencer(_BasePosteriorInferencer):
    def __init__(self, overdispersion: float) -> None:
        super().__init__()
        self.overdispersion = float(overdispersion)

    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
        effective_exposure_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_likelihood = log_negative_binomial_likelihood_support(
            counts,
            effective_exposure_values,
            support,
            overdispersion=self.overdispersion,
        )
        posterior_probabilities = posterior_from_log_likelihood(
            log_likelihood,
            prior_probabilities,
        )
        posterior_probabilities = cast(torch.Tensor, posterior_probabilities)
        map_support, posterior_entropy, prior_entropy, mutual_information = (
            self._summarize(
                posterior_probabilities,
                support,
                prior_probabilities,
            )
        )
        return (
            posterior_probabilities,
            map_support,
            posterior_entropy,
            prior_entropy,
            mutual_information,
        )


class _PoissonPosteriorInferencer(_BasePosteriorInferencer):
    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_likelihood = log_poisson_likelihood_support(counts, support)
        posterior_probabilities = posterior_from_log_likelihood(
            log_likelihood,
            prior_probabilities,
        )
        posterior_probabilities = cast(torch.Tensor, posterior_probabilities)
        map_support, posterior_entropy, prior_entropy, mutual_information = (
            self._summarize(
                posterior_probabilities,
                support,
                prior_probabilities,
            )
        )
        return (
            posterior_probabilities,
            map_support,
            posterior_entropy,
            prior_entropy,
            mutual_information,
        )


_COMPILED_INFERENCERS: dict[
    tuple[str, str, str, float | None, bool],
    torch.nn.Module,
] = {}


def _cache_device_key(device: torch.device) -> str:
    return str(device)


def _maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    try:
        return cast(torch.nn.Module, torch.compile(module))
    except Exception:
        return module


def _get_inferencer(
    distribution: Literal["binomial", "negative_binomial", "poisson"],
    *,
    device: torch.device,
    dtype: torch.dtype,
    nb_overdispersion: float,
    compile_model: bool,
) -> torch.nn.Module:
    dtype_name = str(dtype).removeprefix("torch.")
    overdispersion_key = (
        nb_overdispersion if distribution == "negative_binomial" else None
    )
    cache_key = (
        distribution,
        _cache_device_key(device),
        dtype_name,
        overdispersion_key,
        bool(compile_model),
    )
    cached = _COMPILED_INFERENCERS.get(cache_key)
    if cached is not None:
        return cached

    if distribution == "binomial":
        module: torch.nn.Module = _BinomialPosteriorInferencer()
    elif distribution == "negative_binomial":
        module = _NegativeBinomialPosteriorInferencer(nb_overdispersion)
    else:
        module = _PoissonPosteriorInferencer()

    module = module.to(device=device, dtype=dtype)
    compiled = _maybe_compile(module) if compile_model else module
    _COMPILED_INFERENCERS[cache_key] = compiled
    return compiled


def infer_posteriors(
    batch: ObservationBatch | GeneBatch,
    prior: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: str = "float64",
    posterior_distribution: PosteriorDistribution = "auto",
    nb_overdispersion: float = 0.01,
    compile_model: bool = True,
) -> InferenceResult:
    if isinstance(batch, GeneBatch):
        batch = batch.to_observation_batch()
    batch.check_shape()
    prior = prior.select_genes(batch.gene_names).as_gene_specific()
    resolved_distribution = _resolve_distribution(prior, posterior_distribution)

    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    inferencer = _get_inferencer(
        resolved_distribution,
        device=device_obj,
        dtype=dtype_obj,
        nb_overdispersion=nb_overdispersion,
        compile_model=compile_model,
    )

    counts_t = torch.as_tensor(batch.counts.T, dtype=dtype_obj, device=device_obj)
    support_t = torch.as_tensor(prior.support, dtype=dtype_obj, device=device_obj)
    prior_probabilities_t = torch.as_tensor(
        prior.prior_probabilities,
        dtype=dtype_obj,
        device=device_obj,
    )

    if resolved_distribution == "poisson":
        (
            posterior_probabilities_t,
            map_support_t,
            posterior_entropy_t,
            prior_entropy_t,
            mutual_information_t,
        ) = cast(_PoissonPosteriorInferencer, inferencer)(
            counts_t,
            support_t,
            prior_probabilities_t,
        )
    else:
        exposure_values = effective_exposure(batch.reference_counts, prior.scale)
        if resolved_distribution == "binomial":
            validate_binomial_observations(batch.counts, exposure_values)
        exposure_t = (
            torch.as_tensor(
                exposure_values,
                dtype=dtype_obj,
                device=device_obj,
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )
        (
            posterior_probabilities_t,
            map_support_t,
            posterior_entropy_t,
            prior_entropy_t,
            mutual_information_t,
        ) = cast(
            _BinomialPosteriorInferencer | _NegativeBinomialPosteriorInferencer,
            inferencer,
        )(
            counts_t,
            support_t,
            prior_probabilities_t,
            exposure_t,
        )

    return InferenceResult(
        gene_names=list(batch.gene_names),
        support_domain=prior.support_domain,
        support=np.asarray(prior.support, dtype=DTYPE_NP),
        prior_probabilities=np.asarray(prior.prior_probabilities, dtype=DTYPE_NP),
        map_support=map_support_t.detach().cpu().numpy().T.astype(DTYPE_NP, copy=False),
        posterior_entropy=posterior_entropy_t.detach()
        .cpu()
        .numpy()
        .T.astype(DTYPE_NP, copy=False),
        prior_entropy=prior_entropy_t.detach()
        .cpu()
        .numpy()
        .T.astype(DTYPE_NP, copy=False),
        mutual_information=mutual_information_t.detach()
        .cpu()
        .numpy()
        .T.astype(DTYPE_NP, copy=False),
        posterior_probabilities=(
            posterior_probabilities_t.detach()
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
            .astype(DTYPE_NP, copy=False)
            if include_posterior
            else None
        ),
    )


__all__ = ["PosteriorDistribution", "infer_posteriors"]

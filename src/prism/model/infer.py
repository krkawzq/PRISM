from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch

from .constants import DTYPE_NP, TorchDtypeName, resolve_torch_dtype
from .exposure import effective_exposure
from .numeric import (
    entropy,
    log_binomial_likelihood_grid,
    log_negative_binomial_likelihood_grid,
    log_poisson_likelihood_grid,
    posterior_from_log_likelihood,
)
from .types import GeneBatch, InferenceResult, ObservationBatch, PriorGrid

SignalChannel = Literal[
    "signal",
    "map_p",
    "map_mu",
    "map_rate",
    "posterior_entropy",
    "prior_entropy",
    "mutual_information",
    "posterior",
    "support",
]

CORE_CHANNELS = cast(
    frozenset[SignalChannel],
    frozenset({"signal", "posterior_entropy", "prior_entropy", "mutual_information"}),
)
ALL_CHANNELS = cast(
    frozenset[SignalChannel],
    frozenset(
        set(CORE_CHANNELS) | {"map_p", "map_mu", "map_rate", "posterior", "support"}
    ),
)


def infer_posteriors(
    batch: ObservationBatch | GeneBatch,
    priors: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: TorchDtypeName | str = "float64",
    posterior_distribution: Literal[
        "auto", "binomial", "negative_binomial", "poisson"
    ] = "auto",
    nb_overdispersion: float = 0.01,
) -> InferenceResult:
    if hasattr(batch, "to_observation_batch"):
        batch = batch.to_observation_batch()  # type: ignore[assignment]
    batch.check_shape()
    priors = priors.subset(batch.gene_names).batched()
    priors.check_shape()
    if posterior_distribution == "auto":
        resolved_distribution = priors.distribution
    else:
        resolved_distribution = posterior_distribution
    if priors.distribution != resolved_distribution:
        raise ValueError(
            "posterior distribution mismatch with prior distribution; "
            f"priors.distribution={priors.distribution!r}, "
            f"posterior_distribution={resolved_distribution!r}"
        )
    if priors.grid_domain == "rate" and resolved_distribution != "poisson":
        raise ValueError(
            "rate-grid priors require poisson posterior inference; "
            f"got posterior_distribution={resolved_distribution!r}"
        )
    if priors.grid_domain == "p" and resolved_distribution == "poisson":
        raise ValueError("poisson posterior inference requires grid_domain='rate'")
    device_obj = torch.device(device)
    dtype_obj = resolve_torch_dtype(torch_dtype)
    counts_t = torch.as_tensor(batch.counts.T, dtype=dtype_obj, device=device_obj)
    p_grid_t = torch.as_tensor(priors.p_grid, dtype=dtype_obj, device=device_obj)
    weights_t = torch.as_tensor(priors.weights, dtype=dtype_obj, device=device_obj)
    if resolved_distribution == "poisson":
        log_lik = log_poisson_likelihood_grid(counts_t, p_grid_t)
    else:
        n_eff_t = (
            torch.as_tensor(
                effective_exposure(batch.reference_counts, priors.S),
                dtype=dtype_obj,
                device=device_obj,
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )
        if resolved_distribution == "negative_binomial":
            log_lik = log_negative_binomial_likelihood_grid(
                counts_t, n_eff_t, p_grid_t, overdispersion=nb_overdispersion
            )
        else:
            log_lik = log_binomial_likelihood_grid(counts_t, n_eff_t, p_grid_t)
    posterior_t = posterior_from_log_likelihood(log_lik, weights_t)
    map_idx = torch.argmax(posterior_t, dim=-1)
    p_support = p_grid_t[:, None, :].expand(-1, batch.n_cells, -1)
    support_values = torch.gather(p_support, 2, map_idx.unsqueeze(-1)).squeeze(-1)
    if resolved_distribution == "poisson":
        map_rate = support_values
        map_p = torch.full_like(map_rate, torch.nan)
        map_mu = map_rate
    else:
        map_rate = None
        map_p = support_values
        map_mu = map_p * float(priors.S)
    posterior_entropy = entropy(posterior_t)
    prior_entropy = entropy(weights_t)[:, None].expand(-1, batch.n_cells)
    mutual_information = torch.clamp(prior_entropy - posterior_entropy, min=0.0)
    return InferenceResult(
        gene_names=list(batch.gene_names),
        grid_domain=priors.grid_domain,
        p_grid=np.asarray(priors.p_grid, dtype=DTYPE_NP),
        mu_grid=np.asarray(priors.mu_grid, dtype=DTYPE_NP),
        prior_weights=np.asarray(priors.weights, dtype=DTYPE_NP),
        map_p=map_p.detach().cpu().numpy().T,
        map_mu=map_mu.detach().cpu().numpy().T,
        map_rate=None if map_rate is None else map_rate.detach().cpu().numpy().T,
        posterior_entropy=posterior_entropy.detach().cpu().numpy().T,
        prior_entropy=prior_entropy.detach().cpu().numpy().T,
        mutual_information=mutual_information.detach().cpu().numpy().T,
        posterior=posterior_t.detach().cpu().numpy().transpose(1, 0, 2)
        if include_posterior
        else None,
    )

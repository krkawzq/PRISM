from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch

from .constants import DTYPE_NP, DTYPE_TORCH
from .exposure import effective_exposure
from .numeric import (
    entropy,
    log_binomial_likelihood_grid,
    posterior_from_log_likelihood,
)
from .types import GeneBatch, InferenceResult, ObservationBatch, PriorGrid

SignalChannel = Literal[
    "signal",
    "map_p",
    "map_mu",
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
    frozenset(set(CORE_CHANNELS) | {"map_p", "map_mu", "posterior", "support"}),
)


def infer_posteriors(
    batch: ObservationBatch | GeneBatch,
    priors: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
) -> InferenceResult:
    if hasattr(batch, "to_observation_batch"):
        batch = batch.to_observation_batch()  # type: ignore[assignment]
    batch.check_shape()
    priors = priors.subset(batch.gene_names).batched()
    priors.check_shape()
    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=DTYPE_TORCH, device=device_obj)
    n_eff_t = (
        torch.as_tensor(
            effective_exposure(batch.reference_counts, priors.S),
            dtype=DTYPE_TORCH,
            device=device_obj,
        )
        .unsqueeze(0)
        .expand(batch.n_genes, -1)
    )
    p_grid_t = torch.as_tensor(priors.p_grid, dtype=DTYPE_TORCH, device=device_obj)
    weights_t = torch.as_tensor(priors.weights, dtype=DTYPE_TORCH, device=device_obj)
    log_lik = log_binomial_likelihood_grid(counts_t, n_eff_t, p_grid_t)
    posterior_t = posterior_from_log_likelihood(log_lik, weights_t)
    map_idx = torch.argmax(posterior_t, dim=-1)
    p_support = p_grid_t[:, None, :].expand(-1, batch.n_cells, -1)
    map_p = torch.gather(p_support, 2, map_idx.unsqueeze(-1)).squeeze(-1)
    map_mu = map_p * float(priors.S)
    posterior_entropy = entropy(posterior_t)
    prior_entropy = entropy(weights_t)[:, None].expand(-1, batch.n_cells)
    mutual_information = torch.clamp(prior_entropy - posterior_entropy, min=0.0)
    return InferenceResult(
        gene_names=list(batch.gene_names),
        p_grid=np.asarray(priors.p_grid, dtype=DTYPE_NP),
        mu_grid=np.asarray(priors.mu_grid, dtype=DTYPE_NP),
        prior_weights=np.asarray(priors.weights, dtype=DTYPE_NP),
        map_p=map_p.detach().cpu().numpy().T,
        map_mu=map_mu.detach().cpu().numpy().T,
        posterior_entropy=posterior_entropy.detach().cpu().numpy().T,
        prior_entropy=prior_entropy.detach().cpu().numpy().T,
        mutual_information=mutual_information.detach().cpu().numpy().T,
        posterior=posterior_t.detach().cpu().numpy().transpose(1, 0, 2)
        if include_posterior
        else None,
    )

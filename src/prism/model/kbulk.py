from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import DTYPE_NP, DTYPE_TORCH, EPS
from .numeric import entropy, log_binomial_likelihood_grid
from .types import PriorGrid


@dataclass(frozen=True, slots=True)
class KBulkBatch:
    gene_names: list[str]
    counts: np.ndarray  # (K, G)
    effective_exposure: np.ndarray  # (K,)

    @property
    def k(self) -> int:
        return int(np.asarray(self.effective_exposure).reshape(-1).shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        if not self.gene_names:
            raise ValueError("gene_names cannot be empty")
        if len(self.gene_names) != len(set(self.gene_names)):
            raise ValueError("gene_names must be unique")
        counts = np.asarray(self.counts, dtype=DTYPE_NP)
        n_eff = np.asarray(self.effective_exposure, dtype=DTYPE_NP).reshape(-1)
        if counts.ndim != 2:
            raise ValueError(f"counts must be 2D, got shape={counts.shape}")
        if n_eff.ndim != 1:
            raise ValueError(f"effective_exposure must be 1D, got shape={n_eff.shape}")
        if counts.shape != (n_eff.shape[0], len(self.gene_names)):
            raise ValueError(
                "counts shape must equal (k, n_genes), "
                f"got {counts.shape} vs {(n_eff.shape[0], len(self.gene_names))}"
            )
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("counts must be finite and non-negative")
        if np.any(~np.isfinite(n_eff)) or np.any(n_eff <= 0):
            raise ValueError("effective_exposure must be finite and positive")
        if np.any(counts > n_eff[:, None] + 1e-9):
            raise ValueError(
                "counts must satisfy counts <= effective_exposure elementwise"
            )


@dataclass(frozen=True, slots=True)
class KBulkResult:
    gene_names: list[str]
    p_grid: np.ndarray  # (G, M)
    mu_grid: np.ndarray  # (G, M)
    prior_weights: np.ndarray  # (G, M)
    posterior_weights: np.ndarray | None  # (G, M)
    map_p: np.ndarray  # (G,)
    map_mu: np.ndarray  # (G,)
    posterior_entropy: np.ndarray  # (G,)
    prior_entropy: np.ndarray  # (G,)
    mutual_information: np.ndarray  # (G,)


def infer_kbulk(
    batch: KBulkBatch,
    priors: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
) -> KBulkResult:
    batch.check_shape()
    priors = priors.subset(batch.gene_names).batched()
    priors.check_shape()

    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=DTYPE_TORCH, device=device_obj)
    n_eff_t = (
        torch.as_tensor(
            np.asarray(batch.effective_exposure, dtype=DTYPE_NP),
            dtype=DTYPE_TORCH,
            device=device_obj,
        )
        .unsqueeze(0)
        .expand(batch.n_genes, -1)
    )
    p_grid_t = torch.as_tensor(priors.p_grid, dtype=DTYPE_TORCH, device=device_obj)
    weights_t = torch.as_tensor(priors.weights, dtype=DTYPE_TORCH, device=device_obj)

    log_lik_t = log_binomial_likelihood_grid(counts_t, n_eff_t, p_grid_t)
    joint_log_lik_t = log_lik_t.sum(dim=-2)
    joint_log_post_t = joint_log_lik_t + torch.log(weights_t.clamp_min(EPS))
    joint_log_post_t = joint_log_post_t - torch.logsumexp(
        joint_log_post_t, dim=-1, keepdim=True
    )
    posterior_t = torch.exp(joint_log_post_t)

    map_idx_t = torch.argmax(posterior_t, dim=-1)
    map_p_t = torch.gather(p_grid_t, 1, map_idx_t.unsqueeze(-1)).squeeze(-1)
    map_mu_t = map_p_t * float(priors.S)
    posterior_entropy_t = entropy(posterior_t)
    prior_entropy_t = entropy(weights_t)
    mutual_information_t = torch.clamp(
        prior_entropy_t - posterior_entropy_t,
        min=0.0,
    )

    return KBulkResult(
        gene_names=list(batch.gene_names),
        p_grid=np.asarray(priors.p_grid, dtype=DTYPE_NP),
        mu_grid=np.asarray(priors.mu_grid, dtype=DTYPE_NP),
        prior_weights=np.asarray(priors.weights, dtype=DTYPE_NP),
        posterior_weights=posterior_t.detach().cpu().numpy()
        if include_posterior
        else None,
        map_p=map_p_t.detach().cpu().numpy(),
        map_mu=map_mu_t.detach().cpu().numpy(),
        posterior_entropy=posterior_entropy_t.detach().cpu().numpy(),
        prior_entropy=prior_entropy_t.detach().cpu().numpy(),
        mutual_information=mutual_information_t.detach().cpu().numpy(),
    )


class KBulkAggregator:
    def __init__(
        self,
        gene_names: list[str],
        priors: PriorGrid,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not gene_names:
            raise ValueError("gene_names cannot be empty")
        self.gene_names = list(gene_names)
        self.priors = priors.subset(gene_names)
        self.device = device

    def aggregate(
        self,
        batch: KBulkBatch,
        *,
        include_posterior: bool = False,
    ) -> KBulkResult:
        return infer_kbulk(
            batch,
            self.priors,
            device=self.device,
            include_posterior=include_posterior,
        )

    def query(
        self,
        counts: np.ndarray,
        effective_exposure: np.ndarray,
        *,
        include_posterior: bool = False,
    ) -> KBulkResult:
        batch = KBulkBatch(
            gene_names=list(self.gene_names),
            counts=np.asarray(counts, dtype=DTYPE_NP),
            effective_exposure=np.asarray(effective_exposure, dtype=DTYPE_NP),
        )
        return self.aggregate(batch, include_posterior=include_posterior)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import DTYPE_NP, EPS
from .numeric import entropy, log_binomial_likelihood_grid
from .types import InferenceResult, PriorGrid


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch dtype: {name}")


@dataclass(frozen=True, slots=True)
class KBulkBatch:
    gene_names: list[str]
    counts: np.ndarray
    effective_exposure: np.ndarray

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
            raise ValueError("counts must satisfy counts <= effective_exposure")


@dataclass(frozen=True, slots=True)
class KBulkResult:
    gene_names: list[str]
    p_grid: np.ndarray
    mu_grid: np.ndarray
    prior_weights: np.ndarray
    posterior_weights: np.ndarray | None
    map_p: np.ndarray
    map_mu: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray


def _validate_sample_inputs(
    gene_names: list[str],
    aggregated_counts: np.ndarray,
    effective_exposure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts_np = np.asarray(aggregated_counts, dtype=np.float32)
    exposure_np = np.asarray(effective_exposure, dtype=np.float32).reshape(-1)
    if counts_np.ndim != 2:
        raise ValueError(f"aggregated_counts must be 2D, got shape={counts_np.shape}")
    if exposure_np.ndim != 1:
        raise ValueError(
            f"effective_exposure must be 1D, got shape={exposure_np.shape}"
        )
    if counts_np.shape != (exposure_np.shape[0], len(gene_names)):
        raise ValueError(
            "aggregated_counts shape must equal (n_samples, n_genes), "
            f"got {counts_np.shape} vs {(exposure_np.shape[0], len(gene_names))}"
        )
    if np.any(~np.isfinite(counts_np)) or np.any(counts_np < 0):
        raise ValueError("aggregated_counts must be finite and non-negative")
    if np.any(~np.isfinite(exposure_np)) or np.any(exposure_np <= 0):
        raise ValueError("effective_exposure must be finite and positive")
    if np.any(counts_np > exposure_np[:, None] + 1e-6):
        raise ValueError("aggregated_counts must satisfy counts <= effective_exposure")
    return counts_np, exposure_np


def _run_kbulk_inference(
    *,
    gene_names: list[str],
    counts_np: np.ndarray,
    exposure_np: np.ndarray,
    p_grid_np: np.ndarray,
    weights_np: np.ndarray,
    S: float,
    device: str | torch.device,
    torch_dtype: str,
    include_posterior: bool,
) -> InferenceResult:
    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    counts_t = torch.as_tensor(counts_np.T, dtype=dtype_obj, device=device_obj)
    n_eff_t = (
        torch.as_tensor(exposure_np, dtype=dtype_obj, device=device_obj)
        .unsqueeze(0)
        .expand(len(gene_names), -1)
    )
    p_grid_t = torch.as_tensor(p_grid_np, dtype=dtype_obj, device=device_obj)
    weights_t = torch.as_tensor(weights_np, dtype=dtype_obj, device=device_obj)
    return _run_kbulk_inference_tensors(
        gene_names=gene_names,
        counts_t=counts_t,
        n_eff_t=n_eff_t,
        p_grid_t=p_grid_t,
        weights_t=weights_t,
        p_grid_np=np.asarray(p_grid_np, dtype=DTYPE_NP),
        mu_grid_np=np.asarray(p_grid_np, dtype=DTYPE_NP) * float(S),
        prior_weights_np=np.asarray(weights_np, dtype=DTYPE_NP),
        S=float(S),
        include_posterior=include_posterior,
    )


def _run_kbulk_inference_tensors(
    *,
    gene_names: list[str],
    counts_t: torch.Tensor,
    n_eff_t: torch.Tensor,
    p_grid_t: torch.Tensor,
    weights_t: torch.Tensor,
    p_grid_np: np.ndarray,
    mu_grid_np: np.ndarray,
    prior_weights_np: np.ndarray,
    S: float,
    include_posterior: bool,
) -> InferenceResult:
    log_lik_t = log_binomial_likelihood_grid(counts_t, n_eff_t, p_grid_t)
    joint_log_post_t = log_lik_t + torch.log(weights_t.clamp_min(EPS)).unsqueeze(-2)
    joint_log_post_t = joint_log_post_t - torch.logsumexp(
        joint_log_post_t, dim=-1, keepdim=True
    )
    posterior_t = torch.exp(joint_log_post_t)
    map_idx_t = torch.argmax(posterior_t, dim=-1)
    p_support_t = p_grid_t[:, None, :].expand(-1, counts_t.shape[1], -1)
    map_p_t = torch.gather(p_support_t, 2, map_idx_t.unsqueeze(-1)).squeeze(-1)
    map_mu_t = map_p_t * float(S)
    posterior_entropy_t = entropy(posterior_t)
    prior_entropy_t = entropy(weights_t)[:, None].expand(-1, counts_t.shape[1])
    mutual_information_t = torch.clamp(prior_entropy_t - posterior_entropy_t, min=0.0)
    return InferenceResult(
        gene_names=list(gene_names),
        p_grid=p_grid_np,
        mu_grid=mu_grid_np,
        prior_weights=prior_weights_np,
        map_p=map_p_t.detach().cpu().numpy().T.astype(DTYPE_NP, copy=False),
        map_mu=map_mu_t.detach().cpu().numpy().T.astype(DTYPE_NP, copy=False),
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
        posterior=posterior_t.detach()
        .cpu()
        .numpy()
        .transpose(1, 0, 2)
        .astype(DTYPE_NP, copy=False)
        if include_posterior
        else None,
    )


def infer_kbulk_samples(
    gene_names: list[str],
    aggregated_counts: np.ndarray,
    effective_exposure: np.ndarray,
    priors: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: str = "float32",
) -> InferenceResult:
    counts_np, exposure_np = _validate_sample_inputs(
        gene_names, aggregated_counts, effective_exposure
    )
    priors = priors.subset(gene_names).batched()
    priors.check_shape()
    return _run_kbulk_inference(
        gene_names=gene_names,
        counts_np=counts_np,
        exposure_np=exposure_np,
        p_grid_np=np.asarray(priors.p_grid, dtype=np.float32),
        weights_np=np.asarray(priors.weights, dtype=np.float32),
        S=float(priors.S),
        device=device,
        torch_dtype=torch_dtype,
        include_posterior=include_posterior,
    )


def infer_kbulk_samples_with_priors(
    gene_names: list[str],
    aggregated_counts: np.ndarray,
    effective_exposure: np.ndarray,
    priors: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: str = "float32",
) -> InferenceResult:
    counts_np, exposure_np = _validate_sample_inputs(
        gene_names, aggregated_counts, effective_exposure
    )
    priors.check_shape()
    return _run_kbulk_inference(
        gene_names=gene_names,
        counts_np=counts_np,
        exposure_np=exposure_np,
        p_grid_np=np.asarray(priors.p_grid, dtype=np.float32),
        weights_np=np.asarray(priors.weights, dtype=np.float32),
        S=float(priors.S),
        device=device,
        torch_dtype=torch_dtype,
        include_posterior=include_posterior,
    )


def infer_kbulk(
    batch: KBulkBatch,
    priors: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: str = "float32",
) -> KBulkResult:
    batch.check_shape()
    aggregated_counts = np.asarray(batch.counts, dtype=np.float32).sum(
        axis=0, keepdims=True
    )
    aggregated_exposure = np.asarray(batch.effective_exposure, dtype=np.float32).sum(
        keepdims=True
    )
    result = infer_kbulk_samples(
        batch.gene_names,
        aggregated_counts,
        aggregated_exposure,
        priors,
        device=device,
        include_posterior=include_posterior,
        torch_dtype=torch_dtype,
    )
    return KBulkResult(
        gene_names=list(result.gene_names),
        p_grid=result.p_grid,
        mu_grid=result.mu_grid,
        prior_weights=result.prior_weights,
        posterior_weights=None
        if result.posterior is None
        else np.asarray(result.posterior[0], dtype=DTYPE_NP),
        map_p=np.asarray(result.map_p[0], dtype=DTYPE_NP),
        map_mu=np.asarray(result.map_mu[0], dtype=DTYPE_NP),
        posterior_entropy=np.asarray(result.posterior_entropy[0], dtype=DTYPE_NP),
        prior_entropy=np.asarray(result.prior_entropy[0], dtype=DTYPE_NP),
        mutual_information=np.asarray(result.mutual_information[0], dtype=DTYPE_NP),
    )


class KBulkAggregator:
    def __init__(
        self,
        gene_names: list[str],
        priors: PriorGrid,
        *,
        device: str | torch.device = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        if not gene_names:
            raise ValueError("gene_names cannot be empty")
        self.gene_names = list(gene_names)
        self.priors = priors.subset(gene_names).batched()
        self.priors.check_shape()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.dtype_obj = _resolve_torch_dtype(torch_dtype)
        self.p_grid_np = np.asarray(self.priors.p_grid, dtype=np.float32)
        self.weights_np = np.asarray(self.priors.weights, dtype=np.float32)
        self.mu_grid_np = np.asarray(self.priors.mu_grid, dtype=DTYPE_NP)
        self.prior_weights_np = np.asarray(self.priors.weights, dtype=DTYPE_NP)
        self.p_grid_t = torch.as_tensor(
            self.p_grid_np, dtype=self.dtype_obj, device=self.device
        )
        self.weights_t = torch.as_tensor(
            self.weights_np, dtype=self.dtype_obj, device=self.device
        )

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
            torch_dtype=self.torch_dtype,
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

    def query_samples(
        self,
        aggregated_counts: np.ndarray,
        effective_exposure: np.ndarray,
        *,
        include_posterior: bool = False,
    ) -> InferenceResult:
        counts_np, exposure_np = _validate_sample_inputs(
            self.gene_names, aggregated_counts, effective_exposure
        )
        counts_t = torch.as_tensor(
            counts_np.T, dtype=self.dtype_obj, device=self.device
        )
        n_eff_t = (
            torch.as_tensor(exposure_np, dtype=self.dtype_obj, device=self.device)
            .unsqueeze(0)
            .expand(len(self.gene_names), -1)
        )
        return _run_kbulk_inference_tensors(
            gene_names=self.gene_names,
            counts_t=counts_t,
            n_eff_t=n_eff_t,
            p_grid_t=self.p_grid_t,
            weights_t=self.weights_t,
            p_grid_np=np.asarray(self.priors.p_grid, dtype=DTYPE_NP),
            mu_grid_np=self.mu_grid_np,
            prior_weights_np=self.prior_weights_np,
            S=float(self.priors.S),
            include_posterior=include_posterior,
        )

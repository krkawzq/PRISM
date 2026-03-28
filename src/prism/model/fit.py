from __future__ import annotations

from dataclasses import asdict
from typing import Callable, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from .constants import DTYPE_NP, EPS, OptimizerName, SchedulerName, TorchTensor
from .exposure import effective_exposure, mean_reference_count
from .numeric import (
    aggregate_posterior,
    jsd,
    log_binomial_likelihood_grid,
    posterior_from_log_likelihood,
    smooth_probability_weights,
)
from .types import (
    GeneBatch,
    ObservationBatch,
    PriorFitConfig,
    PriorFitResult,
    PriorGrid,
    ScaleMetadata,
)

FitProgressCallback = Callable[[int, int, float, float, float, bool], None]


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch dtype: {name}")


def _iter_cell_slices(n_cells: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_cells))
        for start in range(0, n_cells, chunk_size)
    ]


def _build_optimizer(
    params: list[torch.nn.Parameter], name: OptimizerName, lr: float
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"unsupported optimizer: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: SchedulerName,
    n_iter: int,
    lr: float,
    lr_min_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_iters = max(n_iter, 1)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters,
            eta_min=lr * lr_min_ratio,
        )
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=max(lr_min_ratio, 0.0),
            total_iters=total_iters,
        )
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_iters
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(total_iters // 3, 1),
            gamma=max(lr_min_ratio, EPS),
        )
    raise ValueError(f"unsupported scheduler: {name}")


def _default_p_grid_max(batch: ObservationBatch, S: float) -> np.ndarray:
    mu_max = (
        np.max(batch.counts, axis=0)
        * float(S)
        / max(mean_reference_count(batch.reference_counts), EPS)
    )
    p_max = np.clip(mu_max / float(S), 1e-8, 1.0 - 1e-8)
    return np.asarray(p_max, dtype=DTYPE_NP)


def _build_p_grid(
    grid_size: int, p_grid_max: np.ndarray, *, dtype: torch.dtype, device: torch.device
) -> TorchTensor:
    base = torch.linspace(0.0, 1.0, grid_size, dtype=dtype, device=device)
    p_max = torch.as_tensor(p_grid_max, dtype=dtype, device=device)
    return (p_max[:, None] * base[None, :]).clamp(0.0, 1.0)


def fit_gene_priors(
    batch: ObservationBatch | GeneBatch,
    *,
    S: float,
    config: PriorFitConfig = PriorFitConfig(),
    device: str | torch.device = "cpu",
    p_grid_max: np.ndarray | None = None,
    progress_callback: FitProgressCallback | None = None,
) -> PriorFitResult:
    if isinstance(batch, GeneBatch):
        batch = ObservationBatch(
            gene_names=list(batch.gene_names),
            counts=np.asarray(batch.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(batch.totals, dtype=DTYPE_NP),
        )
    batch = cast(ObservationBatch, batch)
    batch.check_shape()
    if not np.isfinite(S) or S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    if p_grid_max is None:
        p_grid_max = _default_p_grid_max(batch, S)
    p_grid_max = np.asarray(p_grid_max, dtype=DTYPE_NP).reshape(-1)
    if p_grid_max.shape != (batch.n_genes,):
        raise ValueError(
            f"p_grid_max must have shape {(batch.n_genes,)}, got {p_grid_max.shape}"
        )
    if (
        np.any(~np.isfinite(p_grid_max))
        or np.any(p_grid_max <= 0)
        or np.any(p_grid_max > 1)
    ):
        raise ValueError("p_grid_max must lie in (0, 1]")

    torch_dtype = _resolve_torch_dtype(config.torch_dtype)
    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=torch_dtype, device=device_obj)
    n_eff_t = (
        torch.as_tensor(
            effective_exposure(batch.reference_counts, S),
            dtype=torch_dtype,
            device=device_obj,
        )
        .unsqueeze(0)
        .expand(batch.n_genes, -1)
    )
    p_grid_t = _build_p_grid(
        config.grid_size,
        p_grid_max,
        dtype=torch_dtype,
        device=device_obj,
    )
    cell_slices = _iter_cell_slices(batch.n_cells, config.cell_chunk_size)

    with torch.no_grad():
        uniform = torch.full(
            (batch.n_genes, config.grid_size),
            1.0 / config.grid_size,
            dtype=torch_dtype,
            device=device_obj,
        )
        posterior_sum = torch.zeros_like(uniform)
        for cell_slice in cell_slices:
            log_lik = log_binomial_likelihood_grid(
                counts_t[:, cell_slice],
                n_eff_t[:, cell_slice],
                p_grid_t,
            )
            if config.init_temperature != 1.0:
                log_lik = log_lik / config.init_temperature
            posterior_sum += posterior_from_log_likelihood(log_lik, uniform).sum(dim=-2)
        init_post = (posterior_sum / batch.n_cells).clamp_min(EPS)
        init_post = init_post / init_post.sum(dim=-1, keepdim=True)
        init_logits = torch.log(init_post) / max(config.init_temperature, EPS)

    logits = torch.nn.Parameter(init_logits.clone())
    optimizer = _build_optimizer([logits], config.optimizer, config.lr)
    scheduler = _build_scheduler(
        optimizer,
        name=config.scheduler,
        n_iter=config.n_iter,
        lr=config.lr,
        lr_min_ratio=config.lr_min_ratio,
    )

    best_loss = float("inf")
    best_logits = logits.detach().clone()
    best_q_hat = init_post.detach().cpu().numpy()
    loss_history: list[float] = []
    nll_history: list[float] = []
    align_history: list[float] = []
    final_loss = float("inf")

    for step in range(1, config.n_iter + 1):
        optimizer.zero_grad(set_to_none=True)
        prior_weights = smooth_probability_weights(logits, config.sigma_bins)
        with torch.no_grad():
            posterior_sum = torch.zeros_like(prior_weights)
            for cell_slice in cell_slices:
                log_lik = log_binomial_likelihood_grid(
                    counts_t[:, cell_slice],
                    n_eff_t[:, cell_slice],
                    p_grid_t,
                )
                posterior_sum += posterior_from_log_likelihood(
                    log_lik, prior_weights
                ).sum(dim=-2)
            q_hat = (posterior_sum / batch.n_cells).clamp_min(EPS)
            q_hat = q_hat / q_hat.sum(dim=-1, keepdim=True)

        align = jsd(q_hat, prior_weights).mean() * config.align_loss_weight
        log_prior = torch.log(prior_weights.clamp_min(EPS)).unsqueeze(-2)
        nll_value = 0.0
        total_loss = align
        for cell_slice in cell_slices:
            log_lik = log_binomial_likelihood_grid(
                counts_t[:, cell_slice],
                n_eff_t[:, cell_slice],
                p_grid_t,
            )
            log_marginal = torch.logsumexp(log_lik + log_prior, dim=-1)
            nll_chunk = -(log_marginal.sum(dim=-1) / batch.n_cells).mean()
            total_loss = total_loss + nll_chunk
            nll_value += float(nll_chunk.item())

        total_loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([logits], config.grad_clip)
        optimizer.step()
        scheduler.step()

        final_loss = nll_value + float(align.item())
        loss_history.append(final_loss)
        nll_history.append(nll_value)
        align_history.append(float(align.item()))
        if final_loss < best_loss:
            best_loss = final_loss
            best_logits = logits.detach().clone()
            best_q_hat = q_hat.detach().cpu().numpy()
        if progress_callback is not None:
            progress_callback(
                step,
                config.n_iter,
                final_loss,
                nll_value,
                float(align.item()),
                final_loss <= best_loss,
            )

    best_weights = (
        smooth_probability_weights(best_logits, config.sigma_bins)
        .detach()
        .cpu()
        .numpy()
    )
    p_grid = p_grid_t.detach().cpu().numpy()
    return PriorFitResult(
        gene_names=list(batch.gene_names),
        priors=PriorGrid(
            gene_names=list(batch.gene_names),
            p_grid=p_grid,
            weights=best_weights,
            S=float(S),
        ),
        posterior_average=best_q_hat,
        initial_posterior_average=init_post.detach().cpu().numpy(),
        initial_prior_weights=smooth_probability_weights(init_logits, config.sigma_bins)
        .detach()
        .cpu()
        .numpy(),
        loss_history=loss_history,
        nll_history=nll_history,
        align_history=align_history,
        final_loss=final_loss,
        best_loss=best_loss,
        config=asdict(config),
        scale=ScaleMetadata(
            S=float(S),
            mean_reference_count=mean_reference_count(batch.reference_counts),
        ),
    )


def fit_gene_priors_em(
    batch: ObservationBatch | GeneBatch,
    *,
    S: float,
    config: PriorFitConfig = PriorFitConfig(),
    device: str | torch.device = "cpu",
    p_grid_max: np.ndarray | None = None,
    progress_callback: FitProgressCallback | None = None,
    tol: float = 1e-6,
) -> PriorFitResult:
    if isinstance(batch, GeneBatch):
        batch = ObservationBatch(
            gene_names=list(batch.gene_names),
            counts=np.asarray(batch.counts, dtype=DTYPE_NP),
            reference_counts=np.asarray(batch.totals, dtype=DTYPE_NP),
        )
    batch = cast(ObservationBatch, batch)
    batch.check_shape()
    if not np.isfinite(S) or S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    if p_grid_max is None:
        p_grid_max = _default_p_grid_max(batch, S)
    p_grid_max = np.asarray(p_grid_max, dtype=DTYPE_NP).reshape(-1)
    if p_grid_max.shape != (batch.n_genes,):
        raise ValueError(
            f"p_grid_max must have shape {(batch.n_genes,)}, got {p_grid_max.shape}"
        )
    if (
        np.any(~np.isfinite(p_grid_max))
        or np.any(p_grid_max <= 0)
        or np.any(p_grid_max > 1)
    ):
        raise ValueError("p_grid_max must lie in (0, 1]")

    torch_dtype = _resolve_torch_dtype(config.torch_dtype)
    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=torch_dtype, device=device_obj)
    n_eff_t = (
        torch.as_tensor(
            effective_exposure(batch.reference_counts, S),
            dtype=torch_dtype,
            device=device_obj,
        )
        .unsqueeze(0)
        .expand(batch.n_genes, -1)
    )
    p_grid_t = _build_p_grid(
        config.grid_size,
        p_grid_max,
        dtype=torch_dtype,
        device=device_obj,
    )
    cell_slices = _iter_cell_slices(batch.n_cells, config.cell_chunk_size)

    q = torch.full(
        (batch.n_genes, config.grid_size),
        1.0 / config.grid_size,
        dtype=torch_dtype,
        device=device_obj,
    )
    initial_q = q.detach().cpu().numpy().copy()
    loss_history: list[float] = []
    nll_history: list[float] = []

    with torch.no_grad():
        for step in range(1, config.n_iter + 1):
            posterior_sum = torch.zeros_like(q)
            nll_value = 0.0
            log_prior = torch.log(q.clamp_min(EPS)).unsqueeze(-2)
            for cell_slice in cell_slices:
                log_lik = log_binomial_likelihood_grid(
                    counts_t[:, cell_slice],
                    n_eff_t[:, cell_slice],
                    p_grid_t,
                )
                log_marginal = torch.logsumexp(log_lik + log_prior, dim=-1)
                nll_value += float(
                    -(log_marginal.sum(dim=-1) / batch.n_cells).mean().item()
                )
                posterior_sum += posterior_from_log_likelihood(log_lik, q).sum(dim=-2)

            q_new = (posterior_sum / batch.n_cells).clamp_min(EPS)
            q_new = q_new / q_new.sum(dim=-1, keepdim=True)
            delta = float(torch.max(torch.abs(q_new - q)).item())
            q = q_new

            loss_history.append(nll_value)
            nll_history.append(nll_value)
            if progress_callback is not None:
                progress_callback(step, config.n_iter, nll_value, nll_value, 0.0, True)
            if delta < tol:
                break

    q_np = q.detach().cpu().numpy()
    q_smooth = gaussian_filter1d(q_np, sigma=config.sigma_bins, axis=-1)
    q_smooth = np.clip(q_smooth, EPS, None)
    q_smooth = q_smooth / q_smooth.sum(axis=-1, keepdims=True)
    p_grid = p_grid_t.detach().cpu().numpy()
    final_loss = loss_history[-1] if loss_history else float("inf")
    best_loss = min(loss_history) if loss_history else float("inf")
    return PriorFitResult(
        gene_names=list(batch.gene_names),
        priors=PriorGrid(
            gene_names=list(batch.gene_names),
            p_grid=p_grid,
            weights=q_smooth,
            S=float(S),
        ),
        posterior_average=q_np,
        initial_posterior_average=initial_q,
        initial_prior_weights=initial_q.copy(),
        loss_history=loss_history,
        nll_history=nll_history,
        align_history=[0.0] * len(loss_history),
        final_loss=final_loss,
        best_loss=best_loss,
        config=asdict(config),
        scale=ScaleMetadata(
            S=float(S),
            mean_reference_count=mean_reference_count(batch.reference_counts),
        ),
    )

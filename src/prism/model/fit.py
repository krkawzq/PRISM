from __future__ import annotations

from dataclasses import asdict
from typing import Callable, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from .constants import (
    DTYPE_NP,
    EPS,
    OptimizerName,
    SchedulerName,
    TorchTensor,
    resolve_torch_dtype,
)
from .exposure import effective_exposure, mean_reference_count
from .numeric import (
    aggregate_posterior,
    entropy as _entropy_fn,
    jsd,
    kl_divergence,
    log_binomial_likelihood_grid,
    log_negative_binomial_likelihood_grid,
    log_poisson_likelihood_grid,
    posterior_from_log_likelihood,
    smooth_probability_weights,
    weighted_jsd,
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


def _default_p_grid_max(
    batch: ObservationBatch, S: float, *, method: str = "observed_max"
) -> np.ndarray:
    if method == "observed_max":
        mu_max = (
            np.max(batch.counts, axis=0)
            * float(S)
            / max(mean_reference_count(batch.reference_counts), EPS)
        )
    elif method == "quantile":
        quantile_value = np.nanpercentile(batch.counts, 95, axis=0)
        mu_max = (
            quantile_value
            * float(S)
            / max(mean_reference_count(batch.reference_counts), EPS)
        )
    else:
        raise ValueError(f"unsupported grid_max_method: {method}")
    p_max = np.clip(mu_max / float(S), 1e-8, 1.0 - 1e-8)
    return np.asarray(p_max, dtype=DTYPE_NP)


def _build_p_grid(
    grid_size: int,
    p_grid_max: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
    strategy: str = "linear",
) -> TorchTensor:
    p_max = torch.as_tensor(p_grid_max, dtype=dtype, device=device)
    if strategy == "sqrt":
        t = torch.linspace(0.0, 1.0, grid_size, dtype=dtype, device=device)
        base = torch.sqrt(t)
        base = base / base[-1]
    else:
        base = torch.linspace(0.0, 1.0, grid_size, dtype=dtype, device=device)
    return (p_max[:, None] * base[None, :]).clamp(0.0, 1.0)


def _build_poisson_rate_grid(
    counts_np: np.ndarray,
    grid_size: int,
    *,
    strategy: str = "linear",
    dtype: torch.dtype,
    device: torch.device,
) -> TorchTensor:
    """Build a rate (lambda) grid for Poisson likelihood directly in count space.

    Each gene gets its own grid ranging from 0 to its max observed count.
    """
    if strategy == "sqrt":
        t = torch.linspace(0.0, 1.0, grid_size, dtype=dtype, device=device)
        base = torch.sqrt(t)
        base = base / base[-1]
    else:
        base = torch.linspace(0.0, 1.0, grid_size, dtype=dtype, device=device)
    max_counts = np.nanmax(counts_np, axis=0)
    max_counts = np.clip(max_counts, EPS, None)
    rate_max = torch.as_tensor(max_counts, dtype=dtype, device=device)
    rate_max = rate_max + EPS
    return (rate_max[:, None] * base[None, :]).clamp(min=EPS)


def _compute_log_likelihood_chunks(
    counts_t: TorchTensor,
    n_eff_t: TorchTensor,
    p_grid_t: TorchTensor,
    cell_slices: list[slice],
    *,
    likelihood: str = "binomial",
    nb_overdispersion: float = 0.01,
) -> list[TorchTensor]:
    """Compute per-cell-chunk log-likelihoods for binomial or NB.

    Does NOT handle Poisson (see _compute_poisson_likelihood_chunks).
    """
    if likelihood == "binomial":
        lik_fn = lambda c, n, p: log_binomial_likelihood_grid(c, n, p)
    elif likelihood == "negative_binomial":

        def lik_fn(c, n, p):
            return log_negative_binomial_likelihood_grid(
                c, n, p, overdispersion=nb_overdispersion
            )
    else:
        raise ValueError(
            f"likelihood must be binomial or negative_binomial here, got {likelihood}"
        )
    with torch.no_grad():
        return [
            lik_fn(
                counts_t[:, cell_slice],
                n_eff_t[:, cell_slice],
                p_grid_t,
            )
            for cell_slice in cell_slices
        ]


def _compute_poisson_likelihood_chunks(
    counts_t: TorchTensor,
    rate_grid_t: TorchTensor,
    cell_slices: list[slice],
) -> list[TorchTensor]:
    """Compute per-cell-chunk Poisson log-likelihoods (no N_eff)."""
    with torch.no_grad():
        return [
            log_poisson_likelihood_grid(
                counts_t[:, cell_slice],
                rate_grid_t,
            )
            for cell_slice in cell_slices
        ]


def _sigma_for_step(config: PriorFitConfig, step: int) -> float:
    """Compute sigma_bins for a given step, supporting annealing."""
    if config.sigma_anneal_start is None or config.sigma_anneal_end is None:
        return config.sigma_bins
    t = (step - 1) / max(config.n_iter - 1, 1)
    return config.sigma_anneal_start + t * (
        config.sigma_anneal_end - config.sigma_anneal_start
    )


def _sample_cell_slices(
    n_cells: int,
    chunk_size: int,
    fraction: float,
    rng: np.random.Generator,
) -> list[slice]:
    """Return cell slices, optionally subsampled."""
    all_slices = _iter_cell_slices(n_cells, chunk_size)
    if fraction >= 1.0:
        return all_slices
    n_keep = max(1, int(len(all_slices) * fraction))
    indices = rng.choice(len(all_slices), size=n_keep, replace=False)
    return [all_slices[i] for i in sorted(indices)]


def _compute_align_loss(
    q_hat: TorchTensor,
    prior_weights: TorchTensor,
    mode: str,
    weight: float,
    posterior_entropy: TorchTensor | None = None,
) -> TorchTensor:
    """Compute alignment loss according to the chosen mode."""
    if mode == "kl":
        return kl_divergence(q_hat, prior_weights).mean() * weight
    if mode == "weighted_jsd":
        if posterior_entropy is None:
            return jsd(q_hat, prior_weights).mean() * weight
        return weighted_jsd(q_hat, prior_weights, posterior_entropy) * weight
    return jsd(q_hat, prior_weights).mean() * weight


def _apply_shrinkage(
    weights: TorchTensor,
    shrinkage_weight: float,
) -> TorchTensor:
    """Shrink per-gene weights toward the gene-group mean."""
    if shrinkage_weight <= 0:
        return weights
    group_mean = weights.mean(dim=0, keepdim=True)
    group_mean = group_mean / group_mean.sum(dim=-1, keepdim=True).clamp_min(EPS)
    blended = (1.0 - shrinkage_weight) * weights + shrinkage_weight * group_mean
    return blended / blended.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _adaptive_refine_grid(
    p_grid_t: TorchTensor,
    weights: np.ndarray,
    grid_size: int,
    fraction: float,
    quantile_lo: float,
    quantile_hi: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TorchTensor:
    """Rebuild grid concentrating points around high-posterior-mass regions."""
    p_grid_np = p_grid_t.detach().cpu().numpy()
    n_genes = weights.shape[0]
    new_grid = np.zeros((n_genes, grid_size), dtype=np.float64)
    n_focus = max(1, int(grid_size * fraction))
    n_uniform = grid_size - n_focus
    for g in range(n_genes):
        cdf = np.cumsum(weights[g])
        cdf = cdf / cdf[-1]
        lo_idx = int(np.searchsorted(cdf, quantile_lo))
        hi_idx = int(np.searchsorted(cdf, quantile_hi))
        lo_idx = max(lo_idx, 0)
        hi_idx = min(hi_idx, len(cdf) - 1)
        if hi_idx < lo_idx:
            hi_idx = lo_idx
        p_lo = float(p_grid_np[g, lo_idx])
        p_hi = float(p_grid_np[g, hi_idx])
        p_max = float(p_grid_np[g, -1])
        focus_pts = np.linspace(p_lo, p_hi, n_focus)
        if n_uniform > 0:
            uniform_pts = np.linspace(0.0, p_max, n_uniform + 2)[1:-1]
            all_pts = np.sort(np.unique(np.concatenate([focus_pts, uniform_pts])))
        else:
            all_pts = np.sort(np.unique(focus_pts))
        if len(all_pts) < grid_size:
            extra = np.linspace(0.0, p_max, grid_size - len(all_pts) + 2)[1:-1]
            all_pts = np.sort(np.unique(np.concatenate([all_pts, extra])))
        new_grid[g] = all_pts[:grid_size]
    return torch.as_tensor(new_grid, dtype=dtype, device=device)


def _resolve_initial_weights(
    *,
    batch: ObservationBatch,
    config: PriorFitConfig,
    counts_t: TorchTensor,
    n_eff_t: TorchTensor,
    p_grid_t: TorchTensor,
    cell_slices: list[slice],
    torch_dtype: torch.dtype,
    device_obj: torch.device,
    init_prior_weights: np.ndarray | None,
) -> TorchTensor:
    if init_prior_weights is not None:
        weights_t = torch.as_tensor(
            np.asarray(init_prior_weights, dtype=DTYPE_NP),
            dtype=torch_dtype,
            device=device_obj,
        )
        if weights_t.ndim == 1:
            weights_t = weights_t.unsqueeze(0)
        if tuple(weights_t.shape) != (batch.n_genes, config.grid_size):
            raise ValueError(
                "init_prior_weights must have shape "
                f"{(batch.n_genes, config.grid_size)}, got {tuple(weights_t.shape)}"
            )
        weights_t = weights_t.clamp_min(EPS)
        return weights_t / weights_t.sum(dim=-1, keepdim=True)

    if config.init_method == "uniform":
        return torch.full(
            (batch.n_genes, config.grid_size),
            1.0 / config.grid_size,
            dtype=torch_dtype,
            device=device_obj,
        )

    if config.init_method == "random":
        generator = torch.Generator(device=device_obj.type)
        generator.manual_seed(config.init_seed)
        random_logits = torch.rand(
            (batch.n_genes, config.grid_size),
            dtype=torch_dtype,
            device=device_obj,
            generator=generator,
        )
        random_logits = torch.log(random_logits.clamp_min(EPS))
        return torch.softmax(random_logits, dim=-1)

    uniform = torch.full(
        (batch.n_genes, config.grid_size),
        1.0 / config.grid_size,
        dtype=torch_dtype,
        device=device_obj,
    )
    posterior_sum = torch.zeros_like(uniform)
    if config.likelihood == "poisson":
        init_chunks = _compute_poisson_likelihood_chunks(
            counts_t,
            p_grid_t,
            cell_slices,
        )
    else:
        init_chunks = _compute_log_likelihood_chunks(
            counts_t,
            n_eff_t,
            p_grid_t,
            cell_slices,
            likelihood=config.likelihood,
            nb_overdispersion=config.nb_overdispersion,
        )
    for log_lik in init_chunks:
        tempered = (
            log_lik / config.init_temperature
            if config.init_temperature != 1.0
            else log_lik
        )
        posterior_sum += posterior_from_log_likelihood(tempered, uniform).sum(dim=-2)
    init_post = (posterior_sum / batch.n_cells).clamp_min(EPS)
    return init_post / init_post.sum(dim=-1, keepdim=True)


def fit_gene_priors(
    batch: ObservationBatch | GeneBatch,
    *,
    S: float,
    config: PriorFitConfig = PriorFitConfig(),
    device: str | torch.device = "cpu",
    p_grid_max: np.ndarray | None = None,
    init_prior_weights: np.ndarray | None = None,
    progress_callback: FitProgressCallback | None = None,
) -> PriorFitResult:
    if config.ensemble_restarts > 1:
        return _fit_ensemble(
            batch,
            S=S,
            config=config,
            device=device,
            p_grid_max=p_grid_max,
            init_prior_weights=init_prior_weights,
            progress_callback=progress_callback,
        )
    return _fit_single(
        batch,
        S=S,
        config=config,
        device=device,
        p_grid_max=p_grid_max,
        init_prior_weights=init_prior_weights,
        progress_callback=progress_callback,
    )


def _fit_ensemble(
    batch: ObservationBatch | GeneBatch,
    *,
    S: float,
    config: PriorFitConfig,
    device: str | torch.device,
    p_grid_max: np.ndarray | None,
    init_prior_weights: np.ndarray | None,
    progress_callback: FitProgressCallback | None,
) -> PriorFitResult:
    from dataclasses import replace

    methods: list[str] = ["posterior_mean", "uniform", "random"]
    best_result: PriorFitResult | None = None
    for i in range(config.ensemble_restarts):
        method = methods[i % len(methods)]
        restart_config = replace(
            config,
            init_method=method,  # type: ignore[arg-type]
            init_seed=config.init_seed + i,
            ensemble_restarts=1,
        )
        result = _fit_single(
            batch,
            S=S,
            config=restart_config,
            device=device,
            p_grid_max=p_grid_max,
            init_prior_weights=init_prior_weights if i == 0 else None,
            progress_callback=progress_callback,
        )
        if best_result is None or result.best_loss < best_result.best_loss:
            best_result = result
    assert best_result is not None
    return best_result


def _fit_single(
    batch: ObservationBatch | GeneBatch,
    *,
    S: float,
    config: PriorFitConfig,
    device: str | torch.device,
    p_grid_max: np.ndarray | None,
    init_prior_weights: np.ndarray | None,
    progress_callback: FitProgressCallback | None,
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

    torch_dtype = resolve_torch_dtype(config.torch_dtype)
    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=torch_dtype, device=device_obj)
    is_poisson = config.likelihood == "poisson"

    if is_poisson:
        # Poisson: build rate grid from raw counts, no N_eff
        p_grid_t = _build_poisson_rate_grid(
            batch.counts,
            config.grid_size,
            strategy=config.grid_strategy,
            dtype=torch_dtype,
            device=device_obj,
        )
        n_eff_t = torch.ones_like(counts_t)  # placeholder, not used in likelihood
    else:
        # Binomial / NB: use p_grid and N_eff as before
        if p_grid_max is None:
            p_grid_max = _default_p_grid_max(batch, S, method=config.grid_max_method)
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
        p_grid_t = _build_p_grid(
            config.grid_size,
            p_grid_max,
            dtype=torch_dtype,
            device=device_obj,
            strategy=config.grid_strategy,
        )
        n_eff_t = (
            torch.as_tensor(
                effective_exposure(batch.reference_counts, S),
                dtype=torch_dtype,
                device=device_obj,
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )

    cell_slices = _iter_cell_slices(batch.n_cells, config.cell_chunk_size)
    cell_rng = np.random.default_rng(config.cell_sample_seed)

    with torch.no_grad():
        init_post = _resolve_initial_weights(
            batch=batch,
            config=config,
            counts_t=counts_t,
            n_eff_t=n_eff_t,
            p_grid_t=p_grid_t,
            cell_slices=cell_slices,
            torch_dtype=torch_dtype,
            device_obj=device_obj,
            init_prior_weights=init_prior_weights,
        )
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
    latest_q_hat = init_post.detach().clone()
    latest_post_entropy: TorchTensor | None = None
    no_improve_steps = 0

    for step in range(1, config.n_iter + 1):
        optimizer.zero_grad(set_to_none=True)
        sigma = _sigma_for_step(config, step)
        prior_weights = smooth_probability_weights(logits, sigma)
        if config.shrinkage_weight > 0:
            prior_weights = _apply_shrinkage(prior_weights, config.shrinkage_weight)
        step_slices = _sample_cell_slices(
            batch.n_cells,
            config.cell_chunk_size,
            config.cell_sample_fraction,
            cell_rng,
        )
        if is_poisson:
            log_lik_chunks = _compute_poisson_likelihood_chunks(
                counts_t,
                p_grid_t,
                step_slices,
            )
        else:
            log_lik_chunks = _compute_log_likelihood_chunks(
                counts_t,
                n_eff_t,
                p_grid_t,
                step_slices,
                likelihood=config.likelihood,
                nb_overdispersion=config.nb_overdispersion,
            )
        should_compute_align = config.align_loss_weight > 0 and (
            (step - 1) % config.align_every == 0 or step == config.n_iter
        )
        if should_compute_align:
            with torch.no_grad():
                posterior_sum = torch.zeros_like(prior_weights)
                for log_lik in log_lik_chunks:
                    posterior_sum += posterior_from_log_likelihood(
                        log_lik, prior_weights
                    ).sum(dim=-2)
                n_cells_in_slices = sum(
                    s.stop - s.start
                    for s in step_slices  # type: ignore[union-attr]
                )
                q_hat = (posterior_sum / max(n_cells_in_slices, 1)).clamp_min(EPS)
                q_hat = q_hat / q_hat.sum(dim=-1, keepdim=True)
                latest_q_hat = q_hat.detach().clone()
                if config.align_mode == "weighted_jsd":
                    latest_post_entropy = _entropy_fn(q_hat)
            align = _compute_align_loss(
                q_hat,
                prior_weights,
                config.align_mode,
                config.align_loss_weight,
                latest_post_entropy,
            )
        else:
            q_hat = latest_q_hat
            align = torch.zeros((), dtype=torch_dtype, device=device_obj)
        log_prior = torch.log(prior_weights.clamp_min(EPS)).unsqueeze(-2)
        nll_value = 0.0
        total_loss = align
        for log_lik in log_lik_chunks:
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
        is_new_best = final_loss < best_loss
        if is_new_best:
            best_loss = final_loss
            best_logits = logits.detach().clone()
            best_q_hat = q_hat.detach().cpu().numpy()
            no_improve_steps = 0
        elif (
            config.early_stop_tol is not None
            and config.early_stop_patience is not None
            and final_loss >= best_loss - config.early_stop_tol
        ):
            no_improve_steps += 1
        if progress_callback is not None:
            progress_callback(
                step,
                config.n_iter,
                final_loss,
                nll_value,
                float(align.item()),
                is_new_best,
            )
        if (
            config.early_stop_tol is not None
            and config.early_stop_patience is not None
            and no_improve_steps >= config.early_stop_patience
        ):
            break

    # --- adaptive grid refinement (phase 2) ---
    final_sigma = _sigma_for_step(config, config.n_iter)
    if config.adaptive_grid:
        coarse_weights = (
            smooth_probability_weights(best_logits, final_sigma).detach().cpu().numpy()
        )
        p_grid_t = _adaptive_refine_grid(
            p_grid_t,
            coarse_weights,
            config.grid_size,
            config.adaptive_grid_fraction,
            config.adaptive_grid_quantile_lo,
            config.adaptive_grid_quantile_hi,
            dtype=torch_dtype,
            device=device_obj,
        )
        with torch.no_grad():
            reinit_post = _resolve_initial_weights(
                batch=batch,
                config=config,
                counts_t=counts_t,
                n_eff_t=n_eff_t,
                p_grid_t=p_grid_t,
                cell_slices=cell_slices,
                torch_dtype=torch_dtype,
                device_obj=device_obj,
                init_prior_weights=None,
            )
            reinit_logits = torch.log(reinit_post) / max(config.init_temperature, EPS)
        logits = torch.nn.Parameter(reinit_logits.clone())
        optimizer = _build_optimizer([logits], config.optimizer, config.lr * 0.5)
        scheduler = _build_scheduler(
            optimizer,
            name=config.scheduler,
            n_iter=max(config.n_iter // 3, 10),
            lr=config.lr * 0.5,
            lr_min_ratio=config.lr_min_ratio,
        )
        phase2_iters = max(config.n_iter // 3, 10)
        for step in range(1, phase2_iters + 1):
            optimizer.zero_grad(set_to_none=True)
            sigma = _sigma_for_step(config, step + config.n_iter)
            prior_weights = smooth_probability_weights(logits, sigma)
            if config.shrinkage_weight > 0:
                prior_weights = _apply_shrinkage(prior_weights, config.shrinkage_weight)
            if is_poisson:
                log_lik_chunks = _compute_poisson_likelihood_chunks(
                    counts_t,
                    p_grid_t,
                    cell_slices,
                )
            else:
                log_lik_chunks = _compute_log_likelihood_chunks(
                    counts_t,
                    n_eff_t,
                    p_grid_t,
                    cell_slices,
                    likelihood=config.likelihood,
                    nb_overdispersion=config.nb_overdispersion,
                )
            log_prior = torch.log(prior_weights.clamp_min(EPS)).unsqueeze(-2)
            total_loss_p2 = torch.zeros((), dtype=torch_dtype, device=device_obj)
            for log_lik in log_lik_chunks:
                log_marginal = torch.logsumexp(log_lik + log_prior, dim=-1)
                nll_chunk = -(log_marginal.sum(dim=-1) / batch.n_cells).mean()
                total_loss_p2 = total_loss_p2 + nll_chunk
            total_loss_p2.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([logits], config.grad_clip)
            optimizer.step()
            scheduler.step()
            p2_loss = float(total_loss_p2.item())
            if p2_loss < best_loss:
                best_loss = p2_loss
                best_logits = logits.detach().clone()

    best_weights = (
        smooth_probability_weights(best_logits, final_sigma).detach().cpu().numpy()
    )
    if config.shrinkage_weight > 0:
        best_weights_t = _apply_shrinkage(
            torch.as_tensor(best_weights, dtype=torch_dtype, device=device_obj),
            config.shrinkage_weight,
        )
        best_weights = best_weights_t.detach().cpu().numpy()
    p_grid = p_grid_t.detach().cpu().numpy()
    return PriorFitResult(
        gene_names=list(batch.gene_names),
        priors=PriorGrid(
            gene_names=list(batch.gene_names),
            p_grid=p_grid,
            weights=best_weights,
            S=float(S),
            grid_domain="rate" if is_poisson else "p",
            distribution=config.likelihood,
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
    init_prior_weights: np.ndarray | None = None,
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

    torch_dtype = resolve_torch_dtype(config.torch_dtype)
    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=torch_dtype, device=device_obj)
    is_poisson = config.likelihood == "poisson"

    if is_poisson:
        p_grid_t = _build_poisson_rate_grid(
            batch.counts,
            config.grid_size,
            strategy=config.grid_strategy,
            dtype=torch_dtype,
            device=device_obj,
        )
        n_eff_t = torch.ones_like(counts_t)
    else:
        if p_grid_max is None:
            p_grid_max = _default_p_grid_max(batch, S, method=config.grid_max_method)
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
        p_grid_t = _build_p_grid(
            config.grid_size,
            p_grid_max,
            dtype=torch_dtype,
            device=device_obj,
            strategy=config.grid_strategy,
        )
        n_eff_t = (
            torch.as_tensor(
                effective_exposure(batch.reference_counts, S),
                dtype=torch_dtype,
                device=device_obj,
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )

    cell_slices = _iter_cell_slices(batch.n_cells, config.cell_chunk_size)

    q = _resolve_initial_weights(
        batch=batch,
        config=config,
        counts_t=counts_t,
        n_eff_t=n_eff_t,
        p_grid_t=p_grid_t,
        cell_slices=cell_slices,
        torch_dtype=torch_dtype,
        device_obj=device_obj,
        init_prior_weights=init_prior_weights,
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
                if is_poisson:
                    log_lik = log_poisson_likelihood_grid(
                        counts_t[:, cell_slice],
                        p_grid_t,
                    )
                elif config.likelihood == "negative_binomial":
                    log_lik = log_negative_binomial_likelihood_grid(
                        counts_t[:, cell_slice],
                        n_eff_t[:, cell_slice],
                        p_grid_t,
                        overdispersion=config.nb_overdispersion,
                    )
                else:
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
            grid_domain="rate" if is_poisson else "p",
            distribution=config.likelihood,
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

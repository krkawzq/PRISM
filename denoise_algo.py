"""
scPRISM reference implementation aligned to `docs/scPRISM_re.md`.

Pipeline:
1. Estimate the global sampling-pool scale `S` from total UMI counts using a
   Poisson-LogNormal EM model.
2. Fit each gene prior `F_g` on a discrete grid with softmax + Gaussian
   smoothing, using the documented `NLL + JSD` objective.
3. Expose the posterior-derived signal interface: Signal, Confidence,
   Surprisal, and optional Sharpness.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from scipy import special

import torch
import torch.nn.functional as F

EPS_NP = 1e-12
EPS_T = 1e-12
NEG_INF_T = -1e30
TORCH_DTYPE = torch.float64


def project_root() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_scprism_defaults() -> dict[str, Any]:
    config_path = project_root() / "scprism_defaults.toml"
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=1)
def fit_defaults() -> dict[str, Any]:
    return dict(load_scprism_defaults().get("fit", {}))


# ============================================================
# Stage 0: Poisson-LogNormal sampling-pool scale estimation
# ============================================================


@dataclass
class PoolEstimationResult:
    mu: float
    sigma: float
    loglik: float
    n_iter: int
    loglik_history: list[float]
    rs_hat: float
    s_hat: float
    eta_posterior_mean: np.ndarray
    log_eta_posterior_mean: np.ndarray
    log_eta_posterior_var: np.ndarray
    eta_prior_grid: np.ndarray
    eta_prior_density: np.ndarray


@lru_cache(maxsize=16)
def _hermgauss_cached(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    return nodes.astype(float), weights.astype(float)


def _compress_totals(
    totals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_int = np.asarray(np.rint(totals), dtype=np.int64)
    unique_n, inverse, freq = np.unique(
        n_int,
        return_inverse=True,
        return_counts=True,
    )
    return n_int, unique_n.astype(float), inverse.astype(np.int64), freq.astype(float)


def _pln_quadrature_log_joint(
    n_vals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = _hermgauss_cached(n_quad)
    log_eta = mu + sigma * math.sqrt(2.0) * nodes
    eta = np.exp(log_eta)
    n = n_vals[:, None].astype(float)
    log_poisson = n * log_eta[None, :] - eta[None, :] - special.gammaln(n + 1.0)
    log_w = np.log(weights / math.sqrt(math.pi) + EPS_NP)
    return log_eta, eta, log_poisson + log_w[None, :]


def _pln_log_marginal(
    n_vals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> np.ndarray:
    _, _, log_joint = _pln_quadrature_log_joint(n_vals, mu, sigma, n_quad)
    mx = np.max(log_joint, axis=1, keepdims=True)
    return mx.ravel() + np.log(np.sum(np.exp(log_joint - mx), axis=1) + EPS_NP)


def _pln_posterior_stats(
    n_vals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_eta, eta, log_joint = _pln_quadrature_log_joint(n_vals, mu, sigma, n_quad)
    mx = np.max(log_joint, axis=1, keepdims=True)
    joint = np.exp(log_joint - mx)
    pw = joint / (np.sum(joint, axis=1, keepdims=True) + EPS_NP)
    e_log_eta = np.sum(pw * log_eta[None, :], axis=1)
    e_log_eta2 = np.sum(pw * log_eta[None, :] ** 2, axis=1)
    e_eta = np.sum(pw * eta[None, :], axis=1)
    return e_log_eta, e_log_eta2, e_eta


def _lognormal_density_grid(
    mu: float,
    sigma: float,
    grid_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    log_grid = np.linspace(mu - 4.5 * sigma, mu + 4.5 * sigma, grid_size)
    eta_grid = np.exp(log_grid)
    density = np.exp(-0.5 * ((log_grid - mu) / sigma) ** 2)
    density /= eta_grid * sigma * math.sqrt(2.0 * math.pi)
    density /= max(float(np.trapezoid(density, eta_grid)), EPS_NP)
    return eta_grid, density


def fit_poisson_lognormal(
    totals: np.ndarray,
    r: float = 0.05,
    max_iter: int = 120,
    tol: float = 1e-6,
    n_quad: int = 80,
) -> PoolEstimationResult:
    if not (0.0 < r <= 1.0):
        raise ValueError("r must satisfy 0 < r <= 1")

    totals = np.asarray(totals, dtype=float)
    if totals.ndim != 1 or totals.size == 0:
        raise ValueError("totals must be a non-empty 1D array")

    _, unique_n, inverse, freq = _compress_totals(totals)
    log_n = np.log(np.maximum(unique_n, 1.0))
    mu = float(np.average(log_n, weights=freq))
    sigma = max(float(np.sqrt(np.average((log_n - mu) ** 2, weights=freq))), 0.05)

    prev_ll = -np.inf
    loglik_history: list[float] = []
    n_iter = max_iter

    for it in range(max_iter):
        e_log_eta, e_log_eta2, _ = _pln_posterior_stats(unique_n, mu, sigma, n_quad)
        mu_new = float(np.average(e_log_eta, weights=freq))
        sigma2_new = float(np.average(e_log_eta2, weights=freq) - mu_new**2)
        sigma_new = math.sqrt(max(sigma2_new, 1e-8))

        ll_unique = _pln_log_marginal(unique_n, mu_new, sigma_new, n_quad)
        ll = float(np.sum(freq * ll_unique))
        loglik_history.append(ll)

        if abs(ll - prev_ll) / (abs(prev_ll) + 1.0) < tol:
            mu, sigma = mu_new, sigma_new
            prev_ll = ll
            n_iter = it + 1
            break

        mu, sigma = mu_new, sigma_new
        prev_ll = ll

    e_log_eta_u, e_log_eta2_u, e_eta_u = _pln_posterior_stats(
        unique_n, mu, sigma, n_quad
    )
    eta_prior_grid, eta_prior_density = _lognormal_density_grid(mu, sigma)

    rs_hat = float(math.exp(mu))
    s_hat = float(rs_hat / r)

    log_eta_post_mean = e_log_eta_u[inverse]
    log_eta_post_var = np.maximum(e_log_eta2_u[inverse] - log_eta_post_mean**2, 0.0)
    eta_post_mean = e_eta_u[inverse]

    return PoolEstimationResult(
        mu=mu,
        sigma=sigma,
        loglik=prev_ll,
        n_iter=n_iter,
        loglik_history=loglik_history,
        rs_hat=rs_hat,
        s_hat=s_hat,
        eta_posterior_mean=eta_post_mean.astype(float),
        log_eta_posterior_mean=log_eta_post_mean.astype(float),
        log_eta_posterior_var=log_eta_post_var.astype(float),
        eta_prior_grid=eta_prior_grid.astype(float),
        eta_prior_density=eta_prior_density.astype(float),
    )


# ============================================================
# Stage 1: Per-gene prior estimation on a discrete grid
# ============================================================


@dataclass
class GenePriorFitResult:
    gene_id: str
    support: np.ndarray
    grid_step: float
    support_max: float
    s_hat: float
    init_q_hat: np.ndarray
    init_prior_weights: np.ndarray
    prior_weights: np.ndarray
    q_hat: np.ndarray
    loss_history: list[float]
    nll_history: list[float]
    align_history: list[float]
    counts: np.ndarray
    totals: np.ndarray
    x_eff: np.ndarray
    log_likelihood: np.ndarray
    posterior: np.ndarray
    signal: np.ndarray
    confidence: np.ndarray
    surprisal: np.ndarray
    surprisal_norm: np.ndarray
    sharpness: np.ndarray
    config: dict[str, Any]


def _as_torch(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=TORCH_DTYPE)


def _gaussian_kernel_bins(sigma_bins: float) -> torch.Tensor:
    if sigma_bins <= 0:
        return torch.ones(1, dtype=TORCH_DTYPE)
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    offsets = torch.arange(-radius, radius + 1, dtype=TORCH_DTYPE)
    kernel = torch.exp(-0.5 * (offsets / float(sigma_bins)) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


class SmoothedDiscretePrior(torch.nn.Module):
    def __init__(self, support_max: float, grid_size: int, sigma_bins: float):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        support_max = max(float(support_max), 1e-6)
        support = torch.linspace(0.0, support_max, grid_size, dtype=TORCH_DTYPE)
        step = float(support[1] - support[0])
        kernel = _gaussian_kernel_bins(sigma_bins)

        self.logits = torch.nn.Parameter(torch.zeros(grid_size, dtype=TORCH_DTYPE))
        self.register_buffer("support", support)
        self.register_buffer("grid_step", torch.tensor(step, dtype=TORCH_DTYPE))
        self.register_buffer("kernel", kernel)
        self.sigma_bins = float(sigma_bins)

    def get_weights(self) -> torch.Tensor:
        kernel = cast(torch.Tensor, self.kernel)
        w = F.softmax(self.logits, dim=0)
        if kernel.numel() == 1:
            return w

        pad = kernel.numel() // 2
        padded = F.pad(w.view(1, 1, -1), (pad, pad), mode="replicate")
        smoothed = F.conv1d(padded, kernel.view(1, 1, -1)).view(-1)
        return smoothed / (smoothed.sum() + EPS_T)


def _build_log_binomial_grid(
    x_vals: torch.Tensor,
    n_vals: torch.Tensor,
    s_hat: float,
    support: torch.Tensor,
) -> torch.Tensor:
    x = x_vals.view(-1, 1)
    n = n_vals.view(-1, 1)
    p = (support.view(1, -1) / float(s_hat)).clamp(0.0, 1.0)

    log_coeff = (
        torch.lgamma(n + 1.0) - torch.lgamma(x + 1.0) - torch.lgamma(n - x + 1.0)
    )
    log_binom = log_coeff + torch.xlogy(x, p) + torch.xlogy(n - x, 1.0 - p)
    impossible = support.view(1, -1) > float(s_hat) + 1e-12
    return torch.where(impossible, torch.full_like(log_binom, NEG_INF_T), log_binom)


def _posterior_from_log_likelihood(
    log_likelihood: torch.Tensor,
    prior_weights: torch.Tensor,
) -> torch.Tensor:
    log_prior = torch.log(prior_weights.clamp_min(EPS_T)).view(1, -1)
    log_post = log_likelihood + log_prior
    log_post = log_post - torch.logsumexp(log_post, dim=1, keepdim=True)
    return torch.exp(log_post)


def _map_decisions(posterior: torch.Tensor) -> torch.Tensor:
    return torch.argmax(posterior, dim=1)


def _soft_decisions(
    posterior: torch.Tensor,
    support: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    powered = posterior.clamp_min(EPS_T).pow(1.0 / temperature)
    weights = powered / (powered.sum(dim=1, keepdim=True) + EPS_T)
    return (weights * support.view(1, -1)).sum(dim=1)


def _decision_distribution(
    decision_values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    m = int(support.numel())
    q_hat = torch.zeros(m, dtype=TORCH_DTYPE, device=decision_values.device)
    if m == 1:
        q_hat[0] = 1.0
        return q_hat

    step = float(support[1] - support[0])
    pos = (decision_values - support[0]) / max(step, EPS_T)
    left = torch.floor(pos).to(torch.long).clamp(0, m - 1)
    right = (left + 1).clamp(0, m - 1)
    frac_right = (pos - left.to(TORCH_DTYPE)).clamp(0.0, 1.0)
    frac_left = 1.0 - frac_right

    same = left == right
    frac_left = torch.where(same, torch.ones_like(frac_left), frac_left)
    frac_right = torch.where(same, torch.zeros_like(frac_right), frac_right)

    q_hat.scatter_add_(0, left, frac_left)
    q_hat.scatter_add_(0, right, frac_right)
    return q_hat / max(float(decision_values.numel()), 1.0)


def _jsd(q_fixed: torch.Tensor, q_trainable: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (q_fixed + q_trainable)
    kl_fixed = (q_fixed * torch.log((q_fixed + EPS_T) / (m + EPS_T))).sum()
    kl_train = (q_trainable * torch.log((q_trainable + EPS_T) / (m + EPS_T))).sum()
    return 0.5 * (kl_fixed + kl_train)


def _build_alignment_distribution(
    posterior: torch.Tensor,
    support: torch.Tensor,
    decision_temperature: float,
    mode: str,
) -> torch.Tensor:
    if mode == "posterior_average":
        q_hat = posterior.mean(dim=0)
        return q_hat / (q_hat.sum() + EPS_T)

    if mode != "map_histogram":
        raise ValueError(
            "align_distribution must be 'map_histogram' or 'posterior_average'"
        )

    if decision_temperature <= 0:
        map_idx = _map_decisions(posterior)
        q_hat = torch.bincount(map_idx, minlength=posterior.shape[1]).to(TORCH_DTYPE)
        return q_hat / max(float(map_idx.numel()), 1.0)

    decisions = _soft_decisions(posterior, support, decision_temperature)
    return _decision_distribution(decisions, support)


def _bootstrap_logits_from_distribution(
    model: SmoothedDiscretePrior,
    log_likelihood: torch.Tensor,
    x_vals: torch.Tensor,
    n_vals: torch.Tensor,
    s_hat: float,
    decision_temperature: float,
    warm_start_mode: str,
    init_strategy: str,
    init_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if init_strategy == "uniform":
            model.logits.zero_()
            init_prior = model.get_weights()
            return init_prior, init_prior

        base_prior = model.get_weights()
        support_t = cast(torch.Tensor, model.support)
        init_log_likelihood = log_likelihood.clone()

        if (
            init_strategy == "bootstrap_zero_to_first_positive"
            and support_t.numel() > 1
        ):
            first_positive_mu = float(support_t[1].item())
            pseudo_counts = torch.ceil(
                n_vals * (first_positive_mu / float(s_hat))
            ).clamp(min=1.0)
            pseudo_counts = torch.minimum(pseudo_counts, n_vals)
            zero_mask = x_vals <= 0
            if torch.any(zero_mask):
                init_log_likelihood[zero_mask] = _build_log_binomial_grid(
                    pseudo_counts[zero_mask],
                    n_vals[zero_mask],
                    s_hat,
                    support_t,
                )
        elif init_strategy != "bootstrap_raw_observation":
            raise ValueError(
                "init_strategy must be 'uniform', 'bootstrap_raw_observation', or 'bootstrap_zero_to_first_positive'"
            )

        posterior = _posterior_from_log_likelihood(init_log_likelihood, base_prior)
        q_hat = _build_alignment_distribution(
            posterior,
            support_t,
            decision_temperature,
            warm_start_mode,
        )

        q_hat = q_hat.clamp_min(EPS_T)
        q_hat = q_hat / (q_hat.sum() + EPS_T)
        init_temperature = max(float(init_temperature), 1.0)
        model.logits.copy_(torch.log(q_hat) / init_temperature)
        init_prior = model.get_weights()
        return q_hat, init_prior


def fit_gene_prior(
    gene_id: str,
    x_vals: np.ndarray,
    n_vals: np.ndarray,
    s_hat: float,
    grid_size: int | None = None,
    sigma_bins: float | None = None,
    decision_temperature: float | None = None,
    align_distribution: str | None = None,
    warm_start_mode: str | None = None,
    init_strategy: str | None = None,
    init_temperature: float | None = None,
    align_weight: float | None = None,
    lr: float | None = None,
    n_iter: int | None = None,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> GenePriorFitResult:
    defaults = fit_defaults()
    grid_size = int(defaults["grid_size"] if grid_size is None else grid_size)
    sigma_bins = float(defaults["sigma_bins"] if sigma_bins is None else sigma_bins)
    decision_temperature = float(
        defaults["decision_temperature"]
        if decision_temperature is None
        else decision_temperature
    )
    align_distribution = str(
        defaults["align_distribution"]
        if align_distribution is None
        else align_distribution
    )
    warm_start_mode = str(
        defaults["warm_start_mode"] if warm_start_mode is None else warm_start_mode
    )
    init_strategy = str(
        defaults["init_strategy"] if init_strategy is None else init_strategy
    )
    init_temperature = float(
        defaults["init_temperature"] if init_temperature is None else init_temperature
    )
    align_weight = float(
        defaults["align_weight"] if align_weight is None else align_weight
    )
    lr = float(defaults["lr"] if lr is None else lr)
    n_iter = int(defaults["n_iter"] if n_iter is None else n_iter)

    if s_hat <= 0:
        raise ValueError("s_hat must be positive")

    x_vals = np.asarray(x_vals, dtype=float)
    n_vals = np.asarray(n_vals, dtype=float)
    if x_vals.shape != n_vals.shape:
        raise ValueError("x_vals and n_vals must share shape")
    if x_vals.ndim != 1 or x_vals.size == 0:
        raise ValueError("gene inputs must be non-empty 1D arrays")

    x_eff = np.divide(
        x_vals,
        np.maximum(n_vals, 1.0),
        out=np.zeros_like(x_vals),
    ) * float(s_hat)
    support_max = min(max(float(np.max(x_eff)), 1e-6), float(s_hat))

    model = SmoothedDiscretePrior(
        support_max=support_max,
        grid_size=grid_size,
        sigma_bins=sigma_bins,
    )
    x_t = _as_torch(x_vals)
    n_t = _as_torch(n_vals)
    support_t = cast(torch.Tensor, model.support)
    grid_step_t = cast(torch.Tensor, model.grid_step)
    log_likelihood = _build_log_binomial_grid(x_t, n_t, s_hat, support_t)
    init_q_hat_t, init_prior_weights_t = _bootstrap_logits_from_distribution(
        model,
        log_likelihood,
        x_t,
        n_t,
        s_hat,
        decision_temperature,
        warm_start_mode,
        init_strategy,
        init_temperature,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(n_iter), 1),
        eta_min=lr * 0.1,
    )

    loss_history: list[float] = []
    nll_history: list[float] = []
    align_history: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for it in range(int(n_iter)):
        optimizer.zero_grad()
        prior_weights = model.get_weights()
        log_marginal = torch.logsumexp(
            log_likelihood + torch.log(prior_weights.clamp_min(EPS_T)).view(1, -1),
            dim=1,
        )
        nll = -log_marginal.mean()

        with torch.no_grad():
            posterior = _posterior_from_log_likelihood(log_likelihood, prior_weights)
            q_hat = _build_alignment_distribution(
                posterior,
                support_t,
                decision_temperature,
                align_distribution,
            )

        align = _jsd(q_hat, prior_weights)
        loss = nll + float(align_weight) * align
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        nll_val = float(nll.item())
        align_val = float(align.item())
        loss_history.append(loss_val)
        nll_history.append(nll_val)
        align_history.append(align_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if callback is not None:
            callback(it, {"loss": loss_val, "nll": nll_val, "align": align_val})

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        prior_weights_t = model.get_weights()
        posterior_t = _posterior_from_log_likelihood(log_likelihood, prior_weights_t)
        map_idx_t = _map_decisions(posterior_t)
        if decision_temperature <= 0:
            signal_t = support_t[map_idx_t]
        else:
            signal_t = _soft_decisions(posterior_t, support_t, decision_temperature)
        q_hat_t = _build_alignment_distribution(
            posterior_t,
            support_t,
            decision_temperature,
            align_distribution,
        )

    support = support_t.cpu().numpy()
    init_q_hat = init_q_hat_t.cpu().numpy()
    init_prior_weights = init_prior_weights_t.cpu().numpy()
    prior_weights = prior_weights_t.cpu().numpy()
    posterior = posterior_t.cpu().numpy()
    q_hat = q_hat_t.cpu().numpy()
    signal = signal_t.cpu().numpy()
    map_idx = map_idx_t.cpu().numpy()
    log_likelihood_np = log_likelihood.cpu().numpy()

    entropy = -(posterior * np.log(np.clip(posterior, EPS_NP, None))).sum(axis=1)
    confidence = np.clip(1.0 - entropy / math.log(grid_size), 0.0, 1.0)
    surprisal = -np.log(np.clip(prior_weights[map_idx], EPS_NP, None))
    surprisal_norm = surprisal / max(
        float(np.max(-np.log(np.clip(prior_weights, EPS_NP, None)))), EPS_NP
    )

    log_prior = np.log(np.clip(prior_weights, EPS_NP, None))
    sharpness_curve = np.zeros_like(log_prior)
    if sharpness_curve.size >= 3:
        sharpness_curve[1:-1] = -(
            log_prior[:-2] - 2.0 * log_prior[1:-1] + log_prior[2:]
        )
        sharpness_curve[0] = sharpness_curve[1]
        sharpness_curve[-1] = sharpness_curve[-2]
    sharpness = np.clip(sharpness_curve[map_idx], 0.0, None)

    config = {
        "grid_size": int(grid_size),
        "sigma_bins": float(sigma_bins),
        "decision_temperature": float(decision_temperature),
        "align_distribution": str(align_distribution),
        "warm_start_mode": str(warm_start_mode),
        "init_strategy": str(init_strategy),
        "init_temperature": float(init_temperature),
        "align_weight": float(align_weight),
        "lambda_nll": 1.0,
        "lr": float(lr),
        "n_iter": int(n_iter),
        "s_hat": float(s_hat),
    }

    return GenePriorFitResult(
        gene_id=gene_id,
        support=support.astype(float),
        grid_step=float(grid_step_t.item()),
        support_max=float(support_max),
        s_hat=float(s_hat),
        init_q_hat=init_q_hat.astype(float),
        init_prior_weights=init_prior_weights.astype(float),
        prior_weights=prior_weights.astype(float),
        q_hat=q_hat.astype(float),
        loss_history=loss_history,
        nll_history=nll_history,
        align_history=align_history,
        counts=x_vals.astype(float),
        totals=n_vals.astype(float),
        x_eff=x_eff.astype(float),
        log_likelihood=log_likelihood_np.astype(float),
        posterior=posterior.astype(float),
        signal=signal.astype(float),
        confidence=confidence.astype(float),
        surprisal=surprisal.astype(float),
        surprisal_norm=surprisal_norm.astype(float),
        sharpness=sharpness.astype(float),
        config=config,
    )


# ============================================================
# Data loading helpers
# ============================================================


def default_data_dir() -> Path:
    return project_root() / "data"


def _ensure_csr(matrix):
    import scipy.sparse as sp

    if sp.issparse(matrix):
        return sp.csr_matrix(matrix)
    return sp.csr_matrix(np.asarray(matrix))


@lru_cache(maxsize=1)
def discover_h5ad_datasets(data_dir: str | None = None) -> dict[str, str]:
    base = Path(data_dir) if data_dir else default_data_dir()
    datasets = {p.stem: str(p.resolve()) for p in sorted(base.glob("*.h5ad"))}
    if not datasets:
        raise FileNotFoundError(f"No h5ad under {base}")
    return datasets


@lru_cache(maxsize=4)
def load_dataset_bundle(h5ad_path: str) -> dict[str, Any]:
    import anndata as ad
    import pandas as pd

    adata = ad.read_h5ad(h5ad_path)
    x_matrix = _ensure_csr(adata.X)
    totals = np.asarray(x_matrix.sum(axis=1)).ravel().astype(float)
    detected_genes = np.asarray(x_matrix.getnnz(axis=1)).ravel().astype(int)
    gene_total = np.asarray(x_matrix.sum(axis=0)).ravel().astype(float)
    gene_detected = np.asarray(x_matrix.getnnz(axis=0)).ravel().astype(int)

    obs = pd.DataFrame(adata.obs.copy())
    obs.index = adata.obs_names.astype(str)
    obs["total_umi"] = totals
    obs["detected_genes"] = detected_genes
    obs["cell_sparsity"] = 1.0 - detected_genes / max(adata.n_vars, 1)

    var = pd.DataFrame(adata.var.copy())
    var.index = adata.var_names.astype(str)
    for col, default_fn in [
        ("Geneid", lambda frame: frame.index.astype(str)),
        ("gene_name", lambda frame: frame.index.astype(str)),
        ("gene_index", lambda frame: np.arange(len(frame), dtype=int)),
    ]:
        if col not in var.columns:
            var[col] = default_fn(var)
    var["total_umi"] = gene_total
    var["detected_cells"] = gene_detected
    var["detected_frac"] = gene_detected / max(adata.n_obs, 1)
    var["zero_frac"] = 1.0 - var["detected_frac"]
    var["mean_umi"] = gene_total / max(adata.n_obs, 1)

    return {
        "path": str(Path(h5ad_path).resolve()),
        "dataset_key": Path(h5ad_path).stem,
        "adata": adata,
        "X": x_matrix,
        "obs": obs,
        "var": var,
        "totals": totals,
        "species": str(adata.uns.get("species", "")),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }


def list_dataset_summaries(data_dir: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, path in discover_h5ad_datasets(data_dir).items():
        bundle = load_dataset_bundle(path)
        totals = bundle["totals"]
        out.append(
            {
                "dataset_key": key,
                "path": path,
                "species": bundle["species"],
                "n_cells": bundle["n_obs"],
                "n_genes": bundle["n_vars"],
                "median_total_umi": float(np.median(totals)),
                "mean_total_umi": float(np.mean(totals)),
            }
        )
    return out


def _rank_gene_matches(var, query: str):
    import pandas as pd

    q = str(query).strip().casefold()
    gene_id = var.index.astype(str)
    gene_name = var["gene_name"].astype(str)
    geneid_col = var["Geneid"].astype(str)
    gene_index = var["gene_index"].astype(str)
    scores = pd.Series(np.full(len(var), 999, dtype=int), index=var.index)
    for test, score in [
        (gene_id.str.casefold() == q, 0),
        (geneid_col.str.casefold() == q, 0),
        (gene_name.str.casefold() == q, 1),
        (gene_index.str.casefold() == q, 2),
        (gene_id.str.casefold().str.startswith(q), 3),
        (geneid_col.str.casefold().str.startswith(q), 3),
        (gene_name.str.casefold().str.startswith(q), 4),
        (gene_id.str.casefold().str.contains(q, regex=False), 5),
        (geneid_col.str.casefold().str.contains(q, regex=False), 5),
        (gene_name.str.casefold().str.contains(q, regex=False), 6),
    ]:
        scores[test] = np.minimum(scores[test], score)
    return scores


def search_gene_candidates(
    h5ad_path: str, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    bundle = load_dataset_bundle(h5ad_path)
    var = bundle["var"]
    if query is None or str(query).strip() == "":
        ranked = var.sort_values(["total_umi", "detected_cells"], ascending=False)
    else:
        scores = _rank_gene_matches(var, str(query))
        ranked = var.loc[scores < 999].copy()
        if ranked.empty:
            return []
        ranked["match_score"] = scores.loc[ranked.index]
        ranked = ranked.sort_values(
            ["match_score", "total_umi", "detected_cells"],
            ascending=[True, False, False],
        )
    hits = []
    for gid, row in ranked.head(limit).iterrows():
        hits.append(
            {
                "gene_id": str(gid),
                "gene_name": str(row["gene_name"]),
                "gene_index": int(row["gene_index"]),
                "total_umi": float(row["total_umi"]),
                "detected_cells": int(row["detected_cells"]),
                "detected_frac": float(row["detected_frac"]),
                "zero_frac": float(row["zero_frac"]),
            }
        )
    return hits


def resolve_gene_query(h5ad_path: str, gene_query: str | int) -> dict[str, Any]:
    bundle = load_dataset_bundle(h5ad_path)
    var = bundle["var"]
    if isinstance(gene_query, int) or str(gene_query).strip().isdigit():
        gene_index = int(gene_query)
        if 0 <= gene_index < len(var):
            row = var.iloc[gene_index]
            return {
                "gene_id": str(var.index[gene_index]),
                "gene_name": str(row["gene_name"]),
                "gene_index": int(row["gene_index"]),
                "matched_by": "gene_index",
            }
    hits = search_gene_candidates(h5ad_path, str(gene_query), limit=1)
    if not hits:
        raise KeyError(f"Gene '{gene_query}' not found in {h5ad_path}")
    hit = hits[0]
    return {
        "gene_id": hit["gene_id"],
        "gene_name": hit["gene_name"],
        "gene_index": hit["gene_index"],
        "matched_by": "search",
    }


def load_gene_bundle(h5ad_path: str, gene_query: str | int) -> dict[str, Any]:
    bundle = load_dataset_bundle(h5ad_path)
    resolved = resolve_gene_query(h5ad_path, gene_query)
    x_vals = (
        np.asarray(bundle["X"][:, resolved["gene_index"]].toarray())
        .ravel()
        .astype(float)
    )
    return {
        **resolved,
        "dataset_key": bundle["dataset_key"],
        "species": bundle["species"],
        "x_vals": x_vals,
        "totals": bundle["totals"],
        "obs": bundle["obs"],
        "var_row": bundle["var"].loc[resolved["gene_id"]].to_dict(),
    }


def summarize_gene_expression(
    h5ad_path: str, gene_query: str | int, max_treatments: int = 12
) -> dict[str, Any]:
    import pandas as pd

    gene = load_gene_bundle(h5ad_path, gene_query)
    counts = gene["x_vals"]
    totals = gene["totals"]
    obs = gene["obs"].copy()
    obs["gene_count"] = counts
    detected = counts > 0
    corr = 0.0
    if np.std(counts) > 0 and np.std(totals) > 0:
        corr = float(np.corrcoef(totals, counts)[0, 1])

    treatment_table = []
    if "treatment" in obs.columns:
        grouped = pd.DataFrame(
            obs.groupby("treatment", observed=False).agg(
                cells=("gene_count", "size"),
                total_counts=("gene_count", "sum"),
                mean_count=("gene_count", "mean"),
                detected_frac=("gene_count", lambda series: float((series > 0).mean())),
                mean_total_umi=("total_umi", "mean"),
            )
        )
        grouped = grouped.sort_values(["total_counts", "cells"], ascending=False).head(
            max_treatments
        )
        treatment_table = [
            {
                "treatment": str(name),
                "cells": int(row["cells"]),
                "total_counts": float(row["total_counts"]),
                "mean_count": float(row["mean_count"]),
                "detected_frac": float(row["detected_frac"]),
                "mean_total_umi": float(row["mean_total_umi"]),
            }
            for name, row in grouped.iterrows()
        ]

    return {
        **gene,
        "n_cells": int(counts.size),
        "total_counts": float(np.sum(counts)),
        "mean_count": float(np.mean(counts)),
        "median_count": float(np.median(counts)),
        "p90_count": float(np.quantile(counts, 0.9)),
        "p99_count": float(np.quantile(counts, 0.99)),
        "max_count": float(np.max(counts)),
        "detected_cells": int(np.sum(detected)),
        "detected_frac": float(np.mean(detected)),
        "zero_frac": float(np.mean(~detected)),
        "count_total_correlation": corr,
        "treatment_table": treatment_table,
    }


def _sample_cells(x_vals: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    n = int(x_vals.size)
    if max_cells <= 0 or n <= max_cells:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(x_vals > 0)
    n_pos = min(len(positive), max_cells // 2)
    chosen_pos = (
        rng.choice(positive, size=n_pos, replace=False)
        if n_pos > 0
        else np.empty(0, dtype=int)
    )
    remaining = max_cells - chosen_pos.size
    pool = np.setdiff1d(np.arange(n, dtype=int), chosen_pos)
    chosen_other = rng.choice(pool, size=remaining, replace=False)
    return np.sort(np.concatenate([chosen_pos, chosen_other]).astype(int))


@lru_cache(maxsize=16)
def fit_cached_dataset_pool(h5ad_path: str, r: float = 0.05) -> PoolEstimationResult:
    bundle = load_dataset_bundle(h5ad_path)
    return fit_poisson_lognormal(
        bundle["totals"], r=r, max_iter=100, tol=1e-6, n_quad=80
    )


def run_real_gene_fit(
    h5ad_path: str,
    gene_query: str,
    max_cells_fit: int | None = None,
    r: float | None = None,
    grid_size: int | None = None,
    sigma_bins: float | None = None,
    decision_temperature: float | None = None,
    align_distribution: str | None = None,
    warm_start_mode: str | None = None,
    init_strategy: str | None = None,
    init_temperature: float | None = None,
    align_weight: float | None = None,
    lr: float | None = None,
    n_iter: int | None = None,
    seed: int | None = None,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    defaults = fit_defaults()
    max_cells_fit = int(
        defaults["max_cells_fit"] if max_cells_fit is None else max_cells_fit
    )
    r = float(defaults["r"] if r is None else r)
    grid_size = int(defaults["grid_size"] if grid_size is None else grid_size)
    sigma_bins = float(defaults["sigma_bins"] if sigma_bins is None else sigma_bins)
    decision_temperature = float(
        defaults["decision_temperature"]
        if decision_temperature is None
        else decision_temperature
    )
    align_distribution = str(
        defaults["align_distribution"]
        if align_distribution is None
        else align_distribution
    )
    warm_start_mode = str(
        defaults["warm_start_mode"] if warm_start_mode is None else warm_start_mode
    )
    init_strategy = str(
        defaults["init_strategy"] if init_strategy is None else init_strategy
    )
    init_temperature = float(
        defaults["init_temperature"] if init_temperature is None else init_temperature
    )
    align_weight = float(
        defaults["align_weight"] if align_weight is None else align_weight
    )
    lr = float(defaults["lr"] if lr is None else lr)
    n_iter = int(defaults["n_iter"] if n_iter is None else n_iter)
    seed = int(defaults["seed"] if seed is None else seed)

    gene = summarize_gene_expression(h5ad_path, gene_query)
    stage0 = fit_cached_dataset_pool(h5ad_path, r=r)

    sample_idx = _sample_cells(gene["x_vals"], max_cells_fit, seed)
    sampled_counts = gene["x_vals"][sample_idx].astype(float)
    sampled_totals = gene["totals"][sample_idx].astype(float)
    sampled_obs = gene["obs"].iloc[sample_idx].copy()

    stage1 = fit_gene_prior(
        gene_id=gene["gene_id"],
        x_vals=sampled_counts,
        n_vals=sampled_totals,
        s_hat=stage0.s_hat,
        grid_size=grid_size,
        sigma_bins=sigma_bins,
        decision_temperature=decision_temperature,
        align_distribution=align_distribution,
        warm_start_mode=warm_start_mode,
        init_strategy=init_strategy,
        init_temperature=init_temperature,
        align_weight=align_weight,
        lr=lr,
        n_iter=n_iter,
        callback=callback,
    )

    return {
        "gene": gene,
        "sample_idx": sample_idx,
        "sampled_obs": sampled_obs,
        "sampled_counts": sampled_counts,
        "sampled_totals": sampled_totals,
        "stage0": stage0,
        "stage1": stage1,
    }


# ============================================================
# Quick smoke test
# ============================================================


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n_cells = 240
    r = 0.05
    s_true = 1800.0
    epsilon_c = rng.lognormal(mean=0.0, sigma=0.45, size=n_cells)
    eta_true = r * s_true * epsilon_c
    totals = rng.poisson(eta_true)
    stage0 = fit_poisson_lognormal(totals, r=r, max_iter=40, n_quad=40)

    latent = np.concatenate(
        [
            rng.normal(8.0, 1.5, size=n_cells // 2),
            rng.normal(45.0, 6.0, size=n_cells - n_cells // 2),
        ]
    )
    latent = np.clip(latent, 0.0, None)
    p = np.clip(latent / stage0.s_hat, 0.0, 1.0)
    counts = rng.binomial(np.asarray(totals, dtype=int), p)

    stage1 = fit_gene_prior(
        gene_id="synthetic_gene",
        x_vals=counts.astype(float),
        n_vals=totals.astype(float),
        s_hat=stage0.s_hat,
        grid_size=96,
        sigma_bins=3.0,
        decision_temperature=0.0,
        align_distribution="posterior_average",
        warm_start_mode="posterior_average",
        init_strategy="bootstrap_raw_observation",
        init_temperature=1.0,
        align_weight=1.0,
        lr=0.05,
        n_iter=60,
    )

    print("=== scPRISM smoke test ===")
    print(
        f"Stage0: mu={stage0.mu:.4f} sigma={stage0.sigma:.4f} s_hat={stage0.s_hat:.2f} iter={stage0.n_iter}"
    )
    print(
        f"Stage1: final loss={stage1.loss_history[-1]:.4f} mean_conf={stage1.confidence.mean():.4f} support_max={stage1.support_max:.2f}"
    )

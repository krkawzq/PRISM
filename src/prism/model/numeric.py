from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants import DTYPE_TORCH, EPS, NEG_INF, TorchTensor


def gaussian_kernel_1d(sigma_bins: float) -> TorchTensor:
    if sigma_bins <= 0:
        return torch.ones(1, dtype=DTYPE_TORCH)
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    offsets = torch.arange(-radius, radius + 1, dtype=DTYPE_TORCH)
    kernel = torch.exp(-0.5 * (offsets / sigma_bins) ** 2)
    return kernel / kernel.sum()


def smooth_probability_weights(logits: TorchTensor, sigma_bins: float) -> TorchTensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    elif logits.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"logits must be 1D or 2D, got shape={tuple(logits.shape)}")
    weights = F.softmax(logits, dim=-1)
    kernel = gaussian_kernel_1d(sigma_bins).to(
        device=weights.device, dtype=weights.dtype
    )
    if kernel.numel() == 1:
        return weights[0] if squeeze else weights
    radius = kernel.numel() // 2
    padded = F.pad(weights.unsqueeze(1), (radius, radius), mode="replicate")
    smoothed = F.conv1d(padded, kernel.view(1, 1, -1)).squeeze(1)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return smoothed[0] if squeeze else smoothed


def log_binomial_likelihood_grid(
    counts: TorchTensor,
    effective_exposure: TorchTensor,
    p_grid: TorchTensor,
) -> TorchTensor:
    if counts.shape != effective_exposure.shape:
        raise ValueError(
            f"counts and effective_exposure must match, got {tuple(counts.shape)} != {tuple(effective_exposure.shape)}"
        )
    if p_grid.ndim != 2:
        raise ValueError(f"p_grid must be 2D, got shape={tuple(p_grid.shape)}")
    x = counts.unsqueeze(-1)
    n = effective_exposure.unsqueeze(-1)
    p = p_grid.unsqueeze(-2).clamp(EPS, 1.0 - EPS)
    log_coeff = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(x + 1.0)
        - torch.lgamma((n - x).clamp_min(0.0) + 1.0)
    )
    log_lik = log_coeff + x * torch.log(p) + (n - x) * torch.log1p(-p)
    invalid = x > (n + 1e-9)
    return log_lik.masked_fill(invalid, NEG_INF)


def posterior_from_log_likelihood(
    log_lik: TorchTensor, prior_weights: TorchTensor
) -> TorchTensor:
    log_prior = torch.log(prior_weights.clamp_min(EPS)).unsqueeze(-2)
    log_post = log_lik + log_prior
    log_post = log_post - torch.logsumexp(log_post, dim=-1, keepdim=True)
    return torch.exp(log_post)


def aggregate_posterior(post: TorchTensor) -> TorchTensor:
    q_hat = post.mean(dim=-2)
    return q_hat / q_hat.sum(dim=-1, keepdim=True).clamp_min(EPS)


def jsd(p: TorchTensor, q: TorchTensor) -> TorchTensor:
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {tuple(p.shape)} != {tuple(q.shape)}")
    m = 0.5 * (p + q)
    kl_p = (p * torch.log((p + EPS) / (m + EPS))).sum(dim=-1)
    kl_q = (q * torch.log((q + EPS) / (m + EPS))).sum(dim=-1)
    return 0.5 * (kl_p + kl_q)


def entropy(weights: TorchTensor) -> TorchTensor:
    values = -(weights * torch.log(weights.clamp_min(EPS))).sum(dim=-1)
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def kl_divergence(p: TorchTensor, q: TorchTensor) -> TorchTensor:
    """KL(p || q), per-gene."""
    return (p * torch.log((p + EPS) / (q + EPS))).sum(dim=-1)


def weighted_jsd(
    q_hat: TorchTensor,
    prior: TorchTensor,
    posterior_entropy: TorchTensor,
) -> TorchTensor:
    """JSD weighted by inverse posterior entropy (high-confidence cells contribute more)."""
    max_ent = math.log(prior.shape[-1]) + EPS
    confidence = 1.0 - (posterior_entropy / max_ent).clamp(0.0, 1.0)
    raw_jsd = jsd(q_hat, prior)
    return (raw_jsd * confidence).mean(dim=-1) if confidence.ndim > 0 else raw_jsd


def log_negative_binomial_likelihood_grid(
    counts: TorchTensor,
    effective_exposure: TorchTensor,
    p_grid: TorchTensor,
    overdispersion: float = 0.01,
) -> TorchTensor:
    """Negative binomial log-likelihood on a p-grid.

    Parameterised as NB(x | mu, r) where mu = n_eff * p_grid and
    r = 1/overdispersion.
    """
    if counts.shape != effective_exposure.shape:
        raise ValueError(
            f"counts and effective_exposure must match, "
            f"got {tuple(counts.shape)} != {tuple(effective_exposure.shape)}"
        )
    x = counts.unsqueeze(-1)
    n = effective_exposure.unsqueeze(-1)
    p = p_grid.unsqueeze(-2).clamp(EPS, 1.0 - EPS)
    mu = n * p
    r = 1.0 / overdispersion
    r_tensor = torch.tensor(r, dtype=x.dtype, device=x.device)
    log_lik = (
        torch.lgamma(x + r)
        - torch.lgamma(x + 1.0)
        - torch.lgamma(r_tensor)
        + r * torch.log(r_tensor)
        - r * torch.log(r_tensor + mu)
        + x * torch.log(mu + EPS)
        - x * torch.log(r_tensor + mu)
    )
    return log_lik


def log_poisson_likelihood_grid(
    counts: TorchTensor,
    rate_grid: TorchTensor,
) -> TorchTensor:
    """Poisson log-likelihood on a rate grid.

    Uses un-normalised form: log P(x | lambda) = x * log(lambda) - lambda.
    No use of N_eff or effective_exposure.
    """
    x = counts.unsqueeze(-1)
    lam = rate_grid.unsqueeze(-2).clamp(min=EPS)
    return x * torch.log(lam) - lam

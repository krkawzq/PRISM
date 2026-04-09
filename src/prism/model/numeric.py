from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants import EPS, NEG_INF


def gaussian_kernel_1d(
    sigma_bins: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if sigma_bins <= 0:
        return torch.ones(1, dtype=dtype, device=device)
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma_bins) ** 2)
    return kernel / kernel.sum()


def smooth_probability_weights(
    logits: torch.Tensor,
    sigma_bins: float,
) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    elif logits.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"logits must be 1D or 2D, got shape={tuple(logits.shape)}")
    probabilities = F.softmax(logits, dim=-1)
    kernel = gaussian_kernel_1d(
        sigma_bins,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    if kernel.numel() == 1:
        return probabilities[0] if squeeze else probabilities
    radius = kernel.numel() // 2
    padded = F.pad(probabilities.unsqueeze(1), (radius, radius), mode="replicate")
    smoothed = F.conv1d(padded, kernel.view(1, 1, -1)).squeeze(1)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return smoothed[0] if squeeze else smoothed


def log_binomial_likelihood_support(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    if counts.shape != effective_exposure.shape:
        raise ValueError(
            f"counts and effective_exposure must match, got {tuple(counts.shape)} != {tuple(effective_exposure.shape)}"
        )
    if support.ndim != 2:
        raise ValueError(f"support must be 2D, got shape={tuple(support.shape)}")
    x = counts.unsqueeze(-1)
    n = effective_exposure.unsqueeze(-1)
    p = support.unsqueeze(-2).clamp(EPS, 1.0 - EPS)
    log_coeff = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(x + 1.0)
        - torch.lgamma((n - x).clamp_min(0.0) + 1.0)
    )
    log_likelihood = log_coeff + x * torch.log(p) + (n - x) * torch.log1p(-p)
    invalid = x > (n + 1e-9)
    return log_likelihood.masked_fill(invalid, NEG_INF)


def log_negative_binomial_likelihood_support(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
    support: torch.Tensor,
    overdispersion: float = 0.01,
) -> torch.Tensor:
    if counts.shape != effective_exposure.shape:
        raise ValueError(
            f"counts and effective_exposure must match, got {tuple(counts.shape)} != {tuple(effective_exposure.shape)}"
        )
    x = counts.unsqueeze(-1)
    n = effective_exposure.unsqueeze(-1)
    p = support.unsqueeze(-2).clamp(EPS, 1.0 - EPS)
    mu = n * p
    r = 1.0 / overdispersion
    r_tensor = torch.tensor(r, dtype=x.dtype, device=x.device)
    return (
        torch.lgamma(x + r)
        - torch.lgamma(x + 1.0)
        - torch.lgamma(r_tensor)
        + r * torch.log(r_tensor)
        - r * torch.log(r_tensor + mu)
        + x * torch.log(mu + EPS)
        - x * torch.log(r_tensor + mu)
    )


def log_poisson_likelihood_support(
    counts: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    x = counts.unsqueeze(-1)
    rate = support.unsqueeze(-2).clamp_min(EPS)
    return x * torch.log(rate) - rate


def posterior_from_log_likelihood(
    log_likelihood: torch.Tensor,
    prior_probabilities: torch.Tensor,
) -> torch.Tensor:
    log_prior = torch.log(prior_probabilities.clamp_min(EPS)).unsqueeze(-2)
    log_posterior = log_likelihood + log_prior
    log_normalizer = torch.logsumexp(
        log_posterior,
        dim=-1,
        keepdim=True,
    )
    if torch.any(~torch.isfinite(log_normalizer)):
        raise ValueError(
            "encountered observations with zero likelihood under the current "
            "support/prior configuration"
        )
    log_posterior = log_posterior - log_normalizer
    return torch.exp(log_posterior)


def aggregate_posterior(posterior_probabilities: torch.Tensor) -> torch.Tensor:
    mean_probabilities = posterior_probabilities.mean(dim=-2)
    return mean_probabilities / mean_probabilities.sum(dim=-1, keepdim=True).clamp_min(
        EPS
    )


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    values = -(probabilities * torch.log(probabilities.clamp_min(EPS))).sum(dim=-1)
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * torch.log((p + EPS) / (q + EPS))).sum(dim=-1)


def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {tuple(p.shape)} != {tuple(q.shape)}")
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def weighted_jsd(
    posterior_mean_probabilities: torch.Tensor,
    prior_probabilities: torch.Tensor,
    posterior_entropy: torch.Tensor,
) -> torch.Tensor:
    max_entropy = math.log(prior_probabilities.shape[-1]) + EPS
    confidence = 1.0 - (posterior_entropy / max_entropy).clamp(0.0, 1.0)
    raw_jsd = jsd(posterior_mean_probabilities, prior_probabilities)
    return (raw_jsd * confidence).mean(dim=-1) if confidence.ndim > 0 else raw_jsd


__all__ = [
    "aggregate_posterior",
    "entropy",
    "gaussian_kernel_1d",
    "jsd",
    "kl_divergence",
    "log_binomial_likelihood_support",
    "log_negative_binomial_likelihood_support",
    "log_poisson_likelihood_support",
    "posterior_from_log_likelihood",
    "smooth_probability_weights",
    "weighted_jsd",
]

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import EPS, NEG_INF


@dataclass(frozen=True, slots=True)
class _ProbabilitySupportTerms:
    values: torch.Tensor
    log_values: torch.Tensor
    log1m_values: torch.Tensor


@dataclass(frozen=True, slots=True)
class _RateSupportTerms:
    values: torch.Tensor
    log_values: torch.Tensor


@dataclass(frozen=True, slots=True)
class _BinomialObservationTerms:
    counts: torch.Tensor
    exposure: torch.Tensor
    log_coeff: torch.Tensor
    invalid: torch.Tensor


@dataclass(frozen=True, slots=True)
class _NegativeBinomialObservationTerms:
    counts: torch.Tensor
    exposure: torch.Tensor
    base_term: torch.Tensor
    r: float
    r_tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PoissonObservationTerms:
    counts: torch.Tensor


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


def _validate_count_exposure_shapes(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
) -> None:
    if counts.shape != effective_exposure.shape:
        raise ValueError(
            "counts and effective_exposure must match, got "
            f"{tuple(counts.shape)} != {tuple(effective_exposure.shape)}"
        )


def _probability_support_terms(support: torch.Tensor) -> _ProbabilitySupportTerms:
    if support.ndim != 2:
        raise ValueError(f"support must be 2D, got shape={tuple(support.shape)}")
    values = support.clamp(EPS, 1.0 - EPS)
    return _ProbabilitySupportTerms(
        values=values,
        log_values=torch.log(values),
        log1m_values=torch.log1p(-values),
    )


def _rate_support_terms(support: torch.Tensor) -> _RateSupportTerms:
    if support.ndim != 2:
        raise ValueError(f"support must be 2D, got shape={tuple(support.shape)}")
    values = support.clamp_min(EPS)
    return _RateSupportTerms(
        values=values,
        log_values=torch.log(values),
    )


def _binomial_observation_terms(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
) -> _BinomialObservationTerms:
    _validate_count_exposure_shapes(counts, effective_exposure)
    counts_values = counts.unsqueeze(-1)
    exposure_values = effective_exposure.unsqueeze(-1)
    log_coeff = (
        torch.lgamma(exposure_values + 1.0)
        - torch.lgamma(counts_values + 1.0)
        - torch.lgamma((exposure_values - counts_values).clamp_min(0.0) + 1.0)
    )
    invalid = counts_values > (exposure_values + 1e-9)
    return _BinomialObservationTerms(
        counts=counts_values,
        exposure=exposure_values,
        log_coeff=log_coeff,
        invalid=invalid,
    )


def _negative_binomial_observation_terms(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
    overdispersion: float,
) -> _NegativeBinomialObservationTerms:
    _validate_count_exposure_shapes(counts, effective_exposure)
    counts_values = counts.unsqueeze(-1)
    exposure_values = effective_exposure.unsqueeze(-1)
    r = float(1.0 / overdispersion)
    r_tensor = torch.tensor(r, dtype=counts_values.dtype, device=counts_values.device)
    base_term = (
        torch.lgamma(counts_values + r)
        - torch.lgamma(counts_values + 1.0)
        - torch.lgamma(r_tensor)
        + r * torch.log(r_tensor)
    )
    return _NegativeBinomialObservationTerms(
        counts=counts_values,
        exposure=exposure_values,
        base_term=base_term,
        r=r,
        r_tensor=r_tensor,
    )


def _poisson_observation_terms(counts: torch.Tensor) -> _PoissonObservationTerms:
    return _PoissonObservationTerms(counts=counts.unsqueeze(-1))


def _log_binomial_likelihood_from_terms(
    observation_terms: _BinomialObservationTerms,
    support_terms: _ProbabilitySupportTerms,
) -> torch.Tensor:
    log_likelihood = (
        observation_terms.log_coeff
        + observation_terms.counts * support_terms.log_values.unsqueeze(-2)
        + (observation_terms.exposure - observation_terms.counts)
        * support_terms.log1m_values.unsqueeze(-2)
    )
    return log_likelihood.masked_fill(observation_terms.invalid, NEG_INF)


def _log_negative_binomial_likelihood_from_terms(
    observation_terms: _NegativeBinomialObservationTerms,
    support_terms: _ProbabilitySupportTerms,
) -> torch.Tensor:
    mu = observation_terms.exposure * support_terms.values.unsqueeze(-2)
    log_r_plus_mu = torch.log(observation_terms.r_tensor + mu)
    return (
        observation_terms.base_term
        + observation_terms.counts * torch.log(mu + EPS)
        - (observation_terms.counts + observation_terms.r) * log_r_plus_mu
    )


def _log_poisson_likelihood_from_terms(
    observation_terms: _PoissonObservationTerms,
    support_terms: _RateSupportTerms,
) -> torch.Tensor:
    return (
        observation_terms.counts * support_terms.log_values.unsqueeze(-2)
        - support_terms.values.unsqueeze(-2)
    )


def log_binomial_likelihood_support(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    return _log_binomial_likelihood_from_terms(
        _binomial_observation_terms(counts, effective_exposure),
        _probability_support_terms(support),
    )


def log_negative_binomial_likelihood_support(
    counts: torch.Tensor,
    effective_exposure: torch.Tensor,
    support: torch.Tensor,
    overdispersion: float = 0.01,
) -> torch.Tensor:
    return _log_negative_binomial_likelihood_from_terms(
        _negative_binomial_observation_terms(
            counts,
            effective_exposure,
            overdispersion,
        ),
        _probability_support_terms(support),
    )


def log_poisson_likelihood_support(
    counts: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    return _log_poisson_likelihood_from_terms(
        _poisson_observation_terms(counts),
        _rate_support_terms(support),
    )


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

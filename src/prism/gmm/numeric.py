from __future__ import annotations

import math

import numpy as np
import torch

from prism.model.constants import DTYPE_NP, EPS, SupportDomain
from prism.model.numeric import jsd as torch_jsd


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch_dtype: {name}")


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.special, "ndtr"):
        return torch.special.ndtr(values)
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def build_bin_edges(
    support: np.ndarray,
    *,
    support_domain: SupportDomain,
    midpoint_fraction: float = 0.5,
    single_point_rate_half_width_scale: float = 0.5,
    single_point_rate_half_width_floor: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    points = np.asarray(support, dtype=DTYPE_NP).reshape(-1)
    if points.size == 0:
        raise ValueError("support cannot be empty")
    if points.size == 1:
        value = float(points[0])
        if support_domain == "probability":
            lower_bound = 0.0
            upper_bound = 1.0
        else:
            half_width = max(
                abs(value),
                float(single_point_rate_half_width_floor),
            ) * float(single_point_rate_half_width_scale)
            lower_bound = max(0.0, value - half_width)
            upper_bound = max(lower_bound + EPS, value + half_width)
        return (
            np.asarray([lower_bound, upper_bound], dtype=DTYPE_NP),
            float(lower_bound),
            float(upper_bound),
        )

    fraction = float(midpoint_fraction)
    midpoints = points[:-1] + fraction * (points[1:] - points[:-1])
    first_gap = points[1] - points[0]
    last_gap = points[-1] - points[-2]
    left_edge = points[0] - fraction * first_gap
    right_edge = points[-1] + fraction * last_gap
    edges = np.concatenate(
        [
            np.asarray([left_edge], dtype=DTYPE_NP),
            np.asarray(midpoints, dtype=DTYPE_NP),
            np.asarray([right_edge], dtype=DTYPE_NP),
        ]
    )
    if support_domain == "probability":
        edges = np.clip(edges, 0.0, 1.0)
    else:
        edges = np.clip(edges, 0.0, None)
    edges = np.maximum.accumulate(edges)
    for idx in range(1, edges.shape[0]):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + EPS
    return edges.astype(DTYPE_NP, copy=False), float(edges[0]), float(edges[-1])


def truncated_gaussian_bin_masses(
    bin_edges: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    left_truncations: torch.Tensor,
    right_truncations: torch.Tensor,
    bin_mask: torch.Tensor,
) -> torch.Tensor:
    if bin_edges.ndim != 2:
        raise ValueError(f"bin_edges must be 2D, got shape={tuple(bin_edges.shape)}")
    if means.ndim != 2:
        raise ValueError(f"means must be 2D, got shape={tuple(means.shape)}")
    if stds.shape != means.shape:
        raise ValueError("stds must match means")
    if left_truncations.shape != means.shape:
        raise ValueError("left_truncations must match means")
    if right_truncations.shape != means.shape:
        raise ValueError("right_truncations must match means")
    if bin_mask.shape != (bin_edges.shape[0], bin_edges.shape[1] - 1):
        raise ValueError("bin_mask must match bin_edges")
    if means.shape[0] != bin_edges.shape[0]:
        raise ValueError("means must match the batch size")

    edges_left = bin_edges[:, :-1].unsqueeze(-1)
    edges_right = bin_edges[:, 1:].unsqueeze(-1)
    return truncated_gaussian_bin_masses_from_edges(
        edges_left,
        edges_right,
        means,
        stds,
        left_truncations,
        right_truncations,
        bin_mask.unsqueeze(-1),
    )


def truncated_gaussian_bin_masses_dense_1d(
    bin_edges: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    left_truncations: torch.Tensor,
    right_truncations: torch.Tensor,
) -> torch.Tensor:
    if bin_edges.ndim != 1:
        raise ValueError(f"bin_edges must be 1D, got shape={tuple(bin_edges.shape)}")
    if means.ndim != 1:
        raise ValueError(f"means must be 1D, got shape={tuple(means.shape)}")
    if stds.shape != means.shape:
        raise ValueError("stds must match means")
    if left_truncations.shape != means.shape:
        raise ValueError("left_truncations must match means")
    if right_truncations.shape != means.shape:
        raise ValueError("right_truncations must match means")
    if bin_edges.shape[0] < 2:
        raise ValueError("bin_edges must contain at least one bin")

    edges_left = bin_edges[:-1].unsqueeze(-1)
    edges_right = bin_edges[1:].unsqueeze(-1)
    stds_safe = stds.clamp_min(EPS)
    means_view = means.unsqueeze(0)
    stds_view = stds_safe.unsqueeze(0)
    left_view = left_truncations.unsqueeze(0)
    right_view = right_truncations.unsqueeze(0)
    clipped_left = torch.maximum(edges_left, left_view)
    clipped_right = torch.minimum(edges_right, right_view)
    active_bins = clipped_right > clipped_left

    z_left = (clipped_left - means_view) / stds_view
    z_right = (clipped_right - means_view) / stds_view
    numerator = _normal_cdf(z_right) - _normal_cdf(z_left)
    numerator = torch.where(active_bins, numerator, torch.zeros_like(numerator))
    trunc_left = (left_truncations - means) / stds_safe
    trunc_right = (right_truncations - means) / stds_safe
    denominator = (_normal_cdf(trunc_right) - _normal_cdf(trunc_left)).clamp_min(EPS)
    masses = (numerator / denominator.unsqueeze(0)).clamp_min(0.0)
    normalizer = masses.sum(dim=0, keepdim=True).clamp_min(EPS)
    return masses / normalizer


def truncated_gaussian_bin_masses_from_edges(
    edges_left: torch.Tensor,
    edges_right: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    left_truncations: torch.Tensor,
    right_truncations: torch.Tensor,
    bin_mask: torch.Tensor,
) -> torch.Tensor:
    if edges_left.ndim != 3 or edges_right.ndim != 3:
        raise ValueError("edges_left and edges_right must be 3D")
    if edges_left.shape != edges_right.shape:
        raise ValueError("edges_left and edges_right must match")
    if means.ndim != 2:
        raise ValueError(f"means must be 2D, got shape={tuple(means.shape)}")
    if stds.shape != means.shape:
        raise ValueError("stds must match means")
    if left_truncations.shape != means.shape:
        raise ValueError("left_truncations must match means")
    if right_truncations.shape != means.shape:
        raise ValueError("right_truncations must match means")
    if bin_mask.shape != edges_left.shape:
        raise ValueError("bin_mask must match edge tensors")
    if means.shape[0] != edges_left.shape[0]:
        raise ValueError("means must match the batch size")

    means_view = means.unsqueeze(1)
    stds_view = stds.clamp_min(EPS).unsqueeze(1)
    left_view = left_truncations.unsqueeze(1)
    right_view = right_truncations.unsqueeze(1)
    clipped_left = torch.maximum(edges_left, left_view)
    clipped_right = torch.minimum(edges_right, right_view)
    active_bins = clipped_right > clipped_left

    z_left = (clipped_left - means_view) / stds_view
    z_right = (clipped_right - means_view) / stds_view
    numerator = _normal_cdf(z_right) - _normal_cdf(z_left)
    numerator = torch.where(active_bins, numerator, torch.zeros_like(numerator))
    trunc_left = (left_truncations - means) / stds.clamp_min(EPS)
    trunc_right = (right_truncations - means) / stds.clamp_min(EPS)
    denominator = (_normal_cdf(trunc_right) - _normal_cdf(trunc_left)).clamp_min(EPS)
    masses = numerator / denominator.unsqueeze(1)
    masses = masses * bin_mask
    masses = masses.clamp_min(0.0)
    normalizer = masses.sum(dim=1, keepdim=True).clamp_min(EPS)
    return masses / normalizer


def normalize_active_weights(
    weights: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if weights.shape != active_mask.shape:
        raise ValueError("weights and active_mask must match")
    resolved = torch.where(
        active_mask,
        weights.clamp_min(0.0),
        torch.zeros_like(weights),
    )
    return resolved / resolved.sum(dim=-1, keepdim=True).clamp_min(EPS)


def mixture_bin_masses(
    component_masses: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if component_masses.ndim != 3:
        raise ValueError(
            f"component_masses must be 3D, got shape={tuple(component_masses.shape)}"
        )
    if weights.shape != (component_masses.shape[0], component_masses.shape[2]):
        raise ValueError("weights must match component_masses")
    mixture = torch.sum(component_masses * weights.unsqueeze(1), dim=-1)
    return mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(EPS)


def cross_entropy_loss(
    target: torch.Tensor,
    fitted: torch.Tensor,
) -> torch.Tensor:
    if target.shape != fitted.shape:
        raise ValueError(
            f"shape mismatch: {tuple(target.shape)} != {tuple(fitted.shape)}"
        )
    return -(target * torch.log(fitted.clamp_min(EPS))).sum(dim=-1)


def l1_error(
    target: torch.Tensor,
    fitted: torch.Tensor,
) -> torch.Tensor:
    if target.shape != fitted.shape:
        raise ValueError(
            f"shape mismatch: {tuple(target.shape)} != {tuple(fitted.shape)}"
        )
    return torch.sum(torch.abs(target - fitted), dim=-1)


def jsd_error(
    target: torch.Tensor,
    fitted: torch.Tensor,
) -> torch.Tensor:
    if target.shape != fitted.shape:
        raise ValueError(
            f"shape mismatch: {tuple(target.shape)} != {tuple(fitted.shape)}"
        )
    return torch_jsd(target, fitted)


__all__ = [
    "build_bin_edges",
    "cross_entropy_loss",
    "jsd_error",
    "l1_error",
    "mixture_bin_masses",
    "normalize_active_weights",
    "resolve_torch_dtype",
    "truncated_gaussian_bin_masses",
    "truncated_gaussian_bin_masses_dense_1d",
    "truncated_gaussian_bin_masses_from_edges",
]

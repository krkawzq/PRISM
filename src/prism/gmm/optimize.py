from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from prism.model.constants import DTYPE_NP, EPS

from .numeric import (
    cross_entropy_loss,
    jsd_error,
    l1_error,
    mixture_bin_masses,
    normalize_active_weights,
    resolve_torch_dtype,
    truncated_gaussian_bin_masses,
)


@dataclass(frozen=True, slots=True)
class MixtureOptimizationResult:
    fitted_probabilities: np.ndarray
    component_masses: np.ndarray
    component_weights: np.ndarray
    component_means: np.ndarray
    component_stds: np.ndarray
    component_left_truncations: np.ndarray
    component_right_truncations: np.ndarray
    jsd: np.ndarray
    cross_entropy: np.ndarray
    l1_error: np.ndarray


def _logit(values: np.ndarray, *, clip: float) -> np.ndarray:
    clipped = np.clip(values, clip, 1.0 - clip)
    return np.log(clipped / (1.0 - clipped))


def _inverse_softplus(values: np.ndarray, *, clip: float) -> np.ndarray:
    clipped = np.clip(values, clip, None)
    return np.log(np.expm1(clipped))


def maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    try:
        return module if not hasattr(torch, "compile") else torch.compile(module)
    except Exception:
        return module


def sigma_floor(
    bin_edges: np.ndarray,
    *,
    floor_fraction: float = 0.25,
) -> np.ndarray:
    widths = np.diff(bin_edges, axis=1)
    positive_widths = np.where(widths > EPS, widths, np.inf)
    floor = np.min(positive_widths, axis=1)
    floor = np.where(np.isfinite(floor), floor, 1.0)
    return np.asarray(np.maximum(floor * floor_fraction, EPS), dtype=DTYPE_NP)


def _validate_inputs(
    *,
    probabilities: np.ndarray,
    bin_edges: np.ndarray,
    support_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    selected_k: np.ndarray,
    component_weights: np.ndarray,
    component_means: np.ndarray,
    component_stds: np.ndarray,
    component_left_truncations: np.ndarray,
    component_right_truncations: np.ndarray,
) -> tuple[np.ndarray, ...]:
    probabilities_np = np.asarray(probabilities, dtype=DTYPE_NP)
    bin_edges_np = np.asarray(bin_edges, dtype=DTYPE_NP)
    support_mask_np = np.asarray(support_mask, dtype=bool)
    lower_bounds_np = np.asarray(lower_bounds, dtype=DTYPE_NP).reshape(-1)
    upper_bounds_np = np.asarray(upper_bounds, dtype=DTYPE_NP).reshape(-1)
    selected_k_np = np.asarray(selected_k, dtype=np.int64).reshape(-1)
    component_weights_np = np.asarray(component_weights, dtype=DTYPE_NP)
    component_means_np = np.asarray(component_means, dtype=DTYPE_NP)
    component_stds_np = np.asarray(component_stds, dtype=DTYPE_NP)
    component_left_np = np.asarray(component_left_truncations, dtype=DTYPE_NP)
    component_right_np = np.asarray(component_right_truncations, dtype=DTYPE_NP)

    if probabilities_np.ndim != 2:
        raise ValueError("probabilities must be 2D")
    if bin_edges_np.shape != (probabilities_np.shape[0], probabilities_np.shape[1] + 1):
        raise ValueError("bin_edges must have one more column than probabilities")
    if support_mask_np.shape != probabilities_np.shape:
        raise ValueError("support_mask must match probabilities")
    if lower_bounds_np.shape[0] != probabilities_np.shape[0]:
        raise ValueError("lower_bounds must match the batch size")
    if upper_bounds_np.shape[0] != probabilities_np.shape[0]:
        raise ValueError("upper_bounds must match the batch size")
    if selected_k_np.shape[0] != probabilities_np.shape[0]:
        raise ValueError("selected_k must match the batch size")
    if component_weights_np.shape != component_means_np.shape:
        raise ValueError("component arrays must match")
    if component_weights_np.shape != component_stds_np.shape:
        raise ValueError("component arrays must match")
    if component_weights_np.shape != component_left_np.shape:
        raise ValueError("component arrays must match")
    if component_weights_np.shape != component_right_np.shape:
        raise ValueError("component arrays must match")
    if component_weights_np.shape[0] != probabilities_np.shape[0]:
        raise ValueError("component arrays must match the batch size")
    if np.any(selected_k_np < 1) or np.any(selected_k_np > component_weights_np.shape[1]):
        raise ValueError("selected_k must lie within the component axis")
    return (
        probabilities_np,
        bin_edges_np,
        support_mask_np,
        lower_bounds_np,
        upper_bounds_np,
        selected_k_np,
        component_weights_np,
        component_means_np,
        component_stds_np,
        component_left_np,
        component_right_np,
    )


def _to_numpy_result(
    *,
    fitted_t: torch.Tensor,
    component_masses_t: torch.Tensor,
    weights_t: torch.Tensor,
    means_t: torch.Tensor,
    stds_t: torch.Tensor,
    lefts_t: torch.Tensor,
    rights_t: torch.Tensor,
    target_t: torch.Tensor,
) -> MixtureOptimizationResult:
    jsd_t = jsd_error(target_t, fitted_t)
    ce_t = cross_entropy_loss(target_t, fitted_t)
    l1_t = l1_error(target_t, fitted_t)
    return MixtureOptimizationResult(
        fitted_probabilities=fitted_t.detach()
        .cpu()
        .numpy()
        .astype(DTYPE_NP, copy=False),
        component_masses=component_masses_t.detach()
        .cpu()
        .numpy()
        .astype(DTYPE_NP, copy=False),
        component_weights=weights_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
        component_means=means_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
        component_stds=stds_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
        component_left_truncations=lefts_t.detach()
        .cpu()
        .numpy()
        .astype(DTYPE_NP, copy=False),
        component_right_truncations=rights_t.detach()
        .cpu()
        .numpy()
        .astype(DTYPE_NP, copy=False),
        jsd=jsd_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
        cross_entropy=ce_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
        l1_error=l1_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
    )


def evaluate_mixture_parameters(
    *,
    probabilities: np.ndarray,
    bin_edges: np.ndarray,
    support_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    selected_k: np.ndarray,
    component_weights: np.ndarray,
    component_means: np.ndarray,
    component_stds: np.ndarray,
    component_left_truncations: np.ndarray,
    component_right_truncations: np.ndarray,
    torch_dtype: str,
    device: str | torch.device = "cpu",
) -> MixtureOptimizationResult:
    (
        probabilities_np,
        bin_edges_np,
        support_mask_np,
        lower_bounds_np,
        upper_bounds_np,
        selected_k_np,
        component_weights_np,
        component_means_np,
        component_stds_np,
        component_left_np,
        component_right_np,
    ) = _validate_inputs(
        probabilities=probabilities,
        bin_edges=bin_edges,
        support_mask=support_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        selected_k=selected_k,
        component_weights=component_weights,
        component_means=component_means,
        component_stds=component_stds,
        component_left_truncations=component_left_truncations,
        component_right_truncations=component_right_truncations,
    )

    active_mask_np = (
        np.arange(component_weights_np.shape[1], dtype=np.int64)[None, :]
        < selected_k_np[:, None]
    )
    clipped_left_np = np.maximum(component_left_np, lower_bounds_np[:, None])
    clipped_right_np = np.minimum(component_right_np, upper_bounds_np[:, None])
    clipped_right_np = np.maximum(clipped_right_np, clipped_left_np + EPS)
    dtype_obj = resolve_torch_dtype(torch_dtype)
    device_obj = torch.device(device)
    with torch.no_grad():
        target_t = torch.as_tensor(probabilities_np, dtype=dtype_obj, device=device_obj)
        bin_edges_t = torch.as_tensor(bin_edges_np, dtype=dtype_obj, device=device_obj)
        support_mask_t = torch.as_tensor(
            support_mask_np,
            dtype=torch.bool,
            device=device_obj,
        )
        component_weights_t = torch.as_tensor(
            component_weights_np,
            dtype=dtype_obj,
            device=device_obj,
        )
        active_mask_t = torch.as_tensor(
            active_mask_np,
            dtype=torch.bool,
            device=device_obj,
        )
        weights_t = normalize_active_weights(component_weights_t, active_mask_t)
        means_t = torch.as_tensor(component_means_np, dtype=dtype_obj, device=device_obj)
        stds_t = torch.as_tensor(component_stds_np, dtype=dtype_obj, device=device_obj)
        stds_t = stds_t.clamp_min(EPS)
        lefts_t = torch.as_tensor(clipped_left_np, dtype=dtype_obj, device=device_obj)
        rights_t = torch.as_tensor(clipped_right_np, dtype=dtype_obj, device=device_obj)
        component_masses_t = truncated_gaussian_bin_masses(
            bin_edges_t,
            means_t,
            stds_t,
            lefts_t,
            rights_t,
            support_mask_t,
        )
        fitted_t = mixture_bin_masses(component_masses_t, weights_t)
    return _to_numpy_result(
        fitted_t=fitted_t,
        component_masses_t=component_masses_t,
        weights_t=weights_t,
        means_t=means_t,
        stds_t=stds_t,
        lefts_t=lefts_t,
        rights_t=rights_t,
        target_t=target_t,
    )


def initialize_raw_parameters(
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    bin_edges: np.ndarray,
    selected_k: np.ndarray,
    initial_weights: np.ndarray,
    initial_means: np.ndarray,
    initial_stds: np.ndarray,
    initial_left_truncations: np.ndarray,
    initial_right_truncations: np.ndarray,
    mean_margin_fraction: float,
    sigma_floor_fraction: float,
    min_window_fraction: float,
    initial_weight_logit_floor: float,
    inactive_weight_floor: float,
    logit_clip: float,
    inverse_softplus_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_genes, n_components = initial_weights.shape
    lower = np.asarray(lower_bounds, dtype=DTYPE_NP).reshape(-1, 1)
    upper = np.asarray(upper_bounds, dtype=DTYPE_NP).reshape(-1, 1)
    span = np.maximum(upper - lower, EPS)
    sigma_floor_values = sigma_floor(
        bin_edges,
        floor_fraction=sigma_floor_fraction,
    ).reshape(-1, 1)
    mean_low = lower - mean_margin_fraction * span
    mean_high = upper + mean_margin_fraction * span

    weight_logits = np.full(
        (n_genes, n_components),
        float(initial_weight_logit_floor),
        dtype=DTYPE_NP,
    )
    mean_raw = np.zeros_like(weight_logits)
    std_raw = np.zeros_like(weight_logits)
    left_raw = np.zeros_like(weight_logits)
    window_raw = np.zeros_like(weight_logits)

    for row_idx in range(n_genes):
        k = int(selected_k[row_idx])
        fallback_mean = float(initial_means[row_idx, 0])
        fallback_std = max(
            float(initial_stds[row_idx, 0]),
            float(sigma_floor_values[row_idx, 0]),
        )
        for component_idx in range(n_components):
            active = component_idx < k
            weight = (
                float(initial_weights[row_idx, component_idx])
                if active
                else float(inactive_weight_floor)
            )
            mean = (
                float(initial_means[row_idx, component_idx])
                if active
                else fallback_mean
            )
            std = (
                float(initial_stds[row_idx, component_idx])
                if active
                else fallback_std
            )
            left = (
                float(initial_left_truncations[row_idx, component_idx])
                if active
                else float(lower[row_idx, 0])
            )
            right = (
                float(initial_right_truncations[row_idx, component_idx])
                if active
                else float(upper[row_idx, 0])
            )
            right = max(right, left + EPS)
            weight_logits[row_idx, component_idx] = math.log(
                max(weight, float(inactive_weight_floor))
            )
            mean_norm = (mean - mean_low[row_idx, 0]) / max(
                mean_high[row_idx, 0] - mean_low[row_idx, 0],
                EPS,
            )
            mean_raw[row_idx, component_idx] = _logit(
                np.asarray([mean_norm]),
                clip=logit_clip,
            )[0]
            std_raw[row_idx, component_idx] = _inverse_softplus(
                np.asarray([max(std - sigma_floor_values[row_idx, 0], EPS)]),
                clip=inverse_softplus_clip,
            )[0]
            left_norm = (left - lower[row_idx, 0]) / max(span[row_idx, 0], EPS)
            left_raw[row_idx, component_idx] = _logit(
                np.asarray([left_norm]),
                clip=logit_clip,
            )[0]
            min_window = max(
                float(sigma_floor_values[row_idx, 0]),
                float(span[row_idx, 0] * min_window_fraction),
                EPS,
            )
            max_window = max(float(upper[row_idx, 0] - left), min_window + EPS)
            window_norm = (right - left - min_window) / max(
                max_window - min_window,
                EPS,
            )
            window_raw[row_idx, component_idx] = _logit(
                np.asarray([window_norm]),
                clip=logit_clip,
            )[0]
    return weight_logits, mean_raw, std_raw, left_raw, window_raw


class RefitModule(torch.nn.Module):
    def __init__(
        self,
        *,
        target: torch.Tensor,
        bin_edges: torch.Tensor,
        bin_mask: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        sigma_floor_values: torch.Tensor,
        active_mask: torch.Tensor,
        initial_weight_logits: np.ndarray,
        initial_mean_raw: np.ndarray,
        initial_std_raw: np.ndarray,
        initial_left_raw: np.ndarray,
        initial_window_raw: np.ndarray,
        mean_margin_fraction: float,
        overshoot_penalty: float,
        min_window_fraction: float,
        masked_logit_value: float,
        truncation_mode: str,
        truncation_regularization_strength: float,
        optimize_weights: bool,
        optimize_means: bool,
        optimize_stds: bool,
        optimize_left_truncations: bool,
        optimize_right_truncations: bool,
    ) -> None:
        super().__init__()
        self.weight_logits = torch.nn.Parameter(
            torch.as_tensor(
                initial_weight_logits,
                dtype=target.dtype,
                device=target.device,
            ),
            requires_grad=optimize_weights,
        )
        self.mean_raw = torch.nn.Parameter(
            torch.as_tensor(initial_mean_raw, dtype=target.dtype, device=target.device),
            requires_grad=optimize_means,
        )
        self.std_raw = torch.nn.Parameter(
            torch.as_tensor(initial_std_raw, dtype=target.dtype, device=target.device),
            requires_grad=optimize_stds,
        )
        self.left_raw = torch.nn.Parameter(
            torch.as_tensor(initial_left_raw, dtype=target.dtype, device=target.device),
            requires_grad=optimize_left_truncations,
        )
        self.window_raw = torch.nn.Parameter(
            torch.as_tensor(
                initial_window_raw,
                dtype=target.dtype,
                device=target.device,
            ),
            requires_grad=optimize_right_truncations,
        )
        self.register_buffer("target", target)
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_mask", bin_mask)
        self.register_buffer("lower_bounds", lower_bounds)
        self.register_buffer("upper_bounds", upper_bounds)
        self.register_buffer("sigma_floor", sigma_floor_values)
        self.register_buffer("active_mask", active_mask)
        self.mean_margin_fraction = float(mean_margin_fraction)
        self.overshoot_penalty = float(overshoot_penalty)
        self.min_window_fraction = float(min_window_fraction)
        self.masked_logit_value = float(masked_logit_value)
        self.truncation_mode = str(truncation_mode)
        self.truncation_regularization_strength = float(
            truncation_regularization_strength
        )

    def forward(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        span = (self.upper_bounds - self.lower_bounds).clamp_min(EPS).unsqueeze(-1)
        mean_low = self.lower_bounds.unsqueeze(-1) - self.mean_margin_fraction * span
        mean_high = self.upper_bounds.unsqueeze(-1) + self.mean_margin_fraction * span
        means = mean_low + torch.sigmoid(self.mean_raw) * (mean_high - mean_low)
        stds = self.sigma_floor.unsqueeze(-1) + F.softplus(self.std_raw)
        if self.truncation_mode == "fixed_bounds":
            lefts = self.lower_bounds.unsqueeze(-1).expand_as(means)
            rights = self.upper_bounds.unsqueeze(-1).expand_as(means)
        else:
            lefts = self.lower_bounds.unsqueeze(-1) + torch.sigmoid(self.left_raw) * span
            min_window = torch.maximum(
                self.sigma_floor.unsqueeze(-1),
                span * self.min_window_fraction,
            )
            available_window = (self.upper_bounds.unsqueeze(-1) - lefts).clamp_min(
                min_window + EPS
            )
            windows = min_window + torch.sigmoid(self.window_raw) * (
                available_window - min_window
            )
            rights = torch.minimum(lefts + windows, self.upper_bounds.unsqueeze(-1))
        masked_logits = torch.where(
            self.active_mask,
            self.weight_logits,
            torch.full_like(self.weight_logits, self.masked_logit_value),
        )
        weights = torch.softmax(masked_logits, dim=-1)
        component_masses = truncated_gaussian_bin_masses(
            self.bin_edges,
            means,
            stds,
            lefts,
            rights,
            self.bin_mask,
        )
        fitted = mixture_bin_masses(component_masses, weights)
        loss = cross_entropy_loss(self.target, fitted)
        if self.overshoot_penalty > 0:
            loss = loss + self.overshoot_penalty * torch.sum(
                torch.relu(fitted - self.target) ** 2,
                dim=-1,
            )
        if self.truncation_regularization_strength > 0:
            window_fraction = ((rights - lefts) / span).clamp(0.0, 1.0)
            active_components = self.active_mask.to(dtype=self.target.dtype)
            truncation_penalty = torch.sum(
                ((1.0 - window_fraction) ** 2) * active_components,
                dim=-1,
            ) / active_components.sum(dim=-1).clamp_min(1.0)
            loss = loss + self.truncation_regularization_strength * truncation_penalty
        return fitted, component_masses, weights, means, stds, lefts, rights, loss


def optimize_mixture_parameters(
    *,
    probabilities: np.ndarray,
    bin_edges: np.ndarray,
    support_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    selected_k: np.ndarray,
    initial_weights: np.ndarray,
    initial_means: np.ndarray,
    initial_stds: np.ndarray,
    initial_left_truncations: np.ndarray,
    initial_right_truncations: np.ndarray,
    max_iterations: int,
    learning_rate: float,
    convergence_tolerance: float,
    overshoot_penalty: float,
    mean_margin_fraction: float,
    torch_dtype: str,
    device: str | torch.device = "cpu",
    compile_model: bool = False,
    optimize_weights: bool = True,
    optimize_means: bool = True,
    optimize_stds: bool = True,
    optimize_left_truncations: bool = True,
    optimize_right_truncations: bool = True,
    sigma_floor_fraction: float = 0.25,
    min_window_fraction: float = 1e-6,
    truncation_mode: str = "free",
    truncation_regularization_strength: float = 0.0,
    initial_weight_logit_floor: float = -12.0,
    inactive_weight_floor: float = 1e-12,
    masked_logit_value: float = -1e9,
    logit_clip: float = 1e-6,
    inverse_softplus_clip: float = 1e-8,
) -> MixtureOptimizationResult:
    (
        probabilities_np,
        bin_edges_np,
        support_mask_np,
        lower_bounds_np,
        upper_bounds_np,
        selected_k_np,
        initial_weights_np,
        initial_means_np,
        initial_stds_np,
        initial_left_np,
        initial_right_np,
    ) = _validate_inputs(
        probabilities=probabilities,
        bin_edges=bin_edges,
        support_mask=support_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        selected_k=selected_k,
        component_weights=initial_weights,
        component_means=initial_means,
        component_stds=initial_stds,
        component_left_truncations=initial_left_truncations,
        component_right_truncations=initial_right_truncations,
    )
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if convergence_tolerance < 0:
        raise ValueError("convergence_tolerance must be >= 0")
    if overshoot_penalty < 0:
        raise ValueError("overshoot_penalty must be >= 0")
    if mean_margin_fraction < 0:
        raise ValueError("mean_margin_fraction must be >= 0")
    if sigma_floor_fraction <= 0:
        raise ValueError("sigma_floor_fraction must be > 0")
    if min_window_fraction <= 0:
        raise ValueError("min_window_fraction must be > 0")
    if truncation_mode not in {"free", "fixed_bounds"}:
        raise ValueError(f"unsupported truncation_mode: {truncation_mode}")
    if truncation_regularization_strength < 0:
        raise ValueError("truncation_regularization_strength must be >= 0")
    if inactive_weight_floor <= 0:
        raise ValueError("inactive_weight_floor must be > 0")
    if logit_clip <= 0 or logit_clip >= 0.5:
        raise ValueError("logit_clip must be in (0, 0.5)")
    if inverse_softplus_clip <= 0:
        raise ValueError("inverse_softplus_clip must be > 0")

    active_mask_np = (
        np.arange(initial_weights_np.shape[1], dtype=np.int64)[None, :]
        < selected_k_np[:, None]
    )
    if truncation_mode == "fixed_bounds":
        initial_left_np = np.repeat(lower_bounds_np[:, None], initial_left_np.shape[1], axis=1)
        initial_right_np = np.repeat(
            upper_bounds_np[:, None],
            initial_right_np.shape[1],
            axis=1,
        )
        optimize_left_truncations = False
        optimize_right_truncations = False
    (
        initial_weight_logits,
        initial_mean_raw,
        initial_std_raw,
        initial_left_raw,
        initial_window_raw,
    ) = initialize_raw_parameters(
        lower_bounds=lower_bounds_np,
        upper_bounds=upper_bounds_np,
        bin_edges=bin_edges_np,
        selected_k=selected_k_np,
        initial_weights=initial_weights_np,
        initial_means=initial_means_np,
        initial_stds=initial_stds_np,
        initial_left_truncations=initial_left_np,
        initial_right_truncations=initial_right_np,
        mean_margin_fraction=mean_margin_fraction,
        sigma_floor_fraction=sigma_floor_fraction,
        min_window_fraction=min_window_fraction,
        initial_weight_logit_floor=initial_weight_logit_floor,
        inactive_weight_floor=inactive_weight_floor,
        logit_clip=logit_clip,
        inverse_softplus_clip=inverse_softplus_clip,
    )

    device_obj = torch.device(device)
    dtype_obj = resolve_torch_dtype(torch_dtype)
    sigma_floor_np = sigma_floor(
        bin_edges_np,
        floor_fraction=sigma_floor_fraction,
    )
    module = RefitModule(
        target=torch.as_tensor(probabilities_np, dtype=dtype_obj, device=device_obj),
        bin_edges=torch.as_tensor(bin_edges_np, dtype=dtype_obj, device=device_obj),
        bin_mask=torch.as_tensor(support_mask_np, dtype=torch.bool, device=device_obj),
        lower_bounds=torch.as_tensor(
            lower_bounds_np,
            dtype=dtype_obj,
            device=device_obj,
        ),
        upper_bounds=torch.as_tensor(
            upper_bounds_np,
            dtype=dtype_obj,
            device=device_obj,
        ),
        sigma_floor_values=torch.as_tensor(
            sigma_floor_np,
            dtype=dtype_obj,
            device=device_obj,
        ),
        active_mask=torch.as_tensor(
            active_mask_np,
            dtype=torch.bool,
            device=device_obj,
        ),
        initial_weight_logits=initial_weight_logits,
        initial_mean_raw=initial_mean_raw,
        initial_std_raw=initial_std_raw,
        initial_left_raw=initial_left_raw,
        initial_window_raw=initial_window_raw,
        mean_margin_fraction=mean_margin_fraction,
        overshoot_penalty=overshoot_penalty,
        min_window_fraction=min_window_fraction,
        masked_logit_value=masked_logit_value,
        truncation_mode=truncation_mode,
        truncation_regularization_strength=truncation_regularization_strength,
        optimize_weights=optimize_weights,
        optimize_means=optimize_means,
        optimize_stds=optimize_stds,
        optimize_left_truncations=optimize_left_truncations,
        optimize_right_truncations=optimize_right_truncations,
    )
    compiled = maybe_compile(module) if compile_model else module
    best_state: dict[str, torch.Tensor] | None = None
    trainable_parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    if max_iterations > 0 and trainable_parameters:
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        best_loss = float("inf")
        previous_loss: float | None = None
        for _ in range(max_iterations):
            optimizer.zero_grad(set_to_none=True)
            outputs = compiled()
            loss = outputs[-1].mean()
            loss.backward()
            optimizer.step()
            current_loss = float(loss.detach().cpu().item())
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {
                    name: value.detach().clone()
                    for name, value in module.state_dict().items()
                }
            if (
                previous_loss is not None
                and abs(previous_loss - current_loss) < convergence_tolerance
            ):
                break
            previous_loss = current_loss
    if best_state is not None:
        module.load_state_dict(best_state)
    with torch.no_grad():
        (
            fitted_t,
            component_masses_t,
            weights_t,
            means_t,
            stds_t,
            lefts_t,
            rights_t,
            _,
        ) = module()
    return _to_numpy_result(
        fitted_t=fitted_t,
        component_masses_t=component_masses_t,
        weights_t=weights_t,
        means_t=means_t,
        stds_t=stds_t,
        lefts_t=lefts_t,
        rights_t=rights_t,
        target_t=module.target,
    )


__all__ = [
    "MixtureOptimizationResult",
    "evaluate_mixture_parameters",
    "initialize_raw_parameters",
    "maybe_compile",
    "optimize_mixture_parameters",
    "sigma_floor",
]

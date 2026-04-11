from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

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
    truncated_gaussian_bin_masses_dense_1d,
    truncated_gaussian_bin_masses_from_edges,
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


CompilePolicy = Literal["never", "auto", "always"]


@dataclass(slots=True)
class _TensorWorkspace:
    target: torch.Tensor
    bin_edges: torch.Tensor
    support_mask: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    sigma_floor: torch.Tensor
    active_mask: torch.Tensor
    component_weights: torch.Tensor
    component_means: torch.Tensor
    component_stds: torch.Tensor
    component_lefts: torch.Tensor
    component_rights: torch.Tensor


_WORKSPACE_CACHE_MAXSIZE = 8
_WORKSPACE_CACHE: OrderedDict[tuple[object, ...], _TensorWorkspace] = OrderedDict()


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


def _resolve_compile_policy(
    *,
    compile_policy: CompilePolicy,
    compile_model: bool | None,
) -> CompilePolicy:
    if compile_policy not in {"never", "auto", "always"}:
        raise ValueError(f"unsupported compile_policy: {compile_policy!r}")
    if compile_model is None:
        return compile_policy
    if not isinstance(compile_model, bool):
        raise ValueError("compile_model must be a boolean when provided")
    legacy_policy: CompilePolicy = "always" if compile_model else "never"
    if compile_policy != "never" and compile_policy != legacy_policy:
        raise ValueError("compile_model conflicts with compile_policy")
    return legacy_policy


def _should_compile_module(
    *,
    compile_policy: CompilePolicy,
    device: torch.device,
    batch_size: int,
    n_bins: int,
    n_components: int,
    max_iterations: int,
) -> bool:
    if compile_policy == "never":
        return False
    if compile_policy == "always":
        return True
    if device.type != "cuda":
        return False
    work_units = batch_size * n_bins * max(n_components, 1) * max(max_iterations, 1)
    return max_iterations >= 32 and work_units >= 131072


def _workspace_key(
    *,
    batch_size: int,
    n_bins: int,
    n_components: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[object, ...]:
    return (
        batch_size,
        n_bins,
        n_components,
        str(dtype),
        device.type,
        device.index,
    )


def _new_workspace(
    *,
    batch_size: int,
    n_bins: int,
    n_components: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _TensorWorkspace:
    return _TensorWorkspace(
        target=torch.empty((batch_size, n_bins), dtype=dtype, device=device),
        bin_edges=torch.empty((batch_size, n_bins + 1), dtype=dtype, device=device),
        support_mask=torch.empty((batch_size, n_bins), dtype=torch.bool, device=device),
        lower_bounds=torch.empty(batch_size, dtype=dtype, device=device),
        upper_bounds=torch.empty(batch_size, dtype=dtype, device=device),
        sigma_floor=torch.empty(batch_size, dtype=dtype, device=device),
        active_mask=torch.empty((batch_size, n_components), dtype=torch.bool, device=device),
        component_weights=torch.empty((batch_size, n_components), dtype=dtype, device=device),
        component_means=torch.empty((batch_size, n_components), dtype=dtype, device=device),
        component_stds=torch.empty((batch_size, n_components), dtype=dtype, device=device),
        component_lefts=torch.empty((batch_size, n_components), dtype=dtype, device=device),
        component_rights=torch.empty((batch_size, n_components), dtype=dtype, device=device),
    )


def _get_tensor_workspace(
    *,
    batch_size: int,
    n_bins: int,
    n_components: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _TensorWorkspace:
    key = _workspace_key(
        batch_size=batch_size,
        n_bins=n_bins,
        n_components=n_components,
        dtype=dtype,
        device=device,
    )
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = _new_workspace(
            batch_size=batch_size,
            n_bins=n_bins,
            n_components=n_components,
            dtype=dtype,
            device=device,
        )
        _WORKSPACE_CACHE[key] = workspace
        if len(_WORKSPACE_CACHE) > _WORKSPACE_CACHE_MAXSIZE:
            _WORKSPACE_CACHE.popitem(last=False)
        return workspace
    _WORKSPACE_CACHE.move_to_end(key)
    return workspace


def _copy_numpy_to_tensor(array: np.ndarray, target: torch.Tensor) -> torch.Tensor:
    source = torch.as_tensor(
        np.ascontiguousarray(array),
        dtype=target.dtype,
    )
    target.copy_(source)
    return target


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
        if device_obj.type == "cpu":
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
            means_t = torch.as_tensor(component_means_np, dtype=dtype_obj, device=device_obj)
            stds_t = torch.as_tensor(component_stds_np, dtype=dtype_obj, device=device_obj)
            lefts_t = torch.as_tensor(clipped_left_np, dtype=dtype_obj, device=device_obj)
            rights_t = torch.as_tensor(clipped_right_np, dtype=dtype_obj, device=device_obj)
        else:
            workspace = _get_tensor_workspace(
                batch_size=probabilities_np.shape[0],
                n_bins=probabilities_np.shape[1],
                n_components=component_weights_np.shape[1],
                dtype=dtype_obj,
                device=device_obj,
            )
            target_t = _copy_numpy_to_tensor(probabilities_np, workspace.target)
            bin_edges_t = _copy_numpy_to_tensor(bin_edges_np, workspace.bin_edges)
            support_mask_t = _copy_numpy_to_tensor(
                support_mask_np,
                workspace.support_mask,
            )
            component_weights_t = _copy_numpy_to_tensor(
                component_weights_np,
                workspace.component_weights,
            )
            active_mask_t = _copy_numpy_to_tensor(
                active_mask_np,
                workspace.active_mask,
            )
            means_t = _copy_numpy_to_tensor(component_means_np, workspace.component_means)
            stds_t = _copy_numpy_to_tensor(component_stds_np, workspace.component_stds)
            lefts_t = _copy_numpy_to_tensor(clipped_left_np, workspace.component_lefts)
            rights_t = _copy_numpy_to_tensor(clipped_right_np, workspace.component_rights)
        weights_t = normalize_active_weights(component_weights_t, active_mask_t)
        stds_t = stds_t.clamp_min(EPS)
        if probabilities_np.shape[0] == 1 and bool(np.all(support_mask_np[0])):
            component_masses_t = truncated_gaussian_bin_masses_dense_1d(
                bin_edges_t[0],
                means_t[0],
                stds_t[0],
                lefts_t[0],
                rights_t[0],
            ).unsqueeze(0)
        else:
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

    active_mask = (
        np.arange(n_components, dtype=np.int64)[None, :] < selected_k.reshape(-1, 1)
    )
    fallback_mean = initial_means[:, :1]
    fallback_std = np.maximum(initial_stds[:, :1], sigma_floor_values)

    resolved_weights = np.where(
        active_mask,
        initial_weights,
        float(inactive_weight_floor),
    )
    resolved_means = np.where(active_mask, initial_means, fallback_mean)
    resolved_stds = np.where(active_mask, initial_stds, fallback_std)
    resolved_lefts = np.where(active_mask, initial_left_truncations, lower)
    resolved_rights = np.where(active_mask, initial_right_truncations, upper)
    resolved_rights = np.maximum(resolved_rights, resolved_lefts + EPS)

    weight_logits = np.log(
        np.maximum(resolved_weights, float(inactive_weight_floor))
    ).astype(DTYPE_NP, copy=False)

    mean_denom = np.maximum(mean_high - mean_low, EPS)
    mean_norm = (resolved_means - mean_low) / mean_denom
    mean_raw = _logit(mean_norm, clip=logit_clip).astype(DTYPE_NP, copy=False)

    std_delta = np.maximum(resolved_stds - sigma_floor_values, EPS)
    std_raw = _inverse_softplus(
        std_delta,
        clip=inverse_softplus_clip,
    ).astype(DTYPE_NP, copy=False)

    left_norm = (resolved_lefts - lower) / span
    left_raw = _logit(left_norm, clip=logit_clip).astype(DTYPE_NP, copy=False)

    min_window = np.maximum(
        np.maximum(sigma_floor_values, span * min_window_fraction),
        EPS,
    )
    max_window = np.maximum(upper - resolved_lefts, min_window + EPS)
    window_norm = (resolved_rights - resolved_lefts - min_window) / np.maximum(
        max_window - min_window,
        EPS,
    )
    window_raw = _logit(window_norm, clip=logit_clip).astype(DTYPE_NP, copy=False)
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
        target_dtype = target.dtype
        target_device = target.device
        span = (upper_bounds - lower_bounds).clamp_min(EPS).unsqueeze(-1)
        lower_bounds_expanded = lower_bounds.unsqueeze(-1)
        upper_bounds_expanded = upper_bounds.unsqueeze(-1)
        sigma_floor_expanded = sigma_floor_values.unsqueeze(-1)
        mean_low = lower_bounds_expanded - float(mean_margin_fraction) * span
        mean_span = upper_bounds_expanded + float(mean_margin_fraction) * span - mean_low
        min_window = torch.maximum(
            sigma_floor_expanded,
            span * float(min_window_fraction),
        )
        self.weight_logits = torch.nn.Parameter(
            torch.as_tensor(
                initial_weight_logits,
                dtype=target_dtype,
                device=target_device,
            ),
            requires_grad=optimize_weights,
        )
        self.mean_raw = torch.nn.Parameter(
            torch.as_tensor(initial_mean_raw, dtype=target_dtype, device=target_device),
            requires_grad=optimize_means,
        )
        self.std_raw = torch.nn.Parameter(
            torch.as_tensor(initial_std_raw, dtype=target_dtype, device=target_device),
            requires_grad=optimize_stds,
        )
        self.left_raw = torch.nn.Parameter(
            torch.as_tensor(initial_left_raw, dtype=target_dtype, device=target_device),
            requires_grad=optimize_left_truncations,
        )
        self.window_raw = torch.nn.Parameter(
            torch.as_tensor(
                initial_window_raw,
                dtype=target_dtype,
                device=target_device,
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
        self.register_buffer("span", span)
        self.register_buffer("lower_bounds_expanded", lower_bounds_expanded)
        self.register_buffer("upper_bounds_expanded", upper_bounds_expanded)
        self.register_buffer("sigma_floor_expanded", sigma_floor_expanded)
        self.register_buffer("mean_low", mean_low)
        self.register_buffer("mean_span", mean_span)
        self.register_buffer("min_window", min_window)
        self.register_buffer("bin_edges_left", bin_edges[:, :-1].unsqueeze(-1))
        self.register_buffer("bin_edges_right", bin_edges[:, 1:].unsqueeze(-1))
        self.register_buffer("bin_mask_expanded", bin_mask.unsqueeze(-1))
        self.register_buffer(
            "active_component_count",
            active_mask.sum(dim=-1).clamp_min(1).to(dtype=target_dtype),
        )
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
        means = self.mean_low + torch.sigmoid(self.mean_raw) * self.mean_span
        stds = self.sigma_floor_expanded + F.softplus(self.std_raw)
        if self.truncation_mode == "fixed_bounds":
            lefts = self.lower_bounds_expanded.expand_as(means)
            rights = self.upper_bounds_expanded.expand_as(means)
        else:
            lefts = self.lower_bounds_expanded + torch.sigmoid(self.left_raw) * self.span
            available_window = (self.upper_bounds_expanded - lefts).clamp_min(
                self.min_window + EPS
            )
            windows = self.min_window + torch.sigmoid(self.window_raw) * (
                available_window - self.min_window
            )
            rights = torch.minimum(lefts + windows, self.upper_bounds_expanded)
        masked_logits = torch.where(
            self.active_mask,
            self.weight_logits,
            torch.full_like(self.weight_logits, self.masked_logit_value),
        )
        weights = torch.softmax(masked_logits, dim=-1)
        component_masses = truncated_gaussian_bin_masses_from_edges(
            self.bin_edges_left,
            self.bin_edges_right,
            means,
            stds,
            lefts,
            rights,
            self.bin_mask_expanded,
        )
        fitted = mixture_bin_masses(component_masses, weights)
        loss = cross_entropy_loss(self.target, fitted)
        if self.overshoot_penalty > 0:
            loss = loss + self.overshoot_penalty * torch.sum(
                torch.relu(fitted - self.target) ** 2,
                dim=-1,
            )
        if self.truncation_regularization_strength > 0:
            window_fraction = ((rights - lefts) / self.span).clamp(0.0, 1.0)
            active_components = self.active_mask.to(dtype=self.target.dtype)
            truncation_penalty = torch.sum(
                ((1.0 - window_fraction) ** 2) * active_components,
                dim=-1,
            ) / self.active_component_count
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
    compile_model: bool | None = None,
    compile_policy: CompilePolicy = "never",
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
    compile_policy = _resolve_compile_policy(
        compile_policy=compile_policy,
        compile_model=compile_model,
    )

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
    if max_iterations == 0 or not any(
        [
            optimize_weights,
            optimize_means,
            optimize_stds,
            optimize_left_truncations,
            optimize_right_truncations,
        ]
    ):
        return evaluate_mixture_parameters(
            probabilities=probabilities_np,
            bin_edges=bin_edges_np,
            support_mask=support_mask_np,
            lower_bounds=lower_bounds_np,
            upper_bounds=upper_bounds_np,
            selected_k=selected_k_np,
            component_weights=initial_weights_np,
            component_means=initial_means_np,
            component_stds=initial_stds_np,
            component_left_truncations=initial_left_np,
            component_right_truncations=initial_right_np,
            torch_dtype=torch_dtype,
            device=device,
        )
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
    if device_obj.type == "cpu":
        target_t = torch.as_tensor(probabilities_np, dtype=dtype_obj, device=device_obj)
        bin_edges_t = torch.as_tensor(bin_edges_np, dtype=dtype_obj, device=device_obj)
        bin_mask_t = torch.as_tensor(
            support_mask_np,
            dtype=torch.bool,
            device=device_obj,
        )
        lower_bounds_t = torch.as_tensor(
            lower_bounds_np,
            dtype=dtype_obj,
            device=device_obj,
        )
        upper_bounds_t = torch.as_tensor(
            upper_bounds_np,
            dtype=dtype_obj,
            device=device_obj,
        )
        sigma_floor_t = torch.as_tensor(
            sigma_floor_np,
            dtype=dtype_obj,
            device=device_obj,
        )
        active_mask_t = torch.as_tensor(
            active_mask_np,
            dtype=torch.bool,
            device=device_obj,
        )
    else:
        workspace = _get_tensor_workspace(
            batch_size=probabilities_np.shape[0],
            n_bins=probabilities_np.shape[1],
            n_components=initial_weights_np.shape[1],
            dtype=dtype_obj,
            device=device_obj,
        )
        target_t = _copy_numpy_to_tensor(probabilities_np, workspace.target)
        bin_edges_t = _copy_numpy_to_tensor(bin_edges_np, workspace.bin_edges)
        bin_mask_t = _copy_numpy_to_tensor(support_mask_np, workspace.support_mask)
        lower_bounds_t = _copy_numpy_to_tensor(lower_bounds_np, workspace.lower_bounds)
        upper_bounds_t = _copy_numpy_to_tensor(upper_bounds_np, workspace.upper_bounds)
        sigma_floor_t = _copy_numpy_to_tensor(sigma_floor_np, workspace.sigma_floor)
        active_mask_t = _copy_numpy_to_tensor(active_mask_np, workspace.active_mask)
    module = RefitModule(
        target=target_t,
        bin_edges=bin_edges_t,
        bin_mask=bin_mask_t,
        lower_bounds=lower_bounds_t,
        upper_bounds=upper_bounds_t,
        sigma_floor_values=sigma_floor_t,
        active_mask=active_mask_t,
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
    compiled = (
        maybe_compile(module)
        if _should_compile_module(
            compile_policy=compile_policy,
            device=device_obj,
            batch_size=probabilities_np.shape[0],
            n_bins=probabilities_np.shape[1],
            n_components=initial_weights_np.shape[1],
            max_iterations=max_iterations,
        )
        else module
    )
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

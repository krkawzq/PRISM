from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from prism.model.constants import DTYPE_NP, SupportDomain

SupportAxis = Literal["raw", "scaled"]
RefitPruningMetric = Literal["weight", "peak_mass"]
RefitErrorMetric = Literal["jsd", "l1", "cross_entropy"]
FrontierUpdateStrategy = Literal["full_stage", "best_prefix", "single_step"]
CandidateAlphaStrategy = Literal["min_ratio", "least_squares"]
KSelectionMode = Literal["threshold_first", "marginal_gain", "penalized_error"]
TruncationMode = Literal["free", "fixed_bounds"]
CompilePolicy = Literal["never", "auto", "always"]


def _as_vector(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _as_matrix(values: np.ndarray | list[list[float]], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty, got shape={array.shape}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _as_bool_matrix(values: np.ndarray | list[list[bool]], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty, got shape={array.shape}")
    return array


def _as_index_vector(values: np.ndarray | list[int], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


def _normalize_gene_names(gene_names: list[str]) -> list[str]:
    resolved = [str(name) for name in gene_names]
    if not resolved:
        raise ValueError("gene_names cannot be empty")
    if len(resolved) != len(set(resolved)):
        raise ValueError("gene_names must be unique")
    return resolved


def _freeze_config_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


def _resolve_compile_policy_compat(
    *,
    compile_policy: str,
    compile_model: object,
    compile_policy_name: str,
    compile_model_name: str,
) -> tuple[CompilePolicy, bool | None]:
    if compile_policy not in {"never", "auto", "always"}:
        raise ValueError(f"unsupported {compile_policy_name}: {compile_policy!r}")
    resolved_model = compile_model
    if resolved_model is not None:
        if not isinstance(resolved_model, bool):
            raise ValueError(f"{compile_model_name} must be a boolean when provided")
        legacy_policy: CompilePolicy = "always" if resolved_model else "never"
        if compile_policy != "never" and compile_policy != legacy_policy:
            raise ValueError(
                f"{compile_model_name} conflicts with {compile_policy_name}"
            )
        return legacy_policy, resolved_model
    return compile_policy, None


def _validate_prefix_mask(
    mask: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={mask.shape}")
    if mask.shape[1] > 1 and np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError(f"{name} must be prefix-contiguous along the support axis")
    active_counts = mask.sum(axis=1).astype(np.int64, copy=False)
    if np.any(active_counts < 1):
        raise ValueError(f"{name} must activate at least one support point per row")
    return active_counts


def _validate_support_grid(
    *,
    support: np.ndarray,
    support_mask: np.ndarray,
    probabilities: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    bin_edges: np.ndarray | None = None,
    support_domain: SupportDomain | None = None,
    masked_probability_name: str = "probabilities",
) -> np.ndarray:
    active_counts = _validate_prefix_mask(support_mask, name="support_mask")
    masked_probabilities = np.where(support_mask, 0.0, probabilities)
    if np.any(np.abs(masked_probabilities) > 1e-9):
        raise ValueError(f"masked-out {masked_probability_name} must be zero")
    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("lower_bounds must be strictly smaller than upper_bounds")

    for row_idx, active_count in enumerate(active_counts.tolist()):
        row_support = support[row_idx, :active_count]
        if active_count > 1 and np.any(np.diff(row_support) <= 0):
            raise ValueError("active support points must be strictly increasing")
        if support_domain == "probability":
            if np.any(row_support < -1e-9) or np.any(row_support > 1.0 + 1e-9):
                raise ValueError("probability support must lie within [0, 1]")
        elif support_domain == "rate" and np.any(row_support < -1e-9):
            raise ValueError("rate support must be non-negative")
        if bin_edges is None:
            continue
        row_edges = bin_edges[row_idx, : active_count + 1]
        if np.any(np.diff(row_edges) <= 0):
            raise ValueError("active bin_edges must be strictly increasing")
        if abs(float(row_edges[0]) - float(lower_bounds[row_idx])) > 1e-9:
            raise ValueError("lower_bounds must match the first active bin edge")
        if abs(float(row_edges[-1]) - float(upper_bounds[row_idx])) > 1e-9:
            raise ValueError("upper_bounds must match the last active bin edge")
    return active_counts


def _build_mixture_from_row(
    *,
    support_domain: SupportDomain,
    lower_bound: float,
    upper_bound: float,
    selected_k: int,
    component_weights: np.ndarray,
    component_means: np.ndarray,
    component_stds: np.ndarray,
    component_left: np.ndarray,
    component_right: np.ndarray,
) -> GaussianMixtureDistribution:
    components = tuple(
        GaussianComponent(
            weight=float(component_weights[component_idx]),
            mean=float(component_means[component_idx]),
            std=float(component_stds[component_idx]),
            left_truncation=float(component_left[component_idx]),
            right_truncation=float(component_right[component_idx]),
        )
        for component_idx in range(int(selected_k))
    )
    return GaussianMixtureDistribution(
        support_domain=support_domain,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        components=components,
    )


@dataclass(frozen=True, slots=True)
class GaussianComponent:
    weight: float
    mean: float
    std: float
    left_truncation: float
    right_truncation: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and non-negative")
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be finite and positive")
        if not np.isfinite(self.left_truncation) or not np.isfinite(
            self.right_truncation
        ):
            raise ValueError("truncation bounds must be finite")
        if self.right_truncation <= self.left_truncation:
            raise ValueError("right_truncation must exceed left_truncation")


@dataclass(frozen=True, slots=True)
class GaussianMixtureDistribution:
    support_domain: SupportDomain
    lower_bound: float
    upper_bound: float
    components: tuple[GaussianComponent, ...]

    def __post_init__(self) -> None:
        if self.support_domain not in {"probability", "rate"}:
            raise ValueError(f"unsupported support_domain: {self.support_domain!r}")
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
            raise ValueError("mixture bounds must be finite")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if not self.components:
            raise ValueError("components cannot be empty")
        total_weight = 0.0
        for component in self.components:
            total_weight += float(component.weight)
            if component.left_truncation < self.lower_bound - 1e-9:
                raise ValueError("component truncation must lie within mixture bounds")
            if component.right_truncation > self.upper_bound + 1e-9:
                raise ValueError("component truncation must lie within mixture bounds")
        if total_weight <= 0:
            raise ValueError("mixture weights must sum to a positive value")


@dataclass(frozen=True, slots=True)
class GMMSearchConfig:
    max_components: int = 8
    error_threshold: float = 1e-3
    residual_mass_threshold: float = 1e-4
    residual_peak_threshold: float = 1e-4
    merge_tolerance: float = 1e-12
    peak_min_value: float = 0.0
    peak_plateau_tolerance: float = 1e-12
    include_boundary_peaks: bool = True
    peak_limit_per_stage: int | None = None
    candidate_window_count: int = 6
    candidate_sigma_count: int = 8
    candidate_weight_slack: float = 0.98
    candidate_alpha_strategy: CandidateAlphaStrategy = "least_squares"
    candidate_alpha_cap_quantile: float = 0.1
    candidate_rerank_top_n: int | None = 4
    min_component_mass: float = 1e-4
    mass_floor: float = 1e-8
    min_sigma_factor: float = 0.5
    candidate_sigma_max_scale: float = 1.25
    boundary_mean_margin_scale: float = 0.5
    default_support_gap: float = 1.0
    bin_edge_midpoint_fraction: float = 0.5
    single_point_rate_half_width_scale: float = 0.5
    single_point_rate_half_width_floor: float = 1.0
    support_match_atol: float = 1e-9
    support_match_rtol: float = 1e-9
    selection_metric: RefitErrorMetric = "jsd"
    frontier_update_strategy: FrontierUpdateStrategy = "best_prefix"
    k_selection_mode: KSelectionMode = "threshold_first"
    k_min_improvement: float = 1e-4
    k_min_improvement_patience: int = 2
    k_penalty_weight: float = 0.0
    search_refit_enabled: bool = True
    search_refit_max_iterations: int = 25
    search_refit_min_iterations_first_component: int = 100
    search_refit_learning_rate: float = 0.05
    search_refit_convergence_tolerance: float = 1e-6
    search_refit_overshoot_penalty: float = 1.0
    search_refit_mean_margin_fraction: float = 0.5
    search_refit_compile_policy: CompilePolicy = "never"
    search_refit_compile_model: bool | None = None
    search_refit_optimize_weights: bool = True
    search_refit_optimize_means: bool = True
    search_refit_optimize_stds: bool = True
    search_refit_optimize_left_truncations: bool = True
    search_refit_optimize_right_truncations: bool = True
    search_refit_sigma_floor_fraction: float = 0.25
    search_refit_min_window_fraction: float = 1e-6
    search_refit_truncation_mode: TruncationMode = "free"
    search_refit_truncation_regularization_strength: float = 0.0
    search_refit_initial_weight_logit_floor: float = -12.0
    search_refit_inactive_weight_floor: float = 1e-12
    search_refit_masked_logit_value: float = -1e9
    search_refit_logit_clip: float = 1e-6
    search_refit_inverse_softplus_clip: float = 1e-8

    def __post_init__(self) -> None:
        if self.max_components < 1:
            raise ValueError("max_components must be >= 1")
        if self.error_threshold < 0:
            raise ValueError("error_threshold must be >= 0")
        if self.residual_mass_threshold < 0:
            raise ValueError("residual_mass_threshold must be >= 0")
        if self.residual_peak_threshold < 0:
            raise ValueError("residual_peak_threshold must be >= 0")
        if self.merge_tolerance < 0:
            raise ValueError("merge_tolerance must be >= 0")
        if self.peak_min_value < 0:
            raise ValueError("peak_min_value must be >= 0")
        if self.peak_plateau_tolerance < 0:
            raise ValueError("peak_plateau_tolerance must be >= 0")
        if self.peak_limit_per_stage is not None and self.peak_limit_per_stage < 1:
            raise ValueError("peak_limit_per_stage must be >= 1 when provided")
        if self.candidate_window_count < 1:
            raise ValueError("candidate_window_count must be >= 1")
        if self.candidate_sigma_count < 1:
            raise ValueError("candidate_sigma_count must be >= 1")
        if not (0.0 < self.candidate_weight_slack <= 1.0):
            raise ValueError("candidate_weight_slack must be in (0, 1]")
        if self.candidate_alpha_strategy not in {"min_ratio", "least_squares"}:
            raise ValueError(
                f"unsupported candidate_alpha_strategy: {self.candidate_alpha_strategy}"
            )
        if not (0.0 <= self.candidate_alpha_cap_quantile <= 1.0):
            raise ValueError("candidate_alpha_cap_quantile must be in [0, 1]")
        if self.candidate_rerank_top_n is not None and self.candidate_rerank_top_n < 1:
            raise ValueError("candidate_rerank_top_n must be >= 1 when provided")
        if self.min_component_mass < 0:
            raise ValueError("min_component_mass must be >= 0")
        if self.mass_floor <= 0:
            raise ValueError("mass_floor must be > 0")
        if self.min_sigma_factor <= 0:
            raise ValueError("min_sigma_factor must be > 0")
        if self.candidate_sigma_max_scale < 1.0:
            raise ValueError("candidate_sigma_max_scale must be >= 1")
        if self.boundary_mean_margin_scale < 0:
            raise ValueError("boundary_mean_margin_scale must be >= 0")
        if self.default_support_gap <= 0:
            raise ValueError("default_support_gap must be > 0")
        if not (0.0 < self.bin_edge_midpoint_fraction <= 1.0):
            raise ValueError("bin_edge_midpoint_fraction must be in (0, 1]")
        if self.single_point_rate_half_width_scale <= 0:
            raise ValueError("single_point_rate_half_width_scale must be > 0")
        if self.single_point_rate_half_width_floor <= 0:
            raise ValueError("single_point_rate_half_width_floor must be > 0")
        if self.support_match_atol < 0:
            raise ValueError("support_match_atol must be >= 0")
        if self.support_match_rtol < 0:
            raise ValueError("support_match_rtol must be >= 0")
        if self.selection_metric not in {"jsd", "l1", "cross_entropy"}:
            raise ValueError(f"unsupported selection_metric: {self.selection_metric}")
        if self.frontier_update_strategy not in {"full_stage", "best_prefix", "single_step"}:
            raise ValueError(
                f"unsupported frontier_update_strategy: {self.frontier_update_strategy}"
            )
        if self.k_selection_mode not in {"threshold_first", "marginal_gain", "penalized_error"}:
            raise ValueError(f"unsupported k_selection_mode: {self.k_selection_mode}")
        if self.k_min_improvement < 0:
            raise ValueError("k_min_improvement must be >= 0")
        if self.k_min_improvement_patience < 1:
            raise ValueError("k_min_improvement_patience must be >= 1")
        if self.k_penalty_weight < 0:
            raise ValueError("k_penalty_weight must be >= 0")
        if self.search_refit_max_iterations < 0:
            raise ValueError("search_refit_max_iterations must be >= 0")
        if self.search_refit_min_iterations_first_component < 0:
            raise ValueError(
                "search_refit_min_iterations_first_component must be >= 0"
            )
        if self.search_refit_learning_rate <= 0:
            raise ValueError("search_refit_learning_rate must be > 0")
        if self.search_refit_convergence_tolerance < 0:
            raise ValueError("search_refit_convergence_tolerance must be >= 0")
        if self.search_refit_overshoot_penalty < 0:
            raise ValueError("search_refit_overshoot_penalty must be >= 0")
        if self.search_refit_mean_margin_fraction < 0:
            raise ValueError("search_refit_mean_margin_fraction must be >= 0")
        if self.search_refit_sigma_floor_fraction <= 0:
            raise ValueError("search_refit_sigma_floor_fraction must be > 0")
        if self.search_refit_min_window_fraction <= 0:
            raise ValueError("search_refit_min_window_fraction must be > 0")
        if self.search_refit_truncation_mode not in {"free", "fixed_bounds"}:
            raise ValueError(
                f"unsupported search_refit_truncation_mode: {self.search_refit_truncation_mode}"
            )
        if self.search_refit_truncation_regularization_strength < 0:
            raise ValueError(
                "search_refit_truncation_regularization_strength must be >= 0"
            )
        if self.search_refit_inactive_weight_floor <= 0:
            raise ValueError("search_refit_inactive_weight_floor must be > 0")
        if (
            self.search_refit_logit_clip <= 0
            or self.search_refit_logit_clip >= 0.5
        ):
            raise ValueError("search_refit_logit_clip must be in (0, 0.5)")
        if self.search_refit_inverse_softplus_clip <= 0:
            raise ValueError("search_refit_inverse_softplus_clip must be > 0")
        resolved_compile_policy, resolved_compile_model = _resolve_compile_policy_compat(
            compile_policy=self.search_refit_compile_policy,
            compile_model=self.search_refit_compile_model,
            compile_policy_name="search_refit_compile_policy",
            compile_model_name="search_refit_compile_model",
        )
        object.__setattr__(
            self,
            "search_refit_compile_policy",
            resolved_compile_policy,
        )
        object.__setattr__(
            self,
            "search_refit_compile_model",
            resolved_compile_model,
        )


@dataclass(frozen=True, slots=True)
class GMMTrainingConfig:
    max_iterations: int = 300
    learning_rate: float = 0.05
    convergence_tolerance: float = 1e-6
    overshoot_penalty: float = 1.0
    mean_margin_fraction: float = 0.5
    gene_chunk_size: int = 256
    torch_dtype: Literal["float64", "float32"] = "float64"
    compile_policy: CompilePolicy = "never"
    compile_model: bool | None = None
    optimize_weights: bool = True
    optimize_means: bool = True
    optimize_stds: bool = True
    optimize_left_truncations: bool = True
    optimize_right_truncations: bool = True
    sigma_floor_fraction: float = 0.25
    min_window_fraction: float = 1e-6
    truncation_mode: TruncationMode = "free"
    truncation_regularization_strength: float = 0.0
    initial_weight_logit_floor: float = -12.0
    inactive_weight_floor: float = 1e-12
    masked_logit_value: float = -1e9
    logit_clip: float = 1e-6
    inverse_softplus_clip: float = 1e-8
    multi_start_count: int = 1
    multi_start_trigger_threshold: float = float("inf")
    multi_start_jitter_scale: float = 0.1
    multi_start_metric: RefitErrorMetric = "jsd"
    multi_start_seed: int = 0
    pruning_enabled: bool = False
    pruning_error_metric: RefitErrorMetric = "jsd"
    pruning_error_threshold: float = 1e-3
    pruning_max_refits: int = 0
    pruning_min_components: int = 1
    pruning_significance_metric: RefitPruningMetric = "weight"

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.convergence_tolerance < 0:
            raise ValueError("convergence_tolerance must be >= 0")
        if self.overshoot_penalty < 0:
            raise ValueError("overshoot_penalty must be >= 0")
        if self.mean_margin_fraction < 0:
            raise ValueError("mean_margin_fraction must be >= 0")
        if self.gene_chunk_size < 1:
            raise ValueError("gene_chunk_size must be >= 1")
        if self.torch_dtype not in {"float64", "float32"}:
            raise ValueError(f"unsupported torch_dtype: {self.torch_dtype}")
        if self.sigma_floor_fraction <= 0:
            raise ValueError("sigma_floor_fraction must be > 0")
        if self.min_window_fraction <= 0:
            raise ValueError("min_window_fraction must be > 0")
        if self.truncation_mode not in {"free", "fixed_bounds"}:
            raise ValueError(f"unsupported truncation_mode: {self.truncation_mode}")
        if self.truncation_regularization_strength < 0:
            raise ValueError("truncation_regularization_strength must be >= 0")
        if self.inactive_weight_floor <= 0:
            raise ValueError("inactive_weight_floor must be > 0")
        if self.logit_clip <= 0 or self.logit_clip >= 0.5:
            raise ValueError("logit_clip must be in (0, 0.5)")
        if self.inverse_softplus_clip <= 0:
            raise ValueError("inverse_softplus_clip must be > 0")
        if self.multi_start_count < 1:
            raise ValueError("multi_start_count must be >= 1")
        if self.multi_start_trigger_threshold < 0:
            raise ValueError("multi_start_trigger_threshold must be >= 0")
        if self.multi_start_jitter_scale < 0:
            raise ValueError("multi_start_jitter_scale must be >= 0")
        if self.multi_start_metric not in {"jsd", "l1", "cross_entropy"}:
            raise ValueError(f"unsupported multi_start_metric: {self.multi_start_metric}")
        if self.multi_start_seed < 0:
            raise ValueError("multi_start_seed must be >= 0")
        if self.pruning_error_metric not in {"jsd", "l1", "cross_entropy"}:
            raise ValueError(
                f"unsupported pruning_error_metric: {self.pruning_error_metric}"
            )
        if self.pruning_error_threshold < 0:
            raise ValueError("pruning_error_threshold must be >= 0")
        if self.pruning_max_refits < 0:
            raise ValueError("pruning_max_refits must be >= 0")
        if self.pruning_min_components < 1:
            raise ValueError("pruning_min_components must be >= 1")
        if self.pruning_significance_metric not in {"weight", "peak_mass"}:
            raise ValueError(
                "unsupported pruning_significance_metric: "
                f"{self.pruning_significance_metric}"
            )
        resolved_compile_policy, resolved_compile_model = _resolve_compile_policy_compat(
            compile_policy=self.compile_policy,
            compile_model=self.compile_model,
            compile_policy_name="compile_policy",
            compile_model_name="compile_model",
        )
        object.__setattr__(self, "compile_policy", resolved_compile_policy)
        object.__setattr__(self, "compile_model", resolved_compile_model)


def _validate_component_state(
    *,
    support: np.ndarray,
    probabilities: np.ndarray,
    support_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    selected_k: np.ndarray,
    component_weights: np.ndarray,
    component_means: np.ndarray,
    component_stds: np.ndarray,
    component_left: np.ndarray,
    component_right: np.ndarray,
) -> None:
    if support.shape != support_mask.shape:
        raise ValueError("support and support_mask must match")
    if support.shape != probabilities.shape:
        raise ValueError("support and probabilities must match")
    if lower_bounds.shape[0] != support.shape[0] or upper_bounds.shape[0] != support.shape[0]:
        raise ValueError("bound vectors must match the batch size")
    if selected_k.shape[0] != support.shape[0]:
        raise ValueError("selected_k must match the batch size")
    if component_weights.shape != component_means.shape:
        raise ValueError("component_weights and component_means must match")
    if component_weights.shape != component_stds.shape:
        raise ValueError("component_weights and component_stds must match")
    if component_weights.shape != component_left.shape:
        raise ValueError(
            "component_weights and component_left_truncations must match"
        )
    if component_weights.shape != component_right.shape:
        raise ValueError(
            "component_weights and component_right_truncations must match"
        )
    if component_weights.shape[0] != support.shape[0]:
        raise ValueError("component arrays must match the batch size")
    if np.any(selected_k < 1) or np.any(selected_k > component_weights.shape[1]):
        raise ValueError("selected_k must lie within the component axis")
    if np.any(probabilities < -1e-12):
        raise ValueError("probabilities must be non-negative")
    if np.any(np.abs(probabilities.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("probabilities must sum to 1 across support points")
    if np.any(component_weights < -1e-12):
        raise ValueError("component_weights must be non-negative")
    for row_idx, k in enumerate(selected_k.tolist()):
        if abs(float(component_weights[row_idx, :k].sum()) - 1.0) > 1e-6:
            raise ValueError("active component weights must sum to 1")
        if np.any(component_weights[row_idx, k:] > 1e-9):
            raise ValueError("inactive component weights must be zero")
        if np.any(component_stds[row_idx, :k] <= 0):
            raise ValueError("active component stds must be positive")
        if np.any(component_right[row_idx, :k] <= component_left[row_idx, :k]):
            raise ValueError("component truncation bounds must be ordered")
        if np.any(component_left[row_idx, :k] < lower_bounds[row_idx] - 1e-9):
            raise ValueError("component truncations must lie within lower_bounds")
        if np.any(component_right[row_idx, :k] > upper_bounds[row_idx] + 1e-9):
            raise ValueError("component truncations must lie within upper_bounds")


@dataclass(frozen=True, slots=True)
class DistributionGMMSearch:
    support_domain: SupportDomain
    support: np.ndarray
    support_mask: np.ndarray
    probabilities: np.ndarray
    bin_edges: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    selected_k: np.ndarray
    component_weights: np.ndarray
    component_means: np.ndarray
    component_stds: np.ndarray
    component_left_truncations: np.ndarray
    component_right_truncations: np.ndarray
    greedy_probabilities: np.ndarray
    error_path: np.ndarray
    residual_mass_path: np.ndarray
    residual_peak_path: np.ndarray
    explored_k: np.ndarray
    config: Mapping[str, object]

    def __post_init__(self) -> None:
        support = _as_matrix(self.support, name="support")
        support_mask = _as_bool_matrix(self.support_mask, name="support_mask")
        probabilities = _as_matrix(self.probabilities, name="probabilities")
        bin_edges = _as_matrix(self.bin_edges, name="bin_edges")
        lower_bounds = _as_vector(self.lower_bounds, name="lower_bounds")
        upper_bounds = _as_vector(self.upper_bounds, name="upper_bounds")
        selected_k = _as_index_vector(self.selected_k, name="selected_k")
        component_weights = _as_matrix(
            self.component_weights,
            name="component_weights",
        )
        component_means = _as_matrix(self.component_means, name="component_means")
        component_stds = _as_matrix(self.component_stds, name="component_stds")
        component_left = _as_matrix(
            self.component_left_truncations,
            name="component_left_truncations",
        )
        component_right = _as_matrix(
            self.component_right_truncations,
            name="component_right_truncations",
        )
        greedy_probabilities = _as_matrix(
            self.greedy_probabilities,
            name="greedy_probabilities",
        )
        error_path = _as_matrix(self.error_path, name="error_path")
        residual_mass_path = _as_matrix(
            self.residual_mass_path,
            name="residual_mass_path",
        )
        residual_peak_path = _as_matrix(
            self.residual_peak_path,
            name="residual_peak_path",
        )
        explored_k = _as_index_vector(self.explored_k, name="explored_k")
        if self.support_domain not in {"probability", "rate"}:
            raise ValueError(f"unsupported support_domain: {self.support_domain!r}")
        if bin_edges.shape != (support.shape[0], support.shape[1] + 1):
            raise ValueError("bin_edges must have one more column than support")
        if greedy_probabilities.shape != support.shape:
            raise ValueError("greedy_probabilities must match support")
        if error_path.shape != component_weights.shape:
            raise ValueError("error_path must match component arrays")
        if residual_mass_path.shape != component_weights.shape:
            raise ValueError("residual_mass_path must match component arrays")
        if residual_peak_path.shape != component_weights.shape:
            raise ValueError("residual_peak_path must match component arrays")
        if explored_k.shape[0] != support.shape[0]:
            raise ValueError("explored_k must match the batch size")
        if np.any(explored_k < 1) or np.any(explored_k > component_weights.shape[1]):
            raise ValueError("explored_k must lie within the component axis")
        _validate_support_grid(
            support=support,
            support_mask=support_mask,
            probabilities=probabilities,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            bin_edges=bin_edges,
            support_domain=self.support_domain,
        )
        if np.any(np.abs(greedy_probabilities.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError("greedy_probabilities must sum to 1 across support points")
        if np.any(np.abs(np.where(support_mask, 0.0, greedy_probabilities)) > 1e-9):
            raise ValueError("masked-out greedy_probabilities must be zero")
        _validate_component_state(
            support=support,
            probabilities=probabilities,
            support_mask=support_mask,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            selected_k=selected_k,
            component_weights=component_weights,
            component_means=component_means,
            component_stds=component_stds,
            component_left=component_left,
            component_right=component_right,
        )
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "bin_edges", bin_edges)
        object.__setattr__(self, "lower_bounds", lower_bounds)
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "selected_k", selected_k)
        object.__setattr__(self, "component_weights", component_weights)
        object.__setattr__(self, "component_means", component_means)
        object.__setattr__(self, "component_stds", component_stds)
        object.__setattr__(self, "component_left_truncations", component_left)
        object.__setattr__(self, "component_right_truncations", component_right)
        object.__setattr__(self, "greedy_probabilities", greedy_probabilities)
        object.__setattr__(self, "error_path", error_path)
        object.__setattr__(self, "residual_mass_path", residual_mass_path)
        object.__setattr__(self, "residual_peak_path", residual_peak_path)
        object.__setattr__(self, "explored_k", explored_k)
        object.__setattr__(self, "config", _freeze_config_mapping(self.config))

    def to_mixture(self, index: int = 0) -> GaussianMixtureDistribution:
        row_idx = int(index)
        if row_idx < 0 or row_idx >= self.support.shape[0]:
            raise IndexError(f"index out of range: {index!r}")
        return _build_mixture_from_row(
            support_domain=self.support_domain,
            lower_bound=float(self.lower_bounds[row_idx]),
            upper_bound=float(self.upper_bounds[row_idx]),
            selected_k=int(self.selected_k[row_idx]),
            component_weights=self.component_weights[row_idx],
            component_means=self.component_means[row_idx],
            component_stds=self.component_stds[row_idx],
            component_left=self.component_left_truncations[row_idx],
            component_right=self.component_right_truncations[row_idx],
        )


@dataclass(frozen=True, slots=True)
class DistributionGMMReport:
    support_domain: SupportDomain
    support: np.ndarray
    support_mask: np.ndarray
    probabilities: np.ndarray
    bin_edges: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    selected_k: np.ndarray
    fitted_probabilities: np.ndarray
    residual_probabilities: np.ndarray
    component_weights: np.ndarray
    component_means: np.ndarray
    component_stds: np.ndarray
    component_left_truncations: np.ndarray
    component_right_truncations: np.ndarray
    greedy_probabilities: np.ndarray
    jsd: np.ndarray
    cross_entropy: np.ndarray
    l1_error: np.ndarray
    greedy_error: np.ndarray
    explored_k: np.ndarray
    config: Mapping[str, object]

    def __post_init__(self) -> None:
        support = _as_matrix(self.support, name="support")
        support_mask = _as_bool_matrix(self.support_mask, name="support_mask")
        probabilities = _as_matrix(self.probabilities, name="probabilities")
        bin_edges = _as_matrix(self.bin_edges, name="bin_edges")
        lower_bounds = _as_vector(self.lower_bounds, name="lower_bounds")
        upper_bounds = _as_vector(self.upper_bounds, name="upper_bounds")
        selected_k = _as_index_vector(self.selected_k, name="selected_k")
        component_weights = _as_matrix(
            self.component_weights,
            name="component_weights",
        )
        component_means = _as_matrix(self.component_means, name="component_means")
        component_stds = _as_matrix(self.component_stds, name="component_stds")
        component_left = _as_matrix(
            self.component_left_truncations,
            name="component_left_truncations",
        )
        component_right = _as_matrix(
            self.component_right_truncations,
            name="component_right_truncations",
        )
        greedy_probabilities = _as_matrix(
            self.greedy_probabilities,
            name="greedy_probabilities",
        )
        explored_k = _as_index_vector(self.explored_k, name="explored_k")
        fitted = _as_matrix(self.fitted_probabilities, name="fitted_probabilities")
        residual = _as_matrix(
            self.residual_probabilities,
            name="residual_probabilities",
        )
        jsd = _as_vector(self.jsd, name="jsd")
        cross_entropy = _as_vector(self.cross_entropy, name="cross_entropy")
        l1_error = _as_vector(self.l1_error, name="l1_error")
        greedy_error = _as_vector(self.greedy_error, name="greedy_error")
        if self.support_domain not in {"probability", "rate"}:
            raise ValueError(f"unsupported support_domain: {self.support_domain!r}")
        if bin_edges.shape != (support.shape[0], support.shape[1] + 1):
            raise ValueError("bin_edges must have one more column than support")
        _validate_support_grid(
            support=support,
            support_mask=support_mask,
            probabilities=probabilities,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            bin_edges=bin_edges,
            support_domain=self.support_domain,
        )
        _validate_component_state(
            support=support,
            probabilities=probabilities,
            support_mask=support_mask,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            selected_k=selected_k,
            component_weights=component_weights,
            component_means=component_means,
            component_stds=component_stds,
            component_left=component_left,
            component_right=component_right,
        )
        if greedy_probabilities.shape != support.shape:
            raise ValueError("greedy_probabilities must match support")
        if explored_k.shape[0] != support.shape[0]:
            raise ValueError("explored_k must match the batch size")
        if fitted.shape != support.shape:
            raise ValueError("fitted_probabilities must match support")
        if residual.shape != support.shape:
            raise ValueError("residual_probabilities must match support")
        if jsd.shape[0] != support.shape[0]:
            raise ValueError("jsd must match the batch size")
        if cross_entropy.shape[0] != support.shape[0]:
            raise ValueError("cross_entropy must match the batch size")
        if l1_error.shape[0] != support.shape[0]:
            raise ValueError("l1_error must match the batch size")
        if greedy_error.shape[0] != support.shape[0]:
            raise ValueError("greedy_error must match the batch size")
        if np.any(np.abs(fitted.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError("fitted_probabilities must sum to 1 across support points")
        if np.any(np.abs(greedy_probabilities.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError("greedy_probabilities must sum to 1 across support points")
        if np.any(np.abs(np.where(support_mask, 0.0, fitted)) > 1e-9):
            raise ValueError("masked-out fitted_probabilities must be zero")
        if np.any(np.abs(np.where(support_mask, 0.0, residual)) > 1e-9):
            raise ValueError("masked-out residual_probabilities must be zero")
        if np.any(np.abs(np.where(support_mask, 0.0, greedy_probabilities)) > 1e-9):
            raise ValueError("masked-out greedy_probabilities must be zero")
        if np.any(np.abs((fitted + residual) - probabilities) > 1e-6):
            raise ValueError(
                "fitted_probabilities plus residual_probabilities must match probabilities"
            )
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "bin_edges", bin_edges)
        object.__setattr__(self, "lower_bounds", lower_bounds)
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "selected_k", selected_k)
        object.__setattr__(self, "component_weights", component_weights)
        object.__setattr__(self, "component_means", component_means)
        object.__setattr__(self, "component_stds", component_stds)
        object.__setattr__(self, "component_left_truncations", component_left)
        object.__setattr__(self, "component_right_truncations", component_right)
        object.__setattr__(self, "greedy_probabilities", greedy_probabilities)
        object.__setattr__(self, "explored_k", explored_k)
        object.__setattr__(self, "fitted_probabilities", fitted)
        object.__setattr__(self, "residual_probabilities", residual)
        object.__setattr__(self, "jsd", jsd)
        object.__setattr__(self, "cross_entropy", cross_entropy)
        object.__setattr__(self, "l1_error", l1_error)
        object.__setattr__(self, "greedy_error", greedy_error)
        object.__setattr__(self, "config", _freeze_config_mapping(self.config))

    def to_mixture(self, index: int = 0) -> GaussianMixtureDistribution:
        row_idx = int(index)
        if row_idx < 0 or row_idx >= self.support.shape[0]:
            raise IndexError(f"index out of range: {index!r}")
        return _build_mixture_from_row(
            support_domain=self.support_domain,
            lower_bound=float(self.lower_bounds[row_idx]),
            upper_bound=float(self.upper_bounds[row_idx]),
            selected_k=int(self.selected_k[row_idx]),
            component_weights=self.component_weights[row_idx],
            component_means=self.component_means[row_idx],
            component_stds=self.component_stds[row_idx],
            component_left=self.component_left_truncations[row_idx],
            component_right=self.component_right_truncations[row_idx],
        )


@dataclass(frozen=True, slots=True)
class PriorGMMReport:
    gene_names: list[str]
    support_domain: SupportDomain
    support: np.ndarray
    scaled_support: np.ndarray
    support_mask: np.ndarray
    probabilities: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    selected_k: np.ndarray
    fitted_probabilities: np.ndarray
    residual_probabilities: np.ndarray
    component_weights: np.ndarray
    component_means: np.ndarray
    component_stds: np.ndarray
    component_left_truncations: np.ndarray
    component_right_truncations: np.ndarray
    jsd: np.ndarray
    cross_entropy: np.ndarray
    l1_error: np.ndarray
    explored_k: np.ndarray
    scale: float
    config: Mapping[str, object]

    def __post_init__(self) -> None:
        names = _normalize_gene_names(list(self.gene_names))
        support = _as_matrix(self.support, name="support")
        scaled_support = _as_matrix(self.scaled_support, name="scaled_support")
        support_mask = _as_bool_matrix(self.support_mask, name="support_mask")
        probabilities = _as_matrix(self.probabilities, name="probabilities")
        lower_bounds = _as_vector(self.lower_bounds, name="lower_bounds")
        upper_bounds = _as_vector(self.upper_bounds, name="upper_bounds")
        selected_k = _as_index_vector(self.selected_k, name="selected_k")
        fitted = _as_matrix(self.fitted_probabilities, name="fitted_probabilities")
        residual = _as_matrix(
            self.residual_probabilities,
            name="residual_probabilities",
        )
        component_weights = _as_matrix(
            self.component_weights,
            name="component_weights",
        )
        component_means = _as_matrix(self.component_means, name="component_means")
        component_stds = _as_matrix(self.component_stds, name="component_stds")
        component_left = _as_matrix(
            self.component_left_truncations,
            name="component_left_truncations",
        )
        component_right = _as_matrix(
            self.component_right_truncations,
            name="component_right_truncations",
        )
        jsd = _as_vector(self.jsd, name="jsd")
        cross_entropy = _as_vector(self.cross_entropy, name="cross_entropy")
        l1_error = _as_vector(self.l1_error, name="l1_error")
        explored_k = _as_index_vector(self.explored_k, name="explored_k")
        if self.support_domain not in {"probability", "rate"}:
            raise ValueError(f"unsupported support_domain: {self.support_domain!r}")
        if support.shape != scaled_support.shape:
            raise ValueError("support and scaled_support must match")
        if support.shape != support_mask.shape:
            raise ValueError("support and support_mask must match")
        if support.shape != probabilities.shape:
            raise ValueError("support and probabilities must match")
        if support.shape != fitted.shape:
            raise ValueError("support and fitted_probabilities must match")
        if support.shape != residual.shape:
            raise ValueError("support and residual_probabilities must match")
        if support.shape[0] != len(names):
            raise ValueError("gene_names must match the batch size")
        if explored_k.shape[0] != len(names):
            raise ValueError("explored_k must match the batch size")
        _validate_support_grid(
            support=support,
            support_mask=support_mask,
            probabilities=probabilities,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            support_domain=self.support_domain,
        )
        _validate_component_state(
            support=support,
            probabilities=probabilities,
            support_mask=support_mask,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            selected_k=selected_k,
            component_weights=component_weights,
            component_means=component_means,
            component_stds=component_stds,
            component_left=component_left,
            component_right=component_right,
        )
        if jsd.shape[0] != len(names):
            raise ValueError("jsd must match the batch size")
        if cross_entropy.shape[0] != len(names):
            raise ValueError("cross_entropy must match the batch size")
        if l1_error.shape[0] != len(names):
            raise ValueError("l1_error must match the batch size")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if np.any(np.abs(fitted.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError("fitted_probabilities must sum to 1 across support points")
        if np.any(np.abs(np.where(support_mask, 0.0, fitted)) > 1e-9):
            raise ValueError("masked-out fitted_probabilities must be zero")
        if np.any(np.abs(np.where(support_mask, 0.0, residual)) > 1e-9):
            raise ValueError("masked-out residual_probabilities must be zero")
        if np.any(np.abs((fitted + residual) - probabilities) > 1e-6):
            raise ValueError(
                "fitted_probabilities plus residual_probabilities must match probabilities"
            )
        object.__setattr__(self, "gene_names", names)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "scaled_support", scaled_support)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "lower_bounds", lower_bounds)
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "selected_k", selected_k)
        object.__setattr__(self, "fitted_probabilities", fitted)
        object.__setattr__(self, "residual_probabilities", residual)
        object.__setattr__(self, "component_weights", component_weights)
        object.__setattr__(self, "component_means", component_means)
        object.__setattr__(self, "component_stds", component_stds)
        object.__setattr__(self, "component_left_truncations", component_left)
        object.__setattr__(self, "component_right_truncations", component_right)
        object.__setattr__(self, "jsd", jsd)
        object.__setattr__(self, "cross_entropy", cross_entropy)
        object.__setattr__(self, "l1_error", l1_error)
        object.__setattr__(self, "explored_k", explored_k)
        object.__setattr__(self, "config", _freeze_config_mapping(self.config))

    def to_mixture(self, gene_name: str) -> GaussianMixtureDistribution:
        try:
            row_idx = self.gene_names.index(str(gene_name))
        except ValueError as exc:
            raise KeyError(f"unknown gene_name: {gene_name!r}") from exc
        return _build_mixture_from_row(
            support_domain=self.support_domain,
            lower_bound=float(self.lower_bounds[row_idx]),
            upper_bound=float(self.upper_bounds[row_idx]),
            selected_k=int(self.selected_k[row_idx]),
            component_weights=self.component_weights[row_idx],
            component_means=self.component_means[row_idx],
            component_stds=self.component_stds[row_idx],
            component_left=self.component_left_truncations[row_idx],
            component_right=self.component_right_truncations[row_idx],
        )


__all__ = [
    "CompilePolicy",
    "DistributionGMMReport",
    "DistributionGMMSearch",
    "FrontierUpdateStrategy",
    "GaussianComponent",
    "GaussianMixtureDistribution",
    "GMMSearchConfig",
    "GMMTrainingConfig",
    "CandidateAlphaStrategy",
    "KSelectionMode",
    "PriorGMMReport",
    "RefitErrorMetric",
    "RefitPruningMetric",
    "SupportAxis",
    "TruncationMode",
]

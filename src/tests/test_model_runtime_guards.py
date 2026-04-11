from __future__ import annotations

import numpy as np
import pytest
import torch

import prism.model as model_api
from prism.model import (
    InferenceResult,
    ObservationBatch,
    PriorEngineSetting,
    PriorEngineTrainingConfig,
    PriorFitConfig,
    PriorGrid,
    fit_gene_priors,
    infer_kbulk_samples,
    infer_posteriors,
    make_distribution_grid,
)
from prism.model.checkpoint import checkpoint_from_fit_result
from prism.model import fit as fit_module
from prism.model import infer as infer_module
from prism.model.numeric import posterior_from_log_likelihood


def test_extract_common_reexports_resolve_prior_source() -> None:
    from prism.cli.extract.common import resolve_prior_source

    assert resolve_prior_source("global") == "global"
    assert resolve_prior_source("label") == "label"


def test_posterior_from_log_likelihood_rejects_zero_mass_rows() -> None:
    log_likelihood = torch.full((1, 2, 4), float("-inf"), dtype=torch.float64)
    prior_probabilities = torch.full((1, 4), 0.25, dtype=torch.float64)
    with pytest.raises(ValueError, match="zero likelihood"):
        posterior_from_log_likelihood(log_likelihood, prior_probabilities)


def test_observation_batch_normalizes_python_lists() -> None:
    batch = ObservationBatch(
        gene_names=["g1"],
        counts=[[1.0]],
        reference_counts=[2.0],
    )
    assert isinstance(batch.counts, np.ndarray)
    assert isinstance(batch.reference_counts, np.ndarray)
    assert batch.counts.shape == (1, 1)
    assert batch.reference_counts.shape == (1,)


def test_fit_gene_priors_rejects_binomial_counts_above_exposure() -> None:
    batch = ObservationBatch(
        gene_names=["g1"],
        counts=np.array([[5.0], [7.0]], dtype=np.float64),
        reference_counts=np.array([2.0, 2.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="counts <= effective_exposure"):
        fit_gene_priors(
            batch,
            scale=2.0,
            config=PriorFitConfig(
                n_support_points=8,
                max_em_iterations=2,
                likelihood="binomial",
            ),
            compile_model=False,
        )


def test_infer_posteriors_rejects_binomial_counts_above_exposure() -> None:
    batch = ObservationBatch(
        gene_names=["g1"],
        counts=np.array([[5.0], [7.0]], dtype=np.float64),
        reference_counts=np.array([2.0, 2.0], dtype=np.float64),
    )
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            "binomial",
            support=[0.1, 0.2, 0.3],
            probabilities=[0.2, 0.3, 0.5],
        ),
        scale=2.0,
    )
    with pytest.raises(ValueError, match="counts <= effective_exposure"):
        infer_posteriors(
            batch,
            prior,
            compile_model=False,
            posterior_distribution="binomial",
        )


def test_inference_result_validates_support_domain_semantics() -> None:
    with pytest.raises(ValueError, match="probability support must lie in \\[0, 1\\]"):
        InferenceResult(
            gene_names=["g1"],
            support_domain="probability",
            support=np.array([[1.5, 2.0]], dtype=np.float64),
            prior_probabilities=np.array([[0.5, 0.5]], dtype=np.float64),
            map_support=np.array([[1.5]], dtype=np.float64),
            posterior_entropy=np.array([[0.0]], dtype=np.float64),
            prior_entropy=np.array([[0.0]], dtype=np.float64),
            mutual_information=np.array([[0.0]], dtype=np.float64),
        )


def test_infer_kbulk_rejects_binomial_counts_above_exposure() -> None:
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            "binomial",
            support=[0.1, 0.2, 0.3],
            probabilities=[0.2, 0.3, 0.5],
        ),
        scale=2.0,
    )
    with pytest.raises(ValueError, match="counts <= effective_exposure"):
        infer_kbulk_samples(
            ["g1"],
            aggregated_counts=np.array([[5.0], [7.0]], dtype=np.float64),
            effective_exposure=np.array([2.0, 2.0], dtype=np.float64),
            prior=prior,
            posterior_distribution="binomial",
            compile_model=False,
        )


def test_infer_posteriors_chunking_matches_single_chunk() -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        ),
        reference_counts=np.array([12.0, 12.0, 12.0, 12.0], dtype=np.float64),
    )
    prior = PriorGrid(
        gene_names=["g1", "g2"],
        distribution=make_distribution_grid(
            "binomial",
            support=np.array(
                [
                    [0.05, 0.1, 0.2, 0.4],
                    [0.05, 0.1, 0.2, 0.4],
                ],
                dtype=np.float64,
            ),
            probabilities=np.array(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.4, 0.3, 0.2, 0.1],
                ],
                dtype=np.float64,
            ),
        ),
        scale=12.0,
    )

    single = infer_posteriors(
        batch,
        prior,
        include_posterior=True,
        compile_model=False,
    )
    chunked = infer_posteriors(
        batch,
        prior,
        include_posterior=True,
        compile_model=False,
        observation_chunk_size=2,
    )

    assert np.allclose(single.map_support, chunked.map_support)
    assert np.allclose(single.posterior_entropy, chunked.posterior_entropy)
    assert np.allclose(single.prior_entropy, chunked.prior_entropy)
    assert np.allclose(single.mutual_information, chunked.mutual_information)
    assert single.posterior_probabilities is not None
    assert chunked.posterior_probabilities is not None
    assert np.allclose(
        single.posterior_probabilities,
        chunked.posterior_probabilities,
    )


def test_infer_kbulk_chunking_matches_single_chunk() -> None:
    prior = PriorGrid(
        gene_names=["g1", "g2"],
        distribution=make_distribution_grid(
            "binomial",
            support=np.array(
                [
                    [0.05, 0.1, 0.2, 0.4],
                    [0.05, 0.1, 0.2, 0.4],
                ],
                dtype=np.float64,
            ),
            probabilities=np.array(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.4, 0.3, 0.2, 0.1],
                ],
                dtype=np.float64,
            ),
        ),
        scale=20.0,
    )

    single = infer_kbulk_samples(
        ["g1", "g2"],
        aggregated_counts=np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [4.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        ),
        effective_exposure=np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float64),
        prior=prior,
        include_posterior=True,
        compile_model=False,
    )
    chunked = infer_kbulk_samples(
        ["g1", "g2"],
        aggregated_counts=np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [4.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        ),
        effective_exposure=np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float64),
        prior=prior,
        include_posterior=True,
        compile_model=False,
        observation_chunk_size=2,
    )

    assert np.allclose(single.map_support, chunked.map_support)
    assert np.allclose(single.posterior_entropy, chunked.posterior_entropy)
    assert np.allclose(single.prior_entropy, chunked.prior_entropy)
    assert np.allclose(single.mutual_information, chunked.mutual_information)
    assert single.posterior_probabilities is not None
    assert chunked.posterior_probabilities is not None
    assert np.allclose(
        single.posterior_probabilities,
        chunked.posterior_probabilities,
    )


def test_prior_grid_select_genes_same_order_returns_self() -> None:
    prior = PriorGrid(
        gene_names=["g1", "g2"],
        distribution=make_distribution_grid(
            "binomial",
            support=np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.1, 0.2, 0.3],
                ],
                dtype=np.float64,
            ),
            probabilities=np.array(
                [
                    [0.2, 0.3, 0.5],
                    [0.2, 0.3, 0.5],
                ],
                dtype=np.float64,
            ),
        ),
        scale=1.0,
    )

    assert prior.select_genes(["g1", "g2"]) is prior


def test_inferencer_cache_key_distinguishes_compile_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer_module._COMPILED_INFERENCERS.clear()
    compiled_calls: list[torch.nn.Module] = []

    def fake_compile(module: torch.nn.Module) -> torch.nn.Module:
        compiled_calls.append(module)
        return torch.nn.Sequential(module)

    monkeypatch.setattr(infer_module, "_maybe_compile", fake_compile)

    plain = infer_module._get_inferencer(
        "binomial",
        device=torch.device("cpu"),
        dtype=torch.float64,
        nb_overdispersion=0.01,
        compile_model=False,
    )
    compiled = infer_module._get_inferencer(
        "binomial",
        device=torch.device("cpu"),
        dtype=torch.float64,
        nb_overdispersion=0.01,
        compile_model=True,
    )

    assert plain is not compiled
    assert compiled_calls


def test_em_step_cache_key_distinguishes_compile_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_module._COMPILED_EM_STEPS.clear()
    compiled_calls: list[torch.nn.Module] = []

    def fake_compile(module: torch.nn.Module) -> torch.nn.Module:
        compiled_calls.append(module)
        return torch.nn.Sequential(module)

    monkeypatch.setattr(fit_module, "_maybe_compile", fake_compile)

    plain = fit_module._get_em_step_module(
        "binomial",
        device=torch.device("cpu"),
        dtype=torch.float64,
        nb_overdispersion=0.01,
        compile_model=False,
    )
    compiled = fit_module._get_em_step_module(
        "binomial",
        device=torch.device("cpu"),
        dtype=torch.float64,
        nb_overdispersion=0.01,
        compile_model=True,
    )

    assert plain is not compiled
    assert compiled_calls


def test_log_likelihood_cache_avoids_rebuilding_observation_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 2.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        reference_counts=np.array([12.0, 12.0, 12.0, 12.0], dtype=np.float64),
    )
    build_calls = 0
    original = fit_module._build_observation_terms

    def wrapped_build_observation_terms(*args: object, **kwargs: object) -> object:
        nonlocal build_calls
        build_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit_module, "_build_observation_terms", wrapped_build_observation_terms)
    monkeypatch.setattr(fit_module, "_should_cache_log_likelihood", lambda **_: True)
    monkeypatch.setattr(fit_module, "_should_cache_observation_terms", lambda **_: False)

    fit_gene_priors(
        batch,
        scale=12.0,
        config=PriorFitConfig(
            n_support_points=8,
            max_em_iterations=3,
            cell_chunk_size=2,
            likelihood="binomial",
        ),
        compile_model=False,
    )

    assert build_calls == 2


def test_fit_gene_priors_caps_cell_chunk_size_to_n_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        reference_counts=np.array([12.0, 12.0, 12.0], dtype=np.float64),
    )
    seen: dict[str, int] = {}
    original = fit_module._iter_cell_slices

    def wrapped_iter_cell_slices(n_cells: int, chunk_size: int) -> list[slice]:
        seen["n_cells"] = n_cells
        seen["chunk_size"] = chunk_size
        return original(n_cells, chunk_size)

    monkeypatch.setattr(fit_module, "_iter_cell_slices", wrapped_iter_cell_slices)

    fit_gene_priors(
        batch,
        scale=12.0,
        config=PriorFitConfig(
            n_support_points=8,
            max_em_iterations=2,
            cell_chunk_size=4096,
            likelihood="binomial",
        ),
        compile_model=False,
    )

    assert seen == {"n_cells": 3, "chunk_size": 3}


def test_cache_device_key_keeps_cuda_indices_distinct() -> None:
    assert infer_module._cache_device_key(
        torch.device("cuda:0")
    ) != infer_module._cache_device_key(torch.device("cuda:1"))
    assert fit_module._cache_device_key(
        torch.device("cuda:0")
    ) != fit_module._cache_device_key(torch.device("cuda:1"))


def test_prior_engine_configs_validate_at_construction() -> None:
    with pytest.raises(ValueError, match="support_scale"):
        PriorEngineSetting(support_scale=0.5)
    with pytest.raises(ValueError, match="adaptive_support_scale"):
        PriorEngineSetting(use_adaptive_support=True, adaptive_support_scale=0.5)
    with pytest.raises(ValueError, match="unsupported torch_dtype"):
        PriorEngineTrainingConfig(torch_dtype="float16")  # type: ignore[arg-type]


def test_probability_support_max_uses_effective_exposure() -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array([[1.0, 4.0], [2.0, 2.0]], dtype=np.float64),
        reference_counts=np.array([10.0, 20.0], dtype=np.float64),
    )

    support_max = fit_module._default_probability_support_max(
        batch,
        scale=15.0,
        method="observed_max",
    )

    assert np.allclose(
        support_max,
        np.array([2.0 / 15.0, 0.4], dtype=np.float64),
    )


def test_scaled_support_max_keeps_raw_count_floor() -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array([[1.0, 4.0], [2.0, 2.0]], dtype=np.float64),
        reference_counts=np.array([10.0, 20.0], dtype=np.float64),
    )

    scaled_support_max = fit_module._default_scaled_support_max(
        batch,
        scale=15.0,
        method="observed_max",
    )

    assert np.allclose(
        scaled_support_max,
        np.array([2.0, 6.0], dtype=np.float64),
    )


def test_probability_support_grid_includes_zero() -> None:
    support = fit_module._build_probability_support(
        4,
        np.array([0.6], dtype=np.float64),
        dtype=torch.float64,
        device=torch.device("cpu"),
        spacing="linear",
    )

    assert np.allclose(
        support.detach().cpu().numpy(),
        np.array([[0.0, 0.2, 0.4, 0.6]], dtype=np.float64),
    )


def test_support_scale_expands_probability_support_upper_bound() -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array([[0.0, 0.0]], dtype=np.float64),
        reference_counts=np.array([1.0], dtype=np.float64),
    )

    support = fit_module._build_support(
        batch,
        scale=1.0,
        config=PriorFitConfig(
            n_support_points=4,
            support_scale=1.5,
            likelihood="binomial",
        ),
        dtype=torch.float64,
        device=torch.device("cpu"),
        support_max=np.array([0.6, 0.8], dtype=np.float64),
    )

    assert np.allclose(
        support.detach().cpu().numpy(),
        np.array(
            [
                [0.0, 0.3, 0.6, 0.9],
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def test_adaptive_refine_support_keeps_zero_and_scales_upper_bound() -> None:
    support = torch.as_tensor(
        [[0.0, 0.15, 0.3, 0.45, 0.6]],
        dtype=torch.float64,
    )
    probabilities = np.array([[0.6, 0.2, 0.1, 0.1, 0.0]], dtype=np.float64)

    refined = fit_module._adaptive_refine_support(
        support,
        probabilities,
        config=PriorFitConfig(
            use_adaptive_support=True,
            adaptive_support_scale=2.0,
            adaptive_support_quantile=0.75,
        ),
        dtype=torch.float64,
        device=torch.device("cpu"),
        upper_bound=1.0,
    )

    refined_np = refined.detach().cpu().numpy()
    assert refined_np[0, 0] == 0.0
    assert np.isclose(refined_np[0, -1], 0.3)


def test_adaptive_refine_support_expands_when_right_edge_is_saturated() -> None:
    support = torch.as_tensor(
        [[0.0, 0.05, 0.1, 0.15, 0.2]],
        dtype=torch.float64,
    )
    probabilities = np.array([[0.01, 0.01, 0.03, 0.05, 0.9]], dtype=np.float64)

    refined = fit_module._adaptive_refine_support(
        support,
        probabilities,
        config=PriorFitConfig(
            use_adaptive_support=True,
            adaptive_support_scale=1.5,
            adaptive_support_quantile=0.995,
        ),
        dtype=torch.float64,
        device=torch.device("cpu"),
        upper_bound=1.0,
    )

    refined_np = refined.detach().cpu().numpy()
    assert refined_np[0, 0] == 0.0
    assert np.isclose(refined_np[0, -1], 0.3)


def test_interpolate_probabilities_to_support_accepts_torch_inputs() -> None:
    projected = fit_module._interpolate_probabilities_to_support(
        torch.as_tensor([[0.0, 0.5, 1.0]], dtype=torch.float64),
        torch.as_tensor([[0.2, 0.3, 0.5]], dtype=torch.float64),
        torch.as_tensor([[0.0, 0.25, 0.5, 0.75]], dtype=torch.float64),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert np.allclose(
        projected.detach().cpu().numpy(),
        np.array([[0.17391304, 0.2173913, 0.26086957, 0.34782609]], dtype=np.float64),
    )


def test_fit_gene_priors_adaptive_support_runs_end_to_end() -> None:
    batch = ObservationBatch(
        gene_names=["g1", "g2"],
        counts=np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        reference_counts=np.array([12.0, 12.0, 12.0, 12.0], dtype=np.float64),
    )

    result = fit_gene_priors(
        batch,
        scale=12.0,
        config=PriorFitConfig(
            n_support_points=8,
            max_em_iterations=3,
            cell_chunk_size=2,
            likelihood="binomial",
            use_adaptive_support=True,
            adaptive_support_scale=1.5,
            adaptive_support_quantile=0.9,
        ),
        compile_model=False,
    )

    assert result.posterior_mean_probabilities.shape == (2, 8)
    assert np.allclose(result.posterior_mean_probabilities.sum(axis=-1), 1.0)
    assert result.prior.support.shape == (2, 8)
    diagnostics = result.config["support_diagnostics"]
    assert diagnostics["phase1"]["n_genes"] == 2
    assert diagnostics["final"]["n_genes"] == 2
    assert diagnostics["adaptive"]["enabled"] is True
    assert diagnostics["adaptive"]["rounds_run"] >= 1


def test_checkpoint_from_fit_result_sets_current_distribution_metadata() -> None:
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            "binomial",
            support=[0.1, 0.2, 0.3],
            probabilities=[0.2, 0.3, 0.5],
        ),
        scale=1.0,
    )
    result = fit_module.PriorFitResult(
        gene_names=["g1"],
        prior=prior,
        posterior_mean_probabilities=np.array([0.2, 0.3, 0.5], dtype=np.float64),
        objective_history=[1.0],
        final_objective=1.0,
        config={"mean_reference_count": 2.0},
    )
    checkpoint = checkpoint_from_fit_result(result)
    assert checkpoint.metadata["fit_distribution"] == "binomial"
    assert checkpoint.metadata["posterior_distribution"] == "binomial"
    assert checkpoint.metadata["support_domain"] == "probability"


def test_checkpoint_from_fit_result_carries_support_diagnostics() -> None:
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            "binomial",
            support=[0.1, 0.2, 0.3],
            probabilities=[0.2, 0.3, 0.5],
        ),
        scale=1.0,
    )
    result = fit_module.PriorFitResult(
        gene_names=["g1"],
        prior=prior,
        posterior_mean_probabilities=np.array([0.2, 0.3, 0.5], dtype=np.float64),
        objective_history=[1.0],
        final_objective=1.0,
        config={
            "mean_reference_count": 2.0,
            "support_diagnostics": {
                "phase1": {"n_genes": 1, "n_map_at_right_edge": 1},
                "final": {"n_genes": 1, "n_map_at_right_edge": 0},
                "adaptive": {"enabled": True, "rounds_run": 1},
            },
        },
    )

    checkpoint = checkpoint_from_fit_result(result)

    assert checkpoint.metadata["support_diagnostics"]["phase1"]["n_genes"] == 1
    assert checkpoint.metadata["support_diagnostics"]["adaptive"]["rounds_run"] == 1


def test_public_model_api_removes_legacy_aliases() -> None:
    for name in (
        "PriorFitter",
        "SignalExtractor",
        "KBulkResult",
        "PoolFitReport",
        "fit_pool_scale_report",
        "fit_gene_priors_em",
    ):
        assert not hasattr(model_api, name)

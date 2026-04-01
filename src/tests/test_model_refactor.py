from __future__ import annotations

import numpy as np
import pytest

from prism.model import (
    ObservationBatch,
    Posterior,
    PriorFitConfig,
    PriorGrid,
    fit_gene_priors,
    fit_gene_priors_em,
    infer_posteriors,
)


def _build_batch() -> ObservationBatch:
    return ObservationBatch(
        gene_names=["GeneA", "GeneB"],
        counts=np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]]),
        reference_counts=np.asarray([10.0, 12.0, 14.0, 16.0]),
    )


def _build_priors() -> PriorGrid:
    p_grid = np.asarray([[0.05, 0.15, 0.25], [0.02, 0.08, 0.14]], dtype=np.float64)
    weights = np.asarray([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]], dtype=np.float64)
    return PriorGrid(
        gene_names=["GeneA", "GeneB"],
        p_grid=p_grid,
        weights=weights,
        S=10.0,
    )


def test_infer_posteriors_torch_dtype_matches_float64_baseline() -> None:
    batch = _build_batch()
    priors = _build_priors()

    result64 = infer_posteriors(
        batch,
        priors,
        include_posterior=True,
        torch_dtype="float64",
    )
    result32 = infer_posteriors(
        batch,
        priors,
        include_posterior=True,
        torch_dtype="float32",
    )

    np.testing.assert_allclose(result32.map_p, result64.map_p, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(result32.map_mu, result64.map_mu, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        result32.posterior_entropy,
        result64.posterior_entropy,
        atol=1e-5,
        rtol=1e-5,
    )
    assert result32.posterior is not None
    assert result64.posterior is not None
    np.testing.assert_allclose(
        result32.posterior, result64.posterior, atol=1e-5, rtol=1e-5
    )


def test_posterior_extract_accepts_float32_torch_dtype() -> None:
    batch = _build_batch()
    posterior = Posterior(batch.gene_names, _build_priors(), torch_dtype="float32")

    extracted = posterior.extract(
        batch, channels={"signal", "map_p", "posterior_entropy"}
    )

    assert set(extracted) == {"signal", "map_mu", "map_p", "posterior_entropy"}
    assert extracted["signal"].shape == batch.counts.shape
    assert extracted["map_mu"].shape == batch.counts.shape
    assert extracted["map_p"].shape == batch.counts.shape
    assert extracted["posterior_entropy"].shape == batch.counts.shape


def test_fit_gene_priors_returns_normalized_weights() -> None:
    batch = ObservationBatch(
        gene_names=["GeneA"],
        counts=np.asarray([[0.0], [1.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64),
    )

    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=16,
            n_iter=3,
            cell_chunk_size=2,
            torch_dtype="float32",
        ),
    )

    weights = np.asarray(result.priors.weights, dtype=np.float64)
    np.testing.assert_allclose(
        weights.sum(axis=-1), np.ones(weights.shape[0]), atol=1e-6
    )
    assert np.isfinite(result.final_loss)
    assert np.isfinite(result.best_loss)


def test_fit_gene_priors_supports_warm_start_weights() -> None:
    batch = ObservationBatch(
        gene_names=["GeneA"],
        counts=np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0, 11.0], dtype=np.float64),
    )
    init_weights = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)

    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(grid_size=4, n_iter=1, sigma_bins=0.0),
        init_prior_weights=init_weights,
    )

    np.testing.assert_allclose(result.initial_prior_weights, init_weights, atol=1e-6)


def test_fit_gene_priors_stops_early_when_patience_is_exhausted() -> None:
    batch = ObservationBatch(
        gene_names=["GeneA"],
        counts=np.asarray([[0.0], [1.0], [1.0], [2.0], [2.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64),
    )

    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=10,
            early_stop_tol=1e-6,
            early_stop_patience=1,
        ),
    )

    assert len(result.loss_history) <= 10
    assert np.isfinite(result.final_loss)


def test_fit_gene_priors_grid_max_method_quantile() -> None:
    batch = ObservationBatch(
        gene_names=["GeneA"],
        counts=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0, 11.0], dtype=np.float64),
    )

    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=2,
            grid_max_method="quantile",
        ),
    )

    assert np.isfinite(result.final_loss)
    assert result.priors.weights.shape == (1, 8)


def test_fit_gene_priors_grid_strategy_sqrt() -> None:
    batch = ObservationBatch(
        gene_names=["GeneA"],
        counts=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0, 11.0], dtype=np.float64),
    )

    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=2,
            grid_strategy="sqrt",
        ),
    )

    assert np.isfinite(result.final_loss)
    p_grid = np.asarray(result.priors.p_grid, dtype=np.float64)
    assert p_grid.shape == (1, 8)
    assert float(p_grid[0, 0]) == 0.0
    assert float(p_grid[0, -1]) > 0.0


def _small_batch() -> ObservationBatch:
    return ObservationBatch(
        gene_names=["GeneA"],
        counts=np.asarray([[0.0], [1.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64),
    )


def test_sigma_annealing() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=5,
            sigma_anneal_start=3.0,
            sigma_anneal_end=0.5,
        ),
    )
    assert np.isfinite(result.final_loss)
    assert result.priors.weights.shape == (1, 8)


def test_adaptive_grid() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=16,
            n_iter=5,
            adaptive_grid=True,
            adaptive_grid_fraction=0.5,
        ),
    )
    assert np.isfinite(result.best_loss)
    assert result.priors.p_grid.shape == (1, 16)


def test_adaptive_grid_quantile_window_is_configurable() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=16,
            n_iter=5,
            adaptive_grid=True,
            adaptive_grid_fraction=0.5,
            adaptive_grid_quantile_lo=0.2,
            adaptive_grid_quantile_hi=0.8,
        ),
    )
    assert np.isfinite(result.best_loss)
    assert result.priors.p_grid.shape == (1, 16)


def test_cell_sample_fraction() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=3,
            cell_sample_fraction=0.5,
            cell_chunk_size=2,
        ),
    )
    assert np.isfinite(result.final_loss)


def test_align_mode_kl() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(grid_size=8, n_iter=3, align_mode="kl"),
    )
    assert np.isfinite(result.final_loss)


def test_align_mode_weighted_jsd() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(grid_size=8, n_iter=3, align_mode="weighted_jsd"),
    )
    assert np.isfinite(result.final_loss)


def test_shrinkage_weight() -> None:
    batch = ObservationBatch(
        gene_names=["GeneA", "GeneB"],
        counts=np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
        reference_counts=np.asarray([8.0, 9.0, 10.0], dtype=np.float64),
    )
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(grid_size=8, n_iter=3, shrinkage_weight=0.3),
    )
    assert np.isfinite(result.final_loss)
    assert result.priors.weights.shape == (2, 8)


def test_ensemble_restarts() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(grid_size=8, n_iter=3, ensemble_restarts=3),
    )
    assert np.isfinite(result.best_loss)


def test_negative_binomial_likelihood() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=3,
            likelihood="negative_binomial",
            nb_overdispersion=0.1,
        ),
    )
    assert np.isfinite(result.final_loss)
    assert result.priors.weights.shape == (1, 8)
    assert result.priors.distribution == "negative_binomial"


def test_negative_binomial_priors_are_not_silently_used_by_binomial_inference() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=3,
            likelihood="negative_binomial",
            nb_overdispersion=0.1,
        ),
    )
    with pytest.raises(ValueError, match="posterior distribution mismatch"):
        infer_posteriors(batch, result.priors, posterior_distribution="binomial")


def test_poisson_likelihood() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=3,
            likelihood="poisson",
        ),
    )
    assert np.isfinite(result.final_loss)
    assert result.priors.weights.shape == (1, 8)
    p_grid = np.asarray(result.priors.p_grid, dtype=np.float64)
    assert np.all(p_grid >= 0.0)


def test_poisson_likelihood_em() -> None:
    batch = _small_batch()
    result = fit_gene_priors_em(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=3,
            likelihood="poisson",
        ),
    )
    assert np.isfinite(result.best_loss)


def test_poisson_priors_are_not_silently_used_by_binomial_inference() -> None:
    batch = _small_batch()
    result = fit_gene_priors(
        batch,
        S=10.0,
        config=PriorFitConfig(
            grid_size=8,
            n_iter=3,
            likelihood="poisson",
        ),
    )
    with pytest.raises(ValueError, match="posterior distribution mismatch"):
        infer_posteriors(batch, result.priors, posterior_distribution="binomial")

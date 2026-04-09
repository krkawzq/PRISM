from __future__ import annotations

import numpy as np

from prism.cli.fit.common import build_fit_tasks, parse_shard, resolve_scale
from prism.cli.fit.priors import _select_warm_start_prior
from prism.model import ModelCheckpoint, PriorGrid, make_distribution_grid


def _make_checkpoint(distribution: str) -> ModelCheckpoint:
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            distribution,  # type: ignore[arg-type]
            support=[0.1, 0.2] if distribution != "poisson" else [1.0, 2.0],
            probabilities=[0.5, 0.5],
        ),
        scale=1.0,
    )
    return ModelCheckpoint(gene_names=["g1"], prior=prior)


def test_parse_shard_parses_rank_and_world() -> None:
    assert parse_shard("2/4") == (2, 4)


def test_build_fit_tasks_keeps_global_scope_on_all_cells() -> None:
    label_groups = [("a", np.array([1, 3], dtype=np.int64))]
    tasks = build_fit_tasks("both", label_groups, n_cells=5)
    assert tasks[0].scope_kind == "global"
    assert tasks[0].cell_indices.tolist() == [0, 1, 2, 3, 4]
    assert tasks[1].scope_name == "label:a"


def test_resolve_scale_ignores_zero_reference_cells() -> None:
    resolution = resolve_scale(np.array([0.0, 2.0, 4.0], dtype=np.float64), scale=None)
    assert resolution.default_scale == 3.0
    assert resolution.scale == 3.0
    assert resolution.n_positive_reference_cells == 2


def test_select_warm_start_prior_rejects_distribution_mismatch() -> None:
    checkpoint = _make_checkpoint("poisson")
    result = _select_warm_start_prior(
        checkpoint,
        label_value=None,
        gene_names=["g1"],
        expected_distribution="binomial",
    )
    assert result is None


def test_select_warm_start_prior_returns_probabilities_when_compatible() -> None:
    checkpoint = _make_checkpoint("binomial")
    result = _select_warm_start_prior(
        checkpoint,
        label_value=None,
        gene_names=["g1"],
        expected_distribution="binomial",
    )
    assert result is not None
    assert np.allclose(result, np.array([[0.5, 0.5]], dtype=np.float64))

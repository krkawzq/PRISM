from __future__ import annotations

from pathlib import Path

import pytest

from prism.cli.checkpoint.common import merge_prior_scope, validate_shared_metadata
from prism.model import ModelCheckpoint, PriorGrid, ScaleMetadata, make_distribution_grid


def _make_prior(gene_name: str) -> PriorGrid:
    return PriorGrid(
        gene_names=[gene_name],
        distribution=make_distribution_grid(
            "binomial",
            support=[0.1, 0.2],
            probabilities=[0.5, 0.5],
        ),
        scale=1.0,
    )


def test_merge_prior_scope_rejects_inconsistent_scale_metadata() -> None:
    with pytest.raises(ValueError, match="mean_reference_count"):
        merge_prior_scope(
            [_make_prior("a"), _make_prior("b")],
            [
                ScaleMetadata(scale=1.0, mean_reference_count=1.0),
                ScaleMetadata(scale=1.0, mean_reference_count=2.0),
            ],
            requested_gene_names=["a", "b"],
            allow_partial=False,
            scope_name="global",
        )


def test_validate_shared_metadata_rejects_misaligned_inputs() -> None:
    checkpoint = ModelCheckpoint(
        gene_names=["a"],
        prior=_make_prior("a"),
        fit_config={"likelihood": "binomial"},
        metadata={
            "source_h5ad_path": "dataset.h5ad",
            "layer": None,
            "reference_gene_names": ["r"],
            "requested_fit_gene_names": ["a"],
            "fit_mode": "global",
            "label_key": None,
            "fit_distribution": "binomial",
            "posterior_distribution": "binomial",
            "support_domain": "probability",
        },
    )
    with pytest.raises(ValueError, match="same length"):
        validate_shared_metadata([checkpoint], [])
    with pytest.raises(ValueError, match="at least one checkpoint"):
        validate_shared_metadata([], [])

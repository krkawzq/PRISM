from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from prism.model import (
    ModelCheckpoint,
    PriorGrid,
    ScaleMetadata,
    make_distribution_grid,
    save_checkpoint,
)
from prism.server.services import checkpoints as checkpoint_service
from prism.server.services import datasets as dataset_service


def _prior(*, gene_names: list[str], scale: float = 2.0) -> PriorGrid:
    return PriorGrid(
        gene_names=gene_names,
        distribution=make_distribution_grid(
            "binomial",
            support=np.asarray([[0.1, 0.3, 0.5] for _ in gene_names], dtype=np.float64),
            probabilities=np.asarray(
                [[0.2, 0.3, 0.5] for _ in gene_names],
                dtype=np.float64,
            ),
        ),
        scale=scale,
    )


def _checkpoint(gene_names: list[str]) -> ModelCheckpoint:
    return ModelCheckpoint(
        gene_names=gene_names,
        prior=_prior(gene_names=gene_names, scale=3.0),
        fit_config={"likelihood": "negative_binomial", "nb_overdispersion": 0.2},
        metadata={
            "reference_gene_names": ["GeneA", "GeneB", "MissingGene"],
            "label_key": "condition",
        },
        scale_metadata=ScaleMetadata(scale=3.0, mean_reference_count=6.0),
        label_priors={"treated": _prior(gene_names=gene_names, scale=4.0)},
        label_scale_metadata={
            "treated": ScaleMetadata(scale=4.0, mean_reference_count=7.0)
        },
    )


def test_dataset_service_computes_dense_and_sparse_statistics() -> None:
    dense = np.asarray([[1, 0, 2], [0, 0, 3]], dtype=np.float64)
    csr = sparse.csr_matrix(dense)
    assert np.allclose(dataset_service.compute_totals(dense), [3.0, 3.0])
    assert np.allclose(dataset_service.compute_totals(csr), [3.0, 3.0])
    assert np.allclose(dataset_service.compute_gene_totals(dense), [1.0, 0.0, 5.0])
    assert np.allclose(dataset_service.compute_gene_totals(csr), [1.0, 0.0, 5.0])
    assert np.array_equal(dataset_service.compute_detected_counts(dense), [1, 0, 2])
    assert np.array_equal(dataset_service.compute_detected_counts(csr), [1, 0, 2])
    assert np.allclose(dataset_service.compute_cell_zero_fraction(dense), [1 / 3, 2 / 3])


def test_dataset_service_gene_lookup_and_search() -> None:
    gene_names = np.asarray(["GeneA", "GeneB", "GeneC"])
    gene_lower = tuple(name.lower() for name in gene_names.tolist())
    gene_to_idx = dataset_service.build_gene_to_idx(gene_names)
    gene_lower_to_idx = {name.lower(): idx for idx, name in enumerate(gene_names.tolist())}
    resolved_exact = dataset_service.resolve_gene_query(
        "GeneB",
        gene_names,
        gene_lower,
        gene_to_idx,
        gene_lower_to_idx,
    )
    resolved_index = dataset_service.resolve_gene_query(
        "2",
        gene_names,
        gene_lower,
        gene_to_idx,
        gene_lower_to_idx,
    )
    resolved_casefold = dataset_service.resolve_gene_query(
        "genec",
        gene_names,
        gene_lower,
        gene_to_idx,
        gene_lower_to_idx,
    )
    matches = dataset_service.search_gene_candidates(
        "gene",
        gene_names,
        gene_lower,
        np.asarray([5.0, 10.0, 1.0]),
        np.asarray([2, 1, 1]),
        np.asarray([1, 0, 2]),
        n_cells=4,
        limit=2,
    )

    assert resolved_exact == 1
    assert resolved_index == 2
    assert resolved_casefold == 2
    assert [item.gene_name for item in matches] == ["GeneB", "GeneA"]

    with pytest.raises(dataset_service.GeneNotFoundError, match="not found"):
        dataset_service.resolve_gene_query(
            "missing",
            gene_names,
            gene_lower,
            gene_to_idx,
            gene_lower_to_idx,
        )


def test_dataset_service_detects_label_columns_in_preferred_order() -> None:
    adata = SimpleNamespace(
        n_obs=4,
        obs=pd.DataFrame(
            {
                "sample": ["s1", "s2", "s3", "s4"],
                "condition": ["a", "a", "b", "b"],
                "cell_type": ["t", "b", "t", "b"],
            }
        ),
    )
    labels = dataset_service.detect_label_columns(adata)
    assert tuple(labels) == ("cell_type", "condition", "sample")


def test_checkpoint_service_loads_overlap_and_distribution_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    save_checkpoint(_checkpoint(["GeneA", "GeneB"]), checkpoint_path)
    state = checkpoint_service.load_checkpoint_state(
        checkpoint_path,
        dataset_gene_names=["GeneA", "GeneB", "GeneC"],
        gene_to_idx={"GeneA": 0, "GeneB": 1, "GeneC": 2},
        available_label_keys=("condition",),
    )

    assert state.reference_gene_names == ("GeneA", "GeneB")
    assert state.reference_positions == (0, 1)
    assert state.posterior_distribution == "negative_binomial"
    assert state.nb_overdispersion == 0.2
    assert state.suggested_label_key == "condition"


def test_checkpoint_service_requires_reference_overlap(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    save_checkpoint(_checkpoint(["GeneA", "GeneB"]), checkpoint_path)
    with pytest.raises(ValueError, match="do not overlap"):
        checkpoint_service.load_checkpoint_state(
            checkpoint_path,
            dataset_gene_names=["OtherGene"],
            gene_to_idx={"OtherGene": 0},
            available_label_keys=(),
        )


def test_checkpoint_helpers_validate_metadata_and_distribution() -> None:
    with pytest.raises(ValueError, match="reference_gene_names"):
        checkpoint_service._require_reference_gene_names({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported posterior distribution"):
        checkpoint_service._resolve_posterior_distribution(
            {"fit_distribution": "weird"},
            {},
        )
    assert checkpoint_service._resolve_nb_overdispersion({}, {}) == 0.01
    assert checkpoint_service._resolve_suggested_label_key(
        {"label_key": "missing"},
        ("fallback",),
    ) == "fallback"

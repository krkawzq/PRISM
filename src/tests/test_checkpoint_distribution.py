from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from prism.model.checkpoint import ModelCheckpoint, load_checkpoint, save_checkpoint
from prism.model.types import PriorGrid, ScaleMetadata


def _make_checkpoint(distribution: str) -> ModelCheckpoint:
    if distribution == "poisson":
        grid_domain = "rate"
        p_grid = np.array([[0.1, 0.5, 1.0]], dtype=np.float64)
    else:
        grid_domain = "p"
        p_grid = np.array([[0.1, 0.2, 0.4]], dtype=np.float64)
    priors = PriorGrid(
        gene_names=["GeneA"],
        p_grid=p_grid,
        weights=np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
        S=10.0,
        grid_domain=grid_domain,
        distribution=distribution,
    )
    return ModelCheckpoint(
        gene_names=["GeneA"],
        priors=priors,
        scale=ScaleMetadata(S=10.0, mean_reference_count=100.0),
        fit_config={
            "likelihood": distribution,
            "nb_overdispersion": 0.01,
        },
        metadata={
            "fit_distribution": distribution,
            "posterior_distribution": distribution,
            "grid_domain": grid_domain,
        },
    )


@pytest.mark.parametrize("distribution", ["binomial", "negative_binomial", "poisson"])
def test_checkpoint_roundtrip_preserves_distribution(tmp_path, distribution: str) -> None:
    path = tmp_path / f"{distribution}.pkl"
    checkpoint = _make_checkpoint(distribution)
    save_checkpoint(checkpoint, path)
    loaded = load_checkpoint(path)
    assert loaded.metadata["fit_distribution"] == distribution
    assert loaded.metadata["posterior_distribution"] == distribution
    assert loaded.priors is not None
    assert loaded.priors.distribution == distribution
    expected_domain = "rate" if distribution == "poisson" else "p"
    assert loaded.metadata["grid_domain"] == expected_domain
    assert loaded.priors.grid_domain == expected_domain
    assert loaded.metadata["distribution_resolution"] == "explicit"
    assert loaded.metadata["legacy_compatibility"] is False


def test_load_legacy_checkpoint_uses_compatibility_path(tmp_path) -> None:
    path = tmp_path / "legacy.pkl"
    payload = {
        "schema_version": 1,
        "gene_names": ["GeneA"],
        "priors": {
            "gene_names": ["GeneA"],
            "p_grid": np.array([[0.1, 0.2, 0.4]], dtype=np.float64),
            "weights": np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
            "S": 10.0,
        },
        "scale": {"S": 10.0, "mean_reference_count": 100.0},
        "fit_config": {},
        "metadata": {},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_checkpoint(path)
    assert any("compatibility path" in str(item.message) for item in caught)
    assert loaded.metadata["fit_distribution"] == "binomial"
    assert loaded.metadata["posterior_distribution"] == "binomial"
    assert loaded.metadata["grid_domain"] == "p"
    assert loaded.metadata["distribution_resolution"] == "legacy-compatibility"
    assert loaded.metadata["legacy_compatibility"] is True


def test_schema2_checkpoint_missing_distribution_metadata_fails(tmp_path) -> None:
    path = tmp_path / "bad.pkl"
    payload = {
        "schema_version": 2,
        "gene_names": ["GeneA"],
        "priors": {
            "gene_names": ["GeneA"],
            "p_grid": np.array([[0.1, 0.2, 0.4]], dtype=np.float64),
            "weights": np.array([[0.2, 0.3, 0.5]], dtype=np.float64),
            "S": 10.0,
            "grid_domain": "p",
            "distribution": "binomial",
        },
        "scale": {"S": 10.0, "mean_reference_count": 100.0},
        "fit_config": {},
        "metadata": {},
        "label_priors": {},
        "label_scales": {},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="missing required distribution metadata"):
        load_checkpoint(path)

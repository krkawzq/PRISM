from __future__ import annotations

import numpy as np
import pytest

from prism.cli.extract.kbulk import extract_kbulk_command
from prism.model.checkpoint import ModelCheckpoint
from prism.model.kbulk import infer_kbulk_samples
from prism.model.types import PriorGrid, ScaleMetadata


def test_model_kbulk_poisson_produces_map_rate() -> None:
    priors = PriorGrid(
        gene_names=["GeneA"],
        p_grid=np.array([[0.5, 1.0, 2.0]], dtype=np.float64),
        weights=np.array([[0.2, 0.5, 0.3]], dtype=np.float64),
        S=10.0,
        grid_domain="rate",
        distribution="poisson",
    )
    result = infer_kbulk_samples(
        ["GeneA"],
        aggregated_counts=np.array([[1.0], [2.0]], dtype=np.float64),
        effective_exposure=np.array([5.0, 6.0], dtype=np.float64),
        priors=priors,
        posterior_distribution="poisson",
    )
    assert result.map_rate is not None
    assert result.map_rate.shape == (2, 1)
    assert np.isnan(result.map_p).all()
    np.testing.assert_allclose(result.map_mu, result.map_rate)


def test_extract_kbulk_rejects_poisson_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = ModelCheckpoint(
        gene_names=["GeneA"],
        priors=PriorGrid(
            gene_names=["GeneA"],
            p_grid=np.array([[0.5, 1.0, 2.0]], dtype=np.float64),
            weights=np.array([[0.2, 0.5, 0.3]], dtype=np.float64),
            S=10.0,
            grid_domain="rate",
            distribution="poisson",
        ),
        scale=ScaleMetadata(S=10.0, mean_reference_count=100.0),
        fit_config={"likelihood": "poisson"},
        metadata={
            "schema_version": 2,
            "fit_distribution": "poisson",
            "posterior_distribution": "poisson",
            "grid_domain": "rate",
            "reference_gene_names": ["GeneA"],
        },
    )

    checkpoint_path = tmp_path / "dummy.ckpt"
    h5ad_path = tmp_path / "dummy.h5ad"
    checkpoint_path.write_text("stub", encoding="utf-8")
    h5ad_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "prism.cli.extract.kbulk.load_checkpoint",
        lambda path: checkpoint,
    )

    with pytest.raises(ValueError, match="does not support poisson/rate-grid checkpoints yet"):
        extract_kbulk_command(
            checkpoint_path=checkpoint_path,
            h5ad_path=h5ad_path,
            output_path=tmp_path / "out.h5ad",
            class_key="cell_type",
            k=2,
            n_samples=3,
            prior_source="global",
            genes_path=None,
            reference_genes_path=None,
            layer=None,
            balance=False,
            sample_seed=0,
            sample_batch_size=256,
            S_source="checkpoint",
            navg_source="dataset",
            device="cpu",
            dtype="float32",
            dry_run=False,
        )

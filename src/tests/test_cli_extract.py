from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from prism.cli.extract.kbulk_mean import extract_kbulk_mean_command
from prism.cli.extract.signals import extract_signals_command
from prism.model import ModelCheckpoint, PriorGrid, make_distribution_grid, save_checkpoint


def _prior(gene_names: list[str]) -> PriorGrid:
    return PriorGrid(
        gene_names=gene_names,
        distribution=make_distribution_grid(
            "binomial",
            support=np.asarray([[0.1, 0.2], [0.2, 0.4]], dtype=np.float64),
            probabilities=np.asarray([[0.5, 0.5], [0.4, 0.6]], dtype=np.float64),
        ),
        scale=2.0,
    )


def _checkpoint(gene_names: list[str]) -> ModelCheckpoint:
    return ModelCheckpoint(
        gene_names=gene_names,
        prior=_prior(gene_names),
        fit_config={"likelihood": "binomial"},
        metadata={
            "reference_gene_names": list(gene_names),
            "fit_distribution": "binomial",
            "posterior_distribution": "binomial",
            "support_domain": "probability",
        },
    )


def test_extract_signals_command_writes_requested_layers(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pkl"
    h5ad_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "signals.h5ad"
    gene_names = ["g1", "g2"]
    save_checkpoint(_checkpoint(gene_names), checkpoint_path)

    adata = ad.AnnData(
        X=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=gene_names),
    )
    adata.write_h5ad(h5ad_path)

    def fake_extract_batch(**kwargs):
        batch_names = kwargs["batch_names"]
        batch_counts = kwargs["batch_counts"]
        return {
            "signal": np.full(
                (batch_counts.shape[0], len(batch_names)),
                7.0,
                dtype=np.float64,
            )
        }

    monkeypatch.setattr("prism.cli.extract.signals.extract_batch", fake_extract_batch)

    result = extract_signals_command(
        checkpoint_path=checkpoint_path,
        h5ad_path=h5ad_path,
        output_path=output_path,
        genes_path=None,
        output_mode="fitted-only",
        channels=["signal"],
        batch_size=1,
    )

    assert result == 0
    output = ad.read_h5ad(output_path)
    assert output.var_names.tolist() == gene_names
    assert "signal" in output.layers
    assert np.allclose(output.layers["signal"], 7.0)


def test_extract_kbulk_mean_command_writes_output(tmp_path: Path) -> None:
    h5ad_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "kbulk-mean.h5ad"
    adata = ad.AnnData(
        X=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=np.float64,
        ),
        obs=pd.DataFrame(
            {"label": ["a", "a", "b", "b"]},
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.write_h5ad(h5ad_path)

    result = extract_kbulk_mean_command(
        h5ad_path=h5ad_path,
        output_path=output_path,
        class_key="label",
        k=2,
        n_samples=1,
        balance=False,
    )

    assert result == 0
    output = ad.read_h5ad(output_path)
    assert output.n_obs == 2
    assert output.n_vars == 2
    assert output.uns["kbulk"]["method"] == "kbulk-mean"

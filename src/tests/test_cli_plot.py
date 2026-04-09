from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from prism.cli.plot.batch_grid import plot_batch_grid_command
from prism.cli.plot.distributions import plot_distributions_command
from prism.cli.plot.label_summary import plot_label_summary_command
from prism.cli.plot.priors import plot_priors_command
from prism.model import ModelCheckpoint, PriorGrid, make_distribution_grid, save_checkpoint


def _prior(
    *,
    gene_names: list[str],
    support_rows: list[list[float]],
    probability_rows: list[list[float]],
    scale: float = 2.0,
    distribution: str = "binomial",
) -> PriorGrid:
    return PriorGrid(
        gene_names=gene_names,
        distribution=make_distribution_grid(
            distribution,  # type: ignore[arg-type]
            support=np.asarray(support_rows, dtype=np.float64),
            probabilities=np.asarray(probability_rows, dtype=np.float64),
        ),
        scale=scale,
    )


def _checkpoint() -> ModelCheckpoint:
    gene_names = ["GeneA", "GeneB"]
    return ModelCheckpoint(
        gene_names=gene_names,
        prior=_prior(
            gene_names=gene_names,
            support_rows=[[0.10, 0.20, 0.40], [0.15, 0.25, 0.35]],
            probability_rows=[[0.20, 0.30, 0.50], [0.25, 0.25, 0.50]],
        ),
        label_priors={
            "b1_ctrl": _prior(
                gene_names=gene_names,
                support_rows=[[0.10, 0.20, 0.40], [0.15, 0.25, 0.35]],
                probability_rows=[[0.30, 0.30, 0.40], [0.20, 0.30, 0.50]],
            ),
            "b1_stim": _prior(
                gene_names=gene_names,
                support_rows=[[0.10, 0.20, 0.40], [0.15, 0.25, 0.35]],
                probability_rows=[[0.10, 0.20, 0.70], [0.10, 0.20, 0.70]],
            ),
            "b2_ctrl": _prior(
                gene_names=gene_names,
                support_rows=[[0.10, 0.20, 0.40], [0.15, 0.25, 0.35]],
                probability_rows=[[0.40, 0.20, 0.40], [0.35, 0.25, 0.40]],
            ),
        },
        metadata={"source_h5ad_path": "/tmp/reference_dataset.h5ad"},
    )


def _write_checkpoint(
    tmp_path: Path,
    *,
    name: str = "checkpoint.pkl",
    checkpoint: ModelCheckpoint | None = None,
) -> Path:
    checkpoint_path = tmp_path / name
    save_checkpoint(_checkpoint() if checkpoint is None else checkpoint, checkpoint_path)
    return checkpoint_path


def _checkpoint_variant(
    *,
    source_h5ad_path: str,
    include_ctrl_label: bool = True,
) -> ModelCheckpoint:
    gene_names = ["GeneA", "GeneB"]
    label_priors = {
        "b1_stim": _prior(
            gene_names=gene_names,
            support_rows=[[0.12, 0.20, 0.38], [0.14, 0.22, 0.34]],
            probability_rows=[[0.15, 0.20, 0.65], [0.20, 0.30, 0.50]],
        ),
    }
    if include_ctrl_label:
        label_priors["b1_ctrl"] = _prior(
            gene_names=gene_names,
            support_rows=[[0.08, 0.18, 0.36], [0.12, 0.24, 0.32]],
            probability_rows=[[0.45, 0.25, 0.30], [0.30, 0.30, 0.40]],
        )
    return ModelCheckpoint(
        gene_names=gene_names,
        prior=_prior(
            gene_names=gene_names,
            support_rows=[[0.08, 0.18, 0.36], [0.12, 0.24, 0.32]],
            probability_rows=[[0.25, 0.35, 0.40], [0.20, 0.30, 0.50]],
        ),
        label_priors=label_priors,
        metadata={"source_h5ad_path": source_h5ad_path},
    )


def test_plot_priors_command_writes_figure_and_csvs(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    figure_path = tmp_path / "priors.svg"
    curve_csv = tmp_path / "priors.csv"
    summary_csv = tmp_path / "priors_summary.csv"
    plot_priors_command(
        checkpoint_paths=[checkpoint_path],
        gene_names=["GeneA"],
        output_path=figure_path,
        output_csv_path=curve_csv,
        summary_csv_path=summary_csv,
        labels=["b1_ctrl"],
        layout="facet",
    )
    assert figure_path.exists()
    assert curve_csv.exists()
    assert summary_csv.exists()
    assert "probability" in curve_csv.read_text(encoding="utf-8")


def test_plot_priors_command_compares_multiple_checkpoints_by_label(
    tmp_path: Path,
) -> None:
    checkpoint_a = _write_checkpoint(
        tmp_path,
        name="dataset_a.pkl",
        checkpoint=_checkpoint_variant(source_h5ad_path="/tmp/dataset_a.h5ad"),
    )
    checkpoint_b = _write_checkpoint(
        tmp_path,
        name="dataset_b.pkl",
        checkpoint=_checkpoint_variant(source_h5ad_path="/tmp/dataset_b.h5ad"),
    )
    figure_path = tmp_path / "priors_multi.svg"
    curve_csv = tmp_path / "priors_multi.csv"
    summary_csv = tmp_path / "priors_multi_summary.csv"
    plot_priors_command(
        checkpoint_paths=[checkpoint_a, checkpoint_b],
        checkpoint_names=["dataset-a", "dataset-b"],
        gene_names=["GeneA"],
        output_path=figure_path,
        output_csv_path=curve_csv,
        summary_csv_path=summary_csv,
        labels=["b1_ctrl"],
        include_global=False,
        layout="facet",
    )
    exported = pd.read_csv(curve_csv)
    assert figure_path.exists()
    assert summary_csv.exists()
    assert set(exported["checkpoint_name"]) == {"dataset-a", "dataset-b"}
    assert set(exported["source"]) == {
        "dataset-a/label:b1_ctrl",
        "dataset-b/label:b1_ctrl",
    }


def test_plot_priors_command_drop_missing_policy_skips_missing_label(
    tmp_path: Path,
) -> None:
    checkpoint_a = _write_checkpoint(
        tmp_path,
        name="dataset_a.pkl",
        checkpoint=_checkpoint_variant(source_h5ad_path="/tmp/dataset_a.h5ad"),
    )
    checkpoint_b = _write_checkpoint(
        tmp_path,
        name="dataset_b.pkl",
        checkpoint=_checkpoint_variant(
            source_h5ad_path="/tmp/dataset_b.h5ad",
            include_ctrl_label=False,
        ),
    )
    figure_path = tmp_path / "priors_drop.svg"
    curve_csv = tmp_path / "priors_drop.csv"
    plot_priors_command(
        checkpoint_paths=[checkpoint_a, checkpoint_b],
        checkpoint_names=["dataset-a", "dataset-b"],
        gene_names=["GeneA"],
        output_path=figure_path,
        output_csv_path=curve_csv,
        labels=["b1_ctrl"],
        include_global=False,
        missing_policy="drop",
    )
    exported = pd.read_csv(curve_csv)
    assert figure_path.exists()
    assert set(exported["checkpoint_name"]) == {"dataset-a"}


def test_plot_batch_grid_command_writes_gene_figures_and_csvs(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    output_dir = tmp_path / "batch-grid"
    curve_csv = tmp_path / "batch-grid.csv"
    summary_csv = tmp_path / "batch-grid-summary.csv"
    plot_batch_grid_command(
        checkpoint_path=checkpoint_path,
        gene_names=["GeneA"],
        output_dir=output_dir,
        output_csv_path=curve_csv,
        summary_csv_path=summary_csv,
    )
    assert (output_dir / "GeneA.svg").exists()
    assert curve_csv.exists()
    assert summary_csv.exists()
    assert "batch" in curve_csv.read_text(encoding="utf-8")


def test_plot_distributions_command_writes_grouped_figure(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float64),
        obs=pd.DataFrame(
            {"condition": ["a", "a", "b", "b"]},
            index=["cell0", "cell1", "cell2", "cell3"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    adata.layers["signal"] = np.asarray(
        [[0.1, 0.2], [0.2, 0.3], [0.5, 0.6], [0.7, 0.8]],
        dtype=np.float64,
    )
    h5ad_path = tmp_path / "signals.h5ad"
    output_path = tmp_path / "distributions.svg"
    adata.write_h5ad(h5ad_path)
    plot_distributions_command(
        h5ad_path=h5ad_path,
        output_path=output_path,
        layers=["signal"],
        group_key="condition",
        plot_type="box",
    )
    assert output_path.exists()


def test_plot_label_summary_command_writes_heatmap(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    output_path = tmp_path / "label-summary.svg"
    plot_label_summary_command(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        gene_names=["GeneA", "GeneB"],
        metric="jsd",
    )
    assert output_path.exists()

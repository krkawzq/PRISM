from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import prism.plotting as plotting
import pytest
from prism.model import ModelCheckpoint, PriorGrid, make_distribution_grid
from prism.plotting import (
    LabelGridEntry,
    SUPPORTED_CURVE_MODES,
    SUPPORTED_STAT_FIELDS,
    SUPPORTED_Y_SCALES,
    display_cutoff,
    load_annotation_tables,
    load_label_grid_entries,
    plot_batch_grid_figure,
    plot_prior_facet_figure,
    plot_prior_overlay_figure,
    resolve_batch_grid_curve_sets,
    resolve_multi_checkpoint_prior_curve_sets,
    resolve_prior_curve_sets,
)


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
            distribution,
            support=np.asarray(support_rows, dtype=np.float64),
            probabilities=np.asarray(probability_rows, dtype=np.float64),
        ),
        scale=scale,
    )


def _checkpoint() -> ModelCheckpoint:
    return ModelCheckpoint(
        gene_names=["GeneA"],
        prior=_prior(
            gene_names=["GeneA"],
            support_rows=[[0.10, 0.20, 0.40]],
            probability_rows=[[0.20, 0.30, 0.50]],
        ),
        label_priors={
            "ctrl": _prior(
                gene_names=["GeneA"],
                support_rows=[[0.15, 0.25, 0.35]],
                probability_rows=[[0.30, 0.40, 0.30]],
            ),
            "stim": _prior(
                gene_names=["GeneA"],
                support_rows=[[0.05, 0.30, 0.45]],
                probability_rows=[[0.10, 0.20, 0.70]],
            ),
        },
        metadata={"source_h5ad_path": "/tmp/source_a.h5ad"},
    )


def _checkpoint_variant(*, source_h5ad_path: str, include_ctrl: bool = True) -> ModelCheckpoint:
    label_priors = {
        "stim": _prior(
            gene_names=["GeneA"],
            support_rows=[[0.07, 0.28, 0.44]],
            probability_rows=[[0.12, 0.22, 0.66]],
        )
    }
    if include_ctrl:
        label_priors["ctrl"] = _prior(
            gene_names=["GeneA"],
            support_rows=[[0.16, 0.24, 0.38]],
            probability_rows=[[0.25, 0.35, 0.40]],
        )
    return ModelCheckpoint(
        gene_names=["GeneA"],
        prior=_prior(
            gene_names=["GeneA"],
            support_rows=[[0.11, 0.21, 0.41]],
            probability_rows=[[0.18, 0.32, 0.50]],
        ),
        label_priors=label_priors,
        metadata={"source_h5ad_path": source_h5ad_path},
    )


def test_plotting_package_exports_public_helpers() -> None:
    assert SUPPORTED_CURVE_MODES == ("density", "cdf")
    assert SUPPORTED_Y_SCALES == ("linear", "log")
    assert "scale" in SUPPORTED_STAT_FIELDS
    assert plotting.SUPPORTED_CURVE_MODES == SUPPORTED_CURVE_MODES
    assert plotting.SUPPORTED_Y_SCALES == SUPPORTED_Y_SCALES
    assert "plt" in plotting.__all__
    assert callable(plotting.plot_prior_overlay_figure)
    assert plotting.plt is not None
    assert display_cutoff(
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([0.4, 0.6], dtype=np.float64),
        0.5,
    ) == pytest.approx(0.2)


def test_plotting_imports_do_not_override_backend() -> None:
    env = dict(os.environ)
    env["MPLBACKEND"] = "svg"
    script = """
import matplotlib
print("before", matplotlib.get_backend())
import prism.plotting as plotting
print("after_package", matplotlib.get_backend())
_ = plotting.plot_prior_overlay_figure
print("after_plot_attr", matplotlib.get_backend())
import prism.plotting.prior_curves as prior_curves
print("after_curves", matplotlib.get_backend())
_ = prior_curves.plot_prior_overlay_figure
print("after_curve_attr", matplotlib.get_backend())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
    assert result.stdout.strip().splitlines() == [
        "before svg",
        "after_package svg",
        "after_plot_attr svg",
        "after_curves svg",
        "after_curve_attr svg",
    ]


def test_resolve_prior_curve_sets_treats_empty_labels_as_no_labels() -> None:
    curve_sets = resolve_prior_curve_sets(
        _checkpoint(),
        gene_names=["GeneA"],
        labels=[],
        include_global=True,
    )
    assert [curve.source for curve in curve_sets["GeneA"]] == ["global"]


def test_resolve_multi_checkpoint_prior_curve_sets_prefixes_checkpoint_names() -> None:
    curve_sets = resolve_multi_checkpoint_prior_curve_sets(
        [
            ("dataset-a", Path("/tmp/a.pkl"), _checkpoint_variant(source_h5ad_path="/tmp/a.h5ad")),
            ("dataset-b", Path("/tmp/b.pkl"), _checkpoint_variant(source_h5ad_path="/tmp/b.h5ad")),
        ],
        gene_names=["GeneA"],
        labels=["ctrl"],
        include_global=True,
    )
    assert [curve.source for curve in curve_sets["GeneA"]] == [
        "dataset-a/global",
        "dataset-a/label:ctrl",
        "dataset-b/global",
        "dataset-b/label:ctrl",
    ]


def test_resolve_multi_checkpoint_prior_curve_sets_drop_policy_skips_missing_labels() -> None:
    curve_sets = resolve_multi_checkpoint_prior_curve_sets(
        [
            ("dataset-a", Path("/tmp/a.pkl"), _checkpoint_variant(source_h5ad_path="/tmp/a.h5ad")),
            (
                "dataset-b",
                Path("/tmp/b.pkl"),
                _checkpoint_variant(source_h5ad_path="/tmp/b.h5ad", include_ctrl=False),
            ),
        ],
        gene_names=["GeneA"],
        labels=["ctrl"],
        include_global=False,
        missing_policy="drop",
    )
    assert [curve.source for curve in curve_sets["GeneA"]] == ["dataset-a/label:ctrl"]


def test_resolve_batch_grid_curve_sets_rejects_missing_gene() -> None:
    with pytest.raises(ValueError, match="MissingGene"):
        resolve_batch_grid_curve_sets(
            _checkpoint(),
            gene_names=["MissingGene"],
            entries=[LabelGridEntry(label="ctrl", batch="b1", perturbation="ctrl")],
        )


def test_resolve_batch_grid_curve_sets_rejects_duplicate_grid_cells() -> None:
    with pytest.raises(ValueError, match="duplicate batch-grid cell"):
        resolve_batch_grid_curve_sets(
            _checkpoint(),
            gene_names=["GeneA"],
            entries=[
                LabelGridEntry(label="ctrl", batch="b1", perturbation="shared"),
                LabelGridEntry(label="stim", batch="b1", perturbation="shared"),
            ],
        )


def test_plot_prior_facet_figure_uses_cdf_ylabel() -> None:
    curve_sets = resolve_prior_curve_sets(
        _checkpoint(),
        gene_names=["GeneA"],
        labels=["ctrl"],
        include_global=True,
    )
    fig = plot_prior_facet_figure(
        curve_sets,
        x_axis="support",
        mass_quantile=1.0,
        curve_mode="cdf",
    )
    try:
        assert fig.axes[0].get_ylabel() == "GeneA\nprior CDF"
        assert fig.axes[1].get_ylabel() == "prior CDF"
    finally:
        fig.clear()


def test_plot_prior_overlay_figure_rejects_empty_curve_sets() -> None:
    with pytest.raises(ValueError, match="curve_sets cannot be empty"):
        plot_prior_overlay_figure({}, x_axis="support", mass_quantile=1.0)


def test_plot_batch_grid_figure_requires_axes_lists() -> None:
    curve_sets, _, _ = resolve_batch_grid_curve_sets(
        _checkpoint(),
        gene_names=["GeneA"],
        entries=[LabelGridEntry(label="ctrl", batch="b1", perturbation="ctrl")],
    )
    with pytest.raises(ValueError, match="batches cannot be empty"):
        plot_batch_grid_figure(
            "GeneA",
            curve_sets["GeneA"],
            batches=[],
            perturbations=["ctrl"],
            x_axis="support",
            mass_quantile=1.0,
        )


def test_load_annotation_tables_rejects_duplicate_names(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("gene,label,value\nGeneA,ctrl,1\n", encoding="utf-8")
    csv_b.write_text("gene,label,value\nGeneA,stim,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate annotation table name"):
        load_annotation_tables([csv_a, csv_b], ["shared", "shared"])


def test_load_annotation_tables_supports_checkpoint_and_label_columns(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "annot.csv"
    csv_path.write_text(
        "gene,checkpoint,label,value\nGeneA,dataset-a,ctrl,1.5\n",
        encoding="utf-8",
    )
    tables = load_annotation_tables([csv_path], None)
    assert tables["annot"][("GeneA", "dataset-a/label:ctrl")] == "value=1.50"


def test_load_label_grid_entries_rejects_blank_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "label,batch,perturbation\nctrl,,control\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blank 'batch' value"):
        load_label_grid_entries(csv_path)

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from prism.cli.analyze.overlap_de import (
    overlap_de_command as analyze_overlap_de_command,
)
from prism.cli.plot.batch_grid import plot_batch_grid_command
from prism.cli.plot.label_summary import plot_label_summary_command
from prism.cli.plot.overlap import plot_overlap_command
from prism.cli.plot.priors import plot_priors_command
from prism.cli.checkpoint.overlap_de import overlap_de_command
from prism.model import ModelCheckpoint, PriorGrid, save_checkpoint


def _batched_prior(
    *,
    gene_names: list[str],
    p_grid_rows: list[list[float]],
    weights_rows: list[list[float]],
    S: float,
) -> PriorGrid:
    return PriorGrid(
        gene_names=gene_names,
        p_grid=np.asarray(p_grid_rows, dtype=np.float64),
        weights=np.asarray(weights_rows, dtype=np.float64),
        S=S,
    )


def _batched_rate_prior(
    *,
    gene_names: list[str],
    rate_grid_rows: list[list[float]],
    weights_rows: list[list[float]],
    S: float,
) -> PriorGrid:
    return PriorGrid(
        gene_names=gene_names,
        p_grid=np.asarray(rate_grid_rows, dtype=np.float64),
        weights=np.asarray(weights_rows, dtype=np.float64),
        S=S,
        grid_domain="rate",
        distribution="poisson",
    )


class PlotCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_checkpoint(self) -> Path:
        checkpoint_path = self.root / "toy_checkpoint.pkl"
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA", "GeneB"],
            priors=_batched_prior(
                gene_names=["GeneA", "GeneB"],
                p_grid_rows=[[0.25, 0.5], [0.1, 0.4]],
                weights_rows=[[0.4, 0.6], [0.7, 0.3]],
                S=4.0,
            ),
            scale=None,
            fit_config={},
            metadata={
                "fit_distribution": "binomial",
                "posterior_distribution": "binomial",
                "grid_domain": "p",
            },
            label_priors={
                "batch1_ctrl": _batched_prior(
                    gene_names=["GeneA"],
                    p_grid_rows=[[0.2, 0.4]],
                    weights_rows=[[0.5, 0.5]],
                    S=3.0,
                ),
                "batch1_stim": _batched_prior(
                    gene_names=["GeneA"],
                    p_grid_rows=[[0.3, 0.6]],
                    weights_rows=[[0.25, 0.75]],
                    S=3.0,
                ),
                "batch2_ctrl": _batched_prior(
                    gene_names=["GeneA"],
                    p_grid_rows=[[0.15, 0.45]],
                    weights_rows=[[0.6, 0.4]],
                    S=5.0,
                ),
                "batch2_stim": _batched_prior(
                    gene_names=["GeneA"],
                    p_grid_rows=[[0.35, 0.7]],
                    weights_rows=[[0.2, 0.8]],
                    S=5.0,
                ),
            },
            label_scales={},
        )
        save_checkpoint(checkpoint, checkpoint_path)
        return checkpoint_path

    def _write_poisson_label_checkpoint(self) -> Path:
        checkpoint_path = self.root / "poisson_label_checkpoint.pkl"
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA"],
            priors=_batched_rate_prior(
                gene_names=["GeneA"],
                rate_grid_rows=[[1.0, 2.0, 4.0]],
                weights_rows=[[0.2, 0.5, 0.3]],
                S=4.0,
            ),
            scale=None,
            fit_config={"likelihood": "poisson"},
            metadata={
                "fit_distribution": "poisson",
                "posterior_distribution": "poisson",
                "grid_domain": "rate",
            },
            label_priors={
                "ctrl": _batched_rate_prior(
                    gene_names=["GeneA"],
                    rate_grid_rows=[[1.0, 2.0, 4.0]],
                    weights_rows=[[0.5, 0.3, 0.2]],
                    S=4.0,
                ),
                "stim": _batched_rate_prior(
                    gene_names=["GeneA"],
                    rate_grid_rows=[[0.8, 1.6, 3.2]],
                    weights_rows=[[0.2, 0.5, 0.3]],
                    S=4.0,
                ),
            },
            label_scales={},
        )
        save_checkpoint(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_plot_priors_exports_true_p_axis(self) -> None:
        checkpoint_path = self._write_checkpoint()
        output_svg = self.root / "priors.svg"
        output_csv = self.root / "priors.csv"

        plot_priors_command(
            checkpoint_path=checkpoint_path,
            gene_names=["GeneA"],
            genes_path=None,
            top_n=None,
            output_path=output_svg,
            output_csv_path=output_csv,
            annot_csv_paths=None,
            annot_names=None,
            labels=None,
            labels_path=None,
            x_axis="p",
            mass_quantile=1.0,
            include_global=True,
            layout="overlay",
            show_subplot_labels=False,
        )

        self.assertTrue(output_svg.exists())
        df = pd.read_csv(output_csv)
        global_rows = df[df["source"] == "global"].reset_index(drop=True)
        self.assertTrue(np.allclose(global_rows["p"].to_numpy(), [0.25, 0.5]))
        self.assertTrue(np.allclose(global_rows["x"].to_numpy(), [0.25, 0.5]))
        self.assertTrue(np.allclose(global_rows["mu"].to_numpy(), [1.0, 2.0]))
        self.assertTrue(np.allclose(global_rows["S"].to_numpy(), [4.0, 4.0]))

    def test_plot_batch_grid_outputs_per_gene_figures_and_csv(self) -> None:
        checkpoint_path = self._write_checkpoint()
        output_dir = self.root / "batch_grid"
        output_csv = self.root / "batch_grid.csv"
        summary_csv = self.root / "batch_grid_summary.csv"
        mapping_csv = self.root / "label_grid.csv"
        mapping_csv.write_text(
            (
                "label,batch,perturbation\n"
                "batch1_ctrl,batch1,ctrl\n"
                "batch1_stim,batch1,stim\n"
                "batch2_ctrl,batch2,ctrl\n"
                "batch2_stim,batch2,stim\n"
            ),
            encoding="utf-8",
        )

        plot_batch_grid_command(
            checkpoint_path=checkpoint_path,
            gene_names=["GeneA"],
            genes_path=None,
            top_n=None,
            labels=None,
            labels_path=None,
            label_grid_csv_path=mapping_csv,
            output_dir=output_dir,
            output_csv_path=output_csv,
            summary_csv_path=summary_csv,
            x_axis="p",
            mass_quantile=1.0,
            image_format="svg",
            dpi=120,
            curve_mode="cdf",
            stat_fields=["mean_p"],
            show_axis_ticks=False,
        )

        self.assertTrue((output_dir / "GeneA.svg").exists())
        df = pd.read_csv(output_csv)
        summary_df = pd.read_csv(summary_csv)
        self.assertEqual(sorted(df["batch"].unique().tolist()), ["batch1", "batch2"])
        self.assertEqual(sorted(df["perturbation"].unique().tolist()), ["ctrl", "stim"])
        self.assertEqual(
            sorted(df["label"].dropna().unique().tolist()),
            [
                "batch1_ctrl",
                "batch1_stim",
                "batch2_ctrl",
                "batch2_stim",
            ],
        )
        self.assertEqual(df.shape[0], 8)
        self.assertEqual(sorted(summary_df["gene"].unique().tolist()), ["GeneA"])
        self.assertIn("mean_p", summary_df.columns)

    def test_plot_batch_grid_accepts_poisson_rate_grid_checkpoint(self) -> None:
        checkpoint_path = self._write_poisson_label_checkpoint()
        output_dir = self.root / "batch_grid_poisson"
        output_csv = self.root / "batch_grid_poisson.csv"
        summary_csv = self.root / "batch_grid_poisson_summary.csv"
        mapping_csv = self.root / "poisson_label_grid.csv"
        mapping_csv.write_text(
            (
                "label,batch,perturbation\n"
                "ctrl,batch1,ctrl\n"
                "stim,batch1,stim\n"
            ),
            encoding="utf-8",
        )

        plot_batch_grid_command(
            checkpoint_path=checkpoint_path,
            gene_names=["GeneA"],
            genes_path=None,
            top_n=None,
            labels=None,
            labels_path=None,
            label_grid_csv_path=mapping_csv,
            output_dir=output_dir,
            output_csv_path=output_csv,
            summary_csv_path=summary_csv,
            x_axis="rate",
            mass_quantile=1.0,
            image_format="svg",
            dpi=120,
            curve_mode="density",
            stat_fields=["mean_mu"],
            show_axis_ticks=False,
        )

        self.assertTrue((output_dir / "GeneA.svg").exists())
        df = pd.read_csv(output_csv)
        self.assertTrue(np.allclose(df["x"].to_numpy(), [1.0, 2.0, 4.0, 0.8, 1.6, 3.2]))
        self.assertEqual(sorted(df["label"].dropna().unique().tolist()), ["ctrl", "stim"])
        summary_df = pd.read_csv(summary_csv)
        self.assertIn("mean_mu", summary_df.columns)

    def test_plot_priors_supports_summary_stats_and_cdf_mode(self) -> None:
        checkpoint_path = self._write_checkpoint()
        output_svg = self.root / "priors_cdf.svg"
        output_summary = self.root / "priors_summary.csv"

        plot_priors_command(
            checkpoint_path=checkpoint_path,
            gene_names=["GeneA", "GeneB"],
            genes_path=None,
            top_n=None,
            output_path=output_svg,
            output_csv_path=None,
            summary_csv_path=output_summary,
            annot_csv_paths=None,
            annot_names=None,
            labels=None,
            labels_path=None,
            x_axis="mu",
            curve_mode="cdf",
            y_scale="linear",
            mass_quantile=1.0,
            include_global=True,
            layout="facet",
            show_subplot_labels=True,
            show_legend=False,
            stat_fields=["mean_p", "entropy"],
            panel_width=3.0,
            panel_height=2.4,
        )

        self.assertTrue(output_svg.exists())
        summary_df = pd.read_csv(output_summary)
        self.assertEqual(
            sorted(summary_df["gene"].unique().tolist()), ["GeneA", "GeneB"]
        )
        self.assertIn("entropy", summary_df.columns)
        self.assertIn("S", summary_df.columns)

    def test_plot_priors_auto_axis_uses_rate_for_rate_grid(self) -> None:
        checkpoint_path = self.root / "rate_checkpoint.pkl"
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA"],
            priors=_batched_rate_prior(
                gene_names=["GeneA"],
                rate_grid_rows=[[1.0, 2.0, 4.0]],
                weights_rows=[[0.2, 0.5, 0.3]],
                S=4.0,
            ),
            scale=None,
            fit_config={"likelihood": "poisson"},
            metadata={
                "fit_distribution": "poisson",
                "posterior_distribution": "poisson",
                "grid_domain": "rate",
            },
            label_priors={},
            label_scales={},
        )
        save_checkpoint(checkpoint, checkpoint_path)
        output_svg = self.root / "rate_priors.svg"
        output_csv = self.root / "rate_priors.csv"

        plot_priors_command(
            checkpoint_path=checkpoint_path,
            gene_names=["GeneA"],
            genes_path=None,
            top_n=None,
            output_path=output_svg,
            output_csv_path=output_csv,
            summary_csv_path=None,
            annot_csv_paths=None,
            annot_names=None,
            labels=None,
            labels_path=None,
            x_axis="auto",
            curve_mode="density",
            y_scale="linear",
            mass_quantile=1.0,
            include_global=True,
            layout="overlay",
            show_subplot_labels=False,
            show_legend=True,
            stat_fields=None,
            panel_width=0.0,
            panel_height=0.0,
        )

        self.assertTrue(output_svg.exists())
        df = pd.read_csv(output_csv)
        self.assertTrue(np.allclose(df["x"].to_numpy(), [1.0, 2.0, 4.0]))

    def test_plot_overlap_outputs_heatmap_and_metrics(self) -> None:
        checkpoint_path = self._write_checkpoint()
        output_svg = self.root / "overlap.svg"
        output_csv = self.root / "overlap.csv"

        plot_overlap_command(
            checkpoint_path=checkpoint_path,
            output_path=output_svg,
            output_csv_path=output_csv,
            control_label="batch1_ctrl",
            gene_names=["GeneA"],
            genes_path=None,
            top_k=None,
            labels=["batch1_stim", "batch2_stim"],
            labels_path=None,
            metric="best_scale",
            gene_order="metric",
            label_order="alpha",
            scale_min=0.5,
            scale_max=2.0,
            scale_grid_size=31,
            interp_points=256,
            annotate_cells=True,
            cmap=None,
            panel_width=1.0,
            panel_height=0.8,
        )

        self.assertTrue(output_svg.exists())
        df = pd.read_csv(output_csv)
        self.assertEqual(sorted(df["gene"].unique().tolist()), ["GeneA"])
        self.assertEqual(
            sorted(df["label"].unique().tolist()), ["batch1_stim", "batch2_stim"]
        )
        self.assertIn("best_scale", df.columns)
        self.assertIn("overlap", df.columns)

    def test_analyze_overlap_de_is_formal_entrypoint(self) -> None:
        checkpoint_path = self._write_checkpoint()
        output_csv = self.root / "overlap_de.csv"

        analyze_overlap_de_command(
            checkpoint_path=checkpoint_path,
            output_csv_path=output_csv,
            control_label="batch1_ctrl",
            gene_names=["GeneA"],
            gene_list_path=None,
            labels=["batch1_stim"],
            label_list_path=None,
            scale_min=0.5,
            scale_max=2.0,
            scale_grid_size=31,
            interp_points=256,
        )

        df = pd.read_csv(output_csv)
        self.assertEqual(df.shape[0], 1)
        self.assertEqual(df.iloc[0]["gene"], "GeneA")
        self.assertEqual(df.iloc[0]["label"], "batch1_stim")

    def test_checkpoint_overlap_de_remains_legacy_alias(self) -> None:
        checkpoint_path = self._write_checkpoint()
        output_csv = self.root / "overlap_de_alias.csv"

        overlap_de_command(
            checkpoint_path=checkpoint_path,
            output_csv_path=output_csv,
            control_label="batch1_ctrl",
            gene_names=["GeneA"],
            gene_list_path=None,
            labels=["batch1_stim"],
            label_list_path=None,
            scale_min=0.5,
            scale_max=2.0,
            scale_grid_size=31,
            interp_points=256,
        )

        df = pd.read_csv(output_csv)
        self.assertEqual(df.shape[0], 1)
        self.assertEqual(df.iloc[0]["gene"], "GeneA")
        self.assertEqual(df.iloc[0]["label"], "batch1_stim")

    def test_plot_overlap_rejects_poisson_rate_grid_checkpoint(self) -> None:
        checkpoint_path = self._write_poisson_label_checkpoint()
        with self.assertRaisesRegex(ValueError, "does not support checkpoint grid_domain 'rate'"):
            plot_overlap_command(
                checkpoint_path=checkpoint_path,
                output_path=self.root / "overlap_poisson.svg",
                output_csv_path=self.root / "overlap_poisson.csv",
                control_label="ctrl",
                gene_names=["GeneA"],
                genes_path=None,
                top_k=None,
                labels=["stim"],
                labels_path=None,
                metric="overlap",
                gene_order="input",
                label_order="input",
                scale_min=0.5,
                scale_max=2.0,
                scale_grid_size=31,
                interp_points=256,
                annotate_cells=False,
                cmap=None,
                panel_width=1.0,
                panel_height=0.8,
            )

    def test_plot_label_summary_rejects_poisson_rate_grid_checkpoint(self) -> None:
        checkpoint_path = self._write_poisson_label_checkpoint()
        with self.assertRaisesRegex(ValueError, "does not support checkpoint grid_domain 'rate'"):
            plot_label_summary_command(
                checkpoint_path=checkpoint_path,
                output_path=self.root / "label_summary_poisson.svg",
                gene_names=["GeneA"],
                max_genes=10,
                metric="jsd",
                figsize_w=6.0,
                figsize_h=6.0,
                palette=None,
            )

    def test_analyze_overlap_de_rejects_poisson_rate_grid_checkpoint(self) -> None:
        checkpoint_path = self._write_poisson_label_checkpoint()
        with self.assertRaisesRegex(ValueError, "does not support checkpoint grid_domain 'rate'"):
            analyze_overlap_de_command(
                checkpoint_path=checkpoint_path,
                output_csv_path=self.root / "overlap_de_poisson.csv",
                control_label="ctrl",
                gene_names=["GeneA"],
                gene_list_path=None,
                labels=["stim"],
                label_list_path=None,
                scale_min=0.5,
                scale_max=2.0,
                scale_grid_size=31,
                interp_points=256,
            )


if __name__ == "__main__":
    unittest.main()

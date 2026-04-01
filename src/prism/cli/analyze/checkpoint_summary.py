"""Analyze checkpoint-summary: export checkpoint metadata and statistics as CSV/JSON."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import typer

from prism.model import load_checkpoint
from prism.cli.checkpoint_validation import resolve_cli_checkpoint_distribution

from .common import console, print_analysis_plan, print_analysis_summary


def checkpoint_summary_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    output_csv_path: Path | None = typer.Option(
        None, "--output-csv", help="Optional output CSV path for per-gene statistics."
    ),
    output_json_path: Path | None = typer.Option(
        None, "--output-json", help="Optional output JSON path for checkpoint metadata."
    ),
) -> int:
    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if output_csv_path is not None:
        output_csv_path = output_csv_path.expanduser().resolve()
    if output_json_path is not None:
        output_json_path = output_json_path.expanduser().resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    metadata = resolve_cli_checkpoint_distribution(
        checkpoint,
        command_name="prism analyze checkpoint-summary",
    )

    n_genes = len(checkpoint.gene_names)
    has_global_priors = checkpoint.priors is not None
    n_label_priors = len(checkpoint.label_priors)
    S = None if checkpoint.scale is None else float(checkpoint.scale.S)
    mean_ref = (
        None
        if checkpoint.scale is None
        else float(checkpoint.scale.mean_reference_count)
    )

    print_analysis_plan(
        title="Checkpoint Summary",
        checkpoint_path=checkpoint_path,
        n_genes=n_genes,
        has_global_priors=has_global_priors,
        n_label_priors=n_label_priors,
        fit_distribution=metadata["fit_distribution"],
        posterior_distribution=metadata["posterior_distribution"],
        grid_domain=metadata["grid_domain"],
        distribution_resolution=metadata.get("distribution_resolution", ""),
        S=S,
        mean_reference_count=mean_ref,
        S_source=metadata.get("S_source", ""),
        fit_mode=metadata.get("fit_mode", ""),
        fit_method=metadata.get("fit_method", ""),
        source_h5ad_path=metadata.get("source_h5ad_path", ""),
        layer=metadata.get("layer", ""),
    )

    if output_csv_path is not None:
        _write_gene_csv(checkpoint, output_csv_path)
        console.print(f"[bold green]Saved[/bold green] {output_csv_path}")

    if output_json_path is not None:
        _write_metadata_json(checkpoint, output_json_path)
        console.print(f"[bold green]Saved[/bold green] {output_json_path}")

    print_analysis_summary(elapsed_sec=perf_counter() - start_time)
    return 0


def _write_gene_csv(checkpoint, path: Path) -> None:
    import csv

    priors = checkpoint.priors
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "gene_name",
                "prior_entropy",
                "prior_mean_p",
                "prior_mode_p",
            ]
        )
        if priors is None:
            return
        batched = priors.batched()
        weights = np.asarray(batched.weights, dtype=np.float64)
        p_grid = np.asarray(batched.p_grid, dtype=np.float64)
        for idx, gene_name in enumerate(batched.gene_names):
            w = weights[idx]
            p = p_grid[idx]
            w_safe = np.clip(w, 1e-12, None)
            entropy = float(-np.sum(w_safe * np.log(w_safe)))
            mean_p = float(np.sum(w * p))
            mode_idx = int(np.argmax(w))
            mode_p = float(p[mode_idx])
            writer.writerow(
                [gene_name, f"{entropy:.6f}", f"{mean_p:.6f}", f"{mode_p:.6f}"]
            )


def _write_metadata_json(checkpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_genes": len(checkpoint.gene_names),
        "has_global_priors": checkpoint.priors is not None,
        "n_label_priors": len(checkpoint.label_priors),
        "label_names": sorted(checkpoint.label_priors),
        "S": None if checkpoint.scale is None else float(checkpoint.scale.S),
        "mean_reference_count": None
        if checkpoint.scale is None
        else float(checkpoint.scale.mean_reference_count),
        "fit_config": dict(checkpoint.fit_config),
        "metadata": {
            k: v
            for k, v in checkpoint.metadata.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")


__all__ = ["checkpoint_summary_command"]

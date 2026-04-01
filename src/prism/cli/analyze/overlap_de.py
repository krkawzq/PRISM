from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from prism.cli.common import ensure_mutually_exclusive, print_key_value_table, print_saved_path
from prism.io import read_gene_list, read_string_list
from prism.model import load_checkpoint
from prism.plotting import compute_overlap_dataframe

console = Console()


def _resolve_name_list(
    explicit: list[str] | None,
    file_path: Path | None,
    *,
    default: list[str],
    loader,
) -> list[str]:
    values: list[str] = []
    if explicit:
        values.extend(explicit)
    if file_path is not None:
        values.extend(loader(file_path.expanduser().resolve()))
    resolved = list(dict.fromkeys(values))
    return default if not resolved else resolved


def overlap_de_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    output_csv_path: Path = typer.Option(..., "--output-csv", help="Output CSV path."),
    control_label: str = typer.Option(..., help="Control label used as reference."),
    gene_names: list[str] | None = typer.Option(
        None,
        "--gene",
        help="Optional repeatable genes. Defaults to all checkpoint genes.",
    ),
    gene_list_path: Path | None = typer.Option(
        None,
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON gene list.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable perturbation labels. Defaults to all non-control labels.",
    ),
    label_list_path: Path | None = typer.Option(
        None,
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON file listing perturbation labels.",
    ),
    scale_min: float = typer.Option(0.25, min=1e-12, help="Minimum scale factor."),
    scale_max: float = typer.Option(4.0, min=1e-12, help="Maximum scale factor."),
    scale_grid_size: int = typer.Option(
        201, min=3, help="Number of log-scale factors to search."
    ),
    interp_points: int = typer.Option(
        2048, min=128, help="Interpolation points for overlap metrics."
    ),
) -> int:
    if scale_min > scale_max:
        raise ValueError("--scale-min must be <= --scale-max")
    ensure_mutually_exclusive(("--gene", gene_names), ("--gene-list", gene_list_path))
    ensure_mutually_exclusive(("--label", labels), ("--label-list", label_list_path))
    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    if not checkpoint.label_priors:
        raise ValueError("checkpoint has no label priors")
    if control_label not in checkpoint.label_priors:
        raise ValueError(
            f"control label {control_label!r} not found in checkpoint label priors"
        )

    selected_genes = _resolve_name_list(
        gene_names,
        gene_list_path,
        default=list(checkpoint.gene_names),
        loader=read_gene_list,
    )
    available_labels = sorted(
        label for label in checkpoint.label_priors if label != control_label
    )
    selected_labels = _resolve_name_list(
        labels,
        label_list_path,
        default=available_labels,
        loader=read_string_list,
    )
    unknown_labels = [
        label for label in selected_labels if label not in checkpoint.label_priors
    ]
    if unknown_labels:
        raise ValueError(f"unknown labels: {unknown_labels[:10]}")

    control_priors = checkpoint.label_priors[control_label]
    missing_genes = [
        gene for gene in selected_genes if gene not in control_priors.gene_names
    ]
    if missing_genes:
        console.print(
            f"[yellow]Skipped[/yellow] {len(missing_genes)} genes missing from control priors"
        )
    selected_genes = [
        gene for gene in selected_genes if gene in control_priors.gene_names
    ]
    if not selected_genes:
        raise ValueError(
            "no selected genes remain after intersecting with control priors"
        )

    df = compute_overlap_dataframe(
        checkpoint,
        control_label=control_label,
        gene_names=selected_genes,
        labels=selected_labels,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_grid_size=scale_grid_size,
        interp_points=interp_points,
    )
    output_csv_path = output_csv_path.expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print_key_value_table(
        console,
        title="Overlap-DE",
        values={
            "Genes": df["gene"].nunique(),
            "Labels": df["label"].nunique(),
            "Output": output_csv_path,
        },
    )
    print_saved_path(console, output_csv_path)
    return 0


__all__ = ["overlap_de_command"]

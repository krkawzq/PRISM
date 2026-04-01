from __future__ import annotations

from pathlib import Path

import typer

from prism.model import (
    ModelCheckpoint,
    PriorGrid,
    ScaleMetadata,
    load_checkpoint,
    save_checkpoint,
)
from prism.cli.checkpoint_validation import resolve_cli_checkpoint_distribution

from .common import (
    checkpoint_gene_names,
    merge_prior_scope,
    print_merge_summary,
    safe_string_list,
    validate_shared_metadata,
)


def merge_checkpoints_command(
    checkpoint_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input checkpoint paths."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Merged checkpoint path."
    ),
    allow_partial: bool = typer.Option(
        False,
        help="Allow merged checkpoints to miss genes from the declared requested set.",
    ),
) -> int:
    if len(checkpoint_paths) < 2:
        raise ValueError("merge requires at least two checkpoints")
    checkpoints = [
        load_checkpoint(path.expanduser().resolve()) for path in checkpoint_paths
    ]
    for checkpoint in checkpoints:
        resolve_cli_checkpoint_distribution(
            checkpoint,
            command_name="prism checkpoint merge",
        )
    merged = _merge_checkpoints(
        checkpoints,
        [path.expanduser().resolve() for path in checkpoint_paths],
        allow_partial=allow_partial,
    )
    output_path = output_path.expanduser().resolve()
    save_checkpoint(merged, output_path)
    print_merge_summary(output_path, checkpoints, merged)
    return 0


def _merge_checkpoints(
    checkpoints: list[ModelCheckpoint],
    source_paths: list[Path],
    *,
    allow_partial: bool,
) -> ModelCheckpoint:
    first = checkpoints[0]
    validate_shared_metadata(checkpoints, source_paths)
    global_priors, global_scale = merge_prior_scope(
        [checkpoint.priors for checkpoint in checkpoints],
        [checkpoint.scale for checkpoint in checkpoints],
        requested_gene_names=safe_string_list(
            first.metadata.get("requested_fit_gene_names")
        ),
        allow_partial=allow_partial,
        scope_name="global",
    )
    label_names = sorted(
        {label for checkpoint in checkpoints for label in checkpoint.label_priors}
    )
    merged_label_priors: dict[str, PriorGrid] = {}
    merged_label_scales: dict[str, ScaleMetadata] = {}
    for label in label_names:
        priors, scale = merge_prior_scope(
            [checkpoint.label_priors.get(label) for checkpoint in checkpoints],
            [checkpoint.label_scales.get(label) for checkpoint in checkpoints],
            requested_gene_names=safe_string_list(
                first.metadata.get("requested_fit_gene_names")
            ),
            allow_partial=allow_partial,
            scope_name=f"label:{label}",
        )
        if priors is not None and scale is not None:
            merged_label_priors[label] = priors
            merged_label_scales[label] = scale
    metadata = dict(first.metadata)
    resolved_gene_names = checkpoint_gene_names(
        global_priors,
        merged_label_priors,
        safe_string_list(first.metadata.get("requested_fit_gene_names")),
    )
    metadata.update(
        {
            "shard_rank": 0,
            "shard_world_size": 1,
            "shard_gene_names": list(resolved_gene_names),
            "source_checkpoints": [str(path) for path in source_paths],
            "merge_allow_partial": bool(allow_partial),
            "merged_label_priors": sorted(merged_label_priors),
        }
    )
    return ModelCheckpoint(
        gene_names=resolved_gene_names,
        priors=global_priors,
        scale=global_scale,
        fit_config=dict(first.fit_config),
        metadata=metadata,
        label_priors=merged_label_priors,
        label_scales=merged_label_scales,
    )


__all__ = ["merge_checkpoints_command"]

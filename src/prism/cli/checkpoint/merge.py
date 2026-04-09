from __future__ import annotations

from pathlib import Path

import typer

from prism.cli.common import resolve_bool
from prism.model import ModelCheckpoint, load_checkpoint, save_checkpoint

from .common import (
    checkpoint_gene_names,
    merge_prior_scope,
    print_merge_summary,
    safe_string_list,
    validate_shared_metadata,
)


def _merge_checkpoints(
    checkpoints: list[ModelCheckpoint], source_paths: list[Path], *, allow_partial: bool
) -> ModelCheckpoint:
    first = checkpoints[0]
    validate_shared_metadata(checkpoints, source_paths)
    requested_gene_names = safe_string_list(
        first.metadata.get("requested_fit_gene_names")
    )
    global_prior, global_scale = merge_prior_scope(
        [checkpoint.prior for checkpoint in checkpoints],
        [checkpoint.scale_metadata for checkpoint in checkpoints],
        requested_gene_names=requested_gene_names,
        allow_partial=allow_partial,
        scope_name="global",
    )
    label_names = sorted(
        {label for checkpoint in checkpoints for label in checkpoint.label_priors}
    )
    merged_label_priors = {}
    merged_label_scale_metadata = {}
    for label in label_names:
        prior, scale = merge_prior_scope(
            [checkpoint.label_priors.get(label) for checkpoint in checkpoints],
            [checkpoint.label_scale_metadata.get(label) for checkpoint in checkpoints],
            requested_gene_names=requested_gene_names,
            allow_partial=allow_partial,
            scope_name=f"label:{label}",
        )
        if prior is not None:
            merged_label_priors[label] = prior
        if scale is not None:
            merged_label_scale_metadata[label] = scale
    metadata = dict(first.metadata)
    resolved_gene_names = checkpoint_gene_names(
        global_prior, merged_label_priors, requested_gene_names
    )
    metadata.update(
        {
            "source_checkpoints": [str(path) for path in source_paths],
            "merge_allow_partial": bool(allow_partial),
            "merged_label_priors": sorted(merged_label_priors),
        }
    )
    return ModelCheckpoint(
        gene_names=resolved_gene_names,
        prior=global_prior,
        scale_metadata=global_scale,
        fit_config=dict(first.fit_config),
        metadata=metadata,
        label_priors=merged_label_priors,
        label_scale_metadata=merged_label_scale_metadata,
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
    allow_partial = resolve_bool(allow_partial)
    if len(checkpoint_paths) < 2:
        raise ValueError("merge requires at least two checkpoints")
    checkpoints = [
        load_checkpoint(path.expanduser().resolve()) for path in checkpoint_paths
    ]
    merged = _merge_checkpoints(
        checkpoints,
        [path.expanduser().resolve() for path in checkpoint_paths],
        allow_partial=allow_partial,
    )
    output_path = output_path.expanduser().resolve()
    save_checkpoint(merged, output_path)
    print_merge_summary(output_path, checkpoints, merged)
    return 0


__all__ = ["merge_checkpoints_command"]

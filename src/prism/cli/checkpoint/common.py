from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.cli.common import console, print_key_value_table, print_saved_path
from prism.model import (
    ModelCheckpoint,
    PriorGrid,
    ScaleMetadata,
    make_distribution_grid,
)


def safe_string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return list(value)


def checkpoint_gene_names(
    global_prior: PriorGrid | None,
    label_priors: dict[str, PriorGrid],
    fallback_gene_names: list[str],
) -> list[str]:
    if global_prior is not None:
        return list(global_prior.gene_names)
    if label_priors:
        return list(next(iter(label_priors.values())).gene_names)
    return list(fallback_gene_names)


def validate_shared_metadata(
    checkpoints: list[ModelCheckpoint], paths: list[Path]
) -> None:
    if len(checkpoints) != len(paths):
        raise ValueError("checkpoint/path lists must have the same length")
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    first = checkpoints[0]
    for path, checkpoint in zip(paths[1:], checkpoints[1:], strict=True):
        if checkpoint.fit_config != first.fit_config:
            raise ValueError(f"{path} has a different fit configuration")
        for key in (
            "source_h5ad_path",
            "layer",
            "reference_gene_names",
            "requested_fit_gene_names",
            "fit_mode",
            "label_key",
            "fit_distribution",
            "posterior_distribution",
            "support_domain",
        ):
            if checkpoint.metadata.get(key) != first.metadata.get(key):
                raise ValueError(f"{path} has different metadata for {key!r}")


def merge_prior_scope(
    priors_list: list[PriorGrid | None],
    scales: list[ScaleMetadata | None],
    *,
    requested_gene_names: list[str],
    allow_partial: bool,
    scope_name: str,
) -> tuple[PriorGrid | None, ScaleMetadata | None]:
    present = [
        (prior, scale)
        for prior, scale in zip(priors_list, scales, strict=True)
        if prior is not None
    ]
    if not present:
        return None, None
    first_prior, first_scale = present[0]
    assert first_prior is not None
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prior, scale in present:
        gene_specific = prior.as_gene_specific()
        if prior.scale != first_prior.scale:
            raise ValueError(f"{scope_name} has inconsistent scale across checkpoints")
        if prior.support_domain != first_prior.support_domain:
            raise ValueError(
                f"{scope_name} has inconsistent support_domain across checkpoints"
            )
        if prior.distribution_name != first_prior.distribution_name:
            raise ValueError(
                f"{scope_name} has inconsistent distribution across checkpoints"
            )
        if (scale is None) != (first_scale is None):
            raise ValueError(
                f"{scope_name} has inconsistent scale metadata presence across checkpoints"
            )
        if scale is not None and first_scale is not None:
            if scale.scale != first_scale.scale:
                raise ValueError(
                    f"{scope_name} has inconsistent scale metadata.scale across checkpoints"
                )
            if not np.isclose(
                scale.mean_reference_count,
                first_scale.mean_reference_count,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{scope_name} has inconsistent mean_reference_count across checkpoints"
                )
        for idx, gene_name in enumerate(gene_specific.gene_names):
            if gene_name in rows:
                raise ValueError(
                    f"duplicate fitted gene across checkpoints in {scope_name}: {gene_name}"
                )
            rows[gene_name] = (
                np.asarray(gene_specific.support[idx], dtype=np.float64),
                np.asarray(gene_specific.prior_probabilities[idx], dtype=np.float64),
            )
    ordered_gene_names = (
        requested_gene_names if requested_gene_names else list(rows.keys())
    )
    missing = [name for name in ordered_gene_names if name not in rows]
    if missing and not allow_partial:
        raise ValueError(
            f"{scope_name} is missing {len(missing)} genes, e.g. {missing[:5]}"
        )
    merged_gene_names = [name for name in ordered_gene_names if name in rows]
    if not merged_gene_names:
        return None, None
    support = np.stack([rows[name][0] for name in merged_gene_names], axis=0)
    probabilities = np.stack([rows[name][1] for name in merged_gene_names], axis=0)
    merged_prior = PriorGrid(
        gene_names=list(merged_gene_names),
        distribution=make_distribution_grid(
            first_prior.distribution_name, support=support, probabilities=probabilities
        ),
        scale=float(first_prior.scale),
    )
    return merged_prior, first_scale


def print_merge_summary(
    output_path: Path, checkpoints: list[ModelCheckpoint], merged: ModelCheckpoint
) -> None:
    print_key_value_table(
        console,
        title="Merged Checkpoint",
        values={
            "Input checkpoints": len(checkpoints),
            "Input genes": sum(
                len(checkpoint.gene_names) for checkpoint in checkpoints
            ),
            "Merged genes": len(merged.gene_names),
            "Label priors": len(merged.label_priors),
        },
    )
    print_saved_path(console, output_path)

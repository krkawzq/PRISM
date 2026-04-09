from __future__ import annotations

from pathlib import Path
from typing import cast

from prism.cli.common import normalize_choice, unwrap_typer_value
from prism.io import read_gene_list, read_string_list
from prism.model import ModelCheckpoint
from prism.plotting import (
    LabelGridEntry,
    load_label_grid_entries,
    parse_batch_grid_entries,
)


def _dedupe_names(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = str(value).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return ordered


def resolve_optional_list(value: list[str] | None | object) -> list[str] | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return _dedupe_names(list(cast(list[str], resolved)))


def ensure_mutually_exclusive(
    first: tuple[str, object | None],
    second: tuple[str, object | None],
) -> None:
    first_name, first_value = first
    second_name, second_value = second
    if first_value is not None and second_value is not None:
        raise ValueError(f"{first_name} and {second_name} are mutually exclusive")


def resolve_gene_names(
    *,
    gene_names: list[str] | None,
    genes_path: Path | None,
    top_n: int | None,
) -> list[str]:
    ensure_mutually_exclusive(("--gene", gene_names), ("--genes", genes_path))
    if genes_path is not None:
        values = read_gene_list(genes_path)
        if not values:
            raise ValueError(f"gene list is empty: {genes_path}")
        limit = len(values) if top_n is None else min(top_n, len(values))
        return values[:limit]
    if not gene_names:
        raise ValueError("provide either --gene or --genes")
    return _dedupe_names(gene_names)


def resolve_label_names(
    *,
    labels: list[str] | None,
    labels_path: Path | None,
    default: list[str] | None = None,
) -> list[str] | None:
    ensure_mutually_exclusive(("--label", labels), ("--labels", labels_path))
    if labels_path is not None:
        values = read_string_list(labels_path)
        if not values:
            raise ValueError(f"label list is empty: {labels_path}")
        return values
    if labels:
        return _dedupe_names(labels)
    if default is None:
        return None
    return _dedupe_names(default)


def resolve_label_entries(
    *,
    checkpoint: ModelCheckpoint,
    labels: list[str] | None,
    labels_path: Path | None,
    label_grid_csv_path: Path | None,
) -> list[LabelGridEntry]:
    if label_grid_csv_path is not None:
        entries = load_label_grid_entries(label_grid_csv_path)
        requested = resolve_label_names(labels=labels, labels_path=labels_path)
        if requested is None:
            return entries
        requested_set = set(requested)
        filtered = [entry for entry in entries if entry.label in requested_set]
        if not filtered:
            raise ValueError("no label-grid rows remain after filtering requested labels")
        return filtered
    resolved_labels = resolve_label_names(
        labels=labels,
        labels_path=labels_path,
        default=list(checkpoint.available_labels),
    )
    assert resolved_labels is not None
    return parse_batch_grid_entries(resolved_labels)


def normalize_layout(value: str) -> str:
    return normalize_choice(
        value,
        supported=("overlay", "facet"),
        option_name="--layout",
    )


def normalize_image_format(value: str) -> str:
    return normalize_choice(
        value,
        supported=("svg", "pdf", "eps"),
        option_name="--image-format",
    )


def normalize_distribution_plot_type(value: str) -> str:
    return normalize_choice(
        value,
        supported=("violin", "box", "hist"),
        option_name="--plot-type",
    )


def normalize_label_summary_metric(value: str) -> str:
    return normalize_choice(
        value,
        supported=("jsd", "overlap"),
        option_name="--metric",
    )


__all__ = [
    "ensure_mutually_exclusive",
    "normalize_distribution_plot_type",
    "normalize_image_format",
    "normalize_label_summary_metric",
    "normalize_layout",
    "resolve_gene_names",
    "resolve_label_entries",
    "resolve_label_names",
    "resolve_optional_list",
]

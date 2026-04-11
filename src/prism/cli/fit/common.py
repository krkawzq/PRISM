from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from prism.cli.common import (
    console,
    normalize_choice,
    print_elapsed,
    print_key_value_table,
    print_saved_path,
)
from prism.io import (
    compute_reference_counts,
    read_gene_list,
    read_string_list,
    select_matrix,
    slice_gene_matrix,
)
from prism.model import ModelCheckpoint, PriorGrid, summarize_reference_scale


@dataclass(frozen=True, slots=True)
class FitTask:
    scope_kind: str
    scope_name: str
    cell_indices: np.ndarray
    label_value: str | None = None


@dataclass(frozen=True, slots=True)
class GeneBatch:
    gene_names: list[str]
    gene_positions: list[int]


@dataclass(frozen=True, slots=True)
class ScaleResolution:
    scale: float
    default_scale: float
    scale_source: str
    n_positive_reference_cells: int


def parse_shard(value: str) -> tuple[int, int]:
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError("shard must be formatted as rank/world, e.g. 0/4")
    rank = int(parts[0])
    world_size = int(parts[1])
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid shard specification: {value}")
    return rank, world_size


def resolve_fit_mode(value: str) -> str:
    return normalize_choice(
        value,
        supported=("global", "by-label", "both"),
        option_name="--fit-mode",
    )


def resolve_label_groups(
    *,
    adata: ad.AnnData,
    label_key: str | None,
    label_values: list[str] | None,
    fit_mode: str,
) -> list[tuple[str, np.ndarray]]:
    if fit_mode == "global":
        return []
    if label_key is None:
        raise ValueError("--label-key is required when fit_mode is by-label or both")
    if label_key not in adata.obs.columns:
        raise KeyError(f"obs column {label_key!r} does not exist")
    labels = np.asarray(adata.obs[label_key].astype(str)).reshape(-1)
    chosen_values = (
        list(dict.fromkeys(label_values))
        if label_values
        else sorted(np.unique(labels).tolist())
    )
    groups: list[tuple[str, np.ndarray]] = []
    for value in chosen_values:
        indices = np.flatnonzero(labels == value).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"label value {value!r} has no cells")
        groups.append((value, indices))
    return groups


def resolve_requested_labels(
    label_values: list[str] | None,
    label_list_path: Path | None,
) -> list[str] | None:
    if not label_values and label_list_path is None:
        return None
    merged: list[str] = []
    if label_values:
        merged.extend(label_values)
    if label_list_path is not None:
        merged.extend(read_string_list(label_list_path))
    return list(dict.fromkeys(merged))


def build_fit_tasks(
    fit_mode: str, label_groups: list[tuple[str, np.ndarray]], *, n_cells: int
) -> list[FitTask]:
    tasks: list[FitTask] = []
    if fit_mode in {"global", "both"}:
        tasks.append(
            FitTask(
                scope_kind="global",
                scope_name="global",
                cell_indices=np.arange(n_cells, dtype=np.int64),
            )
        )
    if fit_mode in {"by-label", "both"}:
        tasks.extend(
            FitTask(
                scope_kind="label",
                scope_name=f"label:{label}",
                cell_indices=indices,
                label_value=label,
            )
            for label, indices in label_groups
        )
    return tasks


def sample_indices(
    cell_indices: np.ndarray, *, n_samples: int | None, seed: int
) -> np.ndarray:
    if cell_indices.size == 0:
        return cell_indices
    if n_samples is None or cell_indices.size <= n_samples:
        return np.asarray(cell_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cell_indices, size=n_samples, replace=False)
    return np.sort(np.asarray(chosen, dtype=np.int64))


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


def resolve_gene_list(
    path: Path, gene_to_idx: dict[str, int]
) -> tuple[list[str], list[str]]:
    requested = read_gene_list(path)
    ordered: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        if name in gene_to_idx:
            ordered.append(name)
        else:
            missing.append(name)
    return ordered, missing


def resolve_fit_gene_list(
    fit_genes_path: Path | None,
    dataset_gene_names: list[str],
    gene_to_idx: dict[str, int],
) -> tuple[list[str], list[str]]:
    if fit_genes_path is None:
        return list(dataset_gene_names), []
    return resolve_gene_list(fit_genes_path, gene_to_idx)


def shard_gene_names(gene_names: list[str], *, rank: int, world_size: int) -> list[str]:
    return [name for idx, name in enumerate(gene_names) if idx % world_size == rank]


def build_gene_batches(
    gene_names: list[str], gene_to_idx: dict[str, int], *, batch_size: int
) -> list[GeneBatch]:
    return [
        GeneBatch(
            gene_names=gene_names[start : start + batch_size],
            gene_positions=[
                gene_to_idx[name] for name in gene_names[start : start + batch_size]
            ],
        )
        for start in range(0, len(gene_names), batch_size)
    ]


def resolve_scale(
    reference_counts: np.ndarray, *, scale: float | None
) -> ScaleResolution:
    values = np.asarray(reference_counts, dtype=np.float64).reshape(-1)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("reference genes produced no positive per-cell counts")
    diagnostic = summarize_reference_scale(positive)
    resolved_scale = (
        float(diagnostic.suggested_scale) if scale is None else float(scale)
    )
    return ScaleResolution(
        scale=resolved_scale,
        default_scale=float(diagnostic.suggested_scale),
        scale_source="default:mean_reference_count" if scale is None else "user",
        n_positive_reference_cells=int(positive.size),
    )


def slice_gene_counts(
    matrix,
    positions: list[int],
    *,
    cell_indices: np.ndarray | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    return slice_gene_matrix(matrix, positions, cell_indices=cell_indices, dtype=dtype)


def print_fit_plan(**values: Any) -> None:
    print_key_value_table(console, title="Fit Plan", values=values)


def print_fit_summary(
    *, output_path: Path, elapsed_sec: float, checkpoint: ModelCheckpoint
) -> None:
    s_value = (
        "-"
        if checkpoint.scale_metadata is None
        else f"{checkpoint.scale_metadata.scale:.4f}"
    )
    mean_ref = (
        "-"
        if checkpoint.scale_metadata is None
        else f"{checkpoint.scale_metadata.mean_reference_count:.4f}"
    )
    print_key_value_table(
        console,
        title="Fit Summary",
        values={
            "Genes": len(checkpoint.gene_names),
            "Global prior": checkpoint.prior is not None,
            "Label priors": len(checkpoint.label_priors),
            "S": s_value,
            "Mean ref count": mean_ref,
        },
    )
    print_saved_path(console, output_path)
    print_elapsed(console, elapsed_sec)


__all__ = [
    "FitTask",
    "GeneBatch",
    "ScaleResolution",
    "build_fit_tasks",
    "build_gene_batches",
    "checkpoint_gene_names",
    "compute_reference_counts",
    "console",
    "parse_shard",
    "print_fit_plan",
    "print_fit_summary",
    "resolve_fit_gene_list",
    "resolve_fit_mode",
    "resolve_gene_list",
    "resolve_label_groups",
    "resolve_requested_labels",
    "resolve_scale",
    "sample_indices",
    "select_matrix",
    "shard_gene_names",
    "slice_gene_counts",
]

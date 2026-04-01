from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from rich.console import Console

from prism.cli.common import normalize_choice, print_elapsed, print_key_value_table, print_saved_path
from prism.io import (
    compute_reference_counts as compute_reference_counts_shared,
    ensure_dense_matrix as ensure_dense_matrix_shared,
    read_gene_list as read_gene_list_shared,
    select_matrix as select_matrix_shared,
    slice_gene_matrix as slice_gene_matrix_shared,
)
from prism.model import ModelCheckpoint, PriorGrid

console = Console()


@dataclass(frozen=True, slots=True)
class FitTask:
    scope_kind: str
    scope_name: str
    cell_indices: np.ndarray
    label_value: str | None = None


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


def build_fit_tasks(
    fit_mode: str, label_groups: list[tuple[str, np.ndarray]], *, n_cells: int
) -> list[FitTask]:
    tasks: list[FitTask] = []
    if fit_mode in {"global", "both"}:
        if label_groups:
            all_indices = np.unique(
                np.concatenate([indices for _, indices in label_groups])
            )
        else:
            all_indices = np.arange(n_cells, dtype=np.int64)
        tasks.append(
            FitTask(scope_kind="global", scope_name="global", cell_indices=all_indices)
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
    global_priors: PriorGrid | None,
    label_priors: dict[str, PriorGrid],
    fallback_gene_names: list[str],
) -> list[str]:
    if global_priors is not None:
        return list(global_priors.gene_names)
    if label_priors:
        return list(next(iter(label_priors.values())).gene_names)
    return list(fallback_gene_names)


def read_gene_list(path: Path) -> list[str]:
    return read_gene_list_shared(path)


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


def select_matrix(adata: ad.AnnData, layer: str | None):
    return select_matrix_shared(adata, layer)


def ensure_dense_matrix(matrix) -> np.ndarray:
    return ensure_dense_matrix_shared(matrix, dtype=np.float32)


def slice_gene_counts(
    matrix, positions: list[int], *, cell_indices: np.ndarray | None = None
) -> np.ndarray:
    return slice_gene_matrix_shared(
        matrix,
        positions,
        cell_indices=cell_indices,
        dtype=np.float64,
    )


def compute_reference_counts(
    matrix, positions: list[int], *, cell_indices: np.ndarray | None = None
) -> np.ndarray:
    return compute_reference_counts_shared(
        matrix,
        positions,
        cell_indices=cell_indices,
        dtype=np.float64,
    )


def print_fit_plan(**values: Any) -> None:
    print_key_value_table(console, title="Fit Plan", values=values)


def print_fit_summary(
    *, output_path: Path, elapsed_sec: float, checkpoint: ModelCheckpoint
) -> None:
    s_value = "-" if checkpoint.scale is None else f"{checkpoint.scale.S:.4f}"
    mean_ref = (
        "-"
        if checkpoint.scale is None
        else f"{checkpoint.scale.mean_reference_count:.4f}"
    )
    print_key_value_table(
        console,
        title="Fit Summary",
        values={
            "Genes": len(checkpoint.gene_names),
            "S": s_value,
            "Mean ref count": mean_ref,
        },
    )
    print_saved_path(console, output_path)
    print_elapsed(console, elapsed_sec)

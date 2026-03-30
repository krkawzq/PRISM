from __future__ import annotations

from math import comb
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import sparse

from prism.model import CORE_CHANNELS, ObservationBatch, Posterior, SignalChannel

console = Console()


def require_reference_genes(metadata: dict[str, object]) -> list[str]:
    value = metadata.get("reference_gene_names")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("checkpoint metadata is missing reference_gene_names")
    return list(value)


def resolve_prior_source(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"global", "label"}:
        raise ValueError("prior_source must be either 'global' or 'label'")
    return resolved


def resolve_channels(channels: list[str] | None) -> list[str]:
    if not channels:
        return sorted(CORE_CHANNELS)
    valid = set(CORE_CHANNELS) | {"map_p", "map_mu"}
    unknown = [channel for channel in channels if channel not in valid]
    if unknown:
        raise ValueError(f"unknown channels: {unknown}")
    return list(dict.fromkeys(channels))


def resolve_dtype(value: str) -> np.dtype:
    if value == "float32":
        return np.dtype(np.float32)
    if value == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype: {value}")


def read_gene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def slice_gene_counts(matrix, positions: list[int]) -> np.ndarray:
    subset = matrix[:, positions]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=np.float64)
    return np.asarray(subset, dtype=np.float64)


def compute_reference_counts(matrix, positions: list[int]) -> np.ndarray:
    subset = matrix[:, positions]
    if sparse.issparse(subset):
        totals = np.asarray(subset.sum(axis=1)).reshape(-1)
    else:
        totals = np.asarray(subset, dtype=np.float64).sum(axis=1)
    return np.asarray(totals, dtype=np.float64).reshape(-1)


def extract_batch(
    *,
    checkpoint,
    adata: ad.AnnData,
    batch_names: list[str],
    batch_counts: np.ndarray,
    reference_counts: np.ndarray,
    prior_source: str,
    label_key: str | None,
    device: str,
    selected_channels: list[str],
) -> dict[str, np.ndarray]:
    requested_channels = cast(set[SignalChannel], set(selected_channels))
    if prior_source == "global":
        assert checkpoint.priors is not None
        posterior = Posterior(
            batch_names, checkpoint.priors.subset(batch_names), device=device
        )
        return posterior.extract(
            ObservationBatch(
                gene_names=batch_names,
                counts=batch_counts,
                reference_counts=reference_counts,
            ),
            channels=requested_channels,
        )
    if label_key is None:
        raise ValueError("--label-key is required when --prior-source label")
    if label_key not in adata.obs.columns:
        raise KeyError(f"obs column {label_key!r} does not exist")
    labels = np.asarray(adata.obs[label_key].astype(str)).reshape(-1)
    layer_values = {
        channel: np.full(
            (batch_counts.shape[0], len(batch_names)), np.nan, dtype=np.float64
        )
        for channel in selected_channels
    }
    for label in np.unique(labels).tolist():
        if label not in checkpoint.label_priors:
            raise ValueError(f"checkpoint does not contain priors for label {label!r}")
        cell_indices = np.flatnonzero(labels == label)
        priors = checkpoint.label_priors[label].subset(batch_names)
        posterior = Posterior(batch_names, priors, device=device)
        extracted = posterior.extract(
            ObservationBatch(
                gene_names=batch_names,
                counts=batch_counts[cell_indices],
                reference_counts=reference_counts[cell_indices],
            ),
            channels=requested_channels,
        )
        for channel in selected_channels:
            layer_values[channel][cell_indices] = np.asarray(
                extracted[channel], dtype=np.float64
            )
    return layer_values


def print_extract_plan(**values: object) -> None:
    table = Table(title="Extract Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


def print_extract_summary(
    *, output_path: Path, elapsed_sec: float, n_genes: int, channels: list[str]
) -> None:
    table = Table(title="Extract Summary")
    table.add_column("Genes", justify="right")
    table.add_column("Channels")
    table.add_row(str(n_genes), ", ".join(channels))
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")


def resolve_class_groups(adata: ad.AnnData, class_key: str) -> dict[str, np.ndarray]:
    if class_key not in adata.obs.columns:
        raise KeyError(f"obs column {class_key!r} does not exist")
    labels = np.asarray(adata.obs[class_key].astype(str)).reshape(-1)
    groups: dict[str, np.ndarray] = {}
    for label in sorted(np.unique(labels).tolist()):
        groups[label] = np.flatnonzero(labels == label).astype(np.int64)
    return groups


def strict_label_prior_names(checkpoint) -> set[str]:
    return set(str(label) for label in checkpoint.label_priors)


def n_choose_k(n: int, k: int) -> int:
    if k < 0 or n < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        return 0
    return int(comb(n, k))

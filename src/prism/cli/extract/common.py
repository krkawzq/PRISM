from __future__ import annotations

from math import comb
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
from rich.console import Console

from prism.cli.common import (
    print_elapsed,
    print_key_value_table,
    print_saved_path,
    resolve_numpy_dtype,
    resolve_prior_source as resolve_prior_source_shared,
)
from prism.io import (
    compute_reference_counts as compute_reference_counts_shared,
    read_gene_list as read_gene_list_shared,
    select_matrix as select_matrix_shared,
    slice_gene_matrix as slice_gene_matrix_shared,
)
from prism.model import CORE_CHANNELS, ObservationBatch, Posterior, SignalChannel
from prism.model.checkpoint import resolve_checkpoint_distribution

console = Console()


def require_reference_genes(metadata: dict[str, object]) -> list[str]:
    value = metadata.get("reference_gene_names")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("checkpoint metadata is missing reference_gene_names")
    return list(value)


def resolve_prior_source(value: str) -> str:
    return resolve_prior_source_shared(value)


def resolve_channels(channels: list[str] | None) -> list[str]:
    if not channels:
        return sorted(CORE_CHANNELS)
    valid = set(CORE_CHANNELS) | {"map_p", "map_mu", "map_rate"}
    unknown = [channel for channel in channels if channel not in valid]
    if unknown:
        raise ValueError(f"unknown channels: {unknown}")
    return list(dict.fromkeys(channels))


def resolve_posterior_distribution(metadata: dict[str, object]) -> str:
    resolved, _ = resolve_checkpoint_distribution(
        schema_version=int(metadata.get("schema_version", 2)),
        metadata=dict(metadata),
        priors=None,
        label_priors={},
    )
    return str(resolved["posterior_distribution"])


def resolve_nb_overdispersion(
    metadata: dict[str, object], fit_config: dict[str, object]
) -> float:
    value = fit_config.get("nb_overdispersion", metadata.get("nb_overdispersion", 0.01))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.01


def resolve_dtype(value: str) -> np.dtype:
    return resolve_numpy_dtype(value)


def read_gene_list(path: Path) -> list[str]:
    return read_gene_list_shared(path)


def select_matrix(adata: ad.AnnData, layer: str | None):
    return select_matrix_shared(adata, layer)


def slice_gene_counts(matrix, positions: list[int]) -> np.ndarray:
    return slice_gene_matrix_shared(matrix, positions, dtype=np.float64)


def compute_reference_counts(matrix, positions: list[int]) -> np.ndarray:
    return compute_reference_counts_shared(matrix, positions, dtype=np.float64)


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
    torch_dtype: str,
    selected_channels: list[str],
) -> dict[str, np.ndarray]:
    requested_channels = cast(set[SignalChannel], set(selected_channels))
    posterior_distribution = resolve_posterior_distribution(checkpoint.metadata)
    nb_overdispersion = resolve_nb_overdispersion(
        checkpoint.metadata, checkpoint.fit_config
    )
    if prior_source == "global":
        assert checkpoint.priors is not None
        posterior = Posterior(
            batch_names,
            checkpoint.priors.subset(batch_names),
            device=device,
            torch_dtype=torch_dtype,
            posterior_distribution=posterior_distribution,
            nb_overdispersion=nb_overdispersion,
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
        posterior = Posterior(
            batch_names,
            priors,
            device=device,
            torch_dtype=torch_dtype,
            posterior_distribution=posterior_distribution,
            nb_overdispersion=nb_overdispersion,
        )
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
    print_key_value_table(console, title="Extract Plan", values=values)


def print_extract_summary(
    *, output_path: Path, elapsed_sec: float, n_genes: int, channels: list[str]
) -> None:
    print_key_value_table(
        console,
        title="Extract Summary",
        values={"Genes": n_genes, "Channels": ", ".join(channels)},
    )
    print_saved_path(console, output_path)
    print_elapsed(console, elapsed_sec)


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

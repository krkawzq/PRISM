from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar, cast

import anndata as ad
import numpy as np
import pandas as pd
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from prism.io import write_h5ad
from prism.model import KBulkBatch, infer_kbulk, load_checkpoint

from .common import (
    compute_reference_counts,
    console,
    n_choose_k,
    print_extract_plan,
    read_gene_list,
    require_reference_genes,
    resolve_class_groups,
    resolve_dtype,
    resolve_nb_overdispersion,
    resolve_posterior_distribution,
    resolve_prior_source,
    select_matrix,
    slice_gene_counts,
    strict_label_prior_names,
)

T = TypeVar("T")


def _unwrap_typer_value(value: object) -> object:
    return getattr(value, "default", value)


def _resolve_optional_path(value: Path | None | object) -> Path | None:
    resolved = _unwrap_typer_value(value)
    if resolved is None:
        return None
    return Path(cast(str | Path, resolved)).expanduser().resolve()


def _resolve_selected_genes(
    *,
    genes_path: Path | None,
    dataset_gene_names: list[str],
    checkpoint_gene_names: list[str],
) -> list[str]:
    dataset_set = set(dataset_gene_names)
    checkpoint_set = set(checkpoint_gene_names)
    requested = (
        checkpoint_gene_names if genes_path is None else read_gene_list(genes_path)
    )
    selected = [
        gene for gene in requested if gene in dataset_set and gene in checkpoint_set
    ]
    if not selected:
        raise ValueError("no selected genes overlap between dataset and checkpoint")
    return list(dict.fromkeys(selected))


def _sample_unique_combinations(
    indices: np.ndarray, *, k: int, n_samples: int, rng: np.random.Generator
) -> list[tuple[int, ...]]:
    total = n_choose_k(int(indices.shape[0]), int(k))
    if total <= n_samples:
        return [
            tuple(int(i) for i in combo) for combo in combinations(indices.tolist(), k)
        ]
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < n_samples:
        combo = tuple(
            sorted(int(i) for i in rng.choice(indices, size=k, replace=False).tolist())
        )
        chosen.add(combo)
    return sorted(chosen)


def _resolve_target_samples(
    *,
    n_samples: int,
    balance: bool,
    class_mean_reference: float,
    avg_mean_reference: float,
) -> int:
    if not balance:
        return int(n_samples)
    scaled = n_samples * class_mean_reference / max(avg_mean_reference, 1e-12)
    return max(1, int(round(float(scaled))))


def _approx_comb_count(n: int, k: int) -> float:
    if k < 0 or n < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    log_value = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    if log_value > 700:
        return float("inf")
    return float(math.exp(log_value))


def _format_comb_count(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    if value < 1e6:
        return str(int(round(value)))
    return f"{value:.3e}"


def _serialize_per_class_plan(
    per_class_plan: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    if not per_class_plan:
        return {}
    keys = list(per_class_plan[0].keys())
    return {key: [entry[key] for entry in per_class_plan] for key in keys}


def _build_kbulk_result_adata(
    *,
    rows: list[dict[str, Any]],
    gene_names: list[str],
    map_scaled_support_rows: np.ndarray,
    map_probability_rows: np.ndarray,
    posterior_entropy_rows: np.ndarray,
    prior_entropy_rows: np.ndarray,
    mutual_information_rows: np.ndarray,
    raw_count_rows: np.ndarray,
    dtype: np.dtype,
    metadata: dict[str, object],
    map_support_rows: np.ndarray | None = None,
    map_rate_rows: np.ndarray | None = None,
) -> ad.AnnData:
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    output = ad.AnnData(
        X=np.asarray(map_scaled_support_rows, dtype=dtype),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
    )
    output.layers["signal"] = np.asarray(map_scaled_support_rows, dtype=dtype)
    output.layers["map_mu"] = np.asarray(map_scaled_support_rows, dtype=dtype)
    output.layers["map_probability"] = np.asarray(map_probability_rows, dtype=dtype)
    output.layers["map_p"] = np.asarray(map_probability_rows, dtype=dtype)
    if map_support_rows is not None:
        output.layers["map_support"] = np.asarray(map_support_rows, dtype=dtype)
    if map_rate_rows is not None:
        output.layers["map_rate"] = np.asarray(map_rate_rows, dtype=dtype)
    output.layers["posterior_entropy"] = np.asarray(posterior_entropy_rows, dtype=dtype)
    output.layers["prior_entropy"] = np.asarray(prior_entropy_rows, dtype=dtype)
    output.layers["mutual_information"] = np.asarray(
        mutual_information_rows, dtype=dtype
    )
    output.layers["raw_counts"] = np.asarray(raw_count_rows, dtype=dtype)
    output.uns["kbulk"] = metadata
    return output


def _estimate_total_realized(per_class_plan: list[dict[str, Any]]) -> int:
    return int(sum(int(entry["realized_samples"]) for entry in per_class_plan))


def _iter_chunks(values: list[T], chunk_size: int) -> list[list[T]]:
    return [
        values[start : start + chunk_size]
        for start in range(0, len(values), chunk_size)
    ]


def extract_kbulk_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input checkpoint path."
    ),
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output h5ad path."),
    class_key: str = typer.Option(
        ..., help="Obs column defining classes for kBulk construction."
    ),
    k: int = typer.Option(..., min=1, help="Number of cells per kBulk sample."),
    n_samples: int = typer.Option(
        ...,
        min=1,
        help="Target number of kBulk samples per class before balance scaling.",
    ),
    prior_source: str = typer.Option("global", help="Prior source: global or label."),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        exists=True,
        dir_okay=False,
        help="Optional text file restricting kBulk genes.",
    ),
    reference_genes_path: Path | None = typer.Option(
        None,
        "--reference-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file overriding checkpoint reference genes.",
    ),
    layer: str | None = typer.Option(None, help="Input AnnData layer. Defaults to X."),
    balance: bool = typer.Option(
        False, help="Scale per-class kBulk sample count by mean reference size."
    ),
    sample_seed: int = typer.Option(
        0, min=0, help="Random seed for kBulk combination sampling."
    ),
    sample_batch_size: int = typer.Option(
        256, min=1, help="Number of kBulk samples to infer per batch."
    ),
    scale_source: str = typer.Option(
        "checkpoint", help="Scale source: checkpoint or dataset."
    ),
    navg_source: str = typer.Option(
        "dataset", help="N_avg source: dataset or checkpoint."
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    torch_dtype: str = typer.Option(
        "float32", help="Torch dtype for inference: float32 or float64."
    ),
    dtype: str = typer.Option("float32", help="Output dtype: float32 or float64."),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without writing output."
    ),
) -> int:
    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    genes_path = _resolve_optional_path(genes_path)
    reference_genes_path = _resolve_optional_path(reference_genes_path)
    class_key = str(_unwrap_typer_value(class_key))
    k = int(cast(int | str, _unwrap_typer_value(k)))
    n_samples = int(cast(int | str, _unwrap_typer_value(n_samples)))
    prior_source = str(_unwrap_typer_value(prior_source))
    layer = cast(str | None, _unwrap_typer_value(layer))
    balance = bool(_unwrap_typer_value(balance))
    sample_seed = int(cast(int | str, _unwrap_typer_value(sample_seed)))
    sample_batch_size = int(cast(int | str, _unwrap_typer_value(sample_batch_size)))
    scale_source = str(_unwrap_typer_value(scale_source))
    navg_source = str(_unwrap_typer_value(navg_source))
    device = str(_unwrap_typer_value(device))
    torch_dtype = str(_unwrap_typer_value(torch_dtype))
    dtype = str(_unwrap_typer_value(dtype))
    dry_run = bool(_unwrap_typer_value(dry_run))

    checkpoint = load_checkpoint(checkpoint_path)
    prior_source_resolved = resolve_prior_source(prior_source)
    output_dtype = resolve_dtype(dtype)
    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = select_matrix(adata, layer)
    dataset_gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}
    reference_gene_names = (
        require_reference_genes(checkpoint.metadata)
        if reference_genes_path is None
        else read_gene_list(reference_genes_path)
    )
    selected_genes = _resolve_selected_genes(
        genes_path=genes_path,
        dataset_gene_names=dataset_gene_names,
        checkpoint_gene_names=checkpoint.gene_names,
    )
    resolved_reference_gene_names = [
        name for name in reference_gene_names if name in gene_to_idx
    ]
    reference_positions = [gene_to_idx[name] for name in resolved_reference_gene_names]
    if not reference_positions:
        raise ValueError(
            "checkpoint reference genes do not overlap with the input dataset"
        )
    reference_counts = compute_reference_counts(matrix, reference_positions)
    class_groups = resolve_class_groups(adata, class_key)
    if prior_source_resolved == "global" and not checkpoint.has_global_prior:
        raise ValueError("checkpoint does not contain global priors")
    if prior_source_resolved == "label":
        if not checkpoint.has_label_priors:
            raise ValueError("checkpoint does not contain label-specific priors")
        missing_labels = sorted(
            set(class_groups) - strict_label_prior_names(checkpoint)
        )
        if missing_labels:
            raise ValueError(
                f"checkpoint label priors do not match {class_key!r}; missing labels: {missing_labels[:10]}"
            )

    class_mean_reference = {
        label: float(np.mean(reference_counts[indices]))
        for label, indices in class_groups.items()
    }
    avg_mean_reference = float(np.mean(list(class_mean_reference.values())))
    per_class_plan = []
    total_kbulk = 0
    for label, indices in class_groups.items():
        n_cells = int(indices.shape[0])
        target = _resolve_target_samples(
            n_samples=n_samples,
            balance=balance,
            class_mean_reference=class_mean_reference[label],
            avg_mean_reference=avg_mean_reference,
        )
        realized = min(target, n_choose_k(n_cells, k))
        total_kbulk += realized
        per_class_plan.append(
            {
                "label": label,
                "n_cells": n_cells,
                "n_combinations": _approx_comb_count(n_cells, k),
                "mean_reference_count": class_mean_reference[label],
                "target_samples": target,
                "realized_samples": realized,
            }
        )

    print_extract_plan(
        checkpoint_path=checkpoint_path,
        h5ad_path=h5ad_path,
        layer=layer,
        class_key=class_key,
        k=k,
        n_cells=int(adata.n_obs),
        n_classes=len(class_groups),
        skipped_classes=0,
        n_selected_genes=len(selected_genes),
        n_reference_genes=len(reference_positions),
        reference_genes_source="checkpoint" if reference_genes_path is None else "user",
        prior_source=prior_source_resolved,
        n_samples=n_samples,
        balance=balance,
        sample_batch_size=sample_batch_size,
        scale_source=scale_source,
        N_avg_source=navg_source,
        total_planned_kbulk=total_kbulk,
        output_path=output_path,
        device=device,
    )
    if dry_run:
        return 0

    target_positions = [gene_to_idx[name] for name in selected_genes]
    gene_counts = np.ascontiguousarray(
        slice_gene_counts(matrix, target_positions), dtype=np.float32
    )
    reference_counts = np.ascontiguousarray(reference_counts, dtype=np.float32)
    posterior_distribution = resolve_posterior_distribution(
        checkpoint.metadata, checkpoint.fit_config
    )
    nb_overdispersion = resolve_nb_overdispersion(
        checkpoint.metadata, checkpoint.fit_config
    )
    total_realized = _estimate_total_realized(per_class_plan)
    n_genes = len(selected_genes)
    map_scaled_support_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    map_probability_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    map_support_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    posterior_entropy_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    prior_entropy_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    mutual_information_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    raw_count_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(sample_seed)
    offset = 0
    for entry in per_class_plan:
        label = str(entry["label"])
        realized = int(entry["realized_samples"])
        if realized == 0:
            continue
        indices = class_groups[label]
        combos = _sample_unique_combinations(indices, k=k, n_samples=realized, rng=rng)
        prior = checkpoint.get_prior(
            None if prior_source_resolved == "global" else label
        ).select_genes(selected_genes)
        for chunk in _iter_chunks(combos, sample_batch_size):
            combo_matrix = np.asarray(chunk, dtype=np.int64)
            flat_indices = combo_matrix.reshape(-1)
            chunk_counts = gene_counts[flat_indices].reshape(
                combo_matrix.shape[0], combo_matrix.shape[1], n_genes
            )
            chunk_reference = reference_counts[flat_indices].reshape(combo_matrix.shape)
            aggregated_counts = chunk_counts.sum(axis=1, dtype=np.float32)
            aggregated_exposure = chunk_reference.sum(axis=1, dtype=np.float32)
            result = infer_kbulk(
                KBulkBatch(
                    gene_names=selected_genes,
                    counts=aggregated_counts,
                    effective_exposure=aggregated_exposure,
                ),
                prior,
                device=device,
                include_posterior=False,
                torch_dtype=torch_dtype,
                posterior_distribution=posterior_distribution,
                nb_overdispersion=nb_overdispersion,
            )
            size = combo_matrix.shape[0]
            start = offset
            stop = offset + size
            map_support = np.asarray(result.map_support, dtype=output_dtype)
            map_support_rows[start:stop] = map_support
            if posterior_distribution == "poisson":
                map_scaled_support_rows[start:stop] = map_support
                map_probability_rows[start:stop] = np.full_like(
                    map_support, np.nan, dtype=output_dtype
                )
            else:
                map_scaled_support_rows[start:stop] = map_support * np.float32(
                    prior.scale
                )
                map_probability_rows[start:stop] = map_support
            posterior_entropy_rows[start:stop] = np.asarray(
                result.posterior_entropy, dtype=output_dtype
            )
            prior_entropy_rows[start:stop] = np.asarray(
                result.prior_entropy, dtype=output_dtype
            )
            mutual_information_rows[start:stop] = np.asarray(
                result.mutual_information, dtype=output_dtype
            )
            raw_count_rows[start:stop] = np.asarray(
                aggregated_counts, dtype=output_dtype
            )
            for idx in range(size):
                rows.append(
                    {
                        class_key: label,
                        "kbulk_index": start + idx + 1,
                        "k": int(k),
                        "source_n_cells": int(indices.shape[0]),
                        "effective_exposure_sum": float(aggregated_exposure[idx]),
                        "mean_reference_count": float(
                            np.mean(reference_counts[indices])
                        ),
                        "scale": float(prior.scale),
                    }
                )
            offset = stop
    output = _build_kbulk_result_adata(
        rows=rows,
        gene_names=selected_genes,
        map_scaled_support_rows=map_scaled_support_rows,
        map_probability_rows=map_probability_rows,
        map_support_rows=map_support_rows,
        posterior_entropy_rows=posterior_entropy_rows,
        prior_entropy_rows=prior_entropy_rows,
        mutual_information_rows=mutual_information_rows,
        raw_count_rows=raw_count_rows,
        dtype=output_dtype,
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "source_h5ad_path": str(h5ad_path),
            "class_key": class_key,
            "prior_source": prior_source_resolved,
            "k": int(k),
            "requested_n_samples": int(n_samples),
            "balance": bool(balance),
            "sample_seed": int(sample_seed),
            "device": device,
            "posterior_distribution": posterior_distribution,
            "selected_genes": list(selected_genes),
            "reference_gene_names": list(resolved_reference_gene_names),
            "per_class_plan": _serialize_per_class_plan(per_class_plan),
            "per_class_plan_json": json.dumps(per_class_plan),
        },
    )
    write_h5ad(output, output_path)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(
        f"[bold green]Elapsed[/bold green] {perf_counter() - start_time:.2f}s"
    )
    return 0


__all__ = [
    "_approx_comb_count",
    "_build_kbulk_result_adata",
    "_estimate_total_realized",
    "_format_comb_count",
    "_iter_chunks",
    "_resolve_target_samples",
    "_sample_unique_combinations",
    "_serialize_per_class_plan",
    "extract_kbulk_command",
]

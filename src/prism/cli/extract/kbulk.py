from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

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

from prism.io import write_h5ad_atomic
from prism.model import KBulkAggregator, load_checkpoint
from prism.model.checkpoint import resolve_checkpoint_distribution

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
    attempts = 0
    max_attempts = max(n_samples * 50, 1000)
    while len(chosen) < n_samples:
        combo = tuple(
            sorted(int(i) for i in rng.choice(indices, size=k, replace=False).tolist())
        )
        chosen.add(combo)
        attempts += 1
        if attempts > max_attempts and len(chosen) < n_samples:
            raise RuntimeError(
                f"failed to sample {n_samples} unique combinations without replacement"
            )
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
    map_mu_rows: np.ndarray,
    map_p_rows: np.ndarray,
    posterior_entropy_rows: np.ndarray,
    prior_entropy_rows: np.ndarray,
    mutual_information_rows: np.ndarray,
    raw_count_rows: np.ndarray,
    dtype: np.dtype,
    metadata: dict[str, object],
) -> ad.AnnData:
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    output = ad.AnnData(
        X=np.asarray(map_mu_rows, dtype=dtype),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
    )
    output.layers["map_p"] = np.asarray(map_p_rows, dtype=dtype)
    output.layers["posterior_entropy"] = np.asarray(posterior_entropy_rows, dtype=dtype)
    output.layers["prior_entropy"] = np.asarray(prior_entropy_rows, dtype=dtype)
    output.layers["mutual_information"] = np.asarray(
        mutual_information_rows, dtype=dtype
    )
    output.layers["raw_counts"] = np.asarray(raw_count_rows, dtype=dtype)
    output.uns["kbulk"] = metadata
    return output


def _estimate_total_realized(per_class_plan: list[dict[str, Any]]) -> int:
    return int(sum(_entry_int(entry, "realized_samples") for entry in per_class_plan))


def _iter_chunks(values: list[T], chunk_size: int) -> list[list[T]]:
    return [
        values[start : start + chunk_size]
        for start in range(0, len(values), chunk_size)
    ]


def _entry_int(entry: dict[str, Any], key: str) -> int:
    return int(entry[key])


def _entry_float(entry: dict[str, Any], key: str) -> float:
    return float(entry[key])


def _entry_str(entry: dict[str, Any], key: str) -> str:
    return str(entry[key])


def _resolve_scale_pair(
    *,
    label: str | None,
    checkpoint,
    prior_source: str,
    S_source: str,
    navg_source: str,
    reference_counts: np.ndarray,
    class_indices: np.ndarray | None,
) -> tuple[float, float]:
    if prior_source == "label":
        if label is None:
            raise ValueError("label is required for label prior source")
        scale = checkpoint.label_scales.get(label)
        prior = checkpoint.label_priors.get(label)
        if scale is None and (S_source == "checkpoint" or navg_source == "checkpoint"):
            raise ValueError(
                f"checkpoint is missing label scale metadata for {label!r}"
            )
        checkpoint_S = float(scale.S) if scale is not None else float(prior.S)
        checkpoint_navg = (
            float(scale.mean_reference_count)
            if scale is not None
            else float(np.mean(reference_counts[class_indices]))
        )
        assert class_indices is not None
        dataset_navg = float(np.mean(reference_counts[class_indices]))
        dataset_S = dataset_navg
    else:
        if checkpoint.priors is None and checkpoint.scale is None:
            raise ValueError("checkpoint is missing global priors and scale metadata")
        if checkpoint.scale is None and (
            S_source == "checkpoint" or navg_source == "checkpoint"
        ):
            raise ValueError(
                "checkpoint is missing global scale metadata required by selected scale source"
            )
        checkpoint_S = (
            float(checkpoint.scale.S)
            if checkpoint.scale is not None
            else float(checkpoint.priors.S)
        )
        checkpoint_navg = (
            float(checkpoint.scale.mean_reference_count)
            if checkpoint.scale is not None
            else float(np.mean(reference_counts))
        )
        dataset_navg = float(np.mean(reference_counts))
        dataset_S = dataset_navg
    resolved_navg = dataset_navg if navg_source == "dataset" else checkpoint_navg
    resolved_S = checkpoint_S if S_source == "checkpoint" else dataset_S
    return resolved_S, resolved_navg


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
        256,
        min=1,
        help="Number of kBulk samples to infer per batch. Controls GPU memory usage.",
    ),
    S_source: str = typer.Option(
        "checkpoint",
        help="Scale source for S: checkpoint or dataset.",
    ),
    navg_source: str = typer.Option(
        "dataset",
        help="Scale source for N_avg: dataset or checkpoint.",
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    dtype: str = typer.Option("float32", help="Output dtype: float32 or float64."),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without writing output."
    ),
) -> int:
    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    genes_path = None if genes_path is None else genes_path.expanduser().resolve()
    reference_genes_path = (
        None
        if reference_genes_path is None
        else reference_genes_path.expanduser().resolve()
    )

    checkpoint = load_checkpoint(checkpoint_path)
    resolved_checkpoint_metadata, _ = resolve_checkpoint_distribution(
        schema_version=int(checkpoint.metadata.get("schema_version", 2)),
        metadata=checkpoint.metadata,
        priors=checkpoint.priors,
        label_priors=checkpoint.label_priors,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_distribution = str(
        resolved_checkpoint_metadata["posterior_distribution"]
    )
    checkpoint_grid_domain = str(resolved_checkpoint_metadata["grid_domain"])
    if checkpoint_distribution == "poisson" or checkpoint_grid_domain == "rate":
        raise ValueError(
            "prism extract kbulk does not support poisson/rate-grid checkpoints yet: "
            "the CLI currently does not export poisson-specific map_rate outputs, so "
            "using this checkpoint would silently lose distribution semantics. "
            f"Got posterior_distribution={checkpoint_distribution!r}, grid_domain={checkpoint_grid_domain!r}."
        )
    prior_source_resolved = resolve_prior_source(prior_source)
    output_dtype = resolve_dtype(dtype)
    S_source_resolved = S_source.strip().lower()
    navg_source_resolved = navg_source.strip().lower()
    if S_source_resolved not in {"checkpoint", "dataset"}:
        raise ValueError("S_source must be one of: checkpoint, dataset")
    if navg_source_resolved not in {"checkpoint", "dataset"}:
        raise ValueError("navg_source must be one of: dataset, checkpoint")

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
    if prior_source_resolved == "global":
        if checkpoint.priors is None:
            raise ValueError("checkpoint does not contain global priors")
    else:
        if not checkpoint.label_priors:
            raise ValueError("checkpoint does not contain label-specific priors")
        missing_labels = sorted(
            set(class_groups) - strict_label_prior_names(checkpoint)
        )
        if missing_labels:
            raise ValueError(
                f"checkpoint label priors do not match {class_key!r}; missing labels: {missing_labels[:10]}"
            )

    dataset_navg = float(np.mean(reference_counts))
    resolved_S_global: float | None = None
    resolved_navg_global: float | None = None

    class_mean_reference = {
        label: float(np.mean(reference_counts[indices]))
        for label, indices in class_groups.items()
    }
    avg_mean_reference = float(np.mean(list(class_mean_reference.values())))
    per_class_plan: list[dict[str, Any]] = []
    total_kbulk = 0
    skipped_classes = 0
    for label, indices in class_groups.items():
        n_cells = int(indices.shape[0])
        n_combos = n_choose_k(n_cells, k)
        n_combos_approx = _approx_comb_count(n_cells, k)
        target = _resolve_target_samples(
            n_samples=n_samples,
            balance=balance,
            class_mean_reference=class_mean_reference[label],
            avg_mean_reference=avg_mean_reference,
        )
        realized = min(target, n_combos)
        if n_combos == 0:
            skipped_classes += 1
        total_kbulk += realized
        resolved_S_label, resolved_navg_label = _resolve_scale_pair(
            label=label,
            checkpoint=checkpoint,
            prior_source=prior_source_resolved,
            S_source=S_source_resolved,
            navg_source=navg_source_resolved,
            reference_counts=reference_counts,
            class_indices=indices,
        )
        per_class_plan.append(
            {
                "label": label,
                "n_cells": n_cells,
                "n_combinations": n_combos_approx,
                "mean_reference_count": class_mean_reference[label],
                "resolved_S": resolved_S_label,
                "resolved_N_avg": resolved_navg_label,
                "target_samples": target,
                "realized_samples": realized,
            }
        )
    if prior_source_resolved == "global":
        resolved_S_global, resolved_navg_global = _resolve_scale_pair(
            label=None,
            checkpoint=checkpoint,
            prior_source="global",
            S_source=S_source_resolved,
            navg_source=navg_source_resolved,
            reference_counts=reference_counts,
            class_indices=None,
        )
    else:
        resolved_S_global = float(
            np.mean([_entry_float(entry, "resolved_S") for entry in per_class_plan])
        )
        resolved_navg_global = float(
            np.mean([_entry_float(entry, "resolved_N_avg") for entry in per_class_plan])
        )

    print_extract_plan(
        checkpoint_path=checkpoint_path,
        h5ad_path=h5ad_path,
        layer=layer,
        class_key=class_key,
        k=k,
        n_cells=int(adata.n_obs),
        n_classes=len(class_groups),
        skipped_classes=skipped_classes,
        n_selected_genes=len(selected_genes),
        n_reference_genes=len(reference_positions),
        reference_genes_source="checkpoint" if reference_genes_path is None else "user",
        prior_source=prior_source_resolved,
        n_samples=n_samples,
        balance=balance,
        sample_batch_size=sample_batch_size,
        S_source=S_source_resolved,
        N_avg_source=navg_source_resolved,
        S=resolved_S_global,
        N_avg=resolved_navg_global,
        avg_mean_reference=avg_mean_reference,
        total_planned_kbulk=total_kbulk,
        output_path=output_path,
        device=device,
    )
    if per_class_plan:
        preview_lines = [
            f"{entry['label']}: cells={entry['n_cells']}, combos={_format_comb_count(float(entry['n_combinations']))}, target={entry['target_samples']}, realized={entry['realized_samples']}"
            for entry in per_class_plan[: min(10, len(per_class_plan))]
        ]
        console.print("\n".join(preview_lines))
    if dry_run:
        return 0

    target_positions = [gene_to_idx[name] for name in selected_genes]
    gene_counts = np.ascontiguousarray(
        slice_gene_counts(matrix, target_positions), dtype=np.float32
    )
    reference_counts = np.ascontiguousarray(reference_counts, dtype=np.float32)
    rng = np.random.default_rng(sample_seed)
    total_realized = _estimate_total_realized(per_class_plan)
    n_genes = len(selected_genes)
    map_mu_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    map_p_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    posterior_entropy_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    prior_entropy_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    mutual_information_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    raw_count_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    obs_class: list[str] = [""] * total_realized
    obs_kbulk_index = np.empty(total_realized, dtype=np.int64)
    obs_class_sample_index = np.empty(total_realized, dtype=np.int64)
    obs_k = np.full(total_realized, int(k), dtype=np.int64)
    obs_source_n_cells = np.empty(total_realized, dtype=np.int64)
    obs_n_combinations_total = np.empty(total_realized, dtype=np.float64)
    obs_sampling_mode: list[str] = [""] * total_realized
    obs_effective_exposure_sum = np.empty(total_realized, dtype=np.float32)
    obs_mean_reference_count = np.empty(total_realized, dtype=np.float32)
    obs_S = np.empty(total_realized, dtype=np.float32)
    obs_N_avg = np.empty(total_realized, dtype=np.float32)

    realized_classes = [
        entry for entry in per_class_plan if _entry_int(entry, "realized_samples") > 0
    ]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("constructing kbulk", total=len(realized_classes))
        global_sample_index = 0
        priors_cache: dict[str, KBulkAggregator] = {}
        for class_index, entry in enumerate(realized_classes, start=1):
            label = _entry_str(entry, "label")
            indices = class_groups[label]
            target = _entry_int(entry, "target_samples")
            n_combinations_total = _entry_float(entry, "n_combinations")
            mean_reference = _entry_float(entry, "mean_reference_count")
            resolved_S = _entry_float(entry, "resolved_S")
            resolved_navg = _entry_float(entry, "resolved_N_avg")
            combos = _sample_unique_combinations(
                indices, k=k, n_samples=target, rng=rng
            )
            cache_key = "global" if prior_source_resolved == "global" else label
            aggregator = priors_cache.get(cache_key)
            if aggregator is None:
                priors = (
                    checkpoint.priors
                    if prior_source_resolved == "global"
                    else checkpoint.label_priors[label]
                )
                assert priors is not None
                aggregator = KBulkAggregator(
                    selected_genes,
                    priors,
                    device=device,
                    torch_dtype="float32",
                    posterior_distribution=resolve_posterior_distribution(
                        checkpoint.metadata
                    ),
                    nb_overdispersion=resolve_nb_overdispersion(
                        checkpoint.metadata, checkpoint.fit_config
                    ),
                )
                priors_cache[cache_key] = aggregator
            indices_i64 = np.asarray(indices, dtype=np.int64)
            for chunk_index, combo_chunk in enumerate(
                _iter_chunks(combos, sample_batch_size), start=1
            ):
                combo_matrix = np.asarray(combo_chunk, dtype=np.int64)
                flat_indices = combo_matrix.reshape(-1)
                chunk_counts = gene_counts[flat_indices].reshape(
                    combo_matrix.shape[0], combo_matrix.shape[1], n_genes
                )
                chunk_reference = reference_counts[flat_indices].reshape(
                    combo_matrix.shape
                )
                aggregated_counts = chunk_counts.sum(axis=1, dtype=np.float32)
                aggregated_exposure = chunk_reference.sum(
                    axis=1, dtype=np.float32
                ) * np.float32(resolved_S / max(resolved_navg, 1e-12))
                result = aggregator.query_samples(
                    aggregated_counts,
                    aggregated_exposure,
                    include_posterior=False,
                )
                sampling_mode = (
                    "all" if n_combinations_total <= float(target) else "sampled"
                )
                chunk_size_actual = combo_matrix.shape[0]
                start = global_sample_index
                stop = global_sample_index + chunk_size_actual
                map_mu_rows[start:stop] = np.asarray(result.map_mu, dtype=output_dtype)
                map_p_rows[start:stop] = np.asarray(result.map_p, dtype=output_dtype)
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
                obs_class[start:stop] = [label] * chunk_size_actual
                obs_kbulk_index[start:stop] = np.arange(
                    start + 1, stop + 1, dtype=np.int64
                )
                obs_class_sample_index[start:stop] = np.arange(
                    (chunk_index - 1) * sample_batch_size + 1,
                    (chunk_index - 1) * sample_batch_size + 1 + chunk_size_actual,
                    dtype=np.int64,
                )
                obs_source_n_cells[start:stop] = int(indices_i64.shape[0])
                obs_n_combinations_total[start:stop] = np.float64(n_combinations_total)
                obs_sampling_mode[start:stop] = [sampling_mode] * chunk_size_actual
                obs_effective_exposure_sum[start:stop] = aggregated_exposure
                obs_mean_reference_count[start:stop] = np.float32(mean_reference)
                obs_S[start:stop] = np.float32(resolved_S)
                obs_N_avg[start:stop] = np.float32(resolved_navg)
                global_sample_index = stop
            progress.update(
                task_id,
                advance=1,
                description=f"constructing kbulk ({class_index}/{len(realized_classes)})",
            )

    rows = [
        {
            class_key: obs_class[idx],
            "kbulk_index": int(obs_kbulk_index[idx]),
            "class_sample_index": int(obs_class_sample_index[idx]),
            "k": int(obs_k[idx]),
            "source_n_cells": int(obs_source_n_cells[idx]),
            "n_combinations_total": float(obs_n_combinations_total[idx]),
            "sampling_mode": obs_sampling_mode[idx],
            "effective_exposure_sum": float(obs_effective_exposure_sum[idx]),
            "mean_reference_count": float(obs_mean_reference_count[idx]),
            "S": float(obs_S[idx]),
            "N_avg": float(obs_N_avg[idx]),
        }
        for idx in range(total_realized)
    ]

    output = _build_kbulk_result_adata(
        rows=rows,
        gene_names=selected_genes,
        map_mu_rows=map_mu_rows,
        map_p_rows=map_p_rows,
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
            "selected_genes": list(selected_genes),
            "reference_gene_names": list(resolved_reference_gene_names),
            "reference_genes_source": "checkpoint"
            if reference_genes_path is None
            else "user",
            "S_source": S_source_resolved,
            "N_avg_source": navg_source_resolved,
            "S": float(resolved_S_global),
            "N_avg": float(resolved_navg_global),
            "dataset_S": float(dataset_navg),
            "dataset_N_avg": float(dataset_navg),
            "checkpoint_S": float(
                checkpoint.scale.S
                if checkpoint.scale is not None
                else (
                    checkpoint.priors.S
                    if checkpoint.priors is not None
                    else resolved_S_global
                )
            ),
            "checkpoint_N_avg": float(
                checkpoint.scale.mean_reference_count
                if checkpoint.scale is not None
                else dataset_navg
            ),
            "per_class_plan": _serialize_per_class_plan(per_class_plan),
            "per_class_plan_json": json.dumps(per_class_plan),
        },
    )
    write_h5ad_atomic(output, output_path)
    for entry in per_class_plan:
        console.print(
            f"{entry['label']}: cells={entry['n_cells']}, combos={_format_comb_count(float(entry['n_combinations']))}, target={entry['target_samples']}, realized={entry['realized_samples']}"
        )
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(
        f"[bold green]Elapsed[/bold green] {perf_counter() - start_time:.2f}s"
    )
    return 0


__all__ = ["extract_kbulk_command"]

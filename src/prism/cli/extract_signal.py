from __future__ import annotations

import pickle
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import anndata as ad
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from scipy import sparse

from prism.model import CORE_CHANNELS, GeneBatch, Posterior, PriorEngine
from prism.model._typing import DTYPE_NP
from prism.model.posterior import SignalChannel

console = Console()


@dataclass(frozen=True, slots=True)
class _GeneSelection:
    gene_names: list[str]
    input_positions: np.ndarray
    output_positions: np.ndarray


@dataclass(frozen=True, slots=True)
class _BatchPayload:
    offset: int
    gene_names: list[str]
    input_positions: np.ndarray
    output_positions: np.ndarray
    counts: np.ndarray


@dataclass(slots=True)
class _OutputStore:
    adata: ad.AnnData
    channel_paths: dict[str, Path]
    channel_arrays: dict[str, np.memmap]
    temp_dir: Path


def extract_signal(
    ckpt_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input checkpoint pickle path."
    ),
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Argument(..., help="Output h5ad file path."),
    layer: str | None = typer.Option(
        None, help="AnnData layer name to use as input matrix."
    ),
    batch_size: int = typer.Option(32, min=1, help="Genes per extraction batch."),
    device: str = typer.Option("cuda", help="Torch device, e.g. cpu or cuda."),
    channels: list[str] | None = typer.Option(
        None,
        "--channel",
        help="Channels to write. Repeatable; defaults to core channels.",
    ),
    dtype: str = typer.Option(
        "float32", help="Output layer dtype: float32 or float64."
    ),
    gene_start: int = typer.Option(
        0,
        min=0,
        help="Start fitted-gene index within the checkpoint/h5ad overlap.",
    ),
    gene_end: int | None = typer.Option(
        None,
        min=0,
        help="End fitted-gene index within the overlap, exclusive.",
    ),
    max_genes: int | None = typer.Option(
        None,
        min=1,
        help="Extract only the first N genes from the selected overlap.",
    ),
    output_mode: str = typer.Option(
        "full-matrix",
        help="Output mode: full-matrix or fitted-only.",
    ),
) -> int:
    start_time = perf_counter()
    ckpt_path = ckpt_path.resolve()
    h5ad_path = h5ad_path.resolve()
    output_path = output_path.resolve()

    console.print(f"[bold cyan]Loading[/bold cyan] checkpoint {ckpt_path}")
    with ckpt_path.open("rb") as fh:
        checkpoint = pickle.load(fh)

    engine = checkpoint.get("engine")
    s_hat = checkpoint.get("s_hat")
    if not isinstance(engine, PriorEngine):
        raise TypeError("checkpoint 中缺少合法的 PriorEngine")
    if s_hat is None:
        raise KeyError("checkpoint 中缺少 s_hat")

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = _select_matrix(adata, layer)
    gene_names = np.asarray(adata.var_names.astype(str))
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    overlap_gene_names = [
        name
        for name in engine.gene_names
        if name in gene_to_idx and engine.is_fitted(name)
    ]
    if not overlap_gene_names:
        raise ValueError("h5ad 与 checkpoint 没有可提取的已拟合基因交集")

    selection = _select_overlap_genes(
        overlap_gene_names,
        gene_to_idx=gene_to_idx,
        gene_start=gene_start,
        gene_end=gene_end,
        max_genes=max_genes,
        output_mode=output_mode,
    )
    priors = engine.get_priors(selection.gene_names)
    if priors is None:
        raise ValueError("无法从 checkpoint 中读取已拟合先验")

    requested_channels = _resolve_channels(channels)
    output_dtype = _resolve_dtype(dtype)
    totals = _compute_totals(matrix)
    resolved_device = _resolve_extract_device(requested_device=device)
    posterior = Posterior(selection.gene_names, priors, device=resolved_device)
    current_batch_size = _resolve_initial_batch_size(
        requested_batch_size=batch_size,
        n_cells=int(adata.n_obs),
        grid_size=priors.M,
        torch_dtype=output_dtype.name,
        device=resolved_device,
    )

    store = _create_output_store(
        adata=adata,
        output_path=output_path,
        channels=requested_channels,
        output_dtype=output_dtype,
        selection=selection,
        output_mode=output_mode,
    )
    _print_extract_dataset_summary(
        n_cells=int(adata.n_obs),
        total_genes=int(adata.n_vars),
        overlap_genes=len(overlap_gene_names),
        selected_genes=len(selection.gene_names),
        output_genes=int(store.adata.n_vars),
        layer=layer,
        device=resolved_device,
        batch_size=current_batch_size,
        dtype=output_dtype.name,
        channels=requested_channels,
        output_mode=output_mode,
    )

    try:
        _run_extract_pipeline(
            matrix=matrix,
            posterior=posterior,
            totals=totals,
            s_hat=s_hat,
            channels=requested_channels,
            selection=selection,
            store=store,
            output_dtype=output_dtype,
            initial_batch_size=current_batch_size,
            device=resolved_device,
        )

        console.print(f"[bold cyan]Writing[/bold cyan] {output_path}")
        for channel, values in store.channel_arrays.items():
            values.flush()
            store.adata.layers[channel] = np.asarray(values)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store.adata.write_h5ad(output_path)
    finally:
        _cleanup_output_store(store)

    _print_extract_summary(
        n_cells=int(store.adata.n_obs),
        n_genes=int(store.adata.n_vars),
        extracted_genes=len(selection.gene_names),
        channels=requested_channels,
        output_path=output_path,
        elapsed_sec=perf_counter() - start_time,
        output_mode=output_mode,
    )
    return 0


def _resolve_channels(channels: list[str] | None) -> list[str]:
    if not channels:
        return sorted(CORE_CHANNELS)

    valid = set(CORE_CHANNELS)
    unknown = [channel for channel in channels if channel not in valid]
    if unknown:
        raise ValueError(f"未知 channel: {unknown}")
    return list(dict.fromkeys(channels))


def _channel_set(channels: list[str]) -> set[SignalChannel]:
    values: set[SignalChannel] = set()
    for channel in channels:
        values.add(cast(SignalChannel, channel))
    return values


def _resolve_dtype(dtype: str) -> np.dtype:
    if dtype == "float32":
        return np.dtype(np.float32)
    if dtype == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"不支持的 dtype: {dtype!r}")


def _select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} 不存在")
    return adata.layers[layer]


def _compute_totals(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix, dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP).reshape(-1)


def _select_overlap_genes(
    overlap_gene_names: list[str],
    *,
    gene_to_idx: dict[str, int],
    gene_start: int,
    gene_end: int | None,
    max_genes: int | None,
    output_mode: str,
) -> _GeneSelection:
    if output_mode not in {"full-matrix", "fitted-only"}:
        raise ValueError("output_mode 仅支持 full-matrix 或 fitted-only")
    n_genes = len(overlap_gene_names)
    if gene_start < 0 or gene_start >= n_genes:
        raise ValueError(f"gene_start 超出范围: {gene_start}")
    resolved_end = n_genes if gene_end is None else min(gene_end, n_genes)
    if resolved_end <= gene_start:
        raise ValueError(
            f"gene_end 必须大于 gene_start，收到 {resolved_end} <= {gene_start}"
        )
    selected_names = overlap_gene_names[gene_start:resolved_end]
    if max_genes is not None:
        selected_names = selected_names[:max_genes]
    if not selected_names:
        raise ValueError("筛选后没有可提取的基因")
    input_positions = np.asarray(
        [gene_to_idx[name] for name in selected_names], dtype=np.int64
    )
    output_positions = (
        input_positions.copy()
        if output_mode == "full-matrix"
        else np.arange(len(selected_names), dtype=np.int64)
    )
    return _GeneSelection(
        gene_names=selected_names,
        input_positions=input_positions,
        output_positions=output_positions,
    )


def _create_output_store(
    *,
    adata: ad.AnnData,
    output_path: Path,
    channels: list[str],
    output_dtype: np.dtype,
    selection: _GeneSelection,
    output_mode: str,
) -> _OutputStore:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.stem}.extract.", dir=output_path.parent)
    )
    if output_mode == "full-matrix":
        output_adata = adata
        shape = (int(adata.n_obs), int(adata.n_vars))
    else:
        output_adata = adata[:, selection.input_positions].copy()
        shape = (int(output_adata.n_obs), int(output_adata.n_vars))

    channel_paths: dict[str, Path] = {}
    channel_arrays: dict[str, np.memmap] = {}
    for channel in channels:
        path = temp_dir / f"{channel}.dat"
        values = np.memmap(path, dtype=output_dtype, mode="w+", shape=shape)
        values[:] = np.nan
        channel_paths[channel] = path
        channel_arrays[channel] = values
    return _OutputStore(
        adata=output_adata,
        channel_paths=channel_paths,
        channel_arrays=channel_arrays,
        temp_dir=temp_dir,
    )


def _cleanup_output_store(store: _OutputStore) -> None:
    for channel in list(store.channel_arrays):
        values = store.channel_arrays.pop(channel)
        try:
            values.flush()
        except Exception:
            pass
        del values
    try:
        shutil.rmtree(store.temp_dir, ignore_errors=True)
    except Exception:
        pass


def _run_extract_pipeline(
    *,
    matrix,
    posterior: Posterior,
    totals: np.ndarray,
    s_hat: float | np.ndarray,
    channels: list[str],
    selection: _GeneSelection,
    store: _OutputStore,
    output_dtype: np.dtype,
    initial_batch_size: int,
    device: str,
) -> None:
    total_genes = len(selection.gene_names)
    requested_channels = _channel_set(channels)
    batch_size = initial_batch_size
    offset = 0
    future: Future[_BatchPayload] | None = None

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="extract-prefetch"
    ) as pool:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total} genes[/cyan]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("extract", total=total_genes)
            while offset < total_genes:
                if future is None:
                    future = pool.submit(
                        _load_batch_payload,
                        matrix,
                        selection,
                        offset,
                        batch_size,
                    )

                payload = future.result()
                next_offset = payload.offset + len(payload.gene_names)
                next_future = None
                if next_offset < total_genes:
                    next_future = pool.submit(
                        _load_batch_payload,
                        matrix,
                        selection,
                        next_offset,
                        batch_size,
                    )
                try:
                    extracted = posterior.extract(
                        GeneBatch(
                            gene_names=payload.gene_names,
                            counts=payload.counts,
                            totals=totals,
                        ),
                        s_hat=s_hat,
                        channels=requested_channels,
                    )
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc, device) or len(payload.gene_names) <= 1:
                        raise
                    if next_future is not None:
                        next_future.cancel()
                    batch_size = max(1, len(payload.gene_names) // 2)
                    _clear_cuda_cache(device)
                    console.print(
                        "[yellow]Warning:[/yellow] CUDA OOM during extract; "
                        f"reducing batch size to {batch_size} and retrying offset {payload.offset}."
                    )
                    future = None
                    continue

                _write_batch(
                    store=store,
                    channels=channels,
                    extracted=extracted,
                    output_positions=payload.output_positions,
                    output_dtype=output_dtype,
                )
                offset = next_offset
                future = next_future
                progress.update(
                    task_id,
                    advance=len(payload.gene_names),
                    description=(
                        f"extract genes {offset}/{total_genes} | device={device} | batch={len(payload.gene_names)}"
                    ),
                )


def _load_batch_payload(
    matrix,
    selection: _GeneSelection,
    offset: int,
    batch_size: int,
) -> _BatchPayload:
    end = min(offset + batch_size, len(selection.gene_names))
    input_positions = selection.input_positions[offset:end]
    output_positions = selection.output_positions[offset:end]
    return _BatchPayload(
        offset=offset,
        gene_names=selection.gene_names[offset:end],
        input_positions=input_positions,
        output_positions=output_positions,
        counts=_slice_gene_counts_by_index(matrix, input_positions),
    )


def _slice_gene_counts_by_index(matrix, gene_positions: np.ndarray) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=DTYPE_NP)
    return np.asarray(subset, dtype=DTYPE_NP)


def _write_batch(
    *,
    store: _OutputStore,
    channels: list[str],
    extracted: dict[str, np.ndarray],
    output_positions: np.ndarray,
    output_dtype: np.dtype,
) -> None:
    for channel in channels:
        values = _sanitize_channel_output(channel, extracted[channel])
        store.channel_arrays[channel][:, output_positions] = values.T.astype(
            output_dtype,
            copy=False,
        )


def _sanitize_channel_output(channel: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE_NP)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if channel in {"posterior_entropy", "prior_entropy", "mutual_information"}:
        return np.clip(array, 0.0, None)
    return array


def _resolve_extract_device(*, requested_device: str) -> str:
    if not requested_device.startswith("cuda"):
        return requested_device
    if not torch.cuda.is_available():
        console.print("[yellow]Warning:[/yellow] CUDA 不可用，extract 将回退到 CPU。")
        return "cpu"
    return requested_device


def _resolve_initial_batch_size(
    *,
    requested_batch_size: int,
    n_cells: int,
    grid_size: int,
    torch_dtype: str,
    device: str,
) -> int:
    if not device.startswith("cuda"):
        return requested_batch_size

    bytes_per_value = 8 if torch_dtype == "float64" else 4
    estimated_bytes = n_cells * requested_batch_size * grid_size * bytes_per_value * 4
    estimated_gib = estimated_bytes / (1024**3)
    console.print(
        "[cyan]GPU plan[/cyan] "
        f"batch={requested_batch_size} est_workspace={estimated_gib:.2f} GiB device={device}"
    )
    return requested_batch_size


def _is_cuda_oom(exc: RuntimeError, device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_extract_summary(
    *,
    n_cells: int,
    n_genes: int,
    extracted_genes: int,
    channels: list[str],
    output_path: Path,
    elapsed_sec: float,
    output_mode: str,
) -> None:
    table = Table(title="Extract Summary")
    table.add_column("Cells", justify="right")
    table.add_column("Genes", justify="right")
    table.add_column("Extracted", justify="right")
    table.add_column("Mode")
    table.add_column("Channels")
    table.add_row(
        str(n_cells),
        str(n_genes),
        str(extracted_genes),
        output_mode,
        ", ".join(channels),
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")


def _print_extract_dataset_summary(
    *,
    n_cells: int,
    total_genes: int,
    overlap_genes: int,
    selected_genes: int,
    output_genes: int,
    layer: str | None,
    device: str,
    batch_size: int,
    dtype: str,
    channels: list[str],
    output_mode: str,
) -> None:
    table = Table(title="Extract Dataset")
    table.add_column("Cells", justify="right")
    table.add_column("Genes", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Selected", justify="right")
    table.add_column("OutputGenes", justify="right")
    table.add_column("Layer")
    table.add_column("Device")
    table.add_column("Batch", justify="right")
    table.add_column("DType")
    table.add_column("Mode")
    table.add_column("Channels")
    table.add_row(
        str(n_cells),
        str(total_genes),
        str(overlap_genes),
        str(selected_genes),
        str(output_genes),
        layer or "X",
        device,
        str(batch_size),
        dtype,
        output_mode,
        ", ".join(channels),
    )
    console.print(table)

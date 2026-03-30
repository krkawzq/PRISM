#!/usr/bin/env python3
"""
Train GEARS on a custom perturbation dataset and score with cell-eval.

Uses the standard GEARS public API:
  PertData.new_data_process() → PertData.load() →
  prepare_split() → get_dataloader() →
  GEARS() → model_initialize() → train()

Requirements:
  - cell-gears (pip install cell-gears)
  - torch-geometric
  - cell-eval (for optional downstream scoring)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

console = Console()
install_rich_traceback(show_locals=False)

GEARS_GENE2GO_URL = "https://dataverse.harvard.edu/api/access/datafile/6153417"
GEARS_ESSENTIAL_URL = "https://dataverse.harvard.edu/api/access/datafile/6934320"
GEARS_GO_GRAPH_URL = "https://dataverse.harvard.edu/api/access/datafile/6934319"


def detect_proxy_settings() -> dict[str, str]:
    proxy_env: dict[str, str] = {}
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        value = os.environ.get(key)
        if value:
            proxy_env[key] = value
    return proxy_env


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GEARS on a custom perturbation dataset and score with cell-eval.",
    )
    # data
    p.add_argument(
        "--bulk-h5ad",
        type=Path,
        required=True,
        help="h5ad with perturbation expression (bulk or pseudo-bulk)",
    )
    p.add_argument(
        "--reference-h5ad",
        type=Path,
        required=True,
        help="h5ad used only for gene alignment (e.g. unperturbed scRNA-seq)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--label-column",
        type=str,
        default="perturbation",
        help="obs column with perturbation labels",
    )
    p.add_argument(
        "--control-label",
        type=str,
        default="control",
        help="label that marks control cells in --label-column",
    )
    p.add_argument(
        "--cell-type",
        type=str,
        default="K562",
        help="value to fill adata.obs['cell_type'] (GEARS requirement)",
    )
    p.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="optional gene list (.txt or .json with 'gene_names' key)",
    )
    # GEARS data
    p.add_argument("--split", type=str, default="simulation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-gene-set-size", type=float, default=0.75)
    p.add_argument(
        "--skip-calc-de",
        action="store_true",
        help="skip DE gene calculation in new_data_process",
    )
    # GEARS model
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-go-gnn-layers", type=int, default=1)
    p.add_argument("--num-gene-gnn-layers", type=int, default=1)
    p.add_argument("--decoder-hidden-size", type=int, default=16)
    p.add_argument("--num-similar-genes-go-graph", type=int, default=20)
    p.add_argument("--num-similar-genes-coexpress-graph", type=int, default=20)
    p.add_argument("--coexpress-threshold", type=float, default=0.4)
    p.add_argument("--direction-lambda", type=float, default=1e-1)
    p.add_argument("--uncertainty", action="store_true")
    # eval
    p.add_argument("--eval-profile", type=str, default="full")
    p.add_argument("--eval-num-threads", type=int, default=16)
    p.add_argument("--allow-discrete", action="store_true")
    # misc
    p.add_argument("--keep-cache", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_gene_list(path: Path) -> list[str]:
    """Read gene list from .txt (one per line) or .json (key 'gene_names')."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        names = payload.get("gene_names")
        if not isinstance(names, list) or not all(
            isinstance(x, str) and x for x in names
        ):
            raise ValueError(f"invalid gene_names in {path}")
        return list(dict.fromkeys(names))
    return list(
        dict.fromkeys(line.strip() for line in text.splitlines() if line.strip())
    )


def normalize_condition(label: str, control_label: str) -> str:
    """
    Map raw perturbation labels to GEARS condition format:
      control  → 'ctrl'
      single   → 'ctrl+GENE'
      combo    → 'GENE1+GENE2' (sorted)
    """
    raw = str(label)
    lower = raw.lower()
    if lower == control_label.lower() or lower in {
        "ctrl",
        "control",
        "nt",
        "ntc",
        "non-targeting",
    }:
        return "ctrl"
    # split on '+' or '_' to handle various formats
    if "+" in raw:
        parts = [p.strip() for p in raw.split("+") if p.strip()]
    elif "_" in raw:
        parts = [p.strip() for p in raw.split("_") if p.strip()]
    else:
        parts = [raw]
    parts = sorted(p for p in parts if p.lower() not in {"ctrl", "control"})
    if not parts:
        return "ctrl"
    if len(parts) == 1:
        return f"ctrl+{parts[0]}"
    return "+".join(parts)


def ensure_sparse_float32(adata: ad.AnnData) -> ad.AnnData:
    """Ensure adata.X is a CSR float32 sparse matrix."""
    if sparse.issparse(adata.X):
        adata.X = cast(Any, adata.X).tocsr().astype(np.float32)
    else:
        adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    return adata


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()
    proxies = detect_proxy_settings()
    if proxies:
        console.print(
            f"[cyan]Detected proxy settings[/cyan]: {', '.join(sorted(proxies))}"
        )
    else:
        console.print("[yellow]No proxy detected in environment[/yellow]")
    try:
        with requests.get(url, stream=True, timeout=120, verify=False) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
    except requests.RequestException as exc:
        if temp_path.exists():
            temp_path.unlink()
        proxy_hint = (
            "proxy detected from environment"
            if proxies
            else "no proxy detected; set HTTPS_PROXY/HTTP_PROXY if your network requires one"
        )
        raise RuntimeError(f"failed to download {url}: {exc}; {proxy_hint}") from exc
    temp_path.replace(destination)


def ensure_gears_resources(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    gene2go_path = data_root / "gene2go_all.pkl"
    essential_path = data_root / "essential_all_data_pert_genes.pkl"
    go_dir = data_root / "go_essential_all"
    if not gene2go_path.exists():
        console.print("[cyan]Fetching GEARS gene2go resource...[/cyan]")
        download_file(GEARS_GENE2GO_URL, gene2go_path)
    if not essential_path.exists():
        console.print("[cyan]Fetching GEARS essential perturbation list...[/cyan]")
        download_file(GEARS_ESSENTIAL_URL, essential_path)
    if not go_dir.exists():
        console.print("[cyan]Fetching GEARS GO graph archive...[/cyan]")
        tar_path = data_root / "go_essential_all.tar.gz"
        download_file(GEARS_GO_GRAPH_URL, tar_path)
        with tarfile.open(tar_path) as archive:
            archive.extractall(path=data_root)


def subset_and_align(
    bulk: ad.AnnData,
    reference: ad.AnnData,
    gene_list: list[str] | None,
) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    """
    Subset both AnnDatas to a shared gene set and align gene order.
    """
    bulk_lookup = {str(g): i for i, g in enumerate(bulk.var_names)}
    ref_lookup = {str(g): i for i, g in enumerate(reference.var_names)}

    if gene_list is None:
        selected = [g for g in bulk_lookup if g in ref_lookup]
    else:
        selected = [g for g in gene_list if g in bulk_lookup and g in ref_lookup]

    if not selected:
        raise ValueError("no genes overlap between bulk and reference datasets")

    bulk = bulk[:, [bulk_lookup[g] for g in selected]].copy()
    reference = reference[:, [ref_lookup[g] for g in selected]].copy()
    bulk.var_names = list(selected)
    reference.var_names = list(selected)
    return ensure_sparse_float32(bulk), ensure_sparse_float32(reference), selected


def prepare_gears_adata(
    adata: ad.AnnData,
    *,
    label_column: str,
    control_label: str,
    cell_type: str,
) -> ad.AnnData:
    """
    Prepare an AnnData to meet GEARS requirements:
      adata.obs['condition']  – GEARS-format condition string
      adata.obs['cell_type']  – cell type label
      adata.var['gene_name']  – gene symbols

    Also creates 'condition_name' = 'cell_type_condition' which GEARS uses
    internally for DE gene lookup.
    """
    if label_column not in adata.obs.columns:
        raise KeyError(f"missing obs column {label_column!r}")

    out = adata.copy()
    out.obs = out.obs.copy()
    out.var = out.var.copy()

    out.obs[label_column] = out.obs[label_column].astype(str)
    out.obs["condition"] = [
        normalize_condition(v, control_label) for v in out.obs[label_column]
    ]
    out.obs["cell_type"] = cell_type
    # GEARS internally expects 'condition_name' = '{cell_type}_{condition}'
    out.obs["condition_name"] = (
        out.obs["cell_type"].astype(str) + "_" + out.obs["condition"].astype(str)
    )
    out.var["gene_name"] = out.var_names.astype(str)

    if not np.any(out.obs["condition"].astype(str) == "ctrl"):
        raise ValueError(
            f"no control samples found after mapping {label_column!r} with control label {control_label!r}"
        )
    return out


# ---------------------------------------------------------------------------
# GEARS training via standard API
# ---------------------------------------------------------------------------
def train_gears(adata: ad.AnnData, args: argparse.Namespace, data_root: Path):
    """
    Process data, train GEARS, and return (model, pert_data).

    Uses the official PertData.new_data_process → load → prepare_split →
    get_dataloader → GEARS → model_initialize → train pipeline.
    """
    from gears import PertData, GEARS

    data_root.mkdir(parents=True, exist_ok=True)
    ensure_gears_resources(data_root)
    dataset_name = "bulk_dataset"

    # ── 1. Process new dataset ──────────────────────────────────────────
    pert_data = PertData(str(data_root), default_pert_graph=True)
    pert_data.new_data_process(
        dataset_name=dataset_name,
        adata=adata,
        skip_calc_de=args.skip_calc_de,
    )

    # ── 2. Load the processed dataset (populates pyg graphs, filters) ──
    pert_data.load(data_path=str(data_root / dataset_name))

    # ── 3. Split & dataloader ───────────────────────────────────────────
    pert_data.prepare_split(
        split=args.split,
        seed=args.seed,
        train_gene_set_size=args.train_gene_set_size,
    )
    pert_data.get_dataloader(args.batch_size, args.test_batch_size)

    # ── 4. Model ────────────────────────────────────────────────────────
    model = GEARS(pert_data, device=args.device)
    model.model_initialize(
        hidden_size=args.hidden_size,
        num_go_gnn_layers=args.num_go_gnn_layers,
        num_gene_gnn_layers=args.num_gene_gnn_layers,
        decoder_hidden_size=args.decoder_hidden_size,
        num_similar_genes_go_graph=args.num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=args.num_similar_genes_coexpress_graph,
        coexpress_threshold=args.coexpress_threshold,
        uncertainty=args.uncertainty,
        direction_lambda=args.direction_lambda,
    )

    # ── 5. Train ────────────────────────────────────────────────────────
    model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
    return model, pert_data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def run_gears_eval(model, pert_data) -> tuple[dict, dict, dict]:
    """Run GEARS built-in evaluation, return (results, metrics, pert_metrics)."""
    from gears.inference import compute_metrics, evaluate

    loader = pert_data.dataloader.get("test_loader") or pert_data.dataloader.get(
        "val_loader"
    )
    if loader is None:
        raise ValueError("no evaluation dataloader available")

    best = getattr(model, "best_model", None) or model.model
    results = evaluate(loader, best, model.config["uncertainty"], model.device)
    metrics, pert_metrics = compute_metrics(results)
    return results, metrics, pert_metrics


def build_eval_adatas(
    bulk_adata: ad.AnnData,
    results: dict[str, np.ndarray],
    *,
    label_column: str,
    control_label: str,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Construct prediction / ground-truth AnnData pair from GEARS results,
    suitable for cell-eval scoring.
    """
    # map GEARS conditions back to original labels
    obs_df = pd.DataFrame(
        bulk_adata.obs[["condition", label_column]].copy()
    ).drop_duplicates()
    cond2label = dict(
        zip(obs_df["condition"].astype(str), obs_df[label_column].astype(str))
    )

    pert_cats = results["pert_cat"]
    labels = [cond2label.get(str(c), str(c)) for c in pert_cats]
    var_df = pd.DataFrame(bulk_adata.var.copy())

    pred = ad.AnnData(
        X=np.asarray(results["pred"], dtype=np.float32),
        obs=pd.DataFrame({label_column: labels}),
        var=var_df.copy(),
    )
    real = ad.AnnData(
        X=np.asarray(results["truth"], dtype=np.float32),
        obs=pd.DataFrame({label_column: labels}),
        var=var_df.copy(),
    )
    pred.obs.index = pred.obs.index.astype(str)
    real.obs.index = real.obs.index.astype(str)

    # append control cells from original data
    ctrl_mask = (
        bulk_adata.obs[label_column].astype(str).str.lower() == control_label.lower()
    )
    if ctrl_mask.any():
        ctrl = bulk_adata[ctrl_mask].copy()
        ctrl.obs = ctrl.obs[[label_column]].copy()
        ctrl.obs.index = ctrl.obs.index.astype(str)
        ctrl_pred = ctrl.copy()
        pred = ad.concat([pred, ctrl_pred], axis=0)
        real = ad.concat([real, ctrl], axis=0)

    return pred, real


def run_cell_eval(
    pred: ad.AnnData,
    real: ad.AnnData,
    args: argparse.Namespace,
    output_dir: Path,
):
    """Score with cell-eval MetricsEvaluator."""
    from cell_eval import MetricsEvaluator

    evaluator = MetricsEvaluator(
        adata_pred=pred,
        adata_real=real,
        control_pert=args.control_label,
        pert_col=args.label_column,
        num_threads=args.eval_num_threads,
        outdir=str(output_dir),
        allow_discrete=args.allow_discrete,
    )
    return evaluator.compute(profile=args.eval_profile)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── banner ──────────────────────────────────────────────────────────
    info = Table(show_header=False, box=None)
    info.add_row("Bulk", str(args.bulk_h5ad.expanduser().resolve()))
    info.add_row("Reference", str(args.reference_h5ad.expanduser().resolve()))
    info.add_row("Output", str(output_dir))
    info.add_row("Split", args.split)
    info.add_row("Device", args.device)
    console.print(Panel(info, title="Train GEARS", border_style="cyan"))

    start = perf_counter()

    # ── 1. load & align ────────────────────────────────────────────────
    gene_list = (
        None
        if args.gene_list is None
        else read_gene_list(args.gene_list.expanduser().resolve())
    )
    with console.status("Loading and aligning AnnData inputs..."):
        bulk_raw = ad.read_h5ad(args.bulk_h5ad.expanduser().resolve())
        ref_raw = ad.read_h5ad(args.reference_h5ad.expanduser().resolve())
        bulk_raw, ref_raw, selected_genes = subset_and_align(
            bulk_raw,
            ref_raw,
            gene_list,
        )
        bulk = prepare_gears_adata(
            bulk_raw,
            label_column=args.label_column,
            control_label=args.control_label,
            cell_type=args.cell_type,
        )
    console.print(
        f"[green]Aligned[/green] {len(selected_genes)} genes across bulk/reference; "
        f"bulk samples={bulk.n_obs}, controls={(bulk.obs['condition'].astype(str) == 'ctrl').sum()}"
    )
    console.print(
        f"  [green]✓[/green] {len(selected_genes)} genes, {bulk.n_obs} samples aligned"
    )
    bulk_values = (
        cast(Any, bulk.X).data if sparse.issparse(bulk.X) else np.asarray(bulk.X)
    )
    if not np.all(np.isfinite(np.asarray(bulk_values))):
        raise ValueError("bulk expression matrix contains non-finite values")

    # ── 2. train ────────────────────────────────────────────────────────
    data_root = output_dir / "gears_data"
    console.print("[bold]Training GEARS...[/bold]")
    model, pert_data = train_gears(bulk, args, data_root)

    # ── 3. evaluate (GEARS built-in) ───────────────────────────────────
    with console.status("Running GEARS evaluation..."):
        results, gears_metrics, pert_metrics = run_gears_eval(model, pert_data)

    # save predictions
    pred_adata, real_adata = build_eval_adatas(
        bulk,
        results,
        label_column=args.label_column,
        control_label=args.control_label,
    )
    pred_path = output_dir / "predictions.h5ad"
    real_path = output_dir / "targets.h5ad"
    pred_adata.write_h5ad(pred_path)
    real_adata.write_h5ad(real_path)

    # ── 4. cell-eval scoring ────────────────────────────────────────────
    eval_agg = None
    try:
        with console.status("Scoring predictions with cell-eval..."):
            _, eval_agg = run_cell_eval(pred_adata, real_adata, args, output_dir)
    except ImportError:
        console.print("[yellow]cell-eval not installed, skipping scoring[/yellow]")
    except Exception as exc:
        console.print(f"[yellow]cell-eval failed: {exc}[/yellow]")

    # ── 5. save GEARS metrics ───────────────────────────────────────────
    metrics_path = output_dir / "gears_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "overall": {k: float(v) for k, v in gears_metrics.items()},
                "per_perturbation": {
                    k: {m: float(val) for m, val in vs.items()}
                    for k, vs in pert_metrics.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # save model
    model_dir = output_dir / "gears_model"
    model_dir.mkdir(exist_ok=True)
    model.save_model(str(model_dir))

    # ── 6. summary ──────────────────────────────────────────────────────
    summary = Table(title="GEARS Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Selected genes", str(len(selected_genes)))
    summary.add_row("Bulk samples", str(bulk.n_obs))
    summary.add_row("GEARS mse", f"{float(gears_metrics['mse']):.6f}")
    summary.add_row("GEARS pearson", f"{float(gears_metrics['pearson']):.6f}")
    if "mse_de" in gears_metrics:
        summary.add_row("GEARS mse_de", f"{float(gears_metrics['mse_de']):.6f}")
    summary.add_row("Predictions", str(pred_path))
    summary.add_row("Targets", str(real_path))
    summary.add_row("Metrics", str(metrics_path))
    summary.add_row("Model", str(model_dir))
    if eval_agg is not None and hasattr(eval_agg, "shape"):
        summary.add_row("cell-eval rows", str(eval_agg.shape[0]))
    summary.add_row("Elapsed", f"{perf_counter() - start:.2f}s")
    console.print(summary)

    # ── cleanup ─────────────────────────────────────────────────────────
    if not args.keep_cache:
        cache = data_root
        if cache.exists():
            shutil.rmtree(cache)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(Panel(str(exc), title="train_gears failed", border_style="red"))
        raise SystemExit(1) from exc

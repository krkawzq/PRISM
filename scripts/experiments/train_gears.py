#!/usr/bin/env python3
"""
Train GEARS on a perturbation AnnData and export a reusable evaluation pipeline.

Pipeline layout under --output-dir:
  gears/
    model/
    metrics.json
    cache/
  cell_eval/
    input/
      predictions.h5ad
      targets.h5ad
    results/
      results.csv
      agg_results.csv
      run_config.json
  pipeline_manifest.json

GEARS graph resources and graph construction are delegated to GEARS itself.
cell-eval execution is delegated to the local wrapper script
`run_cell_eval.py`, so the same evaluation layer can be reused by other
models later.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from scipy import sparse

console = Console()
install_rich_traceback(show_locals=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GEARS_ROOT = PROJECT_ROOT / "forks" / "GEARS"
CELL_EVAL_RUNNER = PROJECT_ROOT / "scripts" / "experiments" / "run_cell_eval.py"

CELL_EVAL_PERT_COL = "target_gene"
CELL_EVAL_CONTROL = "non-targeting"
CELL_EVAL_CELLTYPE_COL = "celltype"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GEARS and export a reusable cell-eval pipeline.",
    )

    parser.add_argument(
        "--input-h5ad",
        "--bulk-h5ad",
        dest="input_h5ad",
        type=Path,
        required=True,
        help="Input perturbation h5ad. GEARS reads directly from adata.X.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-column",
        type=str,
        default="perturbation",
        help="obs column containing perturbation labels.",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default="control",
        help="Control label in --label-column.",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default="K562",
        help="Value written into adata.obs['cell_type'] for GEARS.",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional gene list (.txt or .json with key 'gene_names') to subset adata.",
    )

    parser.add_argument("--split", type=str, default="simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-gene-set-size", type=float, default=0.75)
    parser.add_argument(
        "--skip-calc-de",
        action="store_true",
        help="Skip DE calculation in PertData.new_data_process().",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-go-gnn-layers", type=int, default=1)
    parser.add_argument("--num-gene-gnn-layers", type=int, default=1)
    parser.add_argument("--decoder-hidden-size", type=int, default=16)
    parser.add_argument("--num-similar-genes-go-graph", type=int, default=20)
    parser.add_argument("--num-similar-genes-coexpress-graph", type=int, default=20)
    parser.add_argument("--coexpress-threshold", type=float, default=0.4)
    parser.add_argument("--direction-lambda", type=float, default=1e-1)
    parser.add_argument("--uncertainty", action="store_true")

    parser.add_argument("--skip-cell-eval", action="store_true")
    parser.add_argument("--eval-profile", type=str, default="full")
    parser.add_argument("--eval-num-threads", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--eval-de-method", type=str, default="wilcoxon")
    parser.add_argument("--eval-skip-metrics", type=str, default=None)
    parser.add_argument("--eval-celltype-col", type=str, default=None)
    parser.add_argument("--eval-embed-key", type=str, default=None)
    parser.add_argument("--allow-discrete", action="store_true")
    parser.add_argument("--break-on-cell-eval-error", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def read_gene_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        names = payload.get("gene_names")
        if not isinstance(names, list) or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError(f"invalid gene_names in {path}")
        return list(dict.fromkeys(names))
    return list(
        dict.fromkeys(line.strip() for line in text.splitlines() if line.strip())
    )


def normalize_condition(label: str, control_label: str) -> str:
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

    if "+" in raw:
        parts = [part.strip() for part in raw.split("+") if part.strip()]
    elif "_" in raw:
        parts = [part.strip() for part in raw.split("_") if part.strip()]
    else:
        parts = [raw]

    parts = sorted(part for part in parts if part.lower() not in {"ctrl", "control"})
    if not parts:
        return "ctrl"
    if len(parts) == 1:
        return f"ctrl+{parts[0]}"
    return "+".join(parts)


def normalize_cell_eval_label(label: str, control_label: str) -> str:
    if str(label).lower() == control_label.lower():
        return CELL_EVAL_CONTROL
    return str(label)


def ensure_sparse_float32(adata: ad.AnnData) -> ad.AnnData:
    if sparse.issparse(adata.X):
        adata.X = cast(Any, adata.X).tocsr().astype(np.float32)
    else:
        adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    return adata


def subset_genes(adata: ad.AnnData, gene_list: list[str] | None) -> tuple[ad.AnnData, int]:
    if gene_list is None:
        return ensure_sparse_float32(adata.copy()), adata.n_vars

    lookup = {str(gene): idx for idx, gene in enumerate(adata.var_names)}
    selected = [gene for gene in gene_list if gene in lookup]
    if not selected:
        raise ValueError("no genes from --gene-list were found in input_h5ad")

    subset = adata[:, [lookup[gene] for gene in selected]].copy()
    subset.var_names = selected
    return ensure_sparse_float32(subset), len(selected)


def prepare_gears_adata(
    adata: ad.AnnData,
    *,
    label_column: str,
    control_label: str,
    cell_type: str,
) -> ad.AnnData:
    if label_column not in adata.obs.columns:
        raise KeyError(f"missing obs column {label_column!r}")

    out = adata.copy()
    out.obs = out.obs.copy()
    out.var = out.var.copy()

    out.obs[label_column] = out.obs[label_column].astype(str)
    out.obs["condition"] = [
        normalize_condition(label, control_label) for label in out.obs[label_column]
    ]
    out.obs["cell_type"] = cell_type
    out.var["gene_name"] = out.var_names.astype(str)

    if not np.any(out.obs["condition"].astype(str) == "ctrl"):
        raise ValueError(
            f"no control samples found after mapping {label_column!r} with control label {control_label!r}"
        )

    values = cast(Any, out.X).data if sparse.issparse(out.X) else np.asarray(out.X)
    if not np.all(np.isfinite(np.asarray(values))):
        raise ValueError("input expression matrix contains non-finite values")

    return out


def ensure_gears_import() -> None:
    try:
        import gears  # noqa: F401

        return
    except ImportError:
        pass

    if not LOCAL_GEARS_ROOT.exists():
        raise ImportError(
            "gears is not installed and local fallback was not found at "
            f"{LOCAL_GEARS_ROOT}"
        )

    sys.path.insert(0, str(LOCAL_GEARS_ROOT))
    import gears  # noqa: F401


def train_gears(adata: ad.AnnData, args: argparse.Namespace, data_root: Path):
    ensure_gears_import()
    from gears import GEARS, PertData

    dataset_name = "custom_dataset"
    data_root.mkdir(parents=True, exist_ok=True)

    pert_data = PertData(str(data_root))
    pert_data.new_data_process(
        dataset_name=dataset_name,
        adata=adata,
        skip_calc_de=args.skip_calc_de,
    )
    pert_data.load(data_path=str(data_root / dataset_name))
    pert_data.prepare_split(
        split=args.split,
        seed=args.seed,
        train_gene_set_size=args.train_gene_set_size,
    )
    pert_data.get_dataloader(args.batch_size, args.test_batch_size)

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
    model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
    return model, pert_data


def run_gears_eval(model, pert_data) -> tuple[dict[str, np.ndarray], dict, dict]:
    from gears.inference import compute_metrics, evaluate

    loader = pert_data.dataloader.get("test_loader") or pert_data.dataloader.get(
        "val_loader"
    )
    if loader is None:
        raise ValueError("no evaluation dataloader available")

    best_model = getattr(model, "best_model", None) or model.model
    results = evaluate(loader, best_model, model.config["uncertainty"], model.device)
    metrics, pert_metrics = compute_metrics(results)
    return results, metrics, pert_metrics


def build_obs_frame(
    labels: list[str],
    *,
    label_column: str,
    control_label: str,
    cell_type: str,
    index_prefix: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            label_column: labels,
            CELL_EVAL_PERT_COL: [
                normalize_cell_eval_label(label, control_label) for label in labels
            ],
            CELL_EVAL_CELLTYPE_COL: [cell_type] * len(labels),
        }
    )
    frame.index = [f"{index_prefix}-{idx}" for idx in range(len(frame))]
    return frame


def build_cell_eval_adatas(
    source_adata: ad.AnnData,
    results: dict[str, np.ndarray],
    *,
    label_column: str,
    control_label: str,
    cell_type: str,
) -> tuple[ad.AnnData, ad.AnnData]:
    obs_map = source_adata.obs[["condition", label_column]].copy().drop_duplicates()
    condition_to_label = dict(
        zip(obs_map["condition"].astype(str), obs_map[label_column].astype(str))
    )

    eval_labels = [
        condition_to_label.get(str(condition), str(condition))
        for condition in results["pert_cat"]
    ]
    var_df = pd.DataFrame(source_adata.var.copy())

    pred = ad.AnnData(
        X=np.asarray(results["pred"], dtype=np.float32),
        obs=build_obs_frame(
            eval_labels,
            label_column=label_column,
            control_label=control_label,
            cell_type=cell_type,
            index_prefix="eval",
        ),
        var=var_df.copy(),
    )
    truth = ad.AnnData(
        X=np.asarray(results["truth"], dtype=np.float32),
        obs=build_obs_frame(
            eval_labels,
            label_column=label_column,
            control_label=control_label,
            cell_type=cell_type,
            index_prefix="eval",
        ),
        var=var_df.copy(),
    )

    ctrl_mask = (
        source_adata.obs[label_column].astype(str).str.lower() == control_label.lower()
    )
    if ctrl_mask.any():
        ctrl = source_adata[ctrl_mask].copy()
        ctrl_labels = ctrl.obs[label_column].astype(str).tolist()
        ctrl.obs = build_obs_frame(
            ctrl_labels,
            label_column=label_column,
            control_label=control_label,
            cell_type=cell_type,
            index_prefix="ctrl",
        )
        pred = ad.concat([pred, ctrl.copy()], axis=0)
        truth = ad.concat([truth, ctrl], axis=0)

    cell_eval_config = {
        "pert_col": CELL_EVAL_PERT_COL,
        "control_pert": CELL_EVAL_CONTROL,
        "celltype_col": CELL_EVAL_CELLTYPE_COL,
        "source_label_column": label_column,
        "source_control_label": control_label,
    }
    pred.uns["cell_eval_config"] = cell_eval_config
    truth.uns["cell_eval_config"] = cell_eval_config
    return pred, truth


def run_cell_eval_pipeline(
    *,
    pred_path: Path,
    truth_path: Path,
    args: argparse.Namespace,
    cell_eval_results_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(CELL_EVAL_RUNNER),
        "--pred-h5ad",
        str(pred_path),
        "--real-h5ad",
        str(truth_path),
        "--output-dir",
        str(cell_eval_results_dir),
        "--pert-col",
        CELL_EVAL_PERT_COL,
        "--control-pert",
        CELL_EVAL_CONTROL,
        "--profile",
        args.eval_profile,
        "--num-threads",
        str(args.eval_num_threads),
        "--batch-size",
        str(args.eval_batch_size),
        "--de-method",
        args.eval_de_method,
    ]
    if args.allow_discrete:
        command.append("--allow-discrete")
    if args.eval_skip_metrics:
        command.extend(["--skip-metrics", args.eval_skip_metrics])
    if args.eval_celltype_col:
        command.extend(["--celltype-col", args.eval_celltype_col])
    if args.eval_embed_key:
        command.extend(["--embed-key", args.eval_embed_key])
    if args.break_on_cell_eval_error:
        command.append("--break-on-error")

    subprocess.run(command, check=True)
    return command


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gears_dir = output_dir / "gears"
    gears_model_dir = gears_dir / "model"
    gears_cache_dir = gears_dir / "cache"
    gears_metrics_path = gears_dir / "metrics.json"
    cell_eval_dir = output_dir / "cell_eval"
    cell_eval_input_dir = cell_eval_dir / "input"
    cell_eval_results_dir = cell_eval_dir / "results"
    pred_path = cell_eval_input_dir / "predictions.h5ad"
    truth_path = cell_eval_input_dir / "targets.h5ad"
    manifest_path = output_dir / "pipeline_manifest.json"

    for path in [gears_dir, gears_model_dir, cell_eval_input_dir, cell_eval_results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    info = Table(show_header=False, box=None)
    info.add_row("Input", str(args.input_h5ad.expanduser().resolve()))
    info.add_row("Output", str(output_dir))
    info.add_row("Split", args.split)
    info.add_row("Device", args.device)
    console.print(Panel(info, title="Train GEARS Pipeline", border_style="cyan"))

    start = perf_counter()
    gene_list = (
        None
        if args.gene_list is None
        else read_gene_list(args.gene_list.expanduser().resolve())
    )

    with console.status("Loading perturbation AnnData..."):
        raw = ad.read_h5ad(args.input_h5ad.expanduser().resolve())
        raw, num_genes = subset_genes(raw, gene_list)
        gears_adata = prepare_gears_adata(
            raw,
            label_column=args.label_column,
            control_label=args.control_label,
            cell_type=args.cell_type,
        )

    num_controls = int((gears_adata.obs["condition"].astype(str) == "ctrl").sum())
    console.print(
        f"[green]Loaded[/green] {gears_adata.n_obs} samples, {num_genes} genes, {num_controls} controls"
    )

    console.print("[bold]Training GEARS...[/bold]")
    model, pert_data = train_gears(gears_adata, args, gears_cache_dir)

    with console.status("Running GEARS evaluation..."):
        results, gears_metrics, pert_metrics = run_gears_eval(model, pert_data)

    pred_adata, truth_adata = build_cell_eval_adatas(
        gears_adata,
        results,
        label_column=args.label_column,
        control_label=args.control_label,
        cell_type=args.cell_type,
    )
    pred_adata.write_h5ad(pred_path)
    truth_adata.write_h5ad(truth_path)

    gears_metrics_path.write_text(
        json.dumps(
            {
                "overall": {name: float(value) for name, value in gears_metrics.items()},
                "per_perturbation": {
                    condition: {name: float(value) for name, value in values.items()}
                    for condition, values in pert_metrics.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model.save_model(str(gears_model_dir))

    cell_eval_command = None
    cell_eval_error = None
    cell_eval_succeeded = False
    cell_eval_results_path = cell_eval_results_dir / "results.csv"
    cell_eval_agg_path = cell_eval_results_dir / "agg_results.csv"
    if not args.skip_cell_eval:
        console.print("[bold]Running cell-eval wrapper...[/bold]")
        try:
            cell_eval_command = run_cell_eval_pipeline(
                pred_path=pred_path,
                truth_path=truth_path,
                args=args,
                cell_eval_results_dir=cell_eval_results_dir,
            )
            cell_eval_succeeded = True
        except Exception as exc:
            cell_eval_error = str(exc)
            if args.break_on_cell_eval_error:
                raise
            console.print(f"[yellow]cell-eval failed: {exc}[/yellow]")

    manifest = {
        "input_h5ad": str(args.input_h5ad.expanduser().resolve()),
        "output_dir": str(output_dir),
        "gears": {
            "model_dir": str(gears_model_dir),
            "metrics_json": str(gears_metrics_path),
            "cache_dir": str(gears_cache_dir),
            "cache_kept": args.keep_cache,
        },
        "cell_eval": {
            "pred_h5ad": str(pred_path),
            "real_h5ad": str(truth_path),
            "pert_col": CELL_EVAL_PERT_COL,
            "control_pert": CELL_EVAL_CONTROL,
            "celltype_col": CELL_EVAL_CELLTYPE_COL,
            "results_csv": str(cell_eval_results_path),
            "agg_results_csv": str(cell_eval_agg_path),
            "run_config_json": str(cell_eval_results_dir / "run_config.json"),
            "command": cell_eval_command,
            "skipped": args.skip_cell_eval,
            "succeeded": cell_eval_succeeded,
            "error": cell_eval_error,
        },
        "source_columns": {
            "label_column": args.label_column,
            "control_label": args.control_label,
            "cell_type": args.cell_type,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = Table(title="GEARS Pipeline Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Samples", str(gears_adata.n_obs))
    summary.add_row("Genes", str(gears_adata.n_vars))
    summary.add_row("Controls", str(num_controls))
    summary.add_row("GEARS mse", f"{float(gears_metrics['mse']):.6f}")
    summary.add_row("GEARS pearson", f"{float(gears_metrics['pearson']):.6f}")
    if "mse_de" in gears_metrics:
        summary.add_row("GEARS mse_de", f"{float(gears_metrics['mse_de']):.6f}")
    if "pearson_de" in gears_metrics:
        summary.add_row("GEARS pearson_de", f"{float(gears_metrics['pearson_de']):.6f}")
    summary.add_row("GEARS model", str(gears_model_dir))
    summary.add_row("GEARS metrics", str(gears_metrics_path))
    summary.add_row("cell-eval pred", str(pred_path))
    summary.add_row("cell-eval real", str(truth_path))
    if cell_eval_succeeded:
        summary.add_row("cell-eval results", str(cell_eval_results_path))
        summary.add_row("cell-eval agg", str(cell_eval_agg_path))
    elif cell_eval_error is not None:
        summary.add_row("cell-eval error", cell_eval_error)
    summary.add_row("Manifest", str(manifest_path))
    summary.add_row("Elapsed", f"{perf_counter() - start:.2f}s")
    console.print(summary)

    if not args.keep_cache and gears_cache_dir.exists():
        shutil.rmtree(gears_cache_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(Panel(str(exc), title="train_gears failed", border_style="red"))
        raise SystemExit(1) from exc

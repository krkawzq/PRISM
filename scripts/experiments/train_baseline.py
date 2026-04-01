#!/usr/bin/env python3
"""Baseline classifier trainer for AnnData representations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import anndata as ad
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    from prism.baseline import DeepMLPClassifier, LinearClassifier, MLPClassifier
    from prism.io import read_gene_list
except ImportError as exc:
    raise ImportError(
        "PRISM is not installed in the active environment. Run `pip install -e .` "
        "from the repository root before executing scripts/experiments/train_baseline.py."
    ) from exc

console = Console()
install_rich_traceback(show_locals=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 0
LR = 1e-3
WD = 1e-4
EPOCHS = {"linear": 30, "mlp": 50, "deep-mlp": 100}
HIDDEN = {"mlp": (1024,), "deep-mlp": (1024, 512, 256)}
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(path: Path, label_col: str, gene_list_path: Path | None) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    if gene_list_path is not None:
        requested = read_gene_list(gene_list_path)
        var_names = [str(name) for name in adata.var_names.tolist()]
        lookup = {name: idx for idx, name in enumerate(var_names)}
        names = [gene for gene in requested if gene in lookup]
        missing = [gene for gene in requested if gene not in lookup]
        if missing:
            console.print(
                f"[yellow]Skipped[/yellow] {len(missing)} genes missing from dataset"
            )
        if not names:
            raise ValueError("gene list has no overlap with the dataset")
        idx = [lookup[g] for g in names]
        adata = adata[:, idx].copy()
    if label_col not in adata.obs:
        raise KeyError(f"Label column '{label_col}' not found in obs.")
    return adata


def split(
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    _, counts = np.unique(labels, return_counts=True)
    strat = labels if counts.min() >= 2 else None
    holdout_fraction = val_fraction + test_fraction
    train, tmp, _, tmp_y = train_test_split(
        idx, labels, test_size=holdout_fraction, random_state=seed, stratify=strat
    )
    strat2 = tmp_y if np.unique(tmp_y, return_counts=True)[1].min() >= 2 else None
    val_share = val_fraction / max(holdout_fraction, 1e-12)
    val, test = train_test_split(
        tmp, train_size=val_share, random_state=seed, stratify=strat2
    )
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def to_dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(cast(Any, matrix).toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def align_class_means(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    expr_dim: int,
    eps: float,
    min_scale: float,
    max_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if expr_dim <= 0:
        raise ValueError("expr_dim must be positive for align-avg")
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError(
            "align-avg scale bounds must satisfy 0 < min_scale <= max_scale"
        )
    classes, counts = np.unique(labels, return_counts=True)
    ref_pos = int(np.argmax(counts))
    ref_label = int(classes[ref_pos])
    ref_mask = labels == ref_label
    ref_mean = X[ref_mask, :expr_dim].mean(axis=0)
    scale_min = float("inf")
    scale_max = float("-inf")
    out = X.copy()
    for cls in classes:
        cls_mask = labels == cls
        cls_mean = out[cls_mask, :expr_dim].mean(axis=0)
        scale = np.ones(expr_dim, dtype=np.float32)
        ref_active = ref_mean > eps
        cls_active = cls_mean > eps
        both_active = ref_active & cls_active
        scale[both_active] = ref_mean[both_active] / cls_mean[both_active]
        only_ref_active = ref_active & ~cls_active
        scale[only_ref_active] = max_scale
        only_cls_active = ~ref_active & cls_active
        scale[only_cls_active] = min_scale
        np.clip(scale, min_scale, max_scale, out=scale)
        out[cls_mask, :expr_dim] *= scale
        scale_min = min(scale_min, float(scale.min()))
        scale_max = max(scale_max, float(scale.max()))
    return out, {
        "reference_label": ref_label,
        "reference_size": int(counts[ref_pos]),
        "expr_dim": int(expr_dim),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
    }


def build_features(
    adata: ad.AnnData,
    args: argparse.Namespace,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Build dense float32 feature matrix from AnnData."""
    # Base: one or more layers (concat), or X
    if args.layers:
        parts = []
        for layer in args.layers:
            if layer not in adata.layers:
                raise KeyError(f"layer '{layer}' not found in AnnData")
            parts.append(to_dense_float32(adata.layers[layer]))
        X = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    else:
        X = to_dense_float32(adata.X)

    np.nan_to_num(X, copy=False)

    # Optional normalization + log1p
    if args.normalize_total is not None:
        totals = X.sum(axis=1, keepdims=True).clip(1e-9)
        X = X / totals * args.normalize_total
    if args.log1p:
        np.log1p(X, out=X)

    expr_dim = int(X.shape[1])

    # Append scalar obs columns
    extras = []
    for key in args.append_obs_keys:
        if key not in adata.obs:
            raise KeyError(f"obs key '{key}' not found.")
        extras.append(adata.obs[key].to_numpy(dtype=np.float32)[:, None])
    if extras:
        X = np.concatenate([X] + extras, axis=1)

    align_info = None
    if args.align_avg:
        X, align_info = align_class_means(
            X,
            labels,
            expr_dim=expr_dim,
            eps=args.align_avg_eps,
            min_scale=args.align_avg_min_scale,
            max_scale=args.align_avg_max_scale,
        )

    return X, align_info


def rep_name(args: argparse.Namespace) -> str:
    parts = list(args.layers) if args.layers else ["X"]
    if args.normalize_total is not None:
        parts.append(f"norm{int(args.normalize_total)}")
    if args.log1p:
        parts.append("log1p")
    if args.align_avg:
        parts.append("alignavg")
    parts.extend(args.append_obs_keys)
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def make_model(
    name: str,
    in_dim: int,
    n_classes: int,
    *,
    mlp_hidden_dim: int,
    deep_hidden_dims: tuple[int, ...],
) -> torch.nn.Module:
    if name == "linear":
        return LinearClassifier(input_dim=in_dim, num_classes=n_classes)
    if name == "mlp":
        return MLPClassifier(
            input_dim=in_dim, num_classes=n_classes, hidden_dim=mlp_hidden_dim
        )
    return DeepMLPClassifier(
        input_dim=in_dim, num_classes=n_classes, hidden_dims=deep_hidden_dims
    )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def batches(idx: np.ndarray, bs: int, shuffle: bool):
    order = idx.copy()
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, len(order), bs):
        yield order[i : i + bs]


@torch.inference_mode()
def evaluate(model, X, y, idx, device, bs=1024):
    model.eval()
    preds = []
    for batch in batches(idx, bs, shuffle=False):
        x = torch.from_numpy(X[batch]).to(device)
        preds.append(torch.argmax(model(x), 1).cpu().numpy())
    pred = np.concatenate(preds)
    true = y[idx]
    return accuracy_score(true, pred), f1_score(true, pred, average="macro")


def train(
    model,
    X,
    y,
    train_idx,
    val_idx,
    device,
    model_name,
    *,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs_override: int | None,
    eval_batch_size: int,
):
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = EPOCHS[model_name] if epochs_override is None else int(epochs_override)
    best_acc, best_state = 0.0, copy.deepcopy(model.state_dict())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("training", total=epochs)
        for epoch in range(epochs):
            model.train()
            for batch in batches(train_idx, batch_size, shuffle=True):
                x = torch.from_numpy(X[batch]).to(device)
                t = torch.from_numpy(y[batch]).to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(x), t).backward()
                opt.step()

            val_acc, val_f1 = evaluate(model, X, y, val_idx, device, bs=eval_batch_size)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            progress.update(
                task_id,
                advance=1,
                description=(
                    f"training epoch {epoch + 1}/{epochs} "
                    f"val_acc={val_acc:.4f} best={best_acc:.4f} val_f1={val_f1:.4f}"
                ),
            )

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train baseline classifiers on AnnData features."
    )
    p.add_argument("h5ad", type=Path)
    p.add_argument("--model", choices=("linear", "mlp", "deep-mlp"), required=True)
    p.add_argument("--layer", dest="layers", action="append", default=[])
    p.add_argument("--normalize-total", type=float, default=None)
    p.add_argument("--log1p", action="store_true")
    p.add_argument(
        "--align-avg",
        action="store_true",
        help="Scale each class so per-gene means match the largest class after preprocessing.",
    )
    p.add_argument(
        "--align-avg-eps",
        type=float,
        default=1e-6,
        help="Numerical floor used in align-avg scaling.",
    )
    p.add_argument(
        "--align-avg-min-scale",
        type=float,
        default=0.1,
        help="Lower bound for per-gene class scaling in align-avg.",
    )
    p.add_argument(
        "--align-avg-max-scale",
        type=float,
        default=10.0,
        help="Upper bound for per-gene class scaling in align-avg.",
    )
    p.add_argument(
        "--append-obs-key", dest="append_obs_keys", action="append", default=[]
    )
    p.add_argument("--label-column", default="treatment")
    p.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional gene list text file or gene-list JSON with gene_names.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WD)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--mlp-hidden-dim", type=int, default=HIDDEN["mlp"][0])
    p.add_argument(
        "--deep-hidden-dims",
        type=int,
        nargs="+",
        default=list(HIDDEN["deep-mlp"]),
    )
    return p.parse_args()


def main():
    start_time = perf_counter()
    args = parse_args()
    split_sum = args.train_fraction + args.val_fraction + args.test_fraction
    if min(args.train_fraction, args.val_fraction, args.test_fraction) <= 0:
        raise ValueError("split fractions must all be positive")
    if not np.isclose(split_sum, 1.0):
        raise ValueError("train/val/test fractions must sum to 1")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.align_avg_eps <= 0:
        raise ValueError("--align-avg-eps must be positive")
    if args.align_avg_min_scale <= 0 or args.align_avg_max_scale <= 0:
        raise ValueError("align-avg scale bounds must be positive")
    if args.align_avg_min_scale > args.align_avg_max_scale:
        raise ValueError("--align-avg-min-scale must be <= --align-avg-max-scale")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive when provided")
    if args.mlp_hidden_dim < 1 or any(dim < 1 for dim in args.deep_hidden_dims):
        raise ValueError("hidden dimensions must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    intro = Table(show_header=False, box=None)
    intro.add_row("Input", str(args.h5ad.expanduser().resolve()))
    intro.add_row("Model", args.model)
    intro.add_row("Label column", args.label_column)
    intro.add_row("Layers", ", ".join(args.layers) if args.layers else "X")
    intro.add_row(
        "Gene list",
        str(args.gene_list.expanduser().resolve()) if args.gene_list else "None",
    )
    intro.add_row("Align avg", str(bool(args.align_avg)))
    intro.add_row("Device", str(device))
    intro.add_row("Seed", str(args.seed))
    console.print(Panel(intro, title="Train Baseline", border_style="cyan"))

    with console.status("Loading AnnData and preparing labels..."):
        adata = load(args.h5ad, args.label_column, args.gene_list)
    labels, classes = (
        lambda s: (s.cat.codes.to_numpy(np.int64), s.cat.categories.tolist())
    )(adata.obs[args.label_column].astype("category"))
    class_lookup = {idx: name for idx, name in enumerate(classes)}

    with console.status("Building feature matrix and train/val/test split..."):
        X, align_info = build_features(adata, args, labels)
        train_idx, val_idx, test_idx = split(
            labels,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
    name = rep_name(args)

    data_summary = Table(title="Dataset Summary")
    data_summary.add_column("Field")
    data_summary.add_column("Value", overflow="fold")
    data_summary.add_row("Cells", str(adata.n_obs))
    data_summary.add_row("Genes", str(adata.n_vars))
    data_summary.add_row("Classes", str(len(classes)))
    data_summary.add_row("Representation", name)
    data_summary.add_row("Feature dim", str(X.shape[1]))
    if align_info is not None:
        data_summary.add_row(
            "Align avg ref class",
            f"{class_lookup[align_info['reference_label']]} ({align_info['reference_size']})",
        )
        data_summary.add_row(
            "Align avg scale range",
            f"{align_info['scale_min']:.4g} - {align_info['scale_max']:.4g}",
        )
        data_summary.add_row(
            "Align avg scale bounds",
            f"{align_info['min_scale']:.4g} - {align_info['max_scale']:.4g}",
        )
        data_summary.add_row(
            "Aligned expr dims",
            str(align_info["expr_dim"]),
        )
    data_summary.add_row(
        "Split sizes",
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
    )
    data_summary.add_row(
        "Split fractions",
        f"train={args.train_fraction:.2f} val={args.val_fraction:.2f} test={args.test_fraction:.2f}",
    )
    data_summary.add_row("LR / WD", f"{args.lr:g} / {args.weight_decay:g}")
    data_summary.add_row("Batch size", str(args.batch_size))
    data_summary.add_row("Eval batch size", str(args.eval_batch_size))
    data_summary.add_row(
        "Epochs",
        str(EPOCHS[args.model] if args.epochs is None else args.epochs),
    )
    console.print(data_summary)

    model = make_model(
        args.model,
        X.shape[1],
        len(classes),
        mlp_hidden_dim=args.mlp_hidden_dim,
        deep_hidden_dims=tuple(args.deep_hidden_dims),
    )
    model = train(
        model,
        X,
        labels,
        train_idx,
        val_idx,
        device,
        args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs_override=args.epochs,
        eval_batch_size=args.eval_batch_size,
    )

    acc, f1 = evaluate(model, X, labels, test_idx, device, bs=args.eval_batch_size)
    summary = Table(title="Training Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Test accuracy", f"{acc:.4f}")
    summary.add_row("Test macro F1", f"{f1:.4f}")
    summary.add_row("Elapsed", f"{perf_counter() - start_time:.2f}s")
    console.print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(str(exc), title="train_baseline failed", border_style="red")
        )
        raise SystemExit(1) from exc

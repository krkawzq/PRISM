#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prism.baseline.gene_mae import GeneMAE, GeneMAEConfig, masked_regression_loss
from prism.baseline.metrics import log1p_normalize_total

SIGNAL_LAYER = "signal"
POSTERIOR_ENTROPY_LAYER = "posterior_entropy"
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
WEIGHT_EPS = 1e-8
QUOTA_ERRNOS = {122, 28}
_IO_WARNED_PATHS: set[tuple[str, str]] = set()
MASK_SELECTION_CHOICES = ("random", "confidence_top_p")


@dataclass(frozen=True, slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True, slots=True)
class PretrainEpochRecord:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True, slots=True)
class ClassificationEpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


@dataclass(frozen=True, slots=True)
class GeneListSpec:
    source_path: str
    method: str
    top_k: int
    gene_indices: list[int]
    gene_names: list[str]
    scores: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GeneMAE for pretraining or downstream classification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser(
        "pretrain", help="Run self-supervised MAE pretraining."
    )
    add_shared_data_args(pretrain)
    add_model_args(pretrain)
    pretrain.add_argument("--output-dir", type=Path, required=True)
    pretrain.add_argument(
        "--input-representation",
        choices=("signal", "lognormal"),
        required=True,
        help="Input/target representation for masked reconstruction.",
    )
    pretrain.add_argument(
        "--loss-weighting",
        choices=("none", "posterior_confidence"),
        default="none",
        help="Apply w_gc = 1 - H(post_gc)/log(M), then normalize to mean 1.",
    )
    pretrain.add_argument(
        "--support-size",
        type=int,
        default=None,
        help="Support grid size M. Defaults to inferring from posterior entropy max.",
    )
    add_mask_selection_args(pretrain)
    pretrain.add_argument("--epochs", type=int, default=100)
    pretrain.add_argument("--batch-size", type=int, default=256)
    pretrain.add_argument("--lr", type=float, default=1e-3)
    pretrain.add_argument("--lr-min-ratio", type=float, default=0.01)
    pretrain.add_argument("--weight-decay", type=float, default=1e-4)
    pretrain.add_argument("--device", type=str, default=None)

    downstream = subparsers.add_parser(
        "downstream",
        help="Run downstream classification with pretrained or random backbone.",
    )
    add_shared_data_args(downstream)
    add_model_args(downstream)
    downstream.add_argument("--output-dir", type=Path, required=True)
    downstream.add_argument(
        "--input-representation",
        choices=("signal", "lognormal"),
        required=True,
        help="Feature representation used for downstream classification.",
    )
    downstream.add_argument(
        "--task-mode",
        choices=("zeroshot", "full"),
        required=True,
        help="zeroshot: freeze backbone and train head only; full: train backbone + linear head.",
    )
    downstream.add_argument(
        "--head",
        choices=("linear", "mlp"),
        default="linear",
        help="Classification head type. full mode forces linear regardless of this flag.",
    )
    downstream.add_argument("--label-column", type=str, default="treatment")
    downstream.add_argument("--checkpoint", type=Path, default=None)
    downstream.add_argument("--epochs", type=int, default=30)
    downstream.add_argument("--batch-size", type=int, default=512)
    downstream.add_argument("--lr", type=float, default=1e-4)
    downstream.add_argument("--lr-min-ratio", type=float, default=0.1)
    downstream.add_argument("--weight-decay", type=float, default=1e-4)
    downstream.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("h5ad_path", type=Path)
    parser.add_argument(
        "--gene-list-json",
        type=Path,
        default=None,
        help="Train only on the gene subset stored in this JSON file.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=True,
        help="Enable mixed precision training when supported (default: on).",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--save-every-epochs",
        "--save-every-steps",
        dest="save_every_epochs",
        type=int,
        default=10,
        help="Save a full training checkpoint every N epochs (0 disables).",
    )
    parser.add_argument(
        "--log-every-epochs",
        "--log-every-steps",
        dest="log_every_epochs",
        type=int,
        default=10,
        help="Append one epoch log record every N epochs (0 disables).",
    )
    parser.add_argument(
        "--keep-last-k-epoch-ckpts",
        "--keep-last-k-step-ckpts",
        dest="keep_last_k_epoch_ckpts",
        type=int,
        default=5,
        help="Keep only the most recent K epoch checkpoints (0 keeps all).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a full training checkpoint saved by this script.",
    )


def add_mask_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mask-selection",
        choices=MASK_SELECTION_CHOICES,
        default="random",
        help="Choose masked positions uniformly at random or only from high-confidence genes.",
    )
    parser.add_argument(
        "--mask-confidence-top-p",
        type=float,
        default=0.5,
        help="When using confidence_top_p masking, sample masked genes only from the top-p confidence fraction.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-gene-id", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--head-expand", type=float, default=1.0)
    parser.add_argument("--ffn-expand", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--ffn-dropout", type=float, default=0.1)
    parser.add_argument("--embed-dropout", type=float, default=0.1)
    parser.add_argument("--path-dropout", type=float, default=0.0)
    parser.add_argument("--signal-bins", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.2)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前不可用")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(*, enabled: bool, device: torch.device, dtype: torch.dtype | None):
    if not enabled or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def read_adata(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path.expanduser().resolve())


def load_gene_list_spec(path: Path) -> GeneListSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GeneListSpec(
        source_path=str(payload.get("source_path", "")),
        method=str(payload["method"]),
        top_k=int(payload["top_k"]),
        gene_indices=[int(v) for v in payload["gene_indices"]],
        gene_names=[str(v) for v in payload["gene_names"]],
        scores=[float(v) for v in payload.get("scores", [])],
    )


def subset_adata_by_gene_list(
    adata: ad.AnnData, spec: GeneListSpec
) -> tuple[ad.AnnData, np.ndarray]:
    name_to_idx = {str(name): idx for idx, name in enumerate(adata.var_names.tolist())}
    indices: list[int] = []
    for gene_name in spec.gene_names:
        if gene_name not in name_to_idx:
            raise KeyError(
                f"基因列表中的基因在 adata.var_names 中不存在: {gene_name!r}"
            )
        indices.append(int(name_to_idx[gene_name]))
    return adata[:, np.asarray(indices, dtype=np.int64)].copy(), np.asarray(
        indices, dtype=np.int64
    )


def maybe_subset_gene_list(
    adata: ad.AnnData, gene_list_json: Path | None
) -> tuple[ad.AnnData, GeneListSpec | None]:
    if gene_list_json is None:
        return adata, None
    spec = load_gene_list_spec(gene_list_json.expanduser().resolve())
    subset, _ = subset_adata_by_gene_list(adata, spec)
    return subset, spec


def compute_totals(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if matrix is None:
        raise ValueError("输入 h5ad 的 X 为空")
    if sparse.issparse(matrix):
        totals = np.asarray(cast(Any, matrix).sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix).sum(axis=1)
    return np.asarray(totals, dtype=np.float32)


def _stratify_or_none(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return None
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        return None
    return labels


def build_splits(n_obs: int, labels: np.ndarray | None = None) -> SplitIndices:
    all_idx = np.arange(n_obs, dtype=np.int64)
    stratify = _stratify_or_none(labels)
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    val_relative = VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    temp_labels = _stratify_or_none(labels[temp_idx] if labels is not None else None)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative),
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )
    return SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        val=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
    )


def encode_labels(adata: ad.AnnData, label_column: str) -> tuple[np.ndarray, list[str]]:
    if label_column not in adata.obs:
        raise KeyError(f"输入文件缺少标签列: {label_column!r}")
    series = adata.obs[label_column].astype("category")
    labels = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"标签列 {label_column!r} 包含缺失值")
    classes = [str(value) for value in series.cat.categories.tolist()]
    return labels, classes


def infer_support_size(adata: ad.AnnData) -> int:
    if POSTERIOR_ENTROPY_LAYER not in adata.layers:
        raise KeyError(f"输入文件缺少必需 layer: {POSTERIOR_ENTROPY_LAYER!r}")
    entropy = np.asarray(adata.layers[POSTERIOR_ENTROPY_LAYER], dtype=np.float64)
    finite = entropy[np.isfinite(entropy)]
    if finite.size == 0:
        raise ValueError("posterior_entropy 全部非有限，无法推断 support size")
    inferred = int(round(math.exp(float(finite.max()))))
    if inferred < 2:
        raise ValueError(f"推断得到的 support size 无效: {inferred}")
    return inferred


def build_model_config(
    args: argparse.Namespace, *, n_genes: int, n_classes: int
) -> GeneMAEConfig:
    return GeneMAEConfig(
        n_genes=n_genes,
        n_classes=n_classes,
        d_model=args.d_model,
        d_gene_id=args.d_gene_id,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_expand=args.head_expand,
        ffn_expand=args.ffn_expand,
        attn_dropout=args.attn_dropout,
        ffn_dropout=args.ffn_dropout,
        signal_bins=args.signal_bins,
        mask_ratio=args.mask_ratio,
        classification_head="linear",
        embed_dropout=args.embed_dropout,
        path_dropout=args.path_dropout,
    )


def _to_dense_float32(matrix_slice: Any) -> np.ndarray:
    if sparse.issparse(matrix_slice):
        dense = matrix_slice.toarray()
    else:
        dense = np.asarray(matrix_slice)
    return np.asarray(dense, dtype=np.float32)


def get_input_batch(
    adata: ad.AnnData,
    *,
    representation: str,
    batch_idx: np.ndarray,
    totals: np.ndarray,
    normalize_target: float,
) -> np.ndarray:
    if representation == "signal":
        if SIGNAL_LAYER not in adata.layers:
            raise KeyError(f"输入文件缺少必需 layer: {SIGNAL_LAYER!r}")
        values = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        return np.nan_to_num(values, copy=False)
    if representation == "log_signal":
        if SIGNAL_LAYER not in adata.layers:
            raise KeyError(f"输入文件缺少必需 layer: {SIGNAL_LAYER!r}")
        values = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        values = np.nan_to_num(values, copy=False)
        values = np.clip(values, a_min=0.0, a_max=None)
        return np.asarray(np.log1p(values), dtype=np.float32)
    if representation == "lognormal":
        matrix = adata.X
        if matrix is None:
            raise ValueError("输入 h5ad 的 X 为空")
        counts = _to_dense_float32(matrix[batch_idx])
        return np.asarray(
            log1p_normalize_total(counts, totals[batch_idx], target=normalize_target),
            dtype=np.float32,
        )
    raise ValueError(f"未知 input representation: {representation!r}")


def compute_weight_mean(
    adata: ad.AnnData,
    *,
    train_idx: np.ndarray,
    support_size: int,
    batch_size: int,
) -> float:
    if POSTERIOR_ENTROPY_LAYER not in adata.layers:
        raise KeyError(f"输入文件缺少必需 layer: {POSTERIOR_ENTROPY_LAYER!r}")
    entropy_scale = math.log(float(support_size))
    if entropy_scale <= 0.0:
        raise ValueError(f"support_size 必须 > 1，收到 {support_size}")
    total_sum = 0.0
    total_count = 0
    for batch in iter_batches(train_idx, batch_size, shuffle=False):
        entropy = _to_dense_float32(adata.layers[POSTERIOR_ENTROPY_LAYER][batch])
        weights = (
            1.0
            - np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0) / entropy_scale
        )
        weights = np.clip(weights, 0.0, None)
        total_sum += float(np.sum(weights, dtype=np.float64))
        total_count += int(weights.size)
    if total_count == 0:
        raise ValueError("训练集为空，无法计算损失权重均值")
    mean_weight = total_sum / total_count
    if mean_weight <= 0.0:
        raise ValueError("损失权重均值 <= 0，无法归一化")
    return float(mean_weight)


def get_weight_batch(
    adata: ad.AnnData,
    *,
    batch_idx: np.ndarray,
    support_size: int,
    mean_weight: float,
) -> np.ndarray:
    entropy = _to_dense_float32(adata.layers[POSTERIOR_ENTROPY_LAYER][batch_idx])
    weights = 1.0 - np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0) / math.log(
        float(support_size)
    )
    weights = np.clip(weights, 0.0, None)
    return np.asarray(weights / max(mean_weight, WEIGHT_EPS), dtype=np.float32)


def get_confidence_batch(
    adata: ad.AnnData,
    *,
    batch_idx: np.ndarray,
    support_size: int,
) -> np.ndarray:
    entropy = _to_dense_float32(adata.layers[POSTERIOR_ENTROPY_LAYER][batch_idx])
    weights = 1.0 - np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0) / math.log(
        float(support_size)
    )
    return np.asarray(np.clip(weights, 0.0, None), dtype=np.float32)


def create_confidence_top_p_mask(
    confidence: np.ndarray,
    *,
    mask_ratio: float,
    top_p: float,
    device: torch.device,
) -> torch.Tensor:
    if confidence.ndim != 2:
        raise ValueError(f"confidence 必须为二维，收到 shape={confidence.shape}")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio 必须在 (0, 1) 内，收到 {mask_ratio}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"mask-confidence-top-p 必须在 (0, 1] 内，收到 {top_p}")

    batch_size, n_genes = confidence.shape
    n_mask = max(1, int(round(n_genes * mask_ratio)))
    n_candidates = max(n_mask, int(round(n_genes * top_p)))
    n_candidates = min(n_genes, n_candidates)

    confidence_tensor = torch.from_numpy(confidence).to(
        device=device, dtype=torch.float32
    )
    _, candidate_idx = torch.topk(
        confidence_tensor, k=n_candidates, dim=1, largest=True
    )
    random_scores = torch.full(
        (batch_size, n_genes),
        float("inf"),
        device=device,
        dtype=torch.float32,
    )
    candidate_noise = torch.rand(batch_size, n_candidates, device=device)
    random_scores.scatter_(1, candidate_idx, candidate_noise)
    _, masked_idx = torch.topk(random_scores, k=n_mask, dim=1, largest=False)
    mask = torch.zeros(batch_size, n_genes, dtype=torch.bool, device=device)
    mask.scatter_(1, masked_idx, True)
    return mask


def create_pretrain_mask(
    *,
    model: Any,
    adata: ad.AnnData,
    batch_idx: np.ndarray,
    support_size: int,
    mask_selection: str,
    mask_confidence_top_p: float,
    device: torch.device,
) -> torch.Tensor | None:
    if mask_selection == "random":
        return None
    if mask_selection != "confidence_top_p":
        raise ValueError(f"未知 mask_selection: {mask_selection!r}")
    confidence_np = get_confidence_batch(
        adata,
        batch_idx=batch_idx,
        support_size=support_size,
    )
    return create_confidence_top_p_mask(
        confidence_np,
        mask_ratio=model.config.mask_ratio,
        top_p=mask_confidence_top_p,
        device=device,
    )


def iter_batches(
    indices: np.ndarray, batch_size: int, *, shuffle: bool
) -> list[np.ndarray]:
    order = indices.copy()
    if shuffle:
        np.random.shuffle(order)
    return [
        order[start : start + batch_size] for start in range(0, len(order), batch_size)
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    except OSError as exc:
        if not _is_quota_error(exc):
            raise
        _warn_io_once("json", path, exc)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_ready(payload), ensure_ascii=True) + "\n")
    except OSError as exc:
        if not _is_quota_error(exc):
            raise
        _warn_io_once("jsonl", path, exc)


def _is_quota_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in QUOTA_ERRNOS


def _warn_io_once(kind: str, path: Path, exc: BaseException) -> None:
    key = (kind, str(path))
    if key in _IO_WARNED_PATHS:
        return
    _IO_WARNED_PATHS.add(key)
    print(
        f"warning: skip writing {kind} to {path} ({exc})",
        file=sys.stderr,
    )


def prune_epoch_checkpoints(epoch_dir: Path, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    ckpts = sorted(epoch_dir.glob("epoch_*.pt"))
    if len(ckpts) <= keep_last_k:
        return
    for path in ckpts[: len(ckpts) - keep_last_k]:
        path.unlink(missing_ok=True)


def save_training_checkpoint(
    path: Path,
    *,
    model: GeneMAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    model_config: GeneMAEConfig,
    mode: str,
    input_representation: str,
    normalize_target: float,
    support_size: int | None,
    loss_weighting: str | None,
    weight_mean: float | None,
    label_column: str | None,
    task_mode: str | None,
    class_names: list[str] | None,
    gene_list_spec: GeneListSpec | None,
    epoch: int,
    global_step: int,
    best_metric_name: str,
    best_metric_value: float,
    best_model_state: dict[str, Any] | None,
    run_args: argparse.Namespace,
) -> None:
    payload = {
        "mode": mode,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "model_config": asdict(model_config),
        "input_representation": input_representation,
        "normalize_target": normalize_target,
        "support_size": support_size,
        "loss_weighting": loss_weighting,
        "weight_mean": weight_mean,
        "label_column": label_column,
        "task_mode": task_mode,
        "class_names": class_names,
        "gene_list_spec": None if gene_list_spec is None else asdict(gene_list_spec),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_model_state": best_model_state,
        "run_args": _json_ready(vars(run_args)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            try:
                torch.save(payload, fh)
            except RuntimeError as exc:
                message = str(exc)
                if "unexpected pos" not in message and "iostream error" not in message:
                    raise
                fh.seek(0)
                fh.truncate()
                torch.save(payload, fh, _use_new_zipfile_serialization=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if _is_quota_error(exc):
            _warn_io_once("checkpoint", path, exc)
            return
        raise


def maybe_save_epoch_checkpoint(
    *,
    output_dir: Path,
    save_every_epochs: int,
    keep_last_k_epoch_ckpts: int,
    global_step: int,
    model: GeneMAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    model_config: GeneMAEConfig,
    mode: str,
    input_representation: str,
    normalize_target: float,
    support_size: int | None,
    loss_weighting: str | None,
    weight_mean: float | None,
    label_column: str | None,
    task_mode: str | None,
    class_names: list[str] | None,
    gene_list_spec: GeneListSpec | None,
    epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    best_model_state: dict[str, Any] | None,
    run_args: argparse.Namespace,
) -> None:
    if save_every_epochs <= 0 or epoch % save_every_epochs != 0:
        return
    epoch_dir = output_dir / "epoch_ckpts"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = epoch_dir / f"epoch_{epoch:04d}.pt"
    save_training_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        model_config=model_config,
        mode=mode,
        input_representation=input_representation,
        normalize_target=normalize_target,
        support_size=support_size,
        loss_weighting=loss_weighting,
        weight_mean=weight_mean,
        label_column=label_column,
        task_mode=task_mode,
        class_names=class_names,
        gene_list_spec=gene_list_spec,
        epoch=epoch,
        global_step=global_step,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        best_model_state=best_model_state,
        run_args=run_args,
    )
    prune_epoch_checkpoints(epoch_dir, keep_last_k_epoch_ckpts)


def load_resume_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("resume checkpoint 不是合法字典")
    return checkpoint


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    lr_min_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    if not 0.0 <= lr_min_ratio <= 1.0:
        raise ValueError(f"lr_min_ratio 必须在 [0, 1] 内，收到 {lr_min_ratio}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=optimizer.param_groups[0]["lr"] * lr_min_ratio,
    )


def train_pretrain(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device)
    amp_enabled = bool(args.amp and amp_dtype is not None)
    scaler = cast(Any, torch.amp).GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    adata_full = read_adata(args.h5ad_path)
    totals = compute_totals(adata_full)
    adata, gene_list_spec = maybe_subset_gene_list(adata_full, args.gene_list_json)
    splits = build_splits(int(adata.n_obs))
    normalize_target = float(np.median(totals[splits.train]))
    support_size = (
        int(args.support_size)
        if args.support_size is not None
        else infer_support_size(adata)
    )
    weight_mean = None
    if args.loss_weighting != "none":
        weight_mean = compute_weight_mean(
            adata,
            train_idx=splits.train,
            support_size=support_size,
            batch_size=args.batch_size,
        )

    config = build_model_config(args, n_genes=int(adata.n_vars), n_classes=1)
    model = GeneMAE(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_batches = iter_batches(splits.train, args.batch_size, shuffle=False)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=max(len(train_batches) * args.epochs, 1),
        lr_min_ratio=args.lr_min_ratio,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "pretrain_history.json"
    metrics_path = output_dir / "metrics.jsonl"
    args_path = output_dir / "args.json"
    write_json(args_path, {"args": vars(args)})

    history: list[PretrainEpochRecord] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    global_step = 0
    start_epoch = 1

    if args.resume is not None:
        resume_ckpt = load_resume_checkpoint(args.resume)
        if resume_ckpt.get("mode") != "pretrain":
            raise ValueError("resume checkpoint 不是 pretrain 模式")
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        scaler_state = resume_ckpt.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_val_loss = float(resume_ckpt.get("best_metric_value", float("inf")))
        best_state = copy.deepcopy(
            resume_ckpt.get("best_model_state", model.state_dict())
        )

    print_pretrain_summary(
        adata=adata,
        args=args,
        device=device,
        splits=splits,
        normalize_target=normalize_target,
        support_size=support_size,
        weight_mean=weight_mean,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        gene_list_spec=gene_list_spec,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_idx in tqdm(
            iter_batches(splits.train, args.batch_size, shuffle=True),
            desc=f"pretrain epoch {epoch}/{args.epochs}",
        ):
            inputs_np = get_input_batch(
                adata,
                representation=args.input_representation,
                batch_idx=batch_idx,
                totals=totals,
                normalize_target=normalize_target,
            )
            weight_np = None
            if args.loss_weighting != "none":
                weight_np = get_weight_batch(
                    adata,
                    batch_idx=batch_idx,
                    support_size=support_size,
                    mean_weight=float(weight_mean if weight_mean is not None else 1.0),
                )

            inputs = torch.from_numpy(inputs_np).to(device)
            weights = (
                None if weight_np is None else torch.from_numpy(weight_np).to(device)
            )
            mask = create_pretrain_mask(
                model=model,
                adata=adata,
                batch_idx=batch_idx,
                support_size=support_size,
                mask_selection=args.mask_selection,
                mask_confidence_top_p=args.mask_confidence_top_p,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
                output = model.forward_pretrain(
                    inputs,
                    mask=mask,
                    return_token_embeddings=False,
                )
                if output.mask is None or output.masked_predictions is None:
                    raise RuntimeError("MAE 预训练输出缺少 mask 或 masked_predictions")
                targets, masked_weights = model.masked_targets(
                    inputs, output.mask, weights=weights
                )
                loss = masked_regression_loss(
                    output.masked_predictions, targets, weights=masked_weights
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            train_losses.append(float(loss.item()))

        val_loss = evaluate_pretrain(
            model=model,
            adata=adata,
            indices=splits.val,
            totals=totals,
            normalize_target=normalize_target,
            representation=args.input_representation,
            batch_size=args.batch_size,
            support_size=support_size,
            mean_weight=weight_mean,
            weighted=(args.loss_weighting != "none"),
            mask_selection=args.mask_selection,
            mask_confidence_top_p=args.mask_confidence_top_p,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        history.append(
            PretrainEpochRecord(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        )
        if args.log_every_epochs > 0 and epoch % args.log_every_epochs == 0:
            append_jsonl(
                metrics_path,
                {
                    "mode": "pretrain",
                    "phase": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )
        append_jsonl(
            metrics_path,
            {
                "mode": "pretrain",
                "phase": "val",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss_before_update": best_val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
        )
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=config,
                mode="pretrain",
                input_representation=args.input_representation,
                normalize_target=normalize_target,
                support_size=support_size,
                loss_weighting=args.loss_weighting,
                weight_mean=weight_mean,
                label_column=None,
                task_mode=None,
                class_names=None,
                gene_list_spec=gene_list_spec,
                epoch=epoch,
                global_step=global_step,
                best_metric_name="val_loss",
                best_metric_value=best_val_loss,
                best_model_state=best_state,
                run_args=args,
            )
        save_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=config,
            mode="pretrain",
            input_representation=args.input_representation,
            normalize_target=normalize_target,
            support_size=support_size,
            loss_weighting=args.loss_weighting,
            weight_mean=weight_mean,
            label_column=None,
            task_mode=None,
            class_names=None,
            gene_list_spec=gene_list_spec,
            epoch=epoch,
            global_step=global_step,
            best_metric_name="val_loss",
            best_metric_value=best_val_loss,
            best_model_state=best_state,
            run_args=args,
        )
        maybe_save_epoch_checkpoint(
            output_dir=output_dir,
            save_every_epochs=args.save_every_epochs,
            keep_last_k_epoch_ckpts=args.keep_last_k_epoch_ckpts,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=config,
            mode="pretrain",
            input_representation=args.input_representation,
            normalize_target=normalize_target,
            support_size=support_size,
            loss_weighting=args.loss_weighting,
            weight_mean=weight_mean,
            label_column=None,
            task_mode=None,
            class_names=None,
            gene_list_spec=gene_list_spec,
            epoch=epoch,
            best_metric_name="val_loss",
            best_metric_value=best_val_loss,
            best_model_state=best_state,
            run_args=args,
        )
        history_payload = {
            "records": [asdict(record) for record in history],
            "best_val_loss": best_val_loss,
            "last_epoch": epoch,
            "global_step": global_step,
        }
        write_json(history_path, history_payload)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={best_val_loss:.6f}"
        )

    model.load_state_dict(best_state)
    test_loss = evaluate_pretrain(
        model=model,
        adata=adata,
        indices=splits.test,
        totals=totals,
        normalize_target=normalize_target,
        representation=args.input_representation,
        batch_size=args.batch_size,
        support_size=support_size,
        mean_weight=weight_mean,
        weighted=(args.loss_weighting != "none"),
        mask_selection=args.mask_selection,
        mask_confidence_top_p=args.mask_confidence_top_p,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    history_payload = {
        "records": [asdict(record) for record in history],
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "global_step": global_step,
    }
    write_json(history_path, history_payload)
    append_jsonl(
        metrics_path,
        {
            "mode": "pretrain",
            "phase": "test",
            "epoch": args.epochs,
            "global_step": global_step,
            "test_loss": test_loss,
            "best_val_loss": best_val_loss,
        },
    )
    print(f"saved checkpoint: {best_path}")
    print(f"saved history   : {history_path}")
    print(f"final test loss : {test_loss:.6f}")


@torch.inference_mode()
def evaluate_pretrain(
    *,
    model: GeneMAE,
    adata: ad.AnnData,
    indices: np.ndarray,
    totals: np.ndarray,
    normalize_target: float,
    representation: str,
    batch_size: int,
    support_size: int,
    mean_weight: float | None,
    weighted: bool,
    mask_selection: str,
    mask_confidence_top_p: float,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx in tqdm(
        iter_batches(indices, batch_size, shuffle=False), desc="eval", leave=False
    ):
        inputs_np = get_input_batch(
            adata,
            representation=representation,
            batch_idx=batch_idx,
            totals=totals,
            normalize_target=normalize_target,
        )
        weight_np = None
        if weighted:
            weight_np = get_weight_batch(
                adata,
                batch_idx=batch_idx,
                support_size=support_size,
                mean_weight=float(mean_weight if mean_weight is not None else 1.0),
            )
        inputs = torch.from_numpy(inputs_np).to(device)
        weights = None if weight_np is None else torch.from_numpy(weight_np).to(device)
        mask = create_pretrain_mask(
            model=model,
            adata=adata,
            batch_idx=batch_idx,
            support_size=support_size,
            mask_selection=mask_selection,
            mask_confidence_top_p=mask_confidence_top_p,
            device=device,
        )
        with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
            output = model.forward_pretrain(
                inputs,
                mask=mask,
                return_token_embeddings=False,
            )
            if output.mask is None or output.masked_predictions is None:
                raise RuntimeError("MAE 预训练输出缺少 mask 或 masked_predictions")
            targets, masked_weights = model.masked_targets(
                inputs, output.mask, weights=weights
            )
            loss = masked_regression_loss(
                output.masked_predictions, targets, weights=masked_weights
            )
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


def train_downstream(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device)
    amp_enabled = bool(args.amp and amp_dtype is not None)
    scaler = cast(Any, torch.amp).GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint.expanduser().resolve(), map_location="cpu"
        )
        pretrain_config = GeneMAEConfig(**checkpoint["model_config"])
        checkpoint_gene_list = checkpoint.get("gene_list_spec")
    else:
        checkpoint = None
        pretrain_config = None
        checkpoint_gene_list = None

    adata_full = read_adata(args.h5ad_path)
    totals = compute_totals(adata_full)
    if args.gene_list_json is not None:
        adata, gene_list_spec = maybe_subset_gene_list(adata_full, args.gene_list_json)
    elif checkpoint_gene_list is not None:
        gene_list_spec = GeneListSpec(**checkpoint_gene_list)
        adata, _ = subset_adata_by_gene_list(adata_full, gene_list_spec)
    else:
        adata, gene_list_spec = adata_full, None
    labels, class_names = encode_labels(adata, args.label_column)
    splits = build_splits(int(adata.n_obs), labels=labels)
    normalize_target = float(np.median(totals[splits.train]))

    if checkpoint is not None:
        normalize_target = float(checkpoint.get("normalize_target", normalize_target))
    else:
        pretrain_config = build_model_config(
            args, n_genes=int(adata.n_vars), n_classes=1
        )
    if pretrain_config is None:
        raise RuntimeError("缺少预训练配置")

    classification_head = "linear" if args.task_mode == "full" else args.head
    config = replace(
        pretrain_config,
        n_classes=len(class_names),
        classification_head=classification_head,
    )
    model = GeneMAE(config).to(device)

    if checkpoint is not None:
        state_dict = dict(checkpoint["model_state"])
        for key in [
            name for name in state_dict if name.startswith("classification_head.")
        ]:
            state_dict.pop(key)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"loaded checkpoint: {args.checkpoint}")
        print(f"missing keys   : {missing}")
        print(f"unexpected keys: {unexpected}")

    freeze_backbone = args.task_mode == "zeroshot"
    for name, param in model.named_parameters():
        is_head = name.startswith("classification_head.")
        param.requires_grad = is_head or not freeze_backbone

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_batches = iter_batches(splits.train, args.batch_size, shuffle=False)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=max(len(train_batches) * args.epochs, 1),
        lr_min_ratio=args.lr_min_ratio,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "downstream_history.json"
    metrics_path = output_dir / "metrics.jsonl"
    args_path = output_dir / "args.json"
    write_json(args_path, {"args": vars(args)})

    print_downstream_summary(
        adata=adata,
        args=args,
        device=device,
        splits=splits,
        normalize_target=normalize_target,
        class_names=class_names,
        freeze_backbone=freeze_backbone,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        gene_list_spec=gene_list_spec,
    )

    history: list[ClassificationEpochRecord] = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    global_step = 0
    start_epoch = 1

    if args.resume is not None:
        resume_ckpt = load_resume_checkpoint(args.resume)
        if resume_ckpt.get("mode") != "downstream":
            raise ValueError("resume checkpoint 不是 downstream 模式")
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        scaler_state = resume_ckpt.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_val_acc = float(resume_ckpt.get("best_metric_value", -1.0))
        best_state = copy.deepcopy(
            resume_ckpt.get("best_model_state", model.state_dict())
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for batch_idx in tqdm(
            iter_batches(splits.train, args.batch_size, shuffle=True),
            desc=f"downstream epoch {epoch}/{args.epochs}",
        ):
            features = get_input_batch(
                adata,
                representation=args.input_representation,
                batch_idx=batch_idx,
                totals=totals,
                normalize_target=normalize_target,
            )
            x = torch.from_numpy(features).to(device)
            y = torch.from_numpy(labels[batch_idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
                output = model.forward_classify(x, return_token_embeddings=False)
                if output.logits is None:
                    raise RuntimeError("分类输出缺少 logits")
                loss = loss_fn(output.logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            pred = torch.argmax(output.logits, dim=1)
            train_losses.append(float(loss.item()))
            train_true.append(y.cpu().numpy())
            train_pred.append(pred.cpu().numpy())

        train_acc = float(
            accuracy_score(np.concatenate(train_true), np.concatenate(train_pred))
        )
        val_loss, val_true, val_pred = evaluate_downstream(
            model=model,
            adata=adata,
            indices=splits.val,
            labels=labels,
            totals=totals,
            normalize_target=normalize_target,
            representation=args.input_representation,
            batch_size=args.batch_size,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_acc = float(accuracy_score(val_true, val_pred))
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        history.append(
            ClassificationEpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )
        if args.log_every_epochs > 0 and epoch % args.log_every_epochs == 0:
            append_jsonl(
                metrics_path,
                {
                    "mode": "downstream",
                    "phase": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=config,
                mode="downstream",
                input_representation=args.input_representation,
                normalize_target=normalize_target,
                support_size=None,
                loss_weighting=None,
                weight_mean=None,
                label_column=args.label_column,
                task_mode=args.task_mode,
                class_names=class_names,
                gene_list_spec=gene_list_spec,
                epoch=epoch,
                global_step=global_step,
                best_metric_name="val_acc",
                best_metric_value=best_val_acc,
                best_model_state=best_state,
                run_args=args,
            )
        save_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=config,
            mode="downstream",
            input_representation=args.input_representation,
            normalize_target=normalize_target,
            support_size=None,
            loss_weighting=None,
            weight_mean=None,
            label_column=args.label_column,
            task_mode=args.task_mode,
            class_names=class_names,
            gene_list_spec=gene_list_spec,
            epoch=epoch,
            global_step=global_step,
            best_metric_name="val_acc",
            best_metric_value=best_val_acc,
            best_model_state=best_state,
            run_args=args,
        )
        maybe_save_epoch_checkpoint(
            output_dir=output_dir,
            save_every_epochs=args.save_every_epochs,
            keep_last_k_epoch_ckpts=args.keep_last_k_epoch_ckpts,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=config,
            mode="downstream",
            input_representation=args.input_representation,
            normalize_target=normalize_target,
            support_size=None,
            loss_weighting=None,
            weight_mean=None,
            label_column=args.label_column,
            task_mode=args.task_mode,
            class_names=class_names,
            gene_list_spec=gene_list_spec,
            epoch=epoch,
            best_metric_name="val_acc",
            best_metric_value=best_val_acc,
            best_model_state=best_state,
            run_args=args,
        )
        history_payload = {
            "records": [asdict(record) for record in history],
            "best_val_acc": best_val_acc,
            "last_epoch": epoch,
            "global_step": global_step,
        }
        write_json(history_path, history_payload)
        append_jsonl(
            metrics_path,
            {
                "mode": "downstream",
                "phase": "val",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} best_val={best_val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    test_loss, test_true, test_pred = evaluate_downstream(
        model=model,
        adata=adata,
        indices=splits.test,
        labels=labels,
        totals=totals,
        normalize_target=normalize_target,
        representation=args.input_representation,
        batch_size=args.batch_size,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    result = summarize_classification(test_true, test_pred)
    history_payload = {
        "records": [asdict(record) for record in history],
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_metrics": asdict(result),
        "classes": class_names,
        "global_step": global_step,
    }
    write_json(history_path, history_payload)
    append_jsonl(
        metrics_path,
        {
            "mode": "downstream",
            "phase": "test",
            "epoch": args.epochs,
            "global_step": global_step,
            "test_loss": test_loss,
            "test_metrics": asdict(result),
            "best_val_acc": best_val_acc,
        },
    )
    print(f"saved checkpoint: {best_path}")
    print(f"saved history   : {history_path}")
    print_classification_result(result)


@torch.inference_mode()
def evaluate_downstream(
    *,
    model: GeneMAE,
    adata: ad.AnnData,
    indices: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    normalize_target: float,
    representation: str,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch_idx in tqdm(
        iter_batches(indices, batch_size, shuffle=False), desc="eval", leave=False
    ):
        features = get_input_batch(
            adata,
            representation=representation,
            batch_idx=batch_idx,
            totals=totals,
            normalize_target=normalize_target,
        )
        x = torch.from_numpy(features).to(device)
        y = torch.from_numpy(labels[batch_idx]).to(device)
        with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
            output = model.forward_classify(x, return_token_embeddings=False)
            if output.logits is None:
                raise RuntimeError("分类输出缺少 logits")
            loss = loss_fn(output.logits, y)
        pred = torch.argmax(output.logits, dim=1)
        losses.append(float(loss.item()))
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    return (
        float(np.mean(losses)) if losses else math.nan,
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
    )


def summarize_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> ClassificationResult:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=cast(Any, 0.0),
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=cast(Any, 0.0),
        )
    )
    return ClassificationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
    )


def print_pretrain_summary(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    device: torch.device,
    splits: SplitIndices,
    normalize_target: float,
    support_size: int,
    weight_mean: float | None,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    gene_list_spec: GeneListSpec | None,
) -> None:
    print("=" * 80)
    print("GeneMAE Pretrain")
    print(f"Dataset           : {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"Gene subset       : {gene_list_spec.top_k if gene_list_spec is not None else 'all'}"
    )
    print(f"Input             : {args.input_representation}")
    print(f"Loss weighting    : {args.loss_weighting}")
    print(f"Mask selection    : {args.mask_selection}")
    if args.mask_selection == "confidence_top_p":
        print(f"Mask top-p        : {args.mask_confidence_top_p}")
    print(f"AMP               : {amp_enabled} ({amp_dtype})")
    print(f"Support size      : {support_size}")
    print(f"Weight mean       : {weight_mean}")
    print(f"Normalize target  : {normalize_target:.4f}")
    print(f"Device            : {device}")
    print(f"Resume            : {args.resume}")
    print(f"Save every epochs : {args.save_every_epochs}")
    print(f"Log every epochs  : {args.log_every_epochs}")
    print(
        f"Split sizes       : train={splits.train.size} val={splits.val.size} test={splits.test.size}"
    )
    print(f"Output dir        : {args.output_dir}")
    print("=" * 80)


def print_downstream_summary(
    *,
    adata: ad.AnnData,
    args: argparse.Namespace,
    device: torch.device,
    splits: SplitIndices,
    normalize_target: float,
    class_names: list[str],
    freeze_backbone: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    gene_list_spec: GeneListSpec | None,
) -> None:
    print("=" * 80)
    print("GeneMAE Downstream")
    print(f"Dataset           : {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"Gene subset       : {gene_list_spec.top_k if gene_list_spec is not None else 'all'}"
    )
    print(f"Input             : {args.input_representation}")
    print(f"Task mode         : {args.task_mode}")
    print(f"Head              : {'linear' if args.task_mode == 'full' else args.head}")
    print(f"Freeze backbone   : {freeze_backbone}")
    print(f"AMP               : {amp_enabled} ({amp_dtype})")
    print(f"Normalize target  : {normalize_target:.4f}")
    print(f"Classes           : {len(class_names)} -> {class_names}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Resume            : {args.resume}")
    print(f"Label column      : {args.label_column}")
    print(f"Device            : {device}")
    print(f"Save every epochs : {args.save_every_epochs}")
    print(f"Log every epochs  : {args.log_every_epochs}")
    print(
        f"Split sizes       : train={splits.train.size} val={splits.val.size} test={splits.test.size}"
    )
    print(f"Output dir        : {args.output_dir}")
    print("=" * 80)


def print_classification_result(result: ClassificationResult) -> None:
    print("Test Metrics")
    print(f"  accuracy          : {result.accuracy:.4f}")
    print(f"  precision_macro   : {result.precision_macro:.4f}")
    print(f"  recall_macro      : {result.recall_macro:.4f}")
    print(f"  f1_macro          : {result.f1_macro:.4f}")
    print(f"  precision_weighted: {result.precision_weighted:.4f}")
    print(f"  recall_weighted   : {result.recall_weighted:.4f}")
    print(f"  f1_weighted       : {result.f1_weighted:.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "pretrain":
        train_pretrain(args)
        return
    if args.command == "downstream":
        train_downstream(args)
        return
    raise ValueError(f"未知 command: {args.command!r}")


if __name__ == "__main__":
    main()

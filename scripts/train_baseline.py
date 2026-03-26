#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

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

from prism.baseline import DeepMLPClassifier, LinearClassifier, MLPClassifier
from prism.baseline.metrics import log1p_normalize_total

SIGNAL_LAYER = "signal"
CONFIDENCE_LAYER = "confidence"
LABEL_COLUMN = "treatment"
BATCH_COLUMN = "batch"
RANDOM_SEED = 0
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
PRECOMPUTE_BATCH_SIZE = 2048
LINEAR_EPOCHS = 30
MLP_EPOCHS = 50
DEEP_MLP_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MLP_HIDDEN_DIM = 1024
DEEP_MLP_HIDDEN_DIMS = (1024, 512, 256)

USE_RAW_UMI = True
USE_RAW_UMI_PLUS_TOTAL = False
USE_LOG1P_NORMALIZED = True
USE_LOG1P_NORMALIZED_PLUS_TOTAL = False
USE_SIGNAL = True
USE_SIGNAL_PLUS_TOTAL = False
USE_SIGNAL_X_CONFIDENCE = False
USE_SIGNAL_X_CONFIDENCE_PLUS_TOTAL = False
USE_SIGNAL_CONFIDENCE_CONCAT = False
USE_SIGNAL_CONFIDENCE_CONCAT_PLUS_TOTAL = False
USE_SIGNAL_CONFIDENCE_THRESHOLD = False

CONFIDENCE_ZERO_THRESHOLD_QUANTILE = 0.001
ALL_REPRESENTATIONS = (
    "raw_umi",
    "raw_umi_plus_total",
    "log1p_normalized",
    "log1p_normalized_plus_total",
    "signal",
    "signal_plus_total",
    "signal_x_confidence",
    "signal_x_confidence_plus_total",
    "signal_confidence_concat",
    "signal_confidence_concat_plus_total",
    "signal_confidence_thresholded",
)


@dataclass(frozen=True, slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    split_mode: str = "random"
    train_batch_values: tuple[str, ...] = ()
    val_batch_values: tuple[str, ...] = ()
    test_batch_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RepresentationResult:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


@dataclass(frozen=True, slots=True)
class FeatureCache:
    raw_umi: np.ndarray | None = None
    log1p_normalized: np.ndarray | None = None
    signal: np.ndarray | None = None
    confidence: np.ndarray | None = None
    confidence_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class GeneListSpec:
    source_path: str
    method: str
    top_k: int
    gene_indices: list[int]
    gene_names: list[str]
    scores: list[float]


@dataclass(frozen=True, slots=True)
class LabelFilterSpec:
    label_column: str
    top_k: int
    selected_labels: tuple[str, ...]
    original_n_obs: int
    filtered_n_obs: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    train_batch_size: int
    eval_batch_size: int
    precompute_batch_size: int
    eval_every: int
    cache_dense_representations: bool


def enabled_representations() -> list[str]:
    items: list[tuple[bool, str]] = [
        (USE_RAW_UMI, "raw_umi"),
        (USE_RAW_UMI_PLUS_TOTAL, "raw_umi_plus_total"),
        (USE_LOG1P_NORMALIZED, "log1p_normalized"),
        (USE_LOG1P_NORMALIZED_PLUS_TOTAL, "log1p_normalized_plus_total"),
        (USE_SIGNAL, "signal"),
        (USE_SIGNAL_PLUS_TOTAL, "signal_plus_total"),
        (USE_SIGNAL_X_CONFIDENCE, "signal_x_confidence"),
        (USE_SIGNAL_X_CONFIDENCE_PLUS_TOTAL, "signal_x_confidence_plus_total"),
        (USE_SIGNAL_CONFIDENCE_CONCAT, "signal_confidence_concat"),
        (
            USE_SIGNAL_CONFIDENCE_CONCAT_PLUS_TOTAL,
            "signal_confidence_concat_plus_total",
        ),
        (USE_SIGNAL_CONFIDENCE_THRESHOLD, "signal_confidence_thresholded"),
    ]
    return [name for enabled, name in items if enabled]


def parse_representations(values: list[str] | None) -> list[str]:
    if not values:
        return enabled_representations()
    unknown = sorted(set(values) - set(ALL_REPRESENTATIONS))
    if unknown:
        raise ValueError(f"未知表示: {unknown}")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple linear or MLP baseline on raw UMI, log1p-normalized, and signal features."
        )
    )
    parser.add_argument("h5ad_path", type=Path, help="Input extracted h5ad file.")
    parser.add_argument(
        "--model",
        choices=("linear", "mlp", "deep-mlp"),
        required=True,
        help="Baseline model type.",
    )
    parser.add_argument(
        "--gene-list-json",
        type=Path,
        default=None,
        help="Use only the gene subset stored in this JSON file.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=LABEL_COLUMN,
        help="obs column used as the classification target.",
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=None,
        help="Keep only the top-K most frequent labels in --label-column.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("random", "batch"),
        default="random",
        help="How to split train/val/test: random cells or whole batches.",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default=BATCH_COLUMN,
        help="obs column used to define batches when --split-mode=batch.",
    )
    parser.add_argument(
        "--val-batch",
        dest="val_batches",
        action="append",
        default=None,
        help="Validation batch value. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--test-batch",
        dest="test_batches",
        action="append",
        default=None,
        help="Test batch value. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--representation",
        dest="representations",
        action="append",
        choices=ALL_REPRESENTATIONS,
        help="Representation to train. Repeatable; defaults to the script's enabled set.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=EVAL_BATCH_SIZE,
        help="Validation/test batch size.",
    )
    parser.add_argument(
        "--precompute-batch-size",
        type=int,
        default=PRECOMPUTE_BATCH_SIZE,
        help="Batch size used when precomputing dense features.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--cache-dense-representations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute dense raw/signal matrices once to reduce per-batch CPU overhead.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 cuda，但当前 PyTorch 看不到可用 GPU")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_adata(h5ad_path: Path) -> ad.AnnData:
    return ad.read_h5ad(h5ad_path)


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


def maybe_subset_gene_list(
    adata: ad.AnnData, gene_list_json: Path | None
) -> tuple[ad.AnnData, GeneListSpec | None]:
    if gene_list_json is None:
        return adata, None
    spec = load_gene_list_spec(gene_list_json.expanduser().resolve())
    name_to_idx = {str(name): idx for idx, name in enumerate(adata.var_names.tolist())}
    indices = np.asarray(
        [name_to_idx[name] for name in spec.gene_names], dtype=np.int64
    )
    subset_view = adata[:, indices]
    subset = subset_view.to_memory() if adata.isbacked else subset_view.copy()
    return subset, spec


def maybe_subset_top_labels(
    adata: ad.AnnData,
    *,
    label_column: str,
    label_top_k: int | None,
) -> tuple[ad.AnnData, LabelFilterSpec | None]:
    if label_top_k is None:
        return adata, None
    if label_top_k < 1:
        raise ValueError("--label-top-k 必须 >= 1")
    if label_column not in adata.obs:
        raise KeyError(f"输入文件缺少标签列: {label_column!r}")

    series = adata.obs[label_column].astype(str)
    counts = series.value_counts(sort=True)
    if counts.empty:
        raise ValueError(f"标签列 {label_column!r} 为空，无法筛选 top-k")

    top_labels = tuple(str(label) for label in counts.index[:label_top_k].tolist())
    if not top_labels:
        raise ValueError("筛选 top-k 标签后结果为空")
    mask = series.isin(top_labels).to_numpy(dtype=bool, copy=False)
    filtered_view = adata[mask]
    filtered = filtered_view.to_memory() if adata.isbacked else filtered_view.copy()
    spec = LabelFilterSpec(
        label_column=label_column,
        top_k=min(label_top_k, int(counts.shape[0])),
        selected_labels=top_labels,
        original_n_obs=int(adata.n_obs),
        filtered_n_obs=int(filtered.n_obs),
    )
    return filtered, spec


def encode_labels(adata: ad.AnnData, label_column: str) -> tuple[np.ndarray, list[str]]:
    if label_column not in adata.obs:
        raise KeyError(f"输入文件缺少标签列: {label_column!r}")
    series = adata.obs[label_column].astype("category")
    labels = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"标签列 {label_column!r} 包含缺失值")
    class_names = [str(value) for value in series.cat.categories.tolist()]
    return labels, class_names


def get_batch_values(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    if batch_key not in adata.obs:
        raise KeyError(f"输入文件缺少 batch 列: {batch_key!r}")
    values = adata.obs[batch_key].astype(str).to_numpy(copy=False)
    if np.any(values == ""):
        raise ValueError(f"batch 列 {batch_key!r} 包含空字符串")
    return np.asarray(values, dtype=object)


def compute_totals(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if matrix is None:
        raise ValueError("输入 h5ad 的 X 为空，无法计算 raw/log1p baseline")
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


def build_splits(labels: np.ndarray) -> SplitIndices:
    all_idx = np.arange(labels.shape[0], dtype=np.int64)
    stratify_labels = _stratify_or_none(labels)
    train_idx, temp_idx, _train_y, temp_y = train_test_split(
        all_idx,
        labels,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )
    val_relative = VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    temp_stratify = _stratify_or_none(np.asarray(temp_y, dtype=np.int64))
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative),
        random_state=RANDOM_SEED,
        stratify=temp_stratify,
    )
    return SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        val=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
        split_mode="random",
    )


def _parse_batch_selection(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for item in values:
        for piece in item.split(","):
            value = piece.strip()
            if value:
                parsed.append(value)
    return tuple(dict.fromkeys(parsed))


def _select_batch_subset(
    candidate_batches: np.ndarray,
    *,
    size: int,
    rng: np.random.RandomState,
) -> tuple[str, ...]:
    if size <= 0:
        return ()
    if size >= candidate_batches.size:
        return tuple(sorted(candidate_batches.astype(str).tolist()))
    selected = rng.choice(candidate_batches, size=size, replace=False)
    return tuple(sorted(np.asarray(selected, dtype=str).tolist()))


def _validate_seen_classes(
    *,
    split_name: str,
    split_idx: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    seen_train_labels: set[int],
) -> None:
    missing = sorted(set(np.unique(labels[split_idx]).tolist()) - seen_train_labels)
    if missing:
        missing_names = ", ".join(class_names[idx] for idx in missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise ValueError(
            f"{split_name} split contains classes unseen in train: {missing_names}{suffix}"
        )


def build_batch_splits(
    *,
    labels: np.ndarray,
    class_names: list[str],
    batch_values: np.ndarray,
    val_batches: tuple[str, ...] = (),
    test_batches: tuple[str, ...] = (),
) -> SplitIndices:
    unique_batches = np.unique(batch_values.astype(str))
    if unique_batches.size < 3:
        raise ValueError(
            f"按 batch 切分至少需要 3 个 batch，当前只有 {unique_batches.size} 个"
        )

    available_batches = set(unique_batches.tolist())
    unknown_val = sorted(set(val_batches) - available_batches)
    unknown_test = sorted(set(test_batches) - available_batches)
    if unknown_val:
        raise ValueError(f"未找到这些 val batches: {unknown_val}")
    if unknown_test:
        raise ValueError(f"未找到这些 test batches: {unknown_test}")
    overlap = sorted(set(val_batches) & set(test_batches))
    if overlap:
        raise ValueError(f"val/test batches 不能重叠: {overlap}")

    rng = np.random.RandomState(RANDOM_SEED)
    remaining_after_test = np.asarray(
        sorted(available_batches - set(test_batches)), dtype=object
    )
    if not test_batches:
        test_size = max(int(round(unique_batches.size * TEST_FRACTION)), 1)
        test_size = min(test_size, unique_batches.size - 2)
        test_batches = _select_batch_subset(
            unique_batches.astype(object), size=test_size, rng=rng
        )
        remaining_after_test = np.asarray(
            sorted(available_batches - set(test_batches)), dtype=object
        )

    if remaining_after_test.size < 2:
        raise ValueError("留出 test batches 后，剩余 batch 不足以再划分 train/val")

    if not val_batches:
        val_size = max(
            int(
                round(
                    remaining_after_test.size
                    * (VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION))
                )
            ),
            1,
        )
        val_size = min(val_size, remaining_after_test.size - 1)
        val_batches = _select_batch_subset(remaining_after_test, size=val_size, rng=rng)

    train_batches = tuple(
        sorted(available_batches - set(val_batches) - set(test_batches))
    )
    if not train_batches or not val_batches or not test_batches:
        raise ValueError("按 batch 切分后 train/val/test 中至少有一个为空")

    batch_values_str = batch_values.astype(str)
    train_mask = np.isin(batch_values_str, np.asarray(train_batches, dtype=object))
    val_mask = np.isin(batch_values_str, np.asarray(val_batches, dtype=object))
    test_mask = np.isin(batch_values_str, np.asarray(test_batches, dtype=object))
    train_idx = np.flatnonzero(train_mask).astype(np.int64, copy=False)
    val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
    test_idx = np.flatnonzero(test_mask).astype(np.int64, copy=False)

    seen_train_labels = set(np.unique(labels[train_idx]).tolist())
    _validate_seen_classes(
        split_name="val",
        split_idx=val_idx,
        labels=labels,
        class_names=class_names,
        seen_train_labels=seen_train_labels,
    )
    _validate_seen_classes(
        split_name="test",
        split_idx=test_idx,
        labels=labels,
        class_names=class_names,
        seen_train_labels=seen_train_labels,
    )

    return SplitIndices(
        train=train_idx,
        val=val_idx,
        test=test_idx,
        split_mode="batch",
        train_batch_values=train_batches,
        val_batch_values=val_batches,
        test_batch_values=test_batches,
    )


def resolve_representations(
    adata: ad.AnnData, requested: list[str] | None
) -> list[str]:
    if requested:
        representations = parse_representations(requested)
    else:
        representations = enabled_representations()

    needs_x = {
        "raw_umi",
        "raw_umi_plus_total",
        "log1p_normalized",
        "log1p_normalized_plus_total",
    }
    needs_signal = {
        "signal",
        "signal_plus_total",
        "signal_x_confidence",
        "signal_x_confidence_plus_total",
        "signal_confidence_concat",
        "signal_confidence_concat_plus_total",
        "signal_confidence_thresholded",
    }
    needs_confidence = {
        "signal_x_confidence",
        "signal_x_confidence_plus_total",
        "signal_confidence_concat",
        "signal_confidence_concat_plus_total",
        "signal_confidence_thresholded",
    }

    available = set(ALL_REPRESENTATIONS)
    if adata.X is None:
        available -= needs_x
    if SIGNAL_LAYER not in adata.layers:
        available -= needs_signal
    if CONFIDENCE_LAYER not in adata.layers:
        available -= needs_confidence

    unavailable = [name for name in representations if name not in available]
    if unavailable and requested:
        raise KeyError(f"这些表示在当前 h5ad 中不可用: {unavailable}")
    return [name for name in representations if name in available]


def make_model(model_name: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    if model_name == "linear":
        return LinearClassifier(input_dim=input_dim, num_classes=num_classes)
    if model_name == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=MLP_HIDDEN_DIM,
        )
    if model_name == "deep-mlp":
        return DeepMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=DEEP_MLP_HIDDEN_DIMS,
        )
    raise ValueError(f"未知模型: {model_name!r}")


def epochs_for_model(model_name: str) -> int:
    if model_name == "linear":
        return LINEAR_EPOCHS
    if model_name == "mlp":
        return MLP_EPOCHS
    if model_name == "deep-mlp":
        return DEEP_MLP_EPOCHS
    raise ValueError(f"未知模型: {model_name!r}")


def _to_dense_float32(matrix_slice) -> np.ndarray:
    if sparse.issparse(matrix_slice):
        array = matrix_slice.toarray()
    else:
        array = np.asarray(matrix_slice)
    return np.asarray(array, dtype=np.float32)


def _materialize_dense_matrix(
    matrix,
    *,
    n_obs: int,
    n_vars: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    dense = np.empty((n_obs, n_vars), dtype=np.float32)
    for start in tqdm(range(0, n_obs, batch_size), desc=desc, unit="batch"):
        end = min(start + batch_size, n_obs)
        batch_idx = np.arange(start, end, dtype=np.int64)
        dense[start:end] = _to_dense_float32(matrix[batch_idx])
    return dense


def precompute_feature_cache(
    adata: ad.AnnData,
    totals: np.ndarray,
    normalize_target: float,
    representations: list[str],
    *,
    batch_size: int,
    cache_dense_representations: bool,
) -> FeatureCache:
    need_raw_umi = any(
        name in {"raw_umi", "raw_umi_plus_total"} for name in representations
    )
    need_log1p = any(
        name in {"log1p_normalized", "log1p_normalized_plus_total"}
        for name in representations
    )
    need_signal = any(
        name
        in {
            "signal",
            "signal_plus_total",
            "signal_x_confidence",
            "signal_x_confidence_plus_total",
            "signal_confidence_concat",
            "signal_confidence_concat_plus_total",
            "signal_confidence_thresholded",
        }
        for name in representations
    )
    need_confidence = any(
        name
        in {
            "signal_x_confidence",
            "signal_x_confidence_plus_total",
            "signal_confidence_concat",
            "signal_confidence_concat_plus_total",
            "signal_confidence_thresholded",
        }
        for name in representations
    )
    need_conf_threshold = "signal_confidence_thresholded" in representations
    confidence_threshold: float | None = None
    raw_umi: np.ndarray | None = None
    signal: np.ndarray | None = None
    confidence: np.ndarray | None = None

    if need_confidence and cache_dense_representations:
        confidence = np.asarray(adata.layers["confidence"], dtype=np.float32)
        confidence = np.nan_to_num(confidence, copy=False)
    if need_conf_threshold:
        confidence_for_threshold = confidence
        if confidence_for_threshold is None:
            confidence_for_threshold = np.asarray(
                adata.layers["confidence"], dtype=np.float32
            )
            confidence_for_threshold = np.nan_to_num(
                confidence_for_threshold, copy=False
            )
        confidence_threshold = float(
            np.quantile(
                confidence_for_threshold.reshape(-1), CONFIDENCE_ZERO_THRESHOLD_QUANTILE
            )
        )

    matrix = adata.X
    if (need_raw_umi or need_log1p) and matrix is None:
        raise ValueError("输入 h5ad 的 X 为空，无法预计算 log1p_normalized")

    if need_raw_umi and cache_dense_representations:
        raw_umi = _materialize_dense_matrix(
            matrix,
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars),
            batch_size=batch_size,
            desc="precompute raw_umi",
        )

    if need_signal and cache_dense_representations:
        signal = np.asarray(adata.layers[SIGNAL_LAYER], dtype=np.float32)
        signal = np.nan_to_num(signal, copy=False)

    if not need_log1p:
        return FeatureCache(
            raw_umi=raw_umi,
            signal=signal,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )

    log1p_matrix = np.empty((int(adata.n_obs), int(adata.n_vars)), dtype=np.float32)
    batches = range(0, int(adata.n_obs), batch_size)
    for start in tqdm(batches, desc="precompute log1p_normalized", unit="batch"):
        end = min(start + batch_size, int(adata.n_obs))
        batch_idx = np.arange(start, end, dtype=np.int64)
        if raw_umi is not None:
            counts = raw_umi[start:end]
        else:
            counts = _to_dense_float32(matrix[batch_idx])
        features = log1p_normalize_total(
            counts,
            totals[batch_idx],
            target=normalize_target,
        )
        log1p_matrix[start:end] = np.asarray(features, dtype=np.float32)

    return FeatureCache(
        raw_umi=raw_umi,
        log1p_normalized=log1p_matrix,
        signal=signal,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )


def get_features(
    adata: ad.AnnData,
    representation: str,
    batch_idx: np.ndarray,
    totals: np.ndarray,
    *,
    feature_cache: FeatureCache,
    normalize_target: float,
) -> np.ndarray:
    if representation == "raw_umi":
        if feature_cache.raw_umi is not None:
            return feature_cache.raw_umi[batch_idx]
        matrix = adata.X
        if matrix is None:
            raise ValueError("输入 h5ad 的 X 为空，无法提取 raw_umi")
        return _to_dense_float32(matrix[batch_idx])
    if representation == "raw_umi_plus_total":
        if feature_cache.raw_umi is not None:
            features = feature_cache.raw_umi[batch_idx]
        else:
            matrix = adata.X
            if matrix is None:
                raise ValueError("输入 h5ad 的 X 为空，无法提取 raw_umi_plus_total")
            features = _to_dense_float32(matrix[batch_idx])
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate([features, total_feature], axis=1)
    if representation == "log1p_normalized":
        if feature_cache.log1p_normalized is None:
            raise ValueError("缺少预计算的 log1p_normalized 特征缓存")
        return feature_cache.log1p_normalized[batch_idx]
    if representation == "log1p_normalized_plus_total":
        if feature_cache.log1p_normalized is None:
            raise ValueError("缺少预计算的 log1p_normalized 特征缓存")
        features = feature_cache.log1p_normalized[batch_idx]
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate(
            [np.asarray(features, dtype=np.float32), total_feature], axis=1
        )
    if representation == "signal":
        if feature_cache.signal is not None:
            features = feature_cache.signal[batch_idx]
        else:
            features = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        return np.nan_to_num(features, copy=False)
    if representation == "signal_plus_total":
        if feature_cache.signal is not None:
            features = feature_cache.signal[batch_idx]
        else:
            features = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate(
            [np.nan_to_num(features, copy=False), total_feature], axis=1
        )
    if representation == "signal_x_confidence":
        signal = (
            feature_cache.signal[batch_idx]
            if feature_cache.signal is not None
            else _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        )
        confidence = (
            feature_cache.confidence[batch_idx]
            if feature_cache.confidence is not None
            else _to_dense_float32(adata.layers["confidence"][batch_idx])
        )
        features = np.nan_to_num(signal, copy=False) * np.nan_to_num(
            confidence, copy=False
        )
        return np.asarray(features, dtype=np.float32)
    if representation == "signal_x_confidence_plus_total":
        signal = (
            feature_cache.signal[batch_idx]
            if feature_cache.signal is not None
            else _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        )
        confidence = (
            feature_cache.confidence[batch_idx]
            if feature_cache.confidence is not None
            else _to_dense_float32(adata.layers["confidence"][batch_idx])
        )
        features = np.nan_to_num(signal, copy=False) * np.nan_to_num(
            confidence, copy=False
        )
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate(
            [np.asarray(features, dtype=np.float32), total_feature], axis=1
        )
    if representation == "signal_confidence_concat":
        signal = (
            feature_cache.signal[batch_idx]
            if feature_cache.signal is not None
            else _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        )
        confidence = (
            feature_cache.confidence[batch_idx]
            if feature_cache.confidence is not None
            else _to_dense_float32(adata.layers["confidence"][batch_idx])
        )
        return np.concatenate(
            [np.nan_to_num(signal, copy=False), np.nan_to_num(confidence, copy=False)],
            axis=1,
        ).astype(np.float32, copy=False)
    if representation == "signal_confidence_concat_plus_total":
        signal = (
            feature_cache.signal[batch_idx]
            if feature_cache.signal is not None
            else _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        )
        confidence = (
            feature_cache.confidence[batch_idx]
            if feature_cache.confidence is not None
            else _to_dense_float32(adata.layers["confidence"][batch_idx])
        )
        features = np.concatenate(
            [np.nan_to_num(signal, copy=False), np.nan_to_num(confidence, copy=False)],
            axis=1,
        ).astype(np.float32, copy=False)
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate([features, total_feature], axis=1)
    if representation == "signal_confidence_thresholded":
        if feature_cache.confidence_threshold is None:
            raise ValueError("缺少 confidence threshold 缓存")
        signal = (
            feature_cache.signal[batch_idx]
            if feature_cache.signal is not None
            else _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        )
        confidence = (
            feature_cache.confidence[batch_idx]
            if feature_cache.confidence is not None
            else _to_dense_float32(adata.layers["confidence"][batch_idx])
        )
        signal = np.nan_to_num(signal, copy=False)
        confidence = np.nan_to_num(confidence, copy=False)
        signal[confidence < feature_cache.confidence_threshold] = 0.0
        return signal.astype(np.float32, copy=False)
    raise ValueError(f"未知表示: {representation!r}")


def feature_dim(adata: ad.AnnData, representation: str) -> int:
    base_dim = int(adata.n_vars)
    if representation in {
        "raw_umi",
        "log1p_normalized",
        "signal",
        "signal_x_confidence",
        "signal_confidence_thresholded",
    }:
        return base_dim
    if representation in {
        "raw_umi_plus_total",
        "log1p_normalized_plus_total",
        "signal_plus_total",
        "signal_x_confidence_plus_total",
    }:
        return base_dim + 1
    if representation == "signal_confidence_concat":
        return base_dim * 2
    if representation == "signal_confidence_concat_plus_total":
        return base_dim * 2 + 1
    raise ValueError(f"未知表示: {representation!r}")


def iter_batches(
    indices: np.ndarray, batch_size: int, *, shuffle: bool
) -> Iterator[np.ndarray]:
    order = indices.copy()
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


@torch.inference_mode()
def evaluate_model(
    *,
    model: torch.nn.Module,
    adata: ad.AnnData,
    representation: str,
    indices: np.ndarray,
    labels: np.ndarray,
    totals: np.ndarray,
    feature_cache: FeatureCache,
    normalize_target: float,
    device: torch.device,
    desc: str,
    eval_batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    losses: list[float] = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx in tqdm(
        iter_batches(indices, eval_batch_size, shuffle=False),
        desc=desc,
        leave=False,
    ):
        features = get_features(
            adata,
            representation,
            batch_idx,
            totals,
            feature_cache=feature_cache,
            normalize_target=normalize_target,
        )
        x = torch.from_numpy(features).to(device)
        y = torch.from_numpy(labels[batch_idx]).to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    return (
        float(np.mean(losses)) if losses else math.nan,
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
    )


def summarize_metrics(
    name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> RepresentationResult:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )
    return RepresentationResult(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
    )


def train_representation(
    *,
    adata: ad.AnnData,
    representation: str,
    model_name: str,
    labels: np.ndarray,
    totals: np.ndarray,
    feature_cache: FeatureCache,
    splits: SplitIndices,
    num_classes: int,
    device: torch.device,
    run_cfg: RunConfig,
) -> RepresentationResult:
    input_dim = feature_dim(adata, representation)
    epochs = epochs_for_model(model_name)
    normalize_target = float(np.median(totals[splits.train]))
    model = make_model(model_name, input_dim=input_dim, num_classes=num_classes).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    train_batches_per_epoch = max(
        math.ceil(len(splits.train) / run_cfg.train_batch_size), 1
    )
    total_train_steps = epochs * train_batches_per_epoch
    step_bar = tqdm(
        total=total_train_steps,
        desc=f"train {representation}",
        leave=True,
        unit="step",
    )
    for epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        batch_correct = 0
        batch_seen = 0
        for step_in_epoch, batch_idx in enumerate(
            iter_batches(splits.train, run_cfg.train_batch_size, shuffle=True),
            start=1,
        ):
            features = get_features(
                adata,
                representation,
                batch_idx,
                totals,
                feature_cache=feature_cache,
                normalize_target=normalize_target,
            )
            x = torch.from_numpy(features).to(device, non_blocking=True)
            y = torch.from_numpy(labels[batch_idx]).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            batch_losses.append(float(loss.item()))
            batch_correct += int((pred == y).sum().item())
            batch_seen += int(y.shape[0])
            step_bar.update(1)
            step_bar.set_postfix(
                epoch=f"{epoch + 1}/{epochs}",
                step=f"{step_in_epoch}/{train_batches_per_epoch}",
                loss=f"{loss.item():.4f}",
            )

        train_loss = float(np.mean(batch_losses)) if batch_losses else math.nan
        train_acc = float(batch_correct / max(batch_seen, 1))
        should_eval = (epoch + 1) % run_cfg.eval_every == 0 or epoch + 1 == epochs
        if should_eval:
            _, val_true, val_pred = evaluate_model(
                model=model,
                adata=adata,
                representation=representation,
                indices=splits.val,
                labels=labels,
                totals=totals,
                feature_cache=feature_cache,
                normalize_target=normalize_target,
                device=device,
                desc=f"val {representation}",
                eval_batch_size=run_cfg.eval_batch_size,
            )
            val_acc = float(accuracy_score(val_true, val_pred))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            step_bar.set_postfix(
                epoch=f"{epoch + 1}/{epochs}",
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_acc=f"{val_acc:.4f}",
                best_val=f"{best_val_acc:.4f}",
            )
        else:
            step_bar.set_postfix(
                epoch=f"{epoch + 1}/{epochs}",
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                best_val=f"{best_val_acc:.4f}" if best_val_acc >= 0 else "-",
            )

    step_bar.close()

    model.load_state_dict(best_state)
    _, test_true, test_pred = evaluate_model(
        model=model,
        adata=adata,
        representation=representation,
        indices=splits.test,
        labels=labels,
        totals=totals,
        feature_cache=feature_cache,
        normalize_target=normalize_target,
        device=device,
        desc=f"test {representation}",
        eval_batch_size=run_cfg.eval_batch_size,
    )
    return summarize_metrics(representation, test_true, test_pred)


def print_dataset_summary(
    *,
    adata: ad.AnnData,
    labels: np.ndarray,
    class_names: list[str],
    splits: SplitIndices,
    model_name: str,
    device: torch.device,
    gene_list_spec: GeneListSpec | None,
    label_filter_spec: LabelFilterSpec | None,
    representations: list[str],
    run_cfg: RunConfig,
    label_column: str,
    batch_key: str,
) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print("=" * 80)
    print(f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"Gene subset: {gene_list_spec.top_k if gene_list_spec is not None else 'all'}"
    )
    if label_filter_spec is None:
        print("Label filter: all labels")
    else:
        print(
            f"Label filter: top-{label_filter_spec.top_k} labels by count ({label_filter_spec.filtered_n_obs}/{label_filter_spec.original_n_obs} cells kept)"
        )
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Representations: {', '.join(representations)}")
    print(f"Label column: {label_column}")
    print(f"Signal layer: {SIGNAL_LAYER} (present={SIGNAL_LAYER in adata.layers})")
    print(
        f"Confidence layer: {CONFIDENCE_LAYER} (present={CONFIDENCE_LAYER in adata.layers})"
    )
    print(
        f"Batch sizes: train={run_cfg.train_batch_size} eval={run_cfg.eval_batch_size} precompute={run_cfg.precompute_batch_size}"
    )
    print(f"Eval every: {run_cfg.eval_every} epoch(s)")
    print(f"Cache dense representations: {run_cfg.cache_dense_representations}")
    print(f"Split mode: {splits.split_mode}")
    if splits.split_mode == "batch":
        print(f"Batch key: {batch_key}")
        print(
            f"Split batches: train={len(splits.train_batch_values)} val={len(splits.val_batch_values)} test={len(splits.test_batch_values)}"
        )
        print(f"  train batches: {', '.join(splits.train_batch_values)}")
        print(f"  val batches  : {', '.join(splits.val_batch_values)}")
        print(f"  test batches : {', '.join(splits.test_batch_values)}")
    print(
        f"Split sizes: train={splits.train.size} val={splits.val.size} test={splits.test.size}"
    )
    print("Classes:")
    for cls_idx, count in zip(unique, counts, strict=False):
        print(f"  {cls_idx:>2d} | {class_names[int(cls_idx)]:<20} | n={int(count)}")
    print("=" * 80)


def print_result(result: RepresentationResult) -> None:
    print(f"\n[{result.name}]")
    print(f"  accuracy          : {result.accuracy:.4f}")
    print(f"  precision_macro   : {result.precision_macro:.4f}")
    print(f"  recall_macro      : {result.recall_macro:.4f}")
    print(f"  f1_macro          : {result.f1_macro:.4f}")
    print(f"  precision_weighted: {result.precision_weighted:.4f}")
    print(f"  recall_weighted   : {result.recall_weighted:.4f}")
    print(f"  f1_weighted       : {result.f1_weighted:.4f}")


def main() -> None:
    args = parse_args()
    if args.train_batch_size < 1:
        raise ValueError("--train-batch-size 必须 >= 1")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size 必须 >= 1")
    if args.precompute_batch_size < 1:
        raise ValueError("--precompute-batch-size 必须 >= 1")
    if args.eval_every < 1:
        raise ValueError("--eval-every 必须 >= 1")

    run_cfg = RunConfig(
        train_batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        precompute_batch_size=int(args.precompute_batch_size),
        eval_every=int(args.eval_every),
        cache_dense_representations=bool(args.cache_dense_representations),
    )
    set_seed(RANDOM_SEED)
    adata_full = read_adata(args.h5ad_path.expanduser().resolve())
    representations = resolve_representations(adata_full, args.representations)
    if not representations:
        raise ValueError("当前 h5ad 不支持任何被启用的表示，请显式指定可用表示")
    adata_labels, label_filter_spec = maybe_subset_top_labels(
        adata_full,
        label_column=args.label_column,
        label_top_k=args.label_top_k,
    )
    totals = compute_totals(adata_labels)
    adata, gene_list_spec = maybe_subset_gene_list(adata_labels, args.gene_list_json)
    labels, class_names = encode_labels(adata, args.label_column)
    if args.split_mode == "batch":
        batch_values = get_batch_values(adata, args.batch_key)
        splits = build_batch_splits(
            labels=labels,
            class_names=class_names,
            batch_values=batch_values,
            val_batches=_parse_batch_selection(args.val_batches),
            test_batches=_parse_batch_selection(args.test_batches),
        )
    else:
        splits = build_splits(labels)
    device = resolve_device(args.device)
    normalize_target = float(np.median(totals[splits.train]))

    print_dataset_summary(
        adata=adata,
        labels=labels,
        class_names=class_names,
        splits=splits,
        model_name=args.model,
        device=device,
        gene_list_spec=gene_list_spec,
        label_filter_spec=label_filter_spec,
        representations=representations,
        run_cfg=run_cfg,
        label_column=args.label_column,
        batch_key=args.batch_key,
    )

    results: list[RepresentationResult] = []
    feature_cache = precompute_feature_cache(
        adata,
        totals,
        normalize_target,
        representations,
        batch_size=run_cfg.precompute_batch_size,
        cache_dense_representations=run_cfg.cache_dense_representations,
    )

    for representation in representations:
        print(f"\n>>> Training {args.model} on {representation}")
        result = train_representation(
            adata=adata,
            representation=representation,
            model_name=args.model,
            labels=labels,
            totals=totals,
            feature_cache=feature_cache,
            splits=splits,
            num_classes=len(class_names),
            device=device,
            run_cfg=run_cfg,
        )
        results.append(result)
        print_result(result)

    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    for result in results:
        print_result(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
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

from prism.baseline import DeepMLPClassifier, LinearClassifier, MLPClassifier
from prism.baseline.metrics import log1p_normalize_total

SIGNAL_LAYER = "signal"
LABEL_COLUMN = "treatment"
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
PRECOMPUTE_BATCH_SIZE = 2048
LINEAR_EPOCHS = 5
MLP_EPOCHS = 10
DEEP_MLP_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MLP_HIDDEN_DIM = 1024
DEEP_MLP_HIDDEN_DIMS = (1024, 512, 256)

USE_RAW_UMI = True
USE_RAW_UMI_PLUS_TOTAL = False
USE_LOG1P_NORMALIZED = False
USE_LOG1P_NORMALIZED_PLUS_TOTAL = False
USE_SIGNAL = False
USE_SIGNAL_PLUS_TOTAL = False
USE_SIGNAL_X_CONFIDENCE = False
USE_SIGNAL_X_CONFIDENCE_PLUS_TOTAL = False
USE_SIGNAL_CONFIDENCE_CONCAT = False
USE_SIGNAL_CONFIDENCE_CONCAT_PLUS_TOTAL = False
USE_SIGNAL_CONFIDENCE_THRESHOLD = True

CONFIDENCE_ZERO_THRESHOLD_QUANTILE = 0.001


@dataclass(frozen=True, slots=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


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
    log1p_normalized: np.ndarray | None = None
    confidence_threshold: float | None = None


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
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_adata(h5ad_path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(h5ad_path)
    if SIGNAL_LAYER not in adata.layers:
        raise KeyError(f"输入文件缺少必需 layer: {SIGNAL_LAYER!r}")
    if LABEL_COLUMN not in adata.obs:
        raise KeyError(f"输入文件缺少标签列: {LABEL_COLUMN!r}")
    return adata


def encode_labels(adata: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    series = adata.obs[LABEL_COLUMN].astype("category")
    labels = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"标签列 {LABEL_COLUMN!r} 包含缺失值")
    class_names = [str(value) for value in series.cat.categories.tolist()]
    return labels, class_names


def compute_totals(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if matrix is None:
        raise ValueError("输入 h5ad 的 X 为空，无法计算 raw/log1p baseline")
    if sparse.issparse(matrix):
        totals = np.asarray(cast(Any, matrix).sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix).sum(axis=1)
    return np.asarray(totals, dtype=np.float32)


def build_splits(labels: np.ndarray) -> SplitIndices:
    all_idx = np.arange(labels.shape[0], dtype=np.int64)
    train_idx, temp_idx, _train_y, temp_y = train_test_split(
        all_idx,
        labels,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    val_relative = VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_relative),
        random_state=RANDOM_SEED,
        stratify=temp_y,
    )
    return SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        val=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
    )


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


def precompute_feature_cache(
    adata: ad.AnnData,
    totals: np.ndarray,
    normalize_target: float,
    representations: list[str],
) -> FeatureCache:
    need_log1p = any(
        name in {"log1p_normalized", "log1p_normalized_plus_total"}
        for name in representations
    )
    need_conf_threshold = "signal_confidence_thresholded" in representations
    confidence_threshold: float | None = None

    if need_conf_threshold:
        confidence = np.asarray(adata.layers["confidence"], dtype=np.float32)
        confidence = np.nan_to_num(confidence, copy=False)
        confidence_threshold = float(
            np.quantile(confidence.reshape(-1), CONFIDENCE_ZERO_THRESHOLD_QUANTILE)
        )

    if not need_log1p:
        return FeatureCache(confidence_threshold=confidence_threshold)

    matrix = adata.X
    if matrix is None:
        raise ValueError("输入 h5ad 的 X 为空，无法预计算 log1p_normalized")

    log1p_matrix = np.empty((int(adata.n_obs), int(adata.n_vars)), dtype=np.float32)
    batches = range(0, int(adata.n_obs), PRECOMPUTE_BATCH_SIZE)
    for start in tqdm(batches, desc="precompute log1p_normalized", unit="batch"):
        end = min(start + PRECOMPUTE_BATCH_SIZE, int(adata.n_obs))
        batch_idx = np.arange(start, end, dtype=np.int64)
        counts = _to_dense_float32(matrix[batch_idx])
        features = log1p_normalize_total(
            counts,
            totals[batch_idx],
            target=normalize_target,
        )
        log1p_matrix[start:end] = np.asarray(features, dtype=np.float32)

    return FeatureCache(
        log1p_normalized=log1p_matrix,
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
        matrix = adata.X
        if matrix is None:
            raise ValueError("输入 h5ad 的 X 为空，无法提取 raw_umi")
        return _to_dense_float32(matrix[batch_idx])
    if representation == "raw_umi_plus_total":
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
        features = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        return np.nan_to_num(features, copy=False)
    if representation == "signal_plus_total":
        features = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate(
            [np.nan_to_num(features, copy=False), total_feature], axis=1
        )
    if representation == "signal_x_confidence":
        signal = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        confidence = _to_dense_float32(adata.layers["confidence"][batch_idx])
        features = np.nan_to_num(signal, copy=False) * np.nan_to_num(
            confidence, copy=False
        )
        return np.asarray(features, dtype=np.float32)
    if representation == "signal_x_confidence_plus_total":
        signal = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        confidence = _to_dense_float32(adata.layers["confidence"][batch_idx])
        features = np.nan_to_num(signal, copy=False) * np.nan_to_num(
            confidence, copy=False
        )
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate(
            [np.asarray(features, dtype=np.float32), total_feature], axis=1
        )
    if representation == "signal_confidence_concat":
        signal = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        confidence = _to_dense_float32(adata.layers["confidence"][batch_idx])
        return np.concatenate(
            [np.nan_to_num(signal, copy=False), np.nan_to_num(confidence, copy=False)],
            axis=1,
        ).astype(np.float32, copy=False)
    if representation == "signal_confidence_concat_plus_total":
        signal = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        confidence = _to_dense_float32(adata.layers["confidence"][batch_idx])
        features = np.concatenate(
            [np.nan_to_num(signal, copy=False), np.nan_to_num(confidence, copy=False)],
            axis=1,
        ).astype(np.float32, copy=False)
        total_feature = totals[batch_idx].astype(np.float32, copy=False)[:, None]
        return np.concatenate([features, total_feature], axis=1)
    if representation == "signal_confidence_thresholded":
        if feature_cache.confidence_threshold is None:
            raise ValueError("缺少 confidence threshold 缓存")
        signal = _to_dense_float32(adata.layers[SIGNAL_LAYER][batch_idx])
        confidence = _to_dense_float32(adata.layers["confidence"][batch_idx])
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
) -> list[np.ndarray]:
    order = indices.copy()
    if shuffle:
        np.random.shuffle(order)
    return [
        order[start : start + batch_size] for start in range(0, len(order), batch_size)
    ]


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
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    losses: list[float] = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx in tqdm(
        iter_batches(indices, EVAL_BATCH_SIZE, shuffle=False),
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

    train_batches_per_epoch = max(math.ceil(len(splits.train) / TRAIN_BATCH_SIZE), 1)
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
            iter_batches(splits.train, TRAIN_BATCH_SIZE, shuffle=True),
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
            x = torch.from_numpy(features).to(device)
            y = torch.from_numpy(labels[batch_idx]).to(device)

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
) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print("=" * 80)
    print(f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Label column: {LABEL_COLUMN}")
    print(f"Signal layer: {SIGNAL_LAYER}")
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
    set_seed(RANDOM_SEED)
    adata = read_adata(args.h5ad_path.expanduser().resolve())
    labels, class_names = encode_labels(adata)
    totals = compute_totals(adata)
    splits = build_splits(labels)
    device = resolve_device()
    normalize_target = float(np.median(totals[splits.train]))

    print_dataset_summary(
        adata=adata,
        labels=labels,
        class_names=class_names,
        splits=splits,
        model_name=args.model,
        device=device,
    )

    results: list[RepresentationResult] = []
    representations = enabled_representations()
    if not representations:
        raise ValueError("所有表示开关都已关闭，至少启用一组输入")
    if (
        any(
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
        and "confidence" not in adata.layers
    ):
        raise KeyError("输入文件缺少必需 layer: 'confidence'")
    feature_cache = precompute_feature_cache(
        adata,
        totals,
        normalize_target,
        representations,
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

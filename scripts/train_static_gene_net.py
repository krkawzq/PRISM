#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prism.baseline.static_gene_net import StaticGeneNet, StaticGeneNetConfig

from train_gene_mae import (
    ClassificationEpochRecord,
    ClassificationResult,
    GeneListSpec,
    PretrainEpochRecord,
    SplitIndices,
    add_mask_selection_args,
    add_shared_data_args,
    append_jsonl,
    autocast_context,
    build_cosine_scheduler,
    build_splits,
    create_pretrain_mask,
    compute_totals,
    compute_weight_mean,
    encode_labels,
    get_weight_batch,
    infer_support_size,
    iter_batches,
    load_resume_checkpoint,
    masked_regression_loss,
    maybe_save_epoch_checkpoint,
    maybe_subset_gene_list,
    read_adata,
    resolve_amp_dtype,
    resolve_device,
    save_training_checkpoint,
    set_seed,
    subset_adata_by_gene_list,
    write_json,
)


def get_static_input_batch(
    adata: ad.AnnData,
    *,
    representation: str,
    batch_idx: np.ndarray,
    totals: np.ndarray,
    normalize_target: float,
) -> np.ndarray:
    if representation == "raw_umi":
        matrix = adata.X
        if matrix is None:
            raise ValueError("输入 h5ad 的 X 为空")
        batch = matrix[batch_idx]
        if hasattr(batch, "toarray"):
            batch = cast(Any, batch).toarray()
        return np.asarray(
            batch,
            dtype=np.float32,
        )
    from train_gene_mae import get_input_batch as mae_get_input_batch

    return mae_get_input_batch(
        adata,
        representation=representation,
        batch_idx=batch_idx,
        totals=totals,
        normalize_target=normalize_target,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train StaticGeneNet for pretraining or downstream classification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser(
        "pretrain", help="Run self-supervised StaticGeneNet pretraining."
    )
    add_shared_data_args(pretrain)
    add_model_args(pretrain)
    pretrain.add_argument("--output-dir", type=Path, required=True)
    pretrain.add_argument(
        "--input-representation",
        choices=("signal", "log_signal", "lognormal", "raw_umi"),
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
    pretrain.add_argument("--batch-size", type=int, default=512)
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
        choices=("signal", "log_signal", "lognormal", "raw_umi"),
        required=True,
        help="Feature representation used for downstream classification.",
    )
    downstream.add_argument(
        "--task-mode",
        choices=("freeze_static", "full"),
        required=True,
        help=(
            "freeze_static: freeze the whole attention branch and train FFN + embeddings + head; "
            "full: train the whole network."
        ),
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


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-gene-id", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--static-qk-dim", type=int, default=16)
    parser.add_argument("--ffn-expand", type=float, default=4.0)
    parser.add_argument("--ffn-dropout", type=float, default=0.1)
    parser.add_argument("--embed-dropout", type=float, default=0.1)
    parser.add_argument("--path-dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--signal-bins", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument(
        "--share-layer-params",
        action="store_true",
        help="Share one StaticGeneNet block across all layers.",
    )
    parser.add_argument(
        "--use-gene-id",
        action="store_true",
        help="Inject learned gene identity features into token input and FFN.",
    )


def build_model_config(
    args: argparse.Namespace, *, n_genes: int, n_classes: int
) -> StaticGeneNetConfig:
    return StaticGeneNetConfig(
        n_genes=n_genes,
        n_classes=n_classes,
        d_model=args.d_model,
        d_gene_id=args.d_gene_id,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        static_qk_dim=args.static_qk_dim,
        ffn_expand=args.ffn_expand,
        ffn_dropout=args.ffn_dropout,
        embed_dropout=args.embed_dropout,
        path_dropout=args.path_dropout,
        signal_bins=args.signal_bins,
        mask_ratio=args.mask_ratio,
        classification_head="linear",
        share_layer_params=bool(args.share_layer_params),
        attention_dropout=args.attention_dropout,
        use_gene_id=bool(args.use_gene_id),
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
    model = StaticGeneNet(config).to(device)
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
            inputs_np = get_static_input_batch(
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
                    raise RuntimeError(
                        "StaticGeneNet 预训练输出缺少 mask 或 masked_predictions"
                    )
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            save_training_checkpoint(
                best_path,
                model=cast(Any, model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=cast(Any, config),
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
            model=cast(Any, model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=cast(Any, config),
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
            model=cast(Any, model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=cast(Any, config),
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
        write_json(
            history_path,
            {
                "records": [asdict(record) for record in history],
                "best_val_loss": best_val_loss,
                "last_epoch": epoch,
                "global_step": global_step,
            },
        )
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
    write_json(
        history_path,
        {
            "records": [asdict(record) for record in history],
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "global_step": global_step,
        },
    )
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
    model: StaticGeneNet,
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
        inputs_np = get_static_input_batch(
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
                raise RuntimeError(
                    "StaticGeneNet 预训练输出缺少 mask 或 masked_predictions"
                )
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
        pretrain_config = StaticGeneNetConfig(**checkpoint["model_config"])
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

    classification_head = "linear"
    config = replace(
        pretrain_config,
        n_classes=len(class_names),
        classification_head=classification_head,
    )
    model = StaticGeneNet(config).to(device)

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

    freeze_static = args.task_mode == "freeze_static"
    for name, param in model.named_parameters():
        is_attention_param = ".attention." in name or name.endswith(
            "attn_residual_lambda"
        )
        if freeze_static:
            param.requires_grad = not is_attention_param
        else:
            param.requires_grad = True

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
        freeze_static=freeze_static,
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
            features = get_static_input_batch(
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            save_training_checkpoint(
                best_path,
                model=cast(Any, model),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=cast(Any, config),
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
            model=cast(Any, model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=cast(Any, config),
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
            model=cast(Any, model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=cast(Any, config),
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
        write_json(
            history_path,
            {
                "records": [asdict(record) for record in history],
                "best_val_acc": best_val_acc,
                "last_epoch": epoch,
                "global_step": global_step,
            },
        )
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
            f"epoch={epoch:03d} train_loss={train_loss:.6f} train_acc={train_acc:.4f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} best_val={best_val_acc:.4f}"
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
    write_json(
        history_path,
        {
            "records": [asdict(record) for record in history],
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_metrics": asdict(result),
            "classes": class_names,
            "global_step": global_step,
        },
    )
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
    model: StaticGeneNet,
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
        features = get_static_input_batch(
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
    print("StaticGeneNet Pretrain")
    print(f"Dataset           : {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"Gene subset       : {gene_list_spec.top_k if gene_list_spec is not None else 'all'}"
    )
    print(f"Input             : {args.input_representation}")
    print(f"Loss weighting    : {args.loss_weighting}")
    print(f"Mask selection    : {args.mask_selection}")
    if args.mask_selection == "confidence_top_p":
        print(f"Mask top-p        : {args.mask_confidence_top_p}")
    print(f"Support size      : {support_size}")
    print(f"Weight mean       : {weight_mean}")
    print(f"Share layers      : {args.share_layer_params}")
    print(f"AMP               : {amp_enabled} ({amp_dtype})")
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
    freeze_static: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    gene_list_spec: GeneListSpec | None,
) -> None:
    print("=" * 80)
    print("StaticGeneNet Downstream")
    print(f"Dataset           : {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"Gene subset       : {gene_list_spec.top_k if gene_list_spec is not None else 'all'}"
    )
    print(f"Input             : {args.input_representation}")
    print(f"Task mode         : {args.task_mode}")
    print("Head              : linear")
    print(f"Freeze attention  : {freeze_static}")
    print(f"Share layers      : {args.share_layer_params}")
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

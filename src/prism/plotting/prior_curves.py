from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prism.model import ModelCheckpoint, PriorGrid

from ._shared import dedupe_names, resolve_plot_export
from .prior_annotations import LabelGridEntry
from .prior_stats import resolve_x_values, summarize_prior_curve


@dataclass(frozen=True, slots=True)
class PriorCurve:
    source: str
    scope_name: str
    checkpoint_name: str
    checkpoint_path: str
    label_name: str | None
    support_domain: str
    support: np.ndarray
    scaled_support: np.ndarray
    prior_probabilities: np.ndarray
    scale: float


def _curve_point_rows(
    gene_name: str,
    curve: PriorCurve,
    *,
    x_axis: str,
    batch: str | None = None,
    perturbation: str | None = None,
) -> Iterable[dict[str, object]]:
    x_values = resolve_x_values(curve, x_axis=x_axis)
    for idx, (support_value, scaled_value, x_value, probability_value) in enumerate(
        zip(
            curve.support,
            curve.scaled_support,
            x_values,
            curve.prior_probabilities,
            strict=True,
        ),
        start=1,
    ):
        row: dict[str, object] = {
            "gene": gene_name,
            "source": curve.source,
            "scope": curve.scope_name,
            "checkpoint_name": curve.checkpoint_name,
            "checkpoint_path": curve.checkpoint_path,
            "label": curve.label_name,
            "point_index": idx,
            "support": float(support_value),
            "scaled_support": float(scaled_value),
            "x": float(x_value),
            "probability": float(probability_value),
            "x_axis": x_axis,
            "scale": float(curve.scale),
        }
        if batch is not None:
            row["batch"] = batch
        if perturbation is not None:
            row["perturbation"] = perturbation
        yield row


def _curve_summary_row(
    gene_name: str,
    curve: PriorCurve,
    *,
    batch: str | None = None,
    perturbation: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "gene": gene_name,
        "source": curve.source,
        "scope": curve.scope_name,
        "checkpoint_name": curve.checkpoint_name,
        "checkpoint_path": curve.checkpoint_path,
        "label": curve.label_name,
        **summarize_prior_curve(curve),
    }
    if batch is not None:
        row["batch"] = batch
    if perturbation is not None:
        row["perturbation"] = perturbation
    return row


def _normalize_missing_policy(value: str) -> str:
    resolved = str(value).strip().lower()
    if resolved not in {"error", "drop"}:
        raise ValueError("missing_policy must be one of: error, drop")
    return resolved


def default_checkpoint_name(
    checkpoint: ModelCheckpoint,
    *,
    checkpoint_path: Path,
) -> str:
    source_h5ad_path = checkpoint.metadata.get("source_h5ad_path")
    if source_h5ad_path is not None:
        stem = Path(str(source_h5ad_path)).stem.strip()
        if stem:
            return stem
    stem = checkpoint_path.stem.strip()
    if stem:
        return stem
    return "checkpoint"


def _display_source(
    *,
    checkpoint_name: str,
    scope_name: str,
    include_checkpoint_name: bool,
) -> str:
    return f"{checkpoint_name}/{scope_name}" if include_checkpoint_name else scope_name


def _curve_from_prior(
    priors: PriorGrid,
    gene_name: str,
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    scope_name: str,
    include_checkpoint_name: bool,
) -> PriorCurve:
    prior = priors.select_genes(gene_name)
    return PriorCurve(
        source=_display_source(
            checkpoint_name=checkpoint_name,
            scope_name=scope_name,
            include_checkpoint_name=include_checkpoint_name,
        ),
        scope_name=scope_name,
        checkpoint_name=checkpoint_name,
        checkpoint_path=str(checkpoint_path),
        label_name=None if scope_name == "global" else scope_name.removeprefix("label:"),
        support_domain=prior.support_domain,
        support=np.asarray(prior.support, dtype=np.float64).reshape(-1),
        scaled_support=np.asarray(prior.scaled_support, dtype=np.float64).reshape(-1),
        prior_probabilities=np.asarray(
            prior.prior_probabilities, dtype=np.float64
        ).reshape(-1),
        scale=float(prior.scale),
    )


def resolve_prior_curve_sets(
    checkpoint: ModelCheckpoint,
    *,
    gene_names: list[str],
    labels: list[str] | None,
    include_global: bool,
    checkpoint_name: str | None = None,
    checkpoint_path: Path | None = None,
    include_checkpoint_name: bool = False,
    missing_policy: str = "error",
    allow_empty_genes: bool = False,
) -> dict[str, list[PriorCurve]]:
    requested = dedupe_names(gene_names)
    if not requested:
        raise ValueError("at least one gene is required")
    missing_policy = _normalize_missing_policy(missing_policy)
    resolved_checkpoint_path = (
        Path("checkpoint.pkl") if checkpoint_path is None else checkpoint_path
    )
    resolved_checkpoint_name = (
        default_checkpoint_name(checkpoint, checkpoint_path=resolved_checkpoint_path)
        if checkpoint_name is None
        else str(checkpoint_name).strip()
    )
    if not resolved_checkpoint_name:
        raise ValueError("checkpoint_name cannot be blank")
    selected_labels = (
        sorted(checkpoint.label_priors)
        if labels is None
        else dedupe_names(labels)
    )
    curve_sets: dict[str, list[PriorCurve]] = {}
    for gene_name in requested:
        curves: list[PriorCurve] = []
        if include_global:
            if not checkpoint.has_global_prior:
                if missing_policy == "error":
                    raise ValueError(
                        f"checkpoint {resolved_checkpoint_name!r} has no global prior"
                    )
            elif gene_name in checkpoint.get_prior().gene_names:
                curves.append(
                    _curve_from_prior(
                        checkpoint.get_prior(),
                        gene_name,
                        checkpoint_name=resolved_checkpoint_name,
                        checkpoint_path=resolved_checkpoint_path,
                        scope_name="global",
                        include_checkpoint_name=include_checkpoint_name,
                    )
                )
            elif missing_policy == "error":
                raise ValueError(
                    f"gene {gene_name!r} is missing from the global prior in checkpoint "
                    f"{resolved_checkpoint_name!r}"
                )
        for label in selected_labels:
            if label not in checkpoint.label_priors:
                if missing_policy == "error":
                    raise ValueError(
                        f"checkpoint {resolved_checkpoint_name!r} is missing label prior "
                        f"{label!r}"
                    )
                continue
            priors = checkpoint.get_prior(label)
            if gene_name in priors.gene_names:
                curves.append(
                    _curve_from_prior(
                        priors,
                        gene_name,
                        checkpoint_name=resolved_checkpoint_name,
                        checkpoint_path=resolved_checkpoint_path,
                        scope_name=f"label:{label}",
                        include_checkpoint_name=include_checkpoint_name,
                    )
                )
            elif missing_policy == "error":
                raise ValueError(
                    f"gene {gene_name!r} is missing from label prior {label!r} in "
                    f"checkpoint {resolved_checkpoint_name!r}"
                )
        if not curves and not allow_empty_genes:
            raise ValueError(
                f"gene {gene_name!r} is not present in the selected prior sources"
            )
        curve_sets[gene_name] = curves
    return curve_sets


def resolve_multi_checkpoint_prior_curve_sets(
    checkpoints: list[tuple[str, Path, ModelCheckpoint]],
    *,
    gene_names: list[str],
    labels: list[str] | None,
    include_global: bool,
    missing_policy: str = "error",
) -> dict[str, list[PriorCurve]]:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    requested = dedupe_names(gene_names)
    if not requested:
        raise ValueError("at least one gene is required")
    _normalize_missing_policy(missing_policy)
    checkpoint_names = [checkpoint_name for checkpoint_name, _, _ in checkpoints]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("checkpoint names must be unique")
    merged: dict[str, list[PriorCurve]] = {gene_name: [] for gene_name in requested}
    for checkpoint_name, checkpoint_path, checkpoint in checkpoints:
        curve_sets = resolve_prior_curve_sets(
            checkpoint,
            gene_names=requested,
            labels=labels,
            include_global=include_global,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            include_checkpoint_name=len(checkpoints) > 1,
            missing_policy=missing_policy,
            allow_empty_genes=missing_policy == "drop",
        )
        for gene_name, curves in curve_sets.items():
            merged[gene_name].extend(curves)
    for gene_name, curves in merged.items():
        if not curves:
            raise ValueError(
                f"gene {gene_name!r} is not present in the selected prior sources"
            )
    return merged


def resolve_batch_grid_curve_sets(
    checkpoint: ModelCheckpoint,
    *,
    gene_names: list[str],
    entries: list[LabelGridEntry],
) -> tuple[dict[str, dict[tuple[str, str], PriorCurve]], list[str], list[str]]:
    if not checkpoint.has_label_priors:
        raise ValueError("checkpoint has no label priors")
    requested_genes = dedupe_names(gene_names)
    if not requested_genes:
        raise ValueError("at least one gene is required")
    if not entries:
        raise ValueError("at least one label-grid entry is required")
    unknown = [
        entry.label for entry in entries if entry.label not in checkpoint.label_priors
    ]
    if unknown:
        raise ValueError(f"unknown label priors: {sorted(set(unknown))[:10]}")
    seen_pairs: dict[tuple[str, str], str] = {}
    batches: list[str] = []
    perturbations: list[str] = []
    for entry in entries:
        pair = (entry.batch, entry.perturbation)
        previous_label = seen_pairs.get(pair)
        if previous_label is not None:
            raise ValueError(
                "duplicate batch-grid cell for "
                f"batch={entry.batch!r}, perturbation={entry.perturbation!r}: "
                f"labels {previous_label!r} and {entry.label!r}"
            )
        seen_pairs[pair] = entry.label
        if entry.batch not in batches:
            batches.append(entry.batch)
        if entry.perturbation not in perturbations:
            perturbations.append(entry.perturbation)
    curve_sets: dict[str, dict[tuple[str, str], PriorCurve]] = {}
    for gene_name in requested_genes:
        curve_map: dict[tuple[str, str], PriorCurve] = {}
        for entry in entries:
            priors = checkpoint.get_prior(entry.label)
            if gene_name in priors.gene_names:
                curve_map[(entry.batch, entry.perturbation)] = _curve_from_prior(
                    priors,
                    gene_name,
                    checkpoint_name=default_checkpoint_name(
                        checkpoint,
                        checkpoint_path=Path("checkpoint.pkl"),
                    ),
                    checkpoint_path=Path("checkpoint.pkl"),
                    scope_name=f"label:{entry.label}",
                    include_checkpoint_name=False,
                )
        if not curve_map:
            raise ValueError(
                f"gene {gene_name!r} is not present in the selected prior sources"
            )
        curve_sets[gene_name] = curve_map
    return curve_sets, batches, perturbations


def curve_sets_to_dataframe(
    curve_sets: dict[str, list[PriorCurve]], *, x_axis: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curves in curve_sets.items():
        for curve in curves:
            rows.extend(_curve_point_rows(gene_name, curve, x_axis=x_axis))
    return pd.DataFrame(rows)


def curve_sets_summary_dataframe(
    curve_sets: dict[str, list[PriorCurve]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curves in curve_sets.items():
        for curve in curves:
            rows.append(_curve_summary_row(gene_name, curve))
    return pd.DataFrame(rows)


def batch_grid_curve_sets_to_dataframe(
    curve_sets: dict[str, dict[tuple[str, str], PriorCurve]], *, x_axis: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curve_map in curve_sets.items():
        for (batch, perturbation), curve in curve_map.items():
            rows.extend(
                _curve_point_rows(
                    gene_name,
                    curve,
                    x_axis=x_axis,
                    batch=batch,
                    perturbation=perturbation,
                )
            )
    return pd.DataFrame(rows)


def batch_grid_summary_dataframe(
    curve_sets: dict[str, dict[tuple[str, str], PriorCurve]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curve_map in curve_sets.items():
        for (batch, perturbation), curve in curve_map.items():
            rows.append(
                _curve_summary_row(
                    gene_name,
                    curve,
                    batch=batch,
                    perturbation=perturbation,
                )
            )
    return pd.DataFrame(rows)


def __getattr__(name: str) -> object:
    try:
        return resolve_plot_export(name, package=__package__)
    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc


__all__ = [
    "PriorCurve",
    "batch_grid_curve_sets_to_dataframe",
    "batch_grid_summary_dataframe",
    "curve_sets_summary_dataframe",
    "curve_sets_to_dataframe",
    "default_checkpoint_name",
    "plot_batch_grid_figure",
    "plot_prior_facet_figure",
    "plot_prior_overlay_figure",
    "plt",
    "resolve_batch_grid_curve_sets",
    "resolve_multi_checkpoint_prior_curve_sets",
    "resolve_prior_curve_sets",
]

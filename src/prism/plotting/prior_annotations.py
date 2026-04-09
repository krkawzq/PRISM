from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._shared import dedupe_names


@dataclass(frozen=True, slots=True)
class LabelGridEntry:
    label: str
    batch: str
    perturbation: str


def _format_annotation_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if np.isnan(value_f):
            return "NA"
        return f"{value_f:.2f}"
    text = str(value)
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def _normalize_table_name(name: object, *, path: Path) -> str:
    resolved = str(name).strip()
    if not resolved:
        raise ValueError(f"annotation table name cannot be blank: {path}")
    return resolved


def _require_text_cell(value: object, *, field_name: str, path: Path) -> str:
    na_value = pd.isna(value)
    if bool(na_value.item()) if hasattr(na_value, "item") else bool(na_value):
        raise ValueError(f"label-grid CSV has blank {field_name!r} value: {path}")
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"label-grid CSV has blank {field_name!r} value: {path}")
    return resolved


def _resolve_annotation_source(row: pd.Series, *, path: Path) -> str:
    if "source" in row.index:
        return _require_text_cell(row["source"], field_name="source", path=path)

    label_col = "label" if "label" in row.index else None
    raw_label = None if label_col is None else row[label_col]
    label_text = "" if raw_label is None else str(raw_label).strip()
    label_is_na = False
    if raw_label is not None:
        na_value = pd.isna(raw_label)
        label_is_na = (
            bool(na_value.item()) if hasattr(na_value, "item") else bool(na_value)
        )
    scope = (
        "global"
        if raw_label is None
        or label_text in {"", "NA", "NaN", "nan"}
        or label_is_na
        else f"label:{label_text}"
    )
    if "checkpoint" in row.index:
        checkpoint_name = _require_text_cell(
            row["checkpoint"],
            field_name="checkpoint",
            path=path,
        )
        return f"{checkpoint_name}/{scope}"
    return scope


def _validate_label_grid_entries(
    entries: list[LabelGridEntry],
    *,
    source_description: str,
    empty_message: str,
) -> list[LabelGridEntry]:
    if not entries:
        raise ValueError(empty_message)
    seen_labels: set[str] = set()
    seen_pairs: dict[tuple[str, str], str] = {}
    for entry in entries:
        if not entry.label:
            raise ValueError(f"{source_description} has blank label value")
        if not entry.batch:
            raise ValueError(f"{source_description} has blank batch value")
        if not entry.perturbation:
            raise ValueError(f"{source_description} has blank perturbation value")
        if entry.label in seen_labels:
            raise ValueError(
                f"{source_description} contains duplicate label {entry.label!r}"
            )
        seen_labels.add(entry.label)
        pair = (entry.batch, entry.perturbation)
        previous = seen_pairs.get(pair)
        if previous is not None:
            raise ValueError(
                "duplicate batch-grid cell for "
                f"batch={entry.batch!r}, perturbation={entry.perturbation!r}: "
                f"labels {previous!r} and {entry.label!r}"
            )
        seen_pairs[pair] = entry.label
    return entries


def load_annotation_tables(
    csv_paths: list[Path], annot_names: list[str] | None
) -> dict[str, dict[tuple[str, str], str]]:
    if annot_names and len(annot_names) != len(csv_paths):
        raise ValueError("--annot-name count must match --annot-csv count")
    tables: dict[str, dict[tuple[str, str], str]] = {}
    for idx, path in enumerate(csv_paths):
        name = _normalize_table_name(
            path.stem if not annot_names else annot_names[idx],
            path=path,
        )
        if name in tables:
            raise ValueError(f"duplicate annotation table name: {name!r}")
        df = pd.read_csv(path)
        if "gene" not in df.columns:
            raise ValueError(f"annotation CSV missing required 'gene' column: {path}")
        value_cols = [
            col
            for col in df.columns
            if col not in {"gene", "label", "source", "checkpoint"}
        ]
        if not value_cols:
            raise ValueError(f"annotation CSV has no value columns: {path}")
        mapping: dict[tuple[str, str], str] = {}
        for _, row in df.iterrows():
            gene = str(row["gene"]).strip()
            if not gene:
                raise ValueError(f"annotation CSV has blank gene value: {path}")
            source = _resolve_annotation_source(row, path=path)
            text = ", ".join(
                f"{col}={_format_annotation_value(row[col])}" for col in value_cols
            )
            mapping[(gene, source)] = text
        tables[name] = mapping
    return tables


def parse_batch_grid_entries(labels: list[str]) -> list[LabelGridEntry]:
    entries: list[LabelGridEntry] = []
    for label in dedupe_names(labels):
        if "_" not in label:
            raise ValueError(
                f"could not infer batch/perturbation from label {label!r}; provide a CSV with columns label,batch,perturbation"
            )
        batch, perturbation = label.split("_", 1)
        if not batch or not perturbation:
            raise ValueError(f"invalid batch-grid label: {label!r}")
        entries.append(
            LabelGridEntry(label=label, batch=batch, perturbation=perturbation)
        )
    return _validate_label_grid_entries(
        entries,
        source_description="resolved batch-grid labels",
        empty_message="no batch-grid labels were resolved",
    )


def load_label_grid_entries(path: Path) -> list[LabelGridEntry]:
    df = pd.read_csv(path)
    required = {"label", "batch", "perturbation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"label-grid CSV is missing required columns {missing}: {path}"
        )
    entries = [
        LabelGridEntry(
            label=_require_text_cell(row["label"], field_name="label", path=path),
            batch=_require_text_cell(row["batch"], field_name="batch", path=path),
            perturbation=_require_text_cell(
                row["perturbation"],
                field_name="perturbation",
                path=path,
            ),
        )
        for _, row in df.iterrows()
    ]
    return _validate_label_grid_entries(
        entries,
        source_description=f"label-grid CSV {path}",
        empty_message=f"label-grid CSV is empty: {path}",
    )


__all__ = [
    "LabelGridEntry",
    "load_annotation_tables",
    "load_label_grid_entries",
    "parse_batch_grid_entries",
]

"""Shared readers and writers for ranked gene lists and generic string lists.

This module intentionally keeps two layers of compatibility:

1. Plain-text one-item-per-line files
2. JSON payloads produced by older CLI commands, experiment scripts, and the
   current normalized schema

Write paths always emit the normalized schema. Read paths remain permissive so
older workflows continue to function while commands migrate to the shared IO
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def _invalid_list_file(path: Path, *, kind: str, detail: str) -> ValueError:
    return ValueError(f"invalid {kind} file {path}: {detail}")


def _read_text(path: Path, *, kind: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _invalid_list_file(path, kind=kind, detail=f"failed to read file ({exc})") from exc


def _read_json_payload(path: Path, *, kind: str) -> object:
    text = _read_text(path, kind=kind)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _invalid_list_file(
            path,
            kind=kind,
            detail=f"malformed JSON ({exc.msg})",
        ) from exc


def _dedupe_string_entries(values: list[object], *, kind: str) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        if not isinstance(raw_value, str):
            raise ValueError(f"{kind} entries must contain only strings")
        value = raw_value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dedupe_gene_entries(
    gene_names: list[object],
    scores: list[object] | None,
) -> tuple[list[str], list[float]]:
    deduped_names: list[str] = []
    deduped_scores: list[float] = []
    seen: set[str] = set()
    use_scores = scores is not None
    if use_scores and len(scores) != len(gene_names):
        raise ValueError("scores must align with gene_names")
    for idx, raw_name in enumerate(gene_names):
        if not isinstance(raw_name, str):
            raise ValueError("gene_names must contain only strings")
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        deduped_names.append(name)
        if use_scores:
            deduped_scores.append(float(scores[idx]))
    return deduped_names, deduped_scores


def _resolve_string_items(
    payload: dict[str, Any],
    *,
    source_path: Path,
    kind: str,
    keys: tuple[str, ...],
) -> list[str]:
    for key in keys:
        values = payload.get(key)
        if isinstance(values, list):
            return _dedupe_string_entries(values, kind=key)
    raise _invalid_list_file(
        source_path,
        kind=kind,
        detail=f"missing a valid string list field ({', '.join(keys)})",
    )


def _resolve_scores(payload: dict[str, Any], n_genes: int) -> list[object] | None:
    for key in ("scores", "rank_mean", "rank_sum"):
        values = payload.get(key)
        if isinstance(values, list) and len(values) == n_genes:
            return values
    return None


@dataclass(frozen=True, slots=True)
class StringListSpec:
    items: list[str]
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    kind: str = "string_list"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "items",
            _dedupe_string_entries(list(self.items), kind="items"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "kind", str(self.kind))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "kind": str(self.kind),
            "items": list(self.items),
            "source_path": self.source_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GeneListSpec:
    gene_names: list[str]
    scores: list[float] = field(default_factory=list)
    source_path: str | None = None
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    kind: str = "gene_list"

    def __post_init__(self) -> None:
        names, scores = _dedupe_gene_entries(
            list(self.gene_names),
            list(self.scores) if self.scores else None,
        )
        object.__setattr__(self, "gene_names", names)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))
        if self.method is not None:
            object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "kind", str(self.kind))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "kind": str(self.kind),
            "gene_names": list(self.gene_names),
            "scores": list(self.scores),
            "source_path": self.source_path,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


def _string_list_spec_from_json_payload(
    payload: dict[str, Any],
    *,
    source_path: Path,
) -> StringListSpec:
    metadata_raw = payload.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    items = _resolve_string_items(
        payload,
        source_path=source_path,
        kind="string-list",
        keys=("items", "values", "labels"),
    )
    return StringListSpec(
        items=items,
        source_path=str(payload.get("source_path", str(source_path))),
        metadata=metadata,
        schema_version=int(payload.get("schema_version", 1)),
        kind=str(payload.get("kind", "string_list")),
    )


def _gene_list_spec_from_json_payload(
    payload: dict[str, Any],
    *,
    source_path: Path,
) -> GeneListSpec:
    if "gene_names" in payload:
        gene_names = payload.get("gene_names")
        if not isinstance(gene_names, list):
            raise _invalid_list_file(
                source_path,
                kind="gene-list",
                detail="gene_names must be a string array",
            )
    else:
        gene_names = _resolve_string_items(
            payload,
            source_path=source_path,
            kind="gene-list",
            keys=("items", "values"),
        )
    scores = _resolve_scores(payload, len(gene_names))
    metadata_raw = payload.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    for legacy_key in (
        "top_k",
        "gene_indices",
        "rank_sum",
        "rank_mean",
        "score_definition",
    ):
        if legacy_key in payload and legacy_key not in metadata:
            metadata[legacy_key] = payload[legacy_key]
    return GeneListSpec(
        gene_names=[str(value) for value in gene_names],
        scores=[] if scores is None else [float(value) for value in scores],
        source_path=str(payload.get("source_path", str(source_path))),
        method=None if payload.get("method") is None else str(payload.get("method")),
        metadata=metadata,
        schema_version=int(payload.get("schema_version", 1)),
        kind=str(payload.get("kind", "gene_list")),
    )


def read_string_list_spec(path: str | Path) -> StringListSpec:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        return StringListSpec(
            items=[
                line.strip() for line in _read_text(resolved, kind="string-list").splitlines() if line.strip()
            ],
            source_path=str(resolved),
            metadata={"format": "text"},
        )
    payload = _read_json_payload(resolved, kind="string-list")
    if isinstance(payload, list):
        return StringListSpec(
            items=_dedupe_string_entries(payload, kind="items"),
            source_path=str(resolved),
            metadata={"format": "json-array"},
        )
    if not isinstance(payload, dict):
        raise _invalid_list_file(
            resolved,
            kind="string-list",
            detail="JSON payload must be an object or string array",
        )
    return _string_list_spec_from_json_payload(payload, source_path=resolved)


def read_string_list(path: str | Path) -> list[str]:
    return list(read_string_list_spec(path).items)


def read_gene_list_spec(path: str | Path) -> GeneListSpec:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        return GeneListSpec(
            gene_names=[
                line.strip() for line in _read_text(resolved, kind="gene-list").splitlines() if line.strip()
            ],
            scores=[],
            source_path=str(resolved),
            method="gene-list-text",
            metadata={"format": "text"},
        )
    payload = _read_json_payload(resolved, kind="gene-list")
    if isinstance(payload, list):
        return GeneListSpec(
            gene_names=_dedupe_string_entries(payload, kind="gene_names"),
            scores=[],
            source_path=str(resolved),
            method="gene-list-json-array",
            metadata={"format": "json-array"},
        )
    if not isinstance(payload, dict):
        raise _invalid_list_file(
            resolved,
            kind="gene-list",
            detail="JSON payload must be an object or string array",
        )
    return _gene_list_spec_from_json_payload(payload, source_path=resolved)


def read_gene_list(path: str | Path) -> list[str]:
    return list(read_gene_list_spec(path).gene_names)


def write_string_list_text(path: str | Path, items: list[str]) -> None:
    resolved = Path(path).expanduser().resolve()
    ordered = _dedupe_string_entries(list(items), kind="items")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        "" if not ordered else "\n".join(ordered) + "\n",
        encoding="utf-8",
    )


def write_string_list_spec(path: str | Path, spec: StringListSpec) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(spec.to_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_gene_list_text(path: str | Path, gene_names: list[str]) -> None:
    resolved = Path(path).expanduser().resolve()
    names, _ = _dedupe_gene_entries(list(gene_names), None)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        "" if not names else "\n".join(names) + "\n",
        encoding="utf-8",
    )


def write_gene_list_spec(path: str | Path, spec: GeneListSpec) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(spec.to_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "GeneListSpec",
    "StringListSpec",
    "read_gene_list",
    "read_gene_list_spec",
    "read_string_list",
    "read_string_list_spec",
    "write_gene_list_spec",
    "write_gene_list_text",
    "write_string_list_spec",
    "write_string_list_text",
]

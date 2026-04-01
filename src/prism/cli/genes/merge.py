from __future__ import annotations

from pathlib import Path

import typer

from prism.io import GeneListSpec, read_gene_list_spec, write_gene_list_spec, write_gene_list_text

from .common import console

SUPPORTED_MERGE_METHODS = ("rank-sum",)
SUPPORTED_GENE_SET_MODES = ("exact", "intersection", "union")


def _resolve_methods(specs: list[GeneListSpec]) -> list[str]:
    return [spec.method or "unknown" for spec in specs]


def _resolve_declared_methods(specs: list[GeneListSpec]) -> set[str]:
    return {
        str(spec.method)
        for spec in specs
        if spec.method not in (None, "", "gene-list-text")
    }


def _normalize_choice(value: str, *, supported: tuple[str, ...], option_name: str) -> str:
    resolved = value.strip().lower()
    if resolved not in supported:
        raise ValueError(f"{option_name} must be one of: {', '.join(supported)}")
    return resolved


def _validate_merge_inputs(specs: list[GeneListSpec]) -> None:
    if len(specs) < 2:
        raise ValueError("merge requires at least two input gene lists")
    methods = _resolve_declared_methods(specs)
    if len(methods) != 1:
        raise ValueError(
            "all inputs must use the same method, got: " + ", ".join(sorted(methods))
        )


def _resolve_gene_universe(specs: list[GeneListSpec], *, gene_set_mode: str) -> list[str]:
    ordered_inputs = [list(spec.gene_names) for spec in specs]
    if gene_set_mode == "exact":
        reference = set(ordered_inputs[0])
        for genes in ordered_inputs[1:]:
            current = set(genes)
            if current != reference:
                missing = sorted(reference - current)
                extra = sorted(current - reference)
                details: list[str] = []
                if missing:
                    details.append(f"missing {len(missing)} genes")
                if extra:
                    details.append(f"extra {len(extra)} genes")
                raise ValueError("gene set mismatch: " + ", ".join(details))
        return ordered_inputs[0]
    if gene_set_mode == "intersection":
        overlap = set(ordered_inputs[0])
        for genes in ordered_inputs[1:]:
            overlap &= set(genes)
        return [gene for gene in ordered_inputs[0] if gene in overlap]
    if gene_set_mode == "union":
        ordered: list[str] = []
        seen: set[str] = set()
        for genes in ordered_inputs:
            for gene in genes:
                if gene in seen:
                    continue
                seen.add(gene)
                ordered.append(gene)
        return ordered
    raise ValueError(
        "--gene-set-mode must be one of: " + ", ".join(SUPPORTED_GENE_SET_MODES)
    )


def _merge_rank_sum(
    specs: list[GeneListSpec],
    *,
    gene_set_mode: str,
) -> tuple[list[str], list[float]]:
    merged_genes = _resolve_gene_universe(specs, gene_set_mode=gene_set_mode)
    rank_maps = [
        {gene: idx for idx, gene in enumerate(spec.gene_names)} for spec in specs
    ]
    rows: list[tuple[float, float, str]] = []
    for gene in merged_genes:
        ranks = [
            float(rank_map.get(gene, len(spec.gene_names)))
            for rank_map, spec in zip(rank_maps, specs, strict=True)
        ]
        rank_sum = float(sum(ranks))
        rank_mean = rank_sum / max(len(ranks), 1)
        rows.append((rank_sum, rank_mean, gene))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [gene for _, _, gene in rows], [rank_mean for _, rank_mean, gene in rows]


def merge_genes_command(
    input_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Two or more input gene-list files."
    ),
    output_ranked_genes: Path = typer.Option(
        ..., "--output-ranked-genes", help="Output text file with merged genes."
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional structured gene-list JSON output.",
    ),
    method: str = typer.Option(
        "rank-sum",
        help="Merge method: " + ", ".join(SUPPORTED_MERGE_METHODS) + ".",
    ),
    gene_set_mode: str = typer.Option(
        "exact",
        help="Gene-set policy: " + ", ".join(SUPPORTED_GENE_SET_MODES) + ".",
    ),
) -> int:
    resolved_inputs = [path.expanduser().resolve() for path in input_paths]
    output_ranked_genes = output_ranked_genes.expanduser().resolve()
    output_json = None if output_json is None else output_json.expanduser().resolve()
    method_resolved = _normalize_choice(
        method,
        supported=SUPPORTED_MERGE_METHODS,
        option_name="--method",
    )
    gene_set_mode_resolved = _normalize_choice(
        gene_set_mode,
        supported=SUPPORTED_GENE_SET_MODES,
        option_name="--gene-set-mode",
    )

    specs = [read_gene_list_spec(path) for path in resolved_inputs]
    _validate_merge_inputs(specs)
    merged_genes, merged_scores = _merge_rank_sum(
        specs,
        gene_set_mode=gene_set_mode_resolved,
    )
    declared_methods = sorted(_resolve_declared_methods(specs))

    write_gene_list_text(output_ranked_genes, merged_genes)
    if output_json is not None:
        merged_spec = GeneListSpec(
            gene_names=merged_genes,
            scores=merged_scores,
            source_path=None,
            method=f"merge:{method_resolved}",
            metadata={
                "merge_method": method_resolved,
                "gene_set_mode": gene_set_mode_resolved,
                "score_definition": "mean rank across input lists (lower is better)",
                "score_order": "ascending",
                "input_paths": [str(path) for path in resolved_inputs],
                "input_methods": _resolve_methods(specs),
                "input_declared_methods": declared_methods,
                "merged_source_method": declared_methods[0] if declared_methods else None,
                "n_inputs": len(specs),
                "n_genes": len(merged_genes),
            },
        )
        write_gene_list_spec(output_json, merged_spec)

    console.print(f"[bold green]Saved[/bold green] {output_ranked_genes}")
    if output_json is not None:
        console.print(f"[bold green]Saved[/bold green] {output_json}")
    return 0


__all__ = ["merge_genes_command"]

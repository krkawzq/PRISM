#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

console = Console()

install_rich_traceback(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ranked gene lists by summing ranks across inputs."
    )
    parser.add_argument(
        "input_jsons",
        nargs="+",
        type=Path,
        help="Two or more calc_gene_list JSON outputs.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-ranked-genes", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_jsons) < 2:
        raise ValueError("at least two input JSON files are required")
    return args


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input payload must be a JSON object: {path}")
    for key in ("method", "gene_names", "scores"):
        if key not in payload:
            raise ValueError(f"missing required key {key!r} in {path}")
    gene_names = payload["gene_names"]
    scores = payload["scores"]
    if not isinstance(gene_names, list) or not all(
        isinstance(gene, str) and gene for gene in gene_names
    ):
        raise ValueError(f"gene_names must be a non-empty string list in {path}")
    if not isinstance(scores, list) or len(scores) != len(gene_names):
        raise ValueError(f"scores must align with gene_names in {path}")
    if len(gene_names) != len(set(gene_names)):
        raise ValueError(f"gene_names must be unique in {path}")
    return payload


def validate_payloads(
    payloads: list[dict[str, Any]], input_paths: list[Path]
) -> tuple[str, list[str]]:
    methods = {str(payload["method"]) for payload in payloads}
    if len(methods) != 1:
        raise ValueError(
            "all inputs must use the same method, got: " + ", ".join(sorted(methods))
        )

    reference_genes = [str(gene) for gene in payloads[0]["gene_names"]]
    reference_set = set(reference_genes)
    for path, payload in zip(input_paths[1:], payloads[1:], strict=False):
        current_genes = [str(gene) for gene in payload["gene_names"]]
        current_set = set(current_genes)
        if current_set != reference_set:
            missing = sorted(reference_set - current_set)
            extra = sorted(current_set - reference_set)
            details: list[str] = []
            if missing:
                details.append(f"missing {len(missing)} genes")
            if extra:
                details.append(f"extra {len(extra)} genes")
            raise ValueError(f"gene set mismatch for {path}: " + ", ".join(details))
    return next(iter(methods)), reference_genes


def merge_ranked_lists(
    payloads: list[dict[str, Any]], ordered_genes: list[str]
) -> tuple[list[str], list[int], list[float]]:
    rank_maps: list[dict[str, int]] = []
    for payload in payloads:
        rank_maps.append(
            {str(gene): rank for rank, gene in enumerate(payload["gene_names"])}
        )

    merged_rows: list[tuple[int, float, str]] = []
    n_inputs = len(rank_maps)
    for gene in ordered_genes:
        rank_sum = sum(rank_map[gene] for rank_map in rank_maps)
        rank_mean = rank_sum / n_inputs
        merged_rows.append((rank_sum, rank_mean, gene))

    merged_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    merged_genes = [gene for _, _, gene in merged_rows]
    rank_sums = [rank_sum for rank_sum, _, _ in merged_rows]
    rank_means = [rank_mean for _, rank_mean, _ in merged_rows]
    return merged_genes, rank_sums, rank_means


def write_ranked_gene_names(output_path: Path, gene_names: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(gene_names) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_paths = [path.expanduser().resolve() for path in args.input_jsons]
    output_json = args.output_json.expanduser().resolve()
    output_ranked_genes = args.output_ranked_genes.expanduser().resolve()

    intro = Table(show_header=False, box=None)
    intro.add_row("Inputs", str(len(input_paths)))
    intro.add_row("Output JSON", str(output_json))
    intro.add_row("Output genes", str(output_ranked_genes))
    console.print(Panel(intro, title="Merge Gene Lists", border_style="cyan"))

    with console.status("Loading and validating ranked gene lists..."):
        payloads = [load_payload(path) for path in input_paths]
        method, ordered_genes = validate_payloads(payloads, input_paths)
        merged_genes, rank_sums, rank_means = merge_ranked_lists(
            payloads, ordered_genes
        )

    output_payload = {
        "input_paths": [str(path) for path in input_paths],
        "method": method,
        "gene_names": merged_genes,
        "rank_sum": rank_sums,
        "rank_mean": rank_means,
        "metadata": {
            "merge_method": "rank_sum",
            "tie_breakers": ["rank_mean", "gene_name"],
            "n_inputs": len(input_paths),
            "n_genes": len(merged_genes),
        },
    }

    with console.status("Writing merged outputs..."):
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        write_ranked_gene_names(output_ranked_genes, merged_genes)

    summary = Table(title="Merge Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Method", method)
    summary.add_row("Input lists", str(len(input_paths)))
    summary.add_row("Genes", str(len(merged_genes)))
    summary.add_row(
        "Top merged genes", ", ".join(merged_genes[:5]) if merged_genes else "None"
    )
    summary.add_row("JSON output", str(output_json))
    summary.add_row("Ranked genes output", str(output_ranked_genes))
    console.print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(str(exc), title="merge_gene_list failed", border_style="red")
        )
        raise SystemExit(1) from exc

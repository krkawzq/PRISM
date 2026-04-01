from __future__ import annotations

from pathlib import Path

import typer

from prism.io import write_gene_list_spec, write_gene_list_text

from .common import (
    build_gene_list_spec,
    compute_ranking,
    console,
    filter_gene_scores,
    normalize_hvg_flavor,
    normalize_prior_source,
    normalize_rank_method,
    rank_gene_scores,
    read_gene_list,
    SUPPORTED_HVG_FLAVORS,
    SUPPORTED_PRIOR_SOURCES,
    SUPPORTED_RANK_METHODS,
)


def rank_genes_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad or checkpoint path."
    ),
    method: str = typer.Option(
        ...,
        "--method",
        help="Ranking method: " + ", ".join(SUPPORTED_RANK_METHODS) + ".",
    ),
    output_ranked_genes: Path = typer.Option(
        ..., "--output-ranked-genes", help="Output text file with ranked genes."
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional structured gene-list JSON output.",
    ),
    top_k: int | None = typer.Option(
        None,
        min=1,
        help="Optional preview top-k recorded in the JSON metadata.",
    ),
    restrict_genes_path: Path | None = typer.Option(
        None,
        "--restrict-genes",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON gene list restricting the ranked output.",
    ),
    max_cells: int | None = typer.Option(
        None,
        min=1,
        help="Maximum number of cells to use for h5ad-based methods.",
    ),
    random_seed: int = typer.Option(
        0,
        "--seed",
        min=0,
        help="Random seed used for cell subsampling.",
    ),
    hvg_flavor: str = typer.Option(
        "seurat_v3",
        help="Scanpy HVG flavor: " + ", ".join(SUPPORTED_HVG_FLAVORS) + ".",
    ),
    prior_source: str = typer.Option(
        "global",
        help=(
            "Checkpoint prior source for prior-entropy methods: "
            + ", ".join(SUPPORTED_PRIOR_SOURCES)
            + "."
        ),
    ),
    label: str | None = typer.Option(None, help="Label used when prior_source=label."),
) -> int:
    input_path = input_path.expanduser().resolve()
    output_ranked_genes = output_ranked_genes.expanduser().resolve()
    output_json = None if output_json is None else output_json.expanduser().resolve()
    restrict_genes_path = (
        None
        if restrict_genes_path is None
        else restrict_genes_path.expanduser().resolve()
    )
    method_resolved = normalize_rank_method(method)
    hvg_flavor_resolved = normalize_hvg_flavor(hvg_flavor)
    prior_source_resolved = normalize_prior_source(prior_source)
    restrict_genes = (
        None if restrict_genes_path is None else read_gene_list(restrict_genes_path)
    )

    result = compute_ranking(
        input_path,
        method=method_resolved,
        hvg_flavor=hvg_flavor_resolved,
        prior_source=prior_source_resolved,
        label=label,
        max_cells=max_cells,
        random_seed=random_seed,
    )
    gene_names, scores, filter_metadata = filter_gene_scores(
        result.gene_names,
        result.scores,
        restrict_genes=restrict_genes,
    )
    _, ranked_gene_names, ranked_scores = rank_gene_scores(
        gene_names,
        scores,
        descending=result.descending,
    )

    metadata = dict(result.metadata)
    metadata.update(filter_metadata)
    metadata.update(
        {
            "restrict_genes_path": None
            if restrict_genes_path is None
            else str(restrict_genes_path),
            "preview_top_k": None if top_k is None else int(top_k),
            "score_order": "descending" if result.descending else "ascending",
            "n_ranked_genes": int(len(ranked_gene_names)),
        }
    )

    write_gene_list_text(
        output_ranked_genes,
        [str(gene) for gene in ranked_gene_names.tolist()],
    )
    if output_json is not None:
        spec = build_gene_list_spec(
            input_path=input_path,
            method=method_resolved,
            ranked_gene_names=ranked_gene_names,
            ranked_scores=ranked_scores,
            metadata=metadata,
        )
        write_gene_list_spec(output_json, spec)

    console.print(f"[bold green]Saved[/bold green] {output_ranked_genes}")
    if output_json is not None:
        console.print(f"[bold green]Saved[/bold green] {output_json}")
    return 0


__all__ = ["rank_genes_command"]

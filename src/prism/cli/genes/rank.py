from __future__ import annotations

from pathlib import Path

import typer

from prism.cli.common import (
    print_saved_path,
    resolve_int,
    resolve_optional_int,
    resolve_optional_path,
    resolve_optional_str,
    resolve_str,
)

from .common import (
    SUPPORTED_HVG_FLAVORS,
    SUPPORTED_PRIOR_SOURCES,
    SUPPORTED_RANK_METHODS,
    compute_ranking,
    console,
    filter_gene_scores,
    normalize_hvg_flavor,
    normalize_prior_source,
    normalize_rank_method,
    print_gene_summary,
    rank_gene_scores,
    read_gene_list,
    write_gene_list,
)


def rank_genes_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad or checkpoint path."
    ),
    method: str = typer.Option(..., "--method", help="Ranking method."),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output text file with ranked genes."
    ),
    restrict_genes_path: Path | None = typer.Option(
        None,
        "--restrict-genes",
        exists=True,
        dir_okay=False,
        help="Optional text gene list restricting the ranked output.",
    ),
    max_cells: int | None = typer.Option(
        None, min=1, help="Maximum number of cells to use for h5ad-based methods."
    ),
    random_seed: int = typer.Option(
        0, "--seed", min=0, help="Random seed used for cell subsampling."
    ),
    hvg_flavor: str = typer.Option("seurat_v3", help="Scanpy HVG flavor."),
    prior_source: str = typer.Option(
        "global", help="Checkpoint prior source for prior-entropy methods."
    ),
    label: str | None = typer.Option(None, help="Label used when prior_source=label."),
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    restrict_genes_path = resolve_optional_path(restrict_genes_path)
    method_resolved = normalize_rank_method(resolve_str(method))
    max_cells = resolve_optional_int(max_cells)
    random_seed = resolve_int(random_seed)
    hvg_flavor_resolved = normalize_hvg_flavor(resolve_str(hvg_flavor))
    prior_source_resolved = normalize_prior_source(resolve_str(prior_source))
    label = resolve_optional_str(label)

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
        result.gene_names, result.scores, restrict_genes=restrict_genes
    )
    _, ranked_gene_names, _ = rank_gene_scores(
        gene_names, scores, descending=result.descending
    )
    write_gene_list(output_path, [str(gene) for gene in ranked_gene_names.tolist()])

    print_gene_summary(
        "Rank Genes",
        Input=input_path,
        Method=method_resolved,
        GenesOut=len(ranked_gene_names),
        MissingRestrictGenes=filter_metadata["n_missing_restrict_genes"],
        Output=output_path,
    )
    print_saved_path(console, output_path)
    return 0


__all__ = [
    "SUPPORTED_HVG_FLAVORS",
    "SUPPORTED_PRIOR_SOURCES",
    "SUPPORTED_RANK_METHODS",
    "rank_genes_command",
]

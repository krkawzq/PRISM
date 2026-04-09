from __future__ import annotations

from pathlib import Path

import typer

from prism.cli.common import (
    print_saved_path,
    resolve_bool,
    resolve_optional_path,
    resolve_str,
)

from .common import (
    BUILTIN_SPECIES,
    NuisanceRuleSet,
    SUPPORTED_FILTER_SPECIES,
    console,
    filter_gene_list,
    print_gene_summary,
    read_gene_list,
    resolve_filter_rules,
    write_gene_list,
)


def filter_genes_command(
    input_genes: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input gene-list text file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output filtered gene list."
    ),
    removed_output_path: Path | None = typer.Option(
        None, "--removed-output", help="Optional text file for removed genes."
    ),
    species: str = typer.Option("human", help="Built-in species rule-set."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        help="Optional JSON/YAML rule config.",
    ),
    config_only: bool = typer.Option(
        False, help="Ignore built-in species rules and use only --config."
    ),
    dry_run: bool = typer.Option(False, help="Preview results without writing files."),
) -> int:
    input_genes = input_genes.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    removed_output_path = resolve_optional_path(removed_output_path)
    species = resolve_str(species)
    config_path = resolve_optional_path(config_path)
    config_only = resolve_bool(config_only)
    dry_run = resolve_bool(dry_run)

    rules = resolve_filter_rules(
        species=species, config_path=config_path, config_only=config_only
    )
    genes = read_gene_list(input_genes)
    kept, removed, metadata = filter_gene_list(genes, rules=rules)
    if not kept:
        raise ValueError("filter removed all genes")

    print_gene_summary(
        "Filter Genes",
        Input=input_genes,
        RuleSet=metadata["rule_set"],
        GenesIn=metadata["n_input_genes"],
        GenesOut=metadata["n_kept_genes"],
        Removed=metadata["n_removed_genes"],
    )
    if dry_run:
        return 0

    write_gene_list(output_path, kept)
    print_saved_path(console, output_path)
    if removed_output_path is not None:
        write_gene_list(removed_output_path, removed)
        print_saved_path(console, removed_output_path)
    return 0


__all__ = [
    "BUILTIN_SPECIES",
    "NuisanceRuleSet",
    "SUPPORTED_FILTER_SPECIES",
    "filter_genes_command",
]

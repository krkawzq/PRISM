#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

mkdir -p \
    data/norman/01_counts \
    data/norman/02_guides \
    data/norman/03_cell_cycle

uv run python scripts/data/norman/build_anndata.py \
    --matrix-path data/raw/norman/filtered/GSE133344_filtered_matrix.mtx.gz \
    --barcodes-path data/raw/norman/filtered/GSE133344_filtered_barcodes.tsv.gz \
    --genes-path data/raw/norman/filtered/GSE133344_filtered_genes.tsv.gz \
    --no-filter \
    --output data/norman/01_counts/norman_filtered.h5ad \
    --force

uv run python scripts/data/norman/build_anndata.py \
    --matrix-path data/raw/norman/raw/GSE133344_raw_matrix.mtx.gz \
    --barcodes-path data/raw/norman/raw/GSE133344_raw_barcodes.tsv.gz \
    --genes-path data/raw/norman/raw/GSE133344_raw_genes.tsv.gz \
    --filter \
    --umi-threshold 2000 \
    --output data/norman/01_counts/norman_raw_u2000.h5ad \
    --force

uv run python scripts/data/norman/build_guide_targets.py \
    data/norman/01_counts/norman_filtered.h5ad \
    --cell-identities data/raw/norman/filtered/GSE133344_filtered_cell_identities.csv.gz \
    --output data/norman/02_guides/norman_filtered_guides.h5ad \
    --force

uv run python scripts/data/norman/build_guide_targets.py \
    data/norman/01_counts/norman_raw_u2000.h5ad \
    --cell-identities data/raw/norman/raw/GSE133344_raw_cell_identities.csv.gz \
    --output data/norman/02_guides/norman_raw_u2000_guides.h5ad \
    --force

uv run python scripts/data/norman/annotate_cell_cycle.py \
    data/norman/02_guides/norman_filtered_guides.h5ad \
    --output data/norman/03_cell_cycle/norman_filtered_cell_cycle.h5ad \
    --force

uv run python scripts/data/norman/annotate_cell_cycle.py \
    data/norman/02_guides/norman_raw_u2000_guides.h5ad \
    --output data/norman/03_cell_cycle/norman_raw_u2000_cell_cycle.h5ad \
    --force

echo "Norman default pipeline finished."

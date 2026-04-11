#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

GEO_ACCESSION="GSE133344"
GEO_SUPPL_BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/${GEO_ACCESSION}/suppl"
OUTPUT_DIR="$REPO_ROOT/data/raw/norman"
FORCE_DOWNLOAD=0

FILES=(
    "GSE133344_filtered_barcodes.tsv.gz"
    "GSE133344_filtered_cell_identities.csv.gz"
    "GSE133344_filtered_genes.tsv.gz"
    "GSE133344_filtered_matrix.mtx.gz"
    "GSE133344_raw_barcodes.tsv.gz"
    "GSE133344_raw_cell_identities.csv.gz"
    "GSE133344_raw_genes.tsv.gz"
    "GSE133344_raw_matrix.mtx.gz"
)

usage() {
    cat <<EOF
Usage:
  scripts/data/norman/download_gse133344_supplementary.sh [--output-dir DIR] [--force]

Description:
  Download the 8 supplementary matrix / metadata files for GEO ${GEO_ACCESSION}
  into data/raw/norman/filtered and data/raw/norman/raw by default.

Options:
  --output-dir DIR  Directory where the files will be saved.
                    Default: ${OUTPUT_DIR}
  --force           Re-download files even if they already exist.
  -h, --help        Show this help message.

Examples:
  scripts/data/norman/download_gse133344_supplementary.sh
  scripts/data/norman/download_gse133344_supplementary.sh --force
  scripts/data/norman/download_gse133344_supplementary.sh --output-dir data/raw/norman
EOF
}

log() {
    printf '[norman-download] %s\n' "$1"
}

has_command() {
    command -v "$1" >/dev/null 2>&1
}

download_file() {
    local url="$1"
    local destination="$2"
    local temp_file="${destination}.part"

    rm -f "$temp_file"

    if has_command curl; then
        curl \
            --fail \
            --location \
            --retry 5 \
            --retry-delay 5 \
            --output "$temp_file" \
            "$url"
    elif has_command wget; then
        wget \
            --tries=5 \
            --output-document="$temp_file" \
            "$url"
    else
        log "Neither curl nor wget is available."
        exit 1
    fi

    mv "$temp_file" "$destination"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            if [[ $# -lt 2 ]]; then
                log "--output-dir requires a value."
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_DOWNLOAD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

FILTERED_DIR="${OUTPUT_DIR}/filtered"
RAW_DIR="${OUTPUT_DIR}/raw"

mkdir -p "$FILTERED_DIR" "$RAW_DIR"

log "Downloading ${#FILES[@]} GEO supplementary files to $FILTERED_DIR and $RAW_DIR"

for filename in "${FILES[@]}"; do
    url="${GEO_SUPPL_BASE_URL}/${filename}"
    if [[ "$filename" == GSE133344_filtered_* ]]; then
        destination="${FILTERED_DIR}/${filename}"
    elif [[ "$filename" == GSE133344_raw_* ]]; then
        destination="${RAW_DIR}/${filename}"
    else
        log "Unexpected filename pattern: $filename"
        exit 1
    fi

    if [[ -s "$destination" && "$FORCE_DOWNLOAD" -eq 0 ]]; then
        log "Skipping existing file: $filename"
        continue
    fi

    log "Downloading $filename"
    download_file "$url" "$destination"
done

log "Finished downloading ${GEO_ACCESSION} supplementary files."

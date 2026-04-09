#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/data/prepare_10xgenomics_all.sh [shared_args]

Description:
  Run all 10x Genomics data preparation scripts in sequence.

Order:
  1. introns
  2. anticoagulants
  3. platforms
  4. cell_gradients
  5. tissues

Notes:
  - Any extra arguments are passed through to every Python prepare script.
  - Run from anywhere; the script resolves the repo root automatically.
  - Uses `uv run python`.

Examples:
  scripts/data/prepare_10xgenomics_all.sh
  scripts/data/prepare_10xgenomics_all.sh --force-download
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

run_step() {
    local label="$1"
    local script_path="$2"
    shift 2

    echo "============================================================"
    echo "Preparing ${label}"
    echo "============================================================"
    uv run python "$script_path" "$@"
    echo
}

cd "$REPO_ROOT"

run_step "10x introns" \
    "$SCRIPT_DIR/prepare_10xgenomics_pbmc_10k_introns.py" \
    "$@"
run_step "10x anticoagulants" \
    "$SCRIPT_DIR/prepare_10xgenomics_anticoagulants.py" \
    "$@"
run_step "10x platforms" \
    "$SCRIPT_DIR/prepare_10xgenomics_platforms.py" \
    "$@"
run_step "10x cell gradients" \
    "$SCRIPT_DIR/prepare_10xgenomics_cell_gradients.py" \
    "$@"
run_step "10x tissues" \
    "$SCRIPT_DIR/prepare_10xgenomics_tissues.py" \
    "$@"

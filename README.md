# PRISM

PRISM is a single-cell, gene-level empirical Bayes toolkit.

It fits per-gene priors from reference scRNA-seq counts, stores them as checkpoints, and uses those checkpoints to produce posterior-derived outputs such as `signal`, `map_p`, `posterior_entropy`, and `mutual_information`.

The codebase currently has three main faces:

- A Typer-based CLI exposed as `prism`
- A Python model layer under `src/prism/model`
- A local server app under `src/prism/server`

## Requirements

- Python 3.11+

## Install

```bash
uv pip install -e .
uv run prism --help
```

The console entrypoint is defined in `pyproject.toml`:

```toml
[project.scripts]
prism = "prism.cli.main:main"
```

## What PRISM Does

At a high level, PRISM works with AnnData (`.h5ad`) inputs and produces two core artifact types:

- Checkpoints: serialized fitted priors and fit metadata
- Extracted AnnData outputs: posterior-derived layers written back to `.h5ad`

The main workflow is:

1. Prepare a gene list.
2. Fit priors into a checkpoint with `prism fit priors`.
3. Extract posterior-derived outputs with `prism extract ...`.
4. Inspect checkpoints or render figures with `prism checkpoint ...` and `prism plot ...`.

## Repository Layout

The package layout under `src/prism` is:

- `cli/`: Typer command-line interface
- `model/`: fitting, inference, checkpoint serialization, numeric/model logic
- `io/`: AnnData and list read/write helpers
- `plotting/`: plotting primitives used by CLI plotting commands
- `gmm/`: Gaussian mixture helpers used by parts of the analysis stack
- `server/`: local API/server app
- `analysis/`: higher-level analysis code

If you only want to use the project from the command line, `src/prism/cli/main.py` is the top-level entry.

## CLI Overview

Current top-level commands:

```text
prism
  fit
    priors
  data
    subset-genes
    downsample
  extract
    signals
    kbulk
    kbulk-mean
  checkpoint
    inspect
    merge
  genes
    intersect
    subset
    rank
    merge
    filter
  plot
    priors
    batch-grid
    distributions
    label-summary
  serve
```

Useful discovery commands:

```bash
uv run prism --help
uv run prism fit --help
uv run prism extract signals --help
uv run prism plot priors --help
```

## Recommended Workflow

### 1. Build or prepare a gene list

Examples:

```bash
uv run prism genes rank input.h5ad \
  --method hvg \
  --output outputs/ranked_genes.txt

uv run prism genes filter outputs/ranked_genes.txt \
  --species human \
  --output outputs/filtered_genes.txt

uv run prism genes subset outputs/filtered_genes.txt \
  --top-k 1000 \
  --output outputs/top1000_genes.txt
```

### 2. Fit priors into a checkpoint

Global priors only:

```bash
uv run prism fit priors input.h5ad \
  --output outputs/checkpoint.pkl \
  --reference-genes refs.txt \
  --fit-genes outputs/top1000_genes.txt
```

Global plus label-specific priors:

```bash
uv run prism fit priors input.h5ad \
  --output outputs/checkpoint.pkl \
  --reference-genes refs.txt \
  --fit-genes outputs/top1000_genes.txt \
  --label-key cell_type \
  --fit-mode both \
  --device cuda
```

### 3. Extract posterior-derived outputs

Extract standard signal layers:

```bash
uv run prism extract signals outputs/checkpoint.pkl input.h5ad \
  --output outputs/signals.h5ad
```

Use label-specific priors and select channels explicitly:

```bash
uv run prism extract signals outputs/checkpoint.pkl input.h5ad \
  --output outputs/signals_by_label.h5ad \
  --prior-source label \
  --label-key cell_type \
  --channel signal \
  --channel map_p \
  --channel posterior_entropy
```

### 4. Inspect or visualize outputs

```bash
uv run prism checkpoint inspect outputs/checkpoint.pkl --show-labels

uv run prism plot priors outputs/checkpoint.pkl \
  --gene CD3D \
  --gene MS4A1 \
  --output outputs/priors.svg
```

## Command Reference

### `prism fit`

#### `prism fit priors`

Fits one checkpoint from one AnnData input.

Basic form:

```bash
uv run prism fit priors INPUT.h5ad --output OUTPUT.pkl [OPTIONS]
```

Important options:

- `--reference-genes`: text file used to compute reference counts
- `--fit-genes`: text file restricting which genes are fitted
- `--layer`: use a named AnnData layer instead of `X`
- `--label-key`, `--label-value`, `--label-list`: choose label-aware fitting scopes
- `--fit-mode`: `global`, `by-label`, or `both`
- `--n-samples`, `--sample-seed`: per-scope cell subsampling
- `--scale`: manually override the scale instead of using mean reference count
- `--device`: usually `cpu` or `cuda`
- `--gene-batch-size`, `--cell-chunk-size`: runtime batching controls
- `--shard`: shard specification like `0/4`
- `--n-support-points`, `--max-em-iterations`, `--convergence-tolerance`: EM/grid controls
- `--support-max-from`: `observed_max` or `quantile`
- `--support-spacing`: `linear` or `sqrt`
- `--use-adaptive-support`: two-phase adaptive support refinement
- `--likelihood`: `binomial`, `negative_binomial`, or `poisson`
- `--nb-overdispersion`: overdispersion used for `negative_binomial`
- `--warm-start-checkpoint`: initialize from an existing compatible checkpoint
- `--torch-dtype`: `float32` or `float64`
- `--compile-model`: enable `torch.compile` when available
- `--dry-run`: print the plan without writing output

### `prism extract`

#### `prism extract signals`

Runs posterior inference for a fitted checkpoint and writes a new AnnData with extracted layers.

Basic form:

```bash
uv run prism extract signals CHECKPOINT.pkl INPUT.h5ad \
  --output OUTPUT.h5ad [OPTIONS]
```

Important options:

- `--genes`: optional text file restricting extracted genes
- `--output-mode`: `fitted-only` or `full-matrix`
- `--prior-source`: `global` or `label`
- `--label-key`: required when `--prior-source label`
- `--batch-size`: genes per inference batch
- `--device`, `--torch-dtype`: inference runtime settings
- `--dtype`: output layer dtype
- `--channel`: repeatable channel selection
- `--dry-run`: show the extraction plan only

#### `prism extract kbulk`

Builds k-cell aggregate samples per class and runs checkpoint-backed inference on them.

Basic form:

```bash
uv run prism extract kbulk CHECKPOINT.pkl INPUT.h5ad \
  --output OUTPUT.h5ad \
  --class-key cell_type \
  --k 8 \
  --n-samples 100
```

Important options:

- `--class-key`: class/group column in `obs`
- `--k`: cells per aggregate sample
- `--n-samples`: target samples per class before balancing
- `--prior-source`: `global` or `label`
- `--genes`: optional gene restriction list
- `--reference-genes`: optional override for reference genes
- `--balance`: scale target samples by mean reference size
- `--scale-source`: `checkpoint` or `dataset`
- `--navg-source`: `dataset` or `checkpoint`
- `--sample-batch-size`: inference batch size over aggregate samples

#### `prism extract kbulk-mean`

Constructs k-cell sampled aggregates from the dataset itself and writes per-sample summary layers. This command does not take a checkpoint.

Basic form:

```bash
uv run prism extract kbulk-mean INPUT.h5ad \
  --output OUTPUT.h5ad \
  --class-key cell_type \
  --k 8 \
  --n-samples 100
```

Important options:

- `--genes`: optional gene restriction list
- `--reference-genes`: optional reference list used for balancing/statistics
- `--normalize-total`: per-cell normalization target before averaging
- `--log1p`: apply `log1p` after optional normalization
- `--balance`: scale target samples by mean reference size
- `--sample-batch-size`: combination aggregation batch size

### `prism genes`

Utilities for building and manipulating plain-text gene lists.

#### `prism genes rank`

Ranks genes from either an AnnData file or a checkpoint.

```bash
uv run prism genes rank input.h5ad \
  --method hvg \
  --output outputs/ranked_genes.txt
```

```bash
uv run prism genes rank input.h5ad \
  --method lognorm-variance \
  --obs-key guide_target \
  --obs-value NegCtrl0_NegCtrl0 \
  --obs-value NegCtrl10_NegCtrl0 \
  --top-n 200 \
  --output-gene-column gene_name \
  --output outputs/top200_symbols.txt
```

```bash
uv run prism genes rank outputs/checkpoint.pkl \
  --method prior-entropy \
  --prior-source global \
  --output outputs/prior_entropy_genes.txt
```

Useful options:

- `--method`: ranking method
- `--restrict-genes`: limit output to a supplied gene list
- `--max-cells`, `--seed`: h5ad-based ranking controls
- `--obs-key`, `--obs-value`, `--obs-values`: restrict h5ad ranking to selected cells
- `--top-n`: write only the leading ranked genes
- `--output-gene-column`: emit a `var` column such as `gene_name` instead of `var_names`
- `--prior-source`, `--label`: checkpoint-based ranking controls

#### `prism genes merge`

Merges two or more ranked gene-list files.

```bash
uv run prism genes merge list_a.txt list_b.txt list_c.txt \
  --output outputs/merged.txt
```

Useful options:

- `--method`: currently `rank-sum`
- `--gene-set-mode`: `exact`, `intersection`, or `union`

#### `prism genes filter`

Filters a gene list through built-in and/or config-driven rules.

```bash
uv run prism genes filter outputs/merged.txt \
  --species human \
  --output outputs/filtered.txt
```

Useful options:

- `--removed-output`: write removed genes separately
- `--species`: built-in species rule set
- `--config`: optional JSON/YAML filter rules
- `--config-only`: disable built-in species rules
- `--dry-run`: preview without writing

#### `prism genes subset`

Slices a gene list by range and/or set operations.

```bash
uv run prism genes subset outputs/filtered.txt \
  --top-k 500 \
  --output outputs/top500.txt
```

Useful options:

- `--start`, `--end`: range slicing
- `--intersect`: keep only genes present in another file
- `--exclude`: remove genes present in another file

#### `prism genes intersect`

Intersects `var_names` across two or more `.h5ad` datasets.

```bash
uv run prism genes intersect dataset_a.h5ad dataset_b.h5ad dataset_c.h5ad \
  --output outputs/shared_genes.txt
```

Useful options:

- `--summary-json`: optional JSON summary
- `--sort`: `first` or `alpha`

### `prism data`

Dataset preprocessing helpers.

#### `prism data subset-genes`

Subsets an AnnData object to genes listed in a text file.

```bash
uv run prism data subset-genes input.h5ad \
  --genes outputs/top500.txt \
  --output outputs/subset.h5ad
```

Useful options:

- `--allow-missing`: skip genes missing from the dataset instead of failing

#### `prism data downsample`

Downsamples cells, optionally stratified by an `obs` column.

```bash
uv run prism data downsample input.h5ad \
  --output outputs/downsampled.h5ad \
  --label-key cell_type \
  --fraction 0.25 \
  --per-class-min 20
```

Useful options:

- `--label-key`: stratification key, default `treatment`
- `--fraction`: retained fraction per class
- `--per-class-min`: minimum retained cells per class
- `--seed`: random seed

### `prism checkpoint`

#### `prism checkpoint inspect`

Prints checkpoint metadata and optional label-prior previews.

```bash
uv run prism checkpoint inspect outputs/checkpoint.pkl --show-labels
```

Useful options:

- `--show-labels`: print label prior names
- `--label-limit`: limit label preview count

#### `prism checkpoint merge`

Merges two or more compatible checkpoints, typically from sharded fitting.

```bash
uv run prism checkpoint merge shard0.pkl shard1.pkl shard2.pkl \
  --output outputs/merged.pkl
```

Useful options:

- `--allow-partial`: allow missing genes relative to the declared requested set

### `prism plot`

#### `prism plot priors`

Renders prior curves from one or more checkpoints.

```bash
uv run prism plot priors outputs/checkpoint.pkl \
  --gene CD3D \
  --gene MS4A1 \
  --output outputs/priors.svg
```

You can also rank genes directly from the source h5ad, then plot the resulting prior curves in one step:

```bash
uv run prism plot priors outputs/checkpoint.pkl \
  --rank-method lognorm-variance \
  --rank-obs-key guide_target \
  --rank-obs-value NegCtrl0_NegCtrl0 \
  --rank-obs-value NegCtrl10_NegCtrl0 \
  --rank-top-n 200 \
  --label NegCtrl0_NegCtrl0 \
  --label NegCtrl10_NegCtrl0 \
  --no-include-global \
  --layout facet \
  --output outputs/top200_priors.svg
```

If your checkpoint stores Ensembl-style gene ids, you can map gene symbols through an h5ad `var` column:

```bash
uv run prism plot priors outputs/checkpoint.pkl \
  --gene HBG1 \
  --gene-id-column gene_name \
  --label NegCtrl0_NegCtrl0 \
  --output outputs/hbg1_prior.svg
```

Useful options:

- `--checkpoint-name`: display names aligned to input checkpoints
- `--gene`, `--genes`, `--top-n`: explicit gene selection controls
- `--rank-h5ad`, `--rank-method`, `--rank-obs-key`, `--rank-obs-value`, `--rank-top-n`: rank-driven gene selection from h5ad
- `--gene-annotations-h5ad`, `--gene-id-column`: map gene symbols through h5ad annotations and add facet labels
- `--label`, `--labels`: label prior selection controls
- `--output-csv`, `--summary-csv`: export plotted data/statistics
- `--annot-csv`, `--annot-name`: extra facet annotations
- `--x-axis`: `auto`, `scaled`, `support`, or `rate`
- `--curve-mode`: `density` or `cdf`
- `--y-scale`: `linear` or `log`
- `--mass-quantile`: truncate the displayed support range by cumulative mass
- `--include-global`: include global prior in label-aware plots
- `--layout`: `overlay` or `facet`
- `--stat`: annotate summary statistics

#### `prism plot batch-grid`

Builds a per-gene, per-label batch grid figure from a checkpoint.

```bash
uv run prism plot batch-grid outputs/checkpoint.pkl \
  --gene CD3D \
  --output-dir outputs/batch_grid
```

Useful options:

- `--label-grid-csv`: custom CSV describing batch/perturbation grid layout; recommended when label names do not follow `batch_perturbation`
- `--image-format`: `svg`, `pdf`, or `eps`
- `--output-csv`, `--summary-csv`: export data/statistics
- `--hide-empty`: hide empty grid cells
- `--show-axis-ticks`: render axis tick labels inside cells

#### `prism plot distributions`

Plots layer distributions from an extracted AnnData file.

```bash
uv run prism plot distributions outputs/signals.h5ad \
  --output outputs/distributions.svg
```

```bash
uv run prism plot distributions outputs/signals.h5ad \
  --gene CD3D \
  --gene MS4A1 \
  --group-key cell_type \
  --plot-type violin \
  --output outputs/distributions_markers.svg
```

Useful options:

- `--gene`, `--genes`: explicit gene selection
- `--layer`: repeatable layer selection
- `--group-key`: group by an `obs` column
- `--plot-type`: `violin`, `box`, or `hist`
- `--max-genes`: limit plotted genes when explicit genes are not provided
- `--sample-genes`: opt into random gene sampling instead of deterministic leading-gene selection

#### `prism plot label-summary`

Summarizes similarity or overlap across label-specific priors.

```bash
uv run prism plot label-summary outputs/checkpoint.pkl \
  --output outputs/label_summary.svg
```

Useful options:

- `--gene`: repeatable gene selection
- `--label`, `--labels`: restrict the summary matrix to selected label priors
- `--max-genes`: limit plotted genes
- `--metric`: `jsd` or `overlap`

### `prism serve`

Starts the local server app:

```bash
uv run prism serve --host 127.0.0.1 --port 8000
```

This is a thin CLI wrapper around the server code in `src/prism/server`.

## Distributed Fitting

The repository includes a launcher script for sharded fitting:

```bash
uv run python scripts/dist/run_fit_distributed.py \
  --h5ad input.h5ad \
  --output-prefix outputs/priors_run \
  --world-size 4 \
  --gpus 0,1,2,3 \
  --reference-genes refs.txt \
  --fit-genes outputs/top1000_genes.txt \
  -- \
  --label-key cell_type \
  --fit-mode both \
  --torch-dtype float32
```

Typical outputs:

- `<output-prefix>.dist_<timestamp>/logs/`: one log per rank
- `<output-prefix>.dist_<timestamp>/shards/`: one checkpoint per rank
- `<output-prefix>.merged.pkl`: merged checkpoint after successful completion

Use `--dry-run` on the launcher to inspect generated shard commands without launching them.

## Python API

The underlying model layer is also usable directly from Python.

```python
from prism.model import (
    ObservationBatch,
    PriorFitConfig,
    checkpoint_from_fit_result,
    fit_gene_priors,
    infer_posteriors,
    load_checkpoint,
    save_checkpoint,
)
```

If you are trying to understand the implementation, start here:

- `src/prism/cli/main.py`: top-level CLI registration
- `src/prism/cli/fit/priors.py`: checkpoint fitting command
- `src/prism/cli/extract/signals.py`: posterior extraction command
- `src/prism/model/fit.py`: model fitting logic
- `src/prism/model/infer.py`: posterior inference logic
- `src/prism/model/checkpoint.py`: checkpoint serialization

## Notes

- Most file-based gene list commands expect plain text with one gene per line.
- Most dataset commands operate on `.h5ad` inputs and preserve AnnData structure.
- The CLI is implemented with Typer and uses `no_args_is_help=True`, so running a command group without a subcommand prints help.
- `--dry-run` is available on the heavier fit/extract commands and is useful for validating inputs before long runs.

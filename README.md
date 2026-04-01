# PRISM

PRISM is a single-cell gene-level empirical Bayes modeling toolkit. It fits per-gene prior distributions over expression probability from scRNA-seq count data, then uses those priors to compute posterior-derived signals, entropy, and mutual information for each cell-gene pair.

## Install

Requires Python >= 3.11.

```bash
# with uv (recommended)
uv pip install -e .

# or with pip
pip install -e .
```

Verify the installation:

```bash
prism --help
```

## Core Concepts

PRISM operates on a simple pipeline:

1. **Fit** per-gene prior distributions from a reference scRNA-seq dataset
2. **Save** the fitted priors as a checkpoint
3. **Extract** posterior-derived signals (signal, map_p, entropy, MI) into an output h5ad
4. **Analyze** and **plot** the results

The model treats each gene independently. For each gene, it places a discrete prior over expression probability `p` on a grid, then updates this prior with observed counts via a binomial likelihood to obtain cell-specific posteriors.

## CLI Reference

The `prism` CLI is organized into command groups:

```
prism
  fit          Fit model artifacts
  extract      Extract signals and derived outputs
  analyze      Tabular analyses and statistics
  plot         Render figures
  data         Prepare and transform datasets
  genes        Build and manipulate gene lists
  checkpoint   Inspect and merge checkpoints
  serve        Launch the interactive web server
```

### prism fit

Fit global or label-specific priors from an input h5ad.

```bash
# Fit priors with default settings
prism fit priors input.h5ad -o checkpoint.pkl

# Fit with custom reference genes and label-specific priors
prism fit priors input.h5ad -o checkpoint.pkl \
  --reference-genes refs.txt \
  --label-key cell_type \
  --fit-mode both

# Fit with warm start from a previous checkpoint
prism fit priors input.h5ad -o checkpoint.pkl \
  --warm-start-checkpoint previous.pkl

# Fit with early stopping and sqrt grid spacing
prism fit priors input.h5ad -o checkpoint.pkl \
  --early-stop-tol 1e-4 \
  --early-stop-patience 5 \
  --grid-strategy sqrt
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `--grid-size` | 512 | Number of grid points for the prior |
| `--n-iter` | 100 | Optimization iterations |
| `--fit-mode` | global | Scope: `global`, `by-label`, or `both` |
| `--fit-method` | gradient | Fitting method: `gradient` or `em` |
| `--init-method` | posterior_mean | Initialization: `posterior_mean`, `uniform`, or `random` |
| `--warm-start-checkpoint` | None | Initialize from an existing checkpoint |
| `--early-stop-tol` | None | Loss improvement threshold for early stopping |
| `--early-stop-patience` | None | Steps without improvement before stopping |
| `--align-every` | 1 | Recompute alignment term every N steps |
| `--grid-max-method` | observed_max | Grid bound method: `observed_max` or `quantile` |
| `--grid-strategy` | linear | Grid spacing: `linear` or `sqrt` |
| `--torch-dtype` | float64 | Computation dtype: `float64` or `float32` |
| `--gene-batch-size` | 64 | Genes per fit batch (controls memory) |
| `--cell-chunk-size` | 512 | Cells per likelihood chunk (controls GPU memory) |
| `--device` | cpu | Torch device, e.g. `cpu` or `cuda` |

### prism extract

Extract posterior-derived outputs from a checkpoint into an h5ad.

```bash
# Extract core signal channels
prism extract signals checkpoint.pkl input.h5ad -o output.h5ad

# Extract with float32 inference for speed
prism extract signals checkpoint.pkl input.h5ad -o output.h5ad \
  --torch-dtype float32

# Extract specific channels
prism extract signals checkpoint.pkl input.h5ad -o output.h5ad \
  --channel signal --channel map_p --channel posterior_entropy

# kBulk aggregated extraction
prism extract kbulk checkpoint.pkl input.h5ad -o kbulk_output.h5ad \
  --label-key cell_type --k 8

# kBulk mean extraction
prism extract kbulk-mean checkpoint.pkl input.h5ad -o kbulk_mean.h5ad \
  --label-key cell_type --k 8
```

### prism analyze

Tabular analyses and statistics from checkpoints and extracted outputs.

```bash
# Overlap-based differential expression
prism analyze overlap-de checkpoint.pkl \
  --output-csv overlap_de.csv \
  --control-label ctrl

# Checkpoint summary with per-gene statistics
prism analyze checkpoint-summary checkpoint.pkl \
  --output-csv gene_stats.csv \
  --output-json metadata.json
```

### prism plot

Render figures from checkpoints and extracted outputs.

```bash
# Plot prior distributions
prism plot priors checkpoint.pkl -o priors.svg \
  --gene GeneA --gene GeneB

# Batch-grid layout
prism plot batch-grid checkpoint.pkl -o batch_grid/ \
  --gene GeneA

# Overlap heatmap
prism plot overlap checkpoint.pkl -o overlap.svg \
  --metric jsd

# Distribution plots from extracted h5ad (signal, entropy, MI)
prism plot distributions output.h5ad -o distributions.svg \
  --layer signal --layer posterior_entropy \
  --group-key cell_type \
  --plot-type violin

# Label similarity heatmap from checkpoint
prism plot label-summary checkpoint.pkl -o label_sim.svg \
  --metric jsd
```

### prism data

Prepare and transform AnnData inputs.

```bash
# Subset genes from an h5ad
prism data subset-genes input.h5ad -o subset.h5ad --genes gene_list.txt

# Stratified downsampling
prism data downsample input.h5ad -o downsampled.h5ad \
  --label-key cell_type --max-per-class 500
```

### prism genes

Build and manipulate gene lists.

```bash
# Rank genes by various methods
prism genes rank input.h5ad -o ranked.json --method hvg
prism genes rank input.h5ad -o ranked.json --method signal-variance

# Merge multiple ranked gene lists
prism genes merge list1.json list2.json -o merged.json

# Filter nuisance genes
prism genes filter ranked.json -o filtered.json --species human

# Intersect gene lists
prism genes intersect list1.json list2.json -o common.json

# Subset a gene list
prism genes subset ranked.json -o top500.json --top-k 500
```

Supported ranking methods: `hvg`, `signal-hvg`, `lognorm-variance`, `lognorm-dispersion`, `signal-variance`, `signal-dispersion`, `prior-entropy`, `prior-entropy-reverse`.

### prism checkpoint

Inspect and merge fitted checkpoints.

```bash
# Inspect checkpoint metadata
prism checkpoint inspect checkpoint.pkl

# Merge sharded checkpoints
prism checkpoint merge shard_0.pkl shard_1.pkl -o merged.pkl
```

Legacy aliases: `prism checkpoint plot-fg` maps to `prism plot priors`, and `prism checkpoint overlap-de` maps to `prism analyze overlap-de`.

### prism serve

Launch the interactive web server for checkpoint-aware gene inspection.

```bash
prism serve --h5ad input.h5ad --checkpoint checkpoint.pkl
```

## Project Structure

```
src/prism/
  model/          Core model: prior fitting, posterior inference, kBulk, checkpoint I/O
  cli/            CLI command groups (fit, extract, analyze, plot, data, genes, checkpoint, serve)
  io/             Shared I/O: gene list specs, AnnData helpers, atomic writes
  plotting/       Matplotlib-based plotting backends
  server/         Web server for interactive exploration
  baseline/       Baseline comparison utilities

scripts/
  data/           Dataset-specific preparation utilities
  analysis/       Analysis helpers not yet promoted to CLI
  experiments/    Experimental training entrypoints
  dev/            Developer diagnostics and validation scripts
  dist/           Distributed execution wrappers

src/tests/          Regression tests for model, CLI, checkpoint, and I/O
docs/             Theory notes and refactoring plan
```

## Python API

The model layer is available as a Python library:

```python
from prism.model import (
    ObservationBatch,
    PriorFitConfig,
    PriorGrid,
    Posterior,
    fit_gene_priors,
    infer_posteriors,
    load_checkpoint,
    save_checkpoint,
)

# Fit priors
result = fit_gene_priors(
    ObservationBatch(gene_names=["GeneA"], counts=counts, reference_counts=ref_counts),
    S=10.0,
    config=PriorFitConfig(grid_size=256, n_iter=50),
)

# Save and load checkpoints
save_checkpoint(checkpoint, "checkpoint.pkl")
checkpoint = load_checkpoint("checkpoint.pkl")

# Run posterior inference
result = infer_posteriors(batch, priors, torch_dtype="float32")

# Extract signals via Posterior helper
posterior = Posterior(gene_names, priors, torch_dtype="float32")
extracted = posterior.extract(batch, channels={"signal", "map_p"})
```

## Development

Run the test suite:

```bash
uv run --with pytest python -m pytest src/tests/ -v
```

## Documentation

- [Theory and model notes (Chinese)](docs/scPRISM_zh.md)
- [Refactoring plan and progress (Chinese)](docs/refactor_plan_zh.md)

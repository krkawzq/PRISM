# Phase 2 Spec: kBulk poisson support boundary

## Findings
- `src/prism/model/kbulk.py` already supports `posterior_distribution='poisson'` and returns `map_rate`.
- `src/prism/cli/extract/kbulk.py` does not preserve `map_rate` in output h5ad.
- Current CLI writes `X=map_mu` and `layers['map_p']`, which is semantically incomplete for poisson checkpoints.

## Decision
For now, `prism extract kbulk` should fail fast on poisson/rate-grid checkpoints.

## Behavior
1. Add explicit validation near CLI entry:
   - if resolved checkpoint posterior distribution is `poisson` or checkpoint grid_domain is `rate`, raise ValueError
   - error must explain that kbulk CLI does not yet export poisson `map_rate`, so poisson checkpoints are unsupported for this command
2. Keep model/kbulk unchanged.
3. Add tests covering:
   - model-level kbulk poisson path still produces `map_rate`
   - CLI/checkpoint validation path rejects poisson checkpoint usage

## Non-goals
- Implementing full poisson kbulk output schema (`map_rate` layer, metadata migration, downstream compatibility)

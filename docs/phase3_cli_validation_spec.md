# Phase 3 Spec: shared CLI checkpoint distribution validation

## Goal
Provide a single CLI-side entrypoint for checkpoint distribution validation and use it across checkpoint-consuming commands.

## Shared policy
1. Reuse checkpoint-level strict resolution from `resolve_checkpoint_distribution()`.
2. Add CLI helper returning normalized distribution info and enforcing optional constraints:
   - allow_distributions
   - require_label_priors
   - require_global_priors
   - require_grid_domains
3. Error messages must include command name and resolved distribution/grid_domain.

## Command support matrix
- checkpoint inspect: support all explicit/legacy-resolved checkpoints
- checkpoint merge: support all distributions, but existing metadata/prior consistency checks remain strict
- analyze checkpoint-summary: support all distributions
- plot priors: support all distributions (already supports rate-grid x-axis)
- plot overlap / analyze overlap-de / plot label-summary:
  - support only `grid_domain='p'`
  - reject poisson/rate-grid explicitly because overlap/jsd summaries were designed around p-space prior comparison semantics

## Non-goals
- server changes
- extract kbulk (handled in phase 2)

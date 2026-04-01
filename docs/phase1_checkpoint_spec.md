# Phase 1 Spec: Checkpoint strict distribution resolution

## Goal
Make checkpoint distribution handling explicit and strict for schema>=2 while keeping limited compatibility for legacy checkpoints.

## Scope
- src/prism/model/checkpoint.py
- tests covering save/load roundtrip and legacy compatibility

## Behavior
1. Add a helper to resolve checkpoint distribution metadata from both metadata and PriorGrid payloads.
2. For schema>=2:
   - require explicit distribution/grid_domain information
   - if missing, raise clear ValueError
   - if metadata and PriorGrid disagree, raise clear ValueError
3. For schema==1 or payloads lacking new fields:
   - infer binomial + p-grid compatibility path
   - attach explicit metadata markers indicating legacy compatibility was used
   - emit warning
4. Save path should always write explicit distribution/grid_domain for global and label priors.
5. Roundtrip must preserve binomial / negative_binomial / poisson faithfully.

## Non-goals
- server changes
- broader CLI validation outside checkpoint loading/saving

| Input | Model | Test Acc | Macro F1 |
|---|---:|---:|---:|
| Raw count | linear | 0.4037 | 0.3488 |
| Raw count | mlp | 0.4037 | 0.3352 |
| Raw lognormal | linear | 0.3231 | 0.2382 |
| Raw lognormal | mlp | 0.3637 | 0.3002 |
| map_mu label-specific | linear | 0.9968 | 0.9977 |
| map_mu label-specific | mlp | 0.9938 | 0.9930 |
| map_mu global | linear | 0.3230 | 0.2477 |
| map_mu global | mlp | 0.3920 | 0.3187 |
| kbulk global k=2 | mlp | 0.8744 | 0.8763 |
| kbulk global k=3 | mlp | 0.9484 | 0.9512 |
| kbulk global k=5 | mlp | 0.9749 | 0.9760 |
| kbulk-mean k=2 | mlp | 0.9787 | 0.9793 |
| kbulk-mean k=3 | mlp | 0.9936 | 0.9940 |
| kbulk-mean k=5 | mlp | 0.9984 | 0.9986 |

| Input | Model | Test Acc | Macro F1 |
|---|---:|---:|---:|
| Raw count + align-avg | mlp | 0.1166 | 0.0075 |
| Raw lognormal + align-avg | mlp | 0.1127 | 0.0055 |
| map_mu global + align-avg | mlp | 0.1262 | 0.0120 |
| map_mu label-specific + align-avg | mlp | 0.1353 | 0.0186 |
| kbulk global k=2 + align-avg | mlp | 0.0408 | 0.0303 |
| kbulk global k=3 + align-avg | mlp | 0.0176 | 0.0136 |
| kbulk global k=5 + align-avg | mlp | 0.0068 | 0.0001 |
| kbulk-mean k=2 + align-avg | mlp | 0.0147 | 0.0039 |
| kbulk-mean k=3 + align-avg | mlp | 0.0120 | 0.0016 |
| kbulk-mean k=5 + align-avg | mlp | 0.0131 | 0.0024 |

## Notes

- `map_mu` with label-specific priors is near-perfect, but this likely contains label prior leakage and should not be interpreted as unbiased representation quality.
- `map_mu` with global priors is close to the raw-count baseline and does not clearly outperform it.
- Both `kbulk` and `kbulk-mean` become extremely separable as `k` increases.
- `kbulk-mean` is at least as strong as `kbulk global` in this classification setting.
- After `align-avg`, all methods collapse sharply, suggesting that the dominant classification signal is largely driven by class-level mean shifts.
- Under the current setup, these results do not yet support the claim that higher-order program structure alone is sufficient for strong classification once mean effects are removed.

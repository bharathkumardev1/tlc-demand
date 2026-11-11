# 003. Zone, borough and city forecasts, and whether reconciling them helps

Status: accepted, result is small

## Context
The service forecasts 262 zones. City and borough totals are easier to forecast than any single zone, and a dispatcher who sums zone forecasts should not get a different city number than the city model gives. I trained one LightGBM model per level (zone, 8 borough values as listed in the TLC zone lookup including EWR, N/A and Unknown, and the city total) on the same features and the same horizon (1 hour), then reconciled the three sets of forecasts so every parent equals the sum of its children.

Reconciliation weights come only from forecast errors on earlier folds (expanding window). A fourth, earlier fold (week starting 2023-06-03) supplies the first residuals, so the three scored folds are the same ones as everywhere else (weeks starting 06-10, 06-17, 06-24) and no fold is scored with weights fitted on its own errors. Reconciled values are clipped at zero afterwards. The math is in `src/tlc_demand/hierarchy.py` with property tests; `scripts/run_hierarchy.py` produced `results/hierarchy.json`.

## Results (WAPE, pooled over the three June folds)
| Method | city | borough | zone |
|---|---|---|---|
| seasonal-naive | 0.1031 | 0.1285 | 0.3096 |
| independent base forecasts | 0.0593 | 0.0806 | 0.2412 |
| bottom-up (sum of zone forecasts) | 0.0659 | 0.0869 | 0.2412 |
| OLS | 0.0591 | 0.0835 | 0.2449 |
| variance-weighted | 0.0623 | 0.0825 | 0.2394 |
| MinT, shrunk covariance | 0.0600 | 0.0799 | 0.2384 |

Per fold, MinT beats the independent zone forecasts in all three folds (0.2315 vs 0.2336, 0.2370 vs 0.2397, 0.2476 vs 0.2510). At borough level it wins in two of three, and at city level it loses in two of three.

## Findings
- Summing zone forecasts is clearly worse than a model trained on the aggregate: 0.0659 against 0.0593 at city level, because errors in sparse zones add up instead of cancelling.
- Shrunk MinT improves the zone level by 0.0028 WAPE (about 1% relative) and the borough level by 0.0007, and gives back 0.0007 at the city level. OLS makes zones worse.
- The effect is small and I have three folds and no interval, so I do not claim reconciliation improves accuracy. What it buys is coherence: parent and child forecasts agree exactly, at no measurable cost in accuracy.
- Shrinkage intensity fell from 0.36 to 0.28 as the residual history grew from one fold to three. With 270 nodes and 672 rows of residuals, an unshrunk covariance would not have been usable.

## Limits
Boroughs EWR, N/A and Unknown are single zones, so their borough node duplicates the zone node. Reconciliation used residuals at a one-hour horizon only.

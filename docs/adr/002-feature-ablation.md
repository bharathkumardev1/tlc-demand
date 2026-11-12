# 002. Which feature groups the zone model needs

Status: accepted

## Context
The model has 18 inputs in five groups: lags (7: lag_0 to lag_4, lag_8, lag_16), rolling means (3), same-slot values from one day and one week earlier (2), calendar (5) and zone id (1). I wanted to know which groups pay for themselves before adding more, and whether any can be dropped to make serving cheaper.

`scripts/run_ablation.py` retrains the model with one group removed, on the same three June folds, rows and seasonal-naive baseline as the main backtest. Same LightGBM settings (Poisson, 300 rounds, 31 leaves, one thread), seed 0. Three seeds of the full model give a noise floor. Output is in `results/ablation.json`.

## Results (WAPE pooled over three folds, lower is better; seasonal-naive is 0.3104)
| Run | features | WAPE | change vs full | relative |
|---|---|---|---|---|
| full, seed 0 | 18 | 0.24144 | | |
| full, seed 1 | 18 | 0.24173 | +0.00029 | +0.12% |
| full, seed 2 | 18 | 0.24177 | +0.00033 | +0.14% |
| minus lags | 11 | 0.24215 | +0.00071 | +0.29% |
| minus rolling means | 15 | 0.24327 | +0.00183 | +0.76% |
| minus same-slot | 16 | 0.24734 | +0.00590 | +2.44% |
| minus zone id | 17 | 0.24604 | +0.00460 | +1.91% |
| minus calendar | 13 | 0.25075 | +0.00931 | +3.86% |

Every ablated run is worse than the full model in all three folds.

## Findings
- The noise floor from seeds alone is about 0.0003 WAPE. Removing the lags (+0.0007) is only about twice that, so I would not say the lags matter. Seven lag columns carry almost nothing that the rolling means and same-slot values do not already carry.
- Calendar, same-slot values and zone id each cost between 0.0046 and 0.0093 when removed, 15 to 30 times the noise floor. Calendar is the largest, which fits a demand series dominated by time of day and day of week.
- Groups overlap, so each number understates what the group would be worth with nothing else standing in for it. The numbers show what is redundant given the others and say nothing about what each group is worth alone.
- Removing the seven lags would cut the feature count from 18 to 11 for about 0.3% relative WAPE. I have not measured the serving cost of the lag features, so I am keeping them; they are the first thing to try removing if latency matters.

## An earlier, wrong version of this experiment
My first version dropped columns with LightGBM's `ignore_column` parameter. That parameter only applies when data is loaded from a text file; with an in-memory array it does nothing. Four of five groups came back with predictions identical to the full model to five decimals, which is how I noticed (dropping seven lag columns cannot leave predictions unchanged). The script now slices the columns and asserts that the trained model's feature names equal the requested ones. The invalid run's numbers are not used anywhere.

## Limits
Three folds, one seed per ablated run, no confidence intervals. The 0.0003 floor comes from three seeds of the full model, and the 0.9 feature fraction is the only seed-dependent step in training (no bagging, deterministic mode), so it is a floor and probably not the full run-to-run spread. The zone-id result treats zone as a categorical; a model with zone-level statistics as numeric features might not need it.

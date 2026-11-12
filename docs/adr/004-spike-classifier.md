# 004. Predicting demand spikes, and whether a classifier beats thresholding the forecast

Status: accepted, modest result

## Context
Dispatch cares less about the exact count than about "this zone is about to be unusually busy". I defined a spike as a 15-minute zone count that is above the 0.9 quantile of that zone's training counts for the same weekday and hour, and at least 5 trips (so a zone that normally sees 0 or 1 does not produce spikes out of noise). Quantiles come from training rows only for each fold. The horizon is one hour, the folds and the gap between training and test are the same as in the main backtest (weeks starting 06-10, 06-17, 06-24). Code: `src/tlc_demand/spikes.py` (5 tests), `scripts/run_spikes.py`, output `results/spikes.json`.

Because of the 5-trip floor and the strict inequality, the rate of spikes is far below the 10% a pure quantile would give: about 1.7% in training and 1.6% to 2.7% in the test weeks.

Metric: average precision (area under the precision-recall curve), since accuracy and ROC-AUC are dominated by the 98% non-spike rows. Intervals are a bootstrap over test days (resampling days instead of rows, because rows within a day are strongly correlated).

I compared four scorers. The classifier is a LightGBM binary model on the forecast features plus the slot threshold and the lag and rolling mean divided by it. The other three need no extra model: the Poisson regression forecast divided by the slot threshold, last week's value for the slot divided by the threshold, and the latest count divided by it.

## Results (pooled over three folds, 528,192 rows, 10,808 spikes, base rate 0.0205)
| Scorer | AP | 95% interval | lift over base rate |
|---|---|---|---|
| classifier | 0.239 | 0.210 to 0.266 | 11.7x |
| regression forecast / threshold | 0.213 | 0.185 to 0.238 | 10.4x |
| latest count / threshold | 0.120 | 0.103 to 0.137 | 5.9x |
| same slot last week / threshold | 0.096 | 0.081 to 0.111 | 4.7x |

Per fold, the classifier is ahead of the regression ratio in all three (0.273 vs 0.243, 0.209 vs 0.188, 0.219 vs 0.199).

## Findings
- Both model-based scorers are about twice as good as the naive ones, so forecasting the spike is doing real work.
- The classifier edges out the regression forecast: ahead in every fold, but the two intervals overlap and I have not run a paired test, so I would call the gap "small and consistent" and leave it unestablished. If the service already produces the forecast, thresholding it recovers about 89% of the classifier's AP (0.213 / 0.239) without a second model.
- An AP of 0.24 means precision is low across most of the recall range; I did not compute precision at a specific alert threshold. The output works as a ranking for prioritizing zones and is too imprecise to page anyone.
- The first fold has a much higher test spike rate (2.7%) than training (1.8%), and AP is higher there too, which is partly the base rate. Compare lift across folds, and treat raw AP with care.

## Not done
No calibration of the classifier probabilities, no operating-point choice, no paired bootstrap between classifier and regression ratio, one seed, and one spike definition (0.9 quantile, 5-trip floor) with no sensitivity check.

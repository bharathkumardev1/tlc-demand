# Case study: forecasting taxi demand, and finding out where the model loses

## The problem

Given the last few hours of pickups in each of NYC's 262 taxi zones, forecast the number of pickups in the 15-minute slot one hour ahead. The hard part is not fitting a model. It is knowing, with leakage ruled out and a baseline that is hard to beat, where the model is better and where it is not.

## What I built

A trip-record validator and a dense zone-by-slot panel with explicit zeros. Features are computed relative to each forecast origin: seven lags, three rolling means, the same slot a day and a week earlier, and calendar fields. The model is a Poisson LightGBM, evaluated with a rolling-origin backtest (three June 2023 weeks, each trained only on earlier data with a one-horizon gap) against a seasonal-naive baseline. It is exported to ONNX and served by a FastAPI service behind a Java 17 Spring Boot gateway that times out at 150 ms, trips a circuit breaker on repeated failures, and falls back to the seasonal-naive value. The service exports Prometheus metrics, including a drift monitor.

## What the numbers say

The model's WAPE averages 0.2415 against 0.3104 for seasonal-naive across the three test weeks, a 22% relative reduction, and it wins in every fold. The Poisson loss and the features make busy zones easy.

Quiet zones are the opposite. In the two lowest-volume quartiles of zones, which together carry 0.28% of trips, the model's WAPE is 2.49 and 1.96 against 1.97 and 1.87 for the baseline. A threshold chosen on a separate validation week sends those zones to the baseline, which moves pooled WAPE from 0.2412 to 0.2406. The gain is smaller than my measured seed noise, so I report the weakness and not the fix.

Feature ablation says the seven lag features contribute 0.0007 WAPE, about twice the difference between seeds. Calendar and same-slot features do the work. If I were shaving serving cost, the lags are the first thing I would drop.

## Where I had to correct myself

My first drift monitor compared raw feature counts to a reference period and false-alarmed constantly, because a window that happened to contain Midtown zones looks different from one that doesn't, whatever the demand is. I normalised each value by the zone's expected count for that weekday and hour and computed PSI within tiers of that expectation. The false alarms dropped but did not go away: 36 of 1,776 hourly checks on clean April to June data, mostly weekends, because the two calibration months were calmer than the three after them. It catches a sustained 1.3x shift in about a day, misses a one-hour clock shift by 0.002 PSI, and reacted to the March 2020 collapse about three days after trips fell below 80% of normal. I would put it on a dashboard and not on a pager.

My first ablation silently did nothing. I passed LightGBM an `ignore_column` setting, which has no effect on in-memory arrays, and every run matched the full model. I caught it because the numbers were identical, switched to slicing the columns, and added an assertion on the feature count. I also wrote an ADR that claimed the ONNX parity bounds were asserted in tests when they weren't, noticed in review, corrected the ADR, and then added the assertions.

## Reconciliation, and what it does not show

I trained separate models for zone, borough and city and reconciled them so children sum to parents. Shrunk MinT improved zone WAPE from 0.2412 to 0.2384 and borough from 0.0806 to 0.0799, and worsened the city from 0.0593 to 0.0600. With three folds and no interval, I describe this as coherence at no measurable cost. Summing zone forecasts is plainly worse than modelling the city directly (0.0659 against 0.0593), which is the more useful finding.

## What I would do next

Fix the sparse-zone problem properly, probably with a shrinkage model for low-count series instead of a routing rule; test more horizons and a holiday period; recalibrate the drift threshold on a rolling clean window and model weekends separately; and replace the startup snapshot with an ingestion path. The README's "What is still rough" section lists the remaining gaps, including that CI has not yet run on GitHub.

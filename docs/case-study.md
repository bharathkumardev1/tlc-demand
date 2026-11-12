# Case study: forecasting taxi demand, and finding out where the model loses

## The question

Given the last few hours of pickups in each of NYC's 262 taxi zones, forecast the number of pickups in the 15-minute slot one hour ahead. Fitting a model is the easy part. What I wanted to know, with leakage ruled out and against baselines I had not weakened, was where the model is better and where it is not.

## The pieces

A trip-record validator and a dense zone-by-slot panel with explicit zeros. Features are computed relative to each forecast origin: seven lags, three rolling means, the same slot a day and a week earlier, and calendar fields. The model is a Poisson LightGBM, evaluated with a rolling-origin backtest (three June 2023 weeks, each trained only on earlier data with a one-horizon gap). It is exported to ONNX and served by a FastAPI service behind a Java 17 Spring Boot gateway that times out at 150 ms and trips a circuit breaker on repeated failures. The service exports Prometheus metrics, including a drift monitor.

The gateway's fallback is a stored number per zone, written once from the history snapshot: the count for the same slot a week before the snapshot's next target bin. It is a seasonal-naive value for that one slot, returned unchanged until the file is regenerated, and the gateway does not compute a seasonal-naive forecast per request. I ran the gateway against the real service, killed the service, and recorded the result: 40 of 40 requests were answered from the fallback, and 20 of 20 came from the model again after the restart.

## Results

I scored the model against three simple baselines on the same rows: the same slot one week earlier, the latest observed count, and the same slot one day earlier. Across the three test weeks the model's WAPE averages 0.2415. The baselines score 0.3104, 0.3476 and 0.3716, and the weekly one is the best in every fold. So the 22% reduction I quote is against the strongest baseline of the three, and I did not try anything stronger, such as a seasonal model fitted per zone. The Poisson loss and the features make busy zones easy.

Quiet zones go the other way. In the two lowest-volume quartiles, which together carry 0.28% of trips, the model's WAPE is 2.49 and 1.96. The baselines score between 1.92 and 1.97 in the lowest quartile and between 1.85 and 1.88 in the next. A threshold chosen on a separate validation week sends those zones to seasonal-naive, which moves pooled WAPE from 0.2412 to 0.2406. That gain is smaller than the seed-to-seed spread I measured, so I report the weakness and leave the fix open.

Dropping feature groups one at a time shows the seven lag features contribute 0.0007 WAPE, about twice the difference between seeds. Calendar and same-slot features carry the model. If serving cost mattered, the lags would go first. All of these runs use one fixed set of hyperparameters with no tuning and no early stopping.

## Mistakes along the way

My first drift monitor compared raw feature counts to a reference period and false-alarmed constantly, because a window that happened to contain Midtown zones looks different from one that doesn't, whatever the demand is. I normalised each value by the zone's expected count for that weekday and hour and computed PSI within tiers of that expectation. The false alarms dropped but did not go away: 36 of 1,776 hourly checks on clean April to June data, mostly weekends, because the two calibration months were calmer than the three after them. The control run with nothing injected also alerts, at 42 hours. That puts the 22 h and 31 h detection delays for the injected shifts in the same range as a false alarm, so I cannot cleanly say how fast it detects. It misses a one-hour clock shift by 0.002 PSI, and it reacted to the March 2020 collapse about three days after trips fell below 80% of normal. It belongs on a dashboard.

My first ablation silently did nothing. I passed LightGBM an `ignore_column` setting, which has no effect on in-memory arrays, and every run matched the full model. The identical numbers gave it away; I switched to slicing the columns and added an assertion on the feature count. An ADR of mine also claimed the ONNX parity bounds were asserted in tests when they weren't. I found that in review, corrected the ADR, and then added the assertions.

## Reconciliation

I trained separate models for zone, borough and city and reconciled them so children sum to parents. Shrunk MinT improved zone WAPE from 0.2412 to 0.2384 and borough from 0.0806 to 0.0799, and worsened the city from 0.0593 to 0.0600. With three folds and no interval, I read this as coherence at no measurable cost. Summing zone forecasts is plainly worse than modelling the city directly (0.0659 against 0.0593), which is the more useful finding.

## What I would do next

Fix the sparse-zone problem with a shrinkage model for low-count series in place of a routing rule. Test more horizons and a holiday period. Recalibrate the drift threshold on a rolling clean window and model weekends separately. Replace the startup snapshot with an ingestion path, and compute the gateway fallback per request. The README's "What is still rough" section lists the other gaps, including that CI has not yet run on GitHub.

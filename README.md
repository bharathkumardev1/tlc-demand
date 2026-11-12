# tlc-demand

A forecasting service for NYC taxi pickups: LightGBM predictions per pickup zone, served through a FastAPI model service and a Java gateway that falls back to a stored value when the model is slow or down.

**Problem.** For each of the 262 TLC pickup zones, predict how many yellow-taxi pickups the zone will see in the 15-minute slot an hour from now, and serve that prediction with a plan for when the model is unavailable.

**Approach.** A Poisson LightGBM model on leakage-checked features (recent counts, rolling means, the same slot a day and a week earlier, calendar fields), scored with a rolling-origin backtest against three simple baselines, exported to ONNX, and served behind a Spring Boot gateway with a timeout and a circuit breaker.

**Headline.** Over three test weeks in June 2023 (one-hour horizon, all zones), the model has a WAPE of 0.2415. The strongest of the three baselines is the same slot one week earlier at 0.3104, so the model's error is 22% lower. Repeating the latest count scores 0.3476 and the same slot yesterday 0.3716. In the half of zones with the lowest volume the model loses to every baseline; those zones carry 0.28% of trips.

**Demo.** No hosted demo. After installing, two commands bring the service up locally (see "Run it") and a real session is shown below.

### Example session

Output copied from a local run (model trained on January to June 2023, history snapshot ending 2023-06-30 23:45):

```
$ curl -s localhost:8000/healthz
{"status":"ok","zones":258,"last_origin":"2023-06-30 23:45:00","version":"dev"}

$ curl -s -X POST localhost:8000/v1/forecast -H 'Content-Type: application/json' -d '{"zone": 161}'
{"zone":161,"origin":"2023-06-30 23:45:00","target":"2023-07-01 00:45:00","horizon_bins":4,"trips":18.494,"model_version":"dev"}

$ curl -s -X POST localhost:8000/v1/forecast -H 'Content-Type: application/json' -d '{"zone": 1}'
{"zone":1,"origin":"2023-06-30 23:45:00","target":"2023-07-01 00:45:00","horizon_bins":4,"trips":0.136,"model_version":"dev"}
```

The gateway behaves the same way in front of this service. `scripts/gateway_integration.py` starts both, kills the model service, and records what the gateway does (`results/gateway_integration.json`): with the model up, gateway answers equal direct model answers (largest difference 0.0). With the model killed, all 40 requests came back from the fallback, 10 as `model_error` before the breaker opened and 30 as `breaker_open`. After the model restarted and the breaker's 10 s open interval passed, 20 of 20 requests were served by the model again.

## Results

All numbers come from `results/*.json`, rendered into this file by `docs/readme_tables.py`. Training data is yellow-taxi trips from January to June 2023. The seasonal-naive column repeats the same slot one week earlier. The other two baselines repeat the latest observed 15-minute count (the origin bin, an hour before the target) and the same slot one day earlier; all four forecasts are scored on the same rows. Each test fold trains on everything strictly before its week, minus a gap of one horizon so no training target overlaps the test window.

<!-- table:backtest -->
| Test week starting | WAPE model | WAPE seasonal-naive (same slot last week) | WAPE last count | WAPE same slot yesterday | MAE model | MAE seasonal-naive |
|---|---|---|---|---|---|---|
| 2023-06-10 | 0.2336 | 0.2843 | 0.3466 | 0.3726 | 1.058 | 1.287 |
| 2023-06-17 | 0.2397 | 0.3182 | 0.3469 | 0.3600 | 1.016 | 1.348 |
| 2023-06-24 | 0.2510 | 0.3287 | 0.3494 | 0.3823 | 1.027 | 1.345 |
| mean of folds | 0.2415 | 0.3104 | 0.3476 | 0.3716 | 1.033 | 1.327 |
<!-- /table:backtest -->

WAPE is the sum of absolute errors over the sum of actual trips, so it is dominated by busy zones. Same slot last week is the best baseline in every fold, which is why the headline compares against it. The next table shows where that leaves the quiet zones.

<!-- table:volume_tiers -->
| Zone volume quartile | Share of trips | WAPE model | WAPE seasonal-naive | WAPE last count | WAPE same slot yesterday | Lowest WAPE |
|---|---|---|---|---|---|---|
| q1_lowest | 0.05% | 2.489 | 1.972 | 1.962 | 1.924 | a baseline |
| q2 | 0.23% | 1.960 | 1.871 | 1.885 | 1.846 | a baseline |
| q3 | 1.24% | 1.385 | 1.442 | 1.461 | 1.490 | model |
| q4_highest | 98.47% | 0.222 | 0.291 | 0.329 | 0.353 | model |
<!-- /table:volume_tiers -->

The two lowest-volume quartiles of zones, together 0.28% of trips, score worse than every baseline (same slot yesterday does best there). I tried routing those zones to seasonal-naive with a threshold chosen on a separate validation week (`scripts/sparse_zone_rule.py`):

<!-- table:sparse -->
| Pooled, three June folds | WAPE | MAE |
|---|---|---|
| Seasonal-naive | 0.3096 | 1.3265 |
| LightGBM everywhere | 0.2412 | 1.0333 |
| Routed: seasonal-naive below 0.316 trips/bin | 0.2406 | 1.0311 |
<!-- /table:sparse -->

That is a gain of 0.0006 WAPE, smaller than the seed-to-seed spread I measured (about 0.0003 per run, so I would not claim it generalises). The fix I would want is a different loss or a zone-level shrinkage model for sparse series, and I haven't built it.

### Which features matter

Dropping one feature group at a time and retraining, same folds and settings. The three full-model rows are the same model with different seeds, which gives the noise floor.

<!-- table:ablation -->
| Run (seed 0 unless noted) | Features | WAPE | Change vs full |
|---|---|---|---|
| full_seed0 | 18 | 0.24144 |  |
| full_seed1 | 18 | 0.24173 | +0.00029 |
| full_seed2 | 18 | 0.24177 | +0.00033 |
| minus_lags | 11 | 0.24215 | +0.00071 |
| minus_rolling_means | 15 | 0.24327 | +0.00183 |
| minus_same_slot | 16 | 0.24734 | +0.00590 |
| minus_calendar | 13 | 0.25075 | +0.00931 |
| minus_zone | 17 | 0.24604 | +0.00460 |
<!-- /table:ablation -->

Calendar features and the same-slot values from a day and a week earlier carry most of the signal; the seven lag features add 0.0007, which is about twice the seed-to-seed difference. The model could drop the lags and cost almost nothing. (ADR 002.)

### Zone, borough and city together

One model per level, then reconciled so that zone forecasts sum to the borough and city forecasts. Weights use only earlier folds' errors.

<!-- table:hierarchy -->
| Pooled WAPE, three June folds | Zone | Borough | City |
|---|---|---|---|
| Seasonal-naive | 0.30961 | 0.12848 | 0.10307 |
| Independent LightGBM per level | 0.24117 | 0.08059 | 0.05934 |
| Bottom-up | 0.24117 | 0.08694 | 0.06585 |
| OLS reconciliation | 0.24486 | 0.08354 | 0.05912 |
| WLS (variance) | 0.23942 | 0.08254 | 0.06225 |
| MinT, shrunk covariance | 0.23842 | 0.07994 | 0.06004 |
<!-- /table:hierarchy -->

Shrunk MinT lowers zone WAPE by 0.0028 and borough WAPE by 0.0007, and raises city WAPE by 0.0007. Three folds and no interval, so I read it as a coherence guarantee (children sum to parents) that costs nothing measurable, and I make no accuracy claim. Summing zone forecasts is clearly worse than a city model: 0.0659 against 0.0593. These pooled WAPEs differ slightly from the fold average above because pooling weights folds by volume. (ADR 003.)

### Spikes

A spike is a zone-slot count above that zone's 0.9 quantile for the same weekday and hour, and at least 5 trips.

<!-- table:spikes -->
| Average precision, base rate 0.0205 | AP | 95% interval (bootstrap over days) | Lift |
|---|---|---|---|
| LightGBM classifier | 0.239 | 0.210 to 0.266 | 11.7x |
| Forecast / slot threshold | 0.213 | 0.185 to 0.238 | 10.4x |
| Latest count / slot threshold | 0.120 | 0.103 to 0.137 | 5.9x |
| Same slot last week / threshold | 0.096 | 0.081 to 0.111 | 4.7x |
<!-- /table:spikes -->

The classifier is ahead of thresholding the regression forecast in all three folds, but the intervals overlap and I did not run a paired test. Thresholding the forecast recovers about 89% of the classifier's AP with no second model. With an AP of 0.24 the output is a ranking for prioritising zones. It is too imprecise to page anyone. (ADR 004.)

### Serving parity

The service runs the model as ONNX. The converter drops LightGBM's categorical splits, so zone becomes a numeric input, and inputs are float32.

<!-- table:parity -->
| ONNX against LightGBM, June held-out week | Value |
|---|---|
| WAPE, LightGBM / ONNX | 0.25483 / 0.25484 |
| Absolute difference, trips per zone-interval: median / p99 / p99.9 / max | 0.0 / 0.04292 / 0.45905 / 2.31086 |
| Rows with absolute difference over 0.01 | 3540 of 173376 |
<!-- /table:parity -->

Aggregate accuracy is unchanged. A few rows differ by more than 2 trips because a float32-rounded feature lands on the other side of a split threshold; feeding the same rounded inputs to LightGBM directly shifts predictions by up to 2.09. The tests therefore assert aggregate and 99th-percentile bounds and do not require elementwise equality. (ADR 001.)

### Drift monitor

The service exports a population stability index (PSI) for four features as Prometheus gauges. Features are divided by the zone's expected count for that weekday and hour, and PSI is computed inside four tiers of that expectation, because raw counts false-alarmed on which zones a window happened to contain. The alert threshold is the largest PSI in a clean calibration period, since the textbook 0.25 would fire constantly on small windows.

<!-- table:drift -->
| 24 h window, injection from 2023-06-15 | Hours to first alert | Share of windows alerting | Max PSI |
|---|---|---|---|
| none (control) | 42 h | 0.5% | 0.13641 |
| 2x demand in 20 busy zones | 31 h | 16.7% | 0.16674 |
| 1.3x demand everywhere | 22 h | 55.2% | 0.26272 |
| clock shifted +1 h | never | 0.0% | 0.13327 |

Threshold 0.135. False alarms on clean April to mid-June data: 36 of 1,776 hourly evaluations.
<!-- /table:drift -->

It catches a sustained 1.3x level shift in about a day and misses a one-hour clock shift by a hair (0.133 against 0.135). The control run, with nothing injected, also alerts, at 42 h with a max PSI of 0.13641 against the 0.13475 threshold, so the 22 h and 31 h detection delays cannot be cleanly separated from a false alarm of that size. On clean data the false-alarm rate is too high to page anyone, and I would send it to a dashboard. On the real 2020 collapse it alerted on 14 March, about three days after daily trips fell below 80% of normal, which is slower than someone watching total trips. Each scenario ran once, so the delays carry no interval. (ADR 005.)

## Architecture

```
TLC parquet -> validate schema -> dense zone x 15-min panel (explicit zeros)
            -> origin-relative features (lags, rolling means, same slot, calendar)
            -> rolling-origin backtest / LightGBM (Poisson) -> ONNX

client -> Spring Boot gateway (Java 17, resilience4j)
            |  150 ms timeout, circuit breaker (20 calls, 50% failures, 10 s open)
            |-> FastAPI service: features from history snapshot -> ONNX -> /v1/forecast, /metrics
            '-> on timeout or open breaker: stored per-zone value from fallback.csv
```

The panel holds explicit zeros because a zone with no pickups in a quarter hour is a real observation, and lag features over a sparse panel would silently read the wrong row. Features are computed relative to each forecast origin, and `tests/test_pipeline.py` checks that changing any value after the origin never changes the features, and that the online feature path matches the batch path. The history store is a snapshot loaded at startup, and its age is exported so a stale snapshot shows up in metrics.

The gateway's fallback is one number per zone, a static snapshot: `make_history.py` writes `artifacts/fallback.csv` once, holding the count for the same slot a week before the snapshot's next target bin. It is a seasonal-naive value for that one slot and the gateway returns it unchanged until the file is regenerated. The gateway computes nothing per request on this path, so during a long outage the value drifts away from the right time of day.

## Run it

Python 3.11 or later, plus Java 17 and Maven for the gateway.

```
pip install -e '.[dev]'
ruff check . && pytest -q                   # lint, then 32 tests on synthetic data
mvn -B -f gateway/pom.xml test              # 6 gateway tests (last run: results/gateway_tests.txt)
```

Data is not in the repo (about 320 MB), and the service starts with two commands:

```
scripts/fetch_tlc.sh data/raw               # six 2023 yellow-taxi files and the zone lookup
scripts/serve_local.sh data/raw             # trains, writes the history snapshot, starts the service on :8000
```

I ran the fetch into an empty directory and the files are byte-identical to the ones the results came from. Training and export take a few minutes on 2 vCPU.

The numbers above come from these scripts, each with its exact command in its docstring:

```
python scripts/run_backtest.py data/raw 2023 1 6 --horizon 4        # backtest and baselines, about 5 minutes
python scripts/make_features_cache.py data/raw 2023 1 6 artifacts/features_h4.parquet   # input to the next four
python scripts/run_ablation.py          # ADR 002, about 4 GB of memory
python scripts/run_spikes.py            # ADR 004
python scripts/run_drift_eval.py 2023 data/raw     # ADR 005; "2020 data/raw2020" for the 2020 collapse
python scripts/check_onnx_parity.py artifacts/ data/raw 2023 6      # ADR 001
python scripts/run_hierarchy.py data/raw 2023 1 6 data/raw/taxi_zone_lookup.csv   # ADR 003
python scripts/sparse_zone_rule.py data/raw 2023 1 6
python scripts/gateway_integration.py   # gateway against the real service, then with the service killed
```

For the 2020 check, run `scripts/fetch_tlc.sh data/raw2020 2020 1 6` first. To check this file against the results: `python docs/readme_tables.py --check`.

## What is still rough

- **Sparse zones are worse than every baseline.** See above. The routing rule's gain is inside the noise.
- **Hyperparameters are fixed.** Learning rate 0.05, 31 leaves, 50 rows per leaf, feature fraction 0.9, 300 rounds, Poisson loss; no tuning and no early stopping. The comparisons here (model against baselines, one feature group dropped at a time) all use the same settings, so they do not depend on tuning, but I did not measure how much a tuned model would gain, and a tuned model could shift which feature groups matter.
- **The gateway fallback is a stored number.** One value per zone from `fallback.csv`, as described under Architecture. The gateway was run against the real service once (`results/gateway_integration.json`), from the test classpath because the Spring Boot repackage plugin could not be downloaded in my container, so I have not run the packaged jar.
- **One horizon, one month of test data.** Everything is evaluated at one hour ahead on three weeks of June 2023. I haven't tested other horizons, other seasons, or holidays.
- **No uncertainty on most comparisons.** Folds are three weeks, the seed noise is measured only for the full model, and the hierarchy and spike comparisons have no paired test.
- **Drift monitor false-alarms.** 36 of 1,776 hourly checks on clean data, clustered on weekends. The threshold is only as good as its two-month calibration.
- **The service forecasts from a snapshot.** There is no stream, so "live" forecasts are as old as the snapshot. A real deployment needs an ingestion path I haven't built.
- **Not in this repo.** No container image, orchestration, warehouse, experiment tracker or model registry. The model artifact is a file produced by `export_model.py`.
- **CI has not run on GitHub.** The workflow's two jobs (lint and Python tests, gateway tests) pass locally.
- **Service load numbers are synthetic and from a 2 vCPU container, gateway not measured.** Forecast service, random zones, load generator on the same two CPUs (`results/loadtest.json`, three runs of 3,000 requests each): one request at a time, p50 2.4 ms, p95 3.3 to 3.4 ms, p99 5.7 to 6.1 ms, about 381 to 390 requests per second; four concurrent clients, p50 8.5 to 9.3 ms, p95 14.4 to 15.4 ms, p99 17.7 to 20.1 ms, about 404 to 442 requests per second. Throughput barely rises from one client to four because the two CPUs are shared with the load generator, so treat it as a floor. The Spring Boot gateway has no timing.
- **No hosted demo.**

## Data and licence

NYC TLC trip records are public. Code is MIT licensed (`LICENSE`).

# 005. Input drift monitoring with PSI, and what it did and did not catch

Status: accepted, with known weaknesses

## Context
The service should say when its inputs stop looking like the data it was trained on. I monitor four features (`lag_0`, `roll_mean_16`, `roll_mean_96`, `same_slot_1w`) with the population stability index against a reference period, exposed as Prometheus gauges from `service.py`.

Raw counts cannot be compared across requests: a window that happens to hit Midtown zones, or one taken at 3 am, differs from the reference for reasons that are not drift. The first version of the monitor false-alarmed on exactly that. Each value is now divided by the zone's mean for the same weekday and hour in the reference period, and PSI is computed inside four tiers of that expected count (noisy small-count ratios are kept apart from large-count ones) and averaged by tier share. Code is in `src/tlc_demand/drift.py`; the evaluation is `scripts/run_drift_eval.py`.

## How the alert threshold is set
The threshold is the maximum PSI seen over a clean calibration period, where the textbook value would be 0.25. For a window of 24 hours or 6 hours, PSI has a size-dependent floor of roughly (bins - 1) / n, so a fixed rule of thumb would alert constantly on small windows. Consequence: the threshold is only as good as the calibration period. The 2023 calibration months may have been calmer than the three that followed, which would explain the false alarms below. I did not check that.

## Test 1: injected drift on 2023 data
Reference January 8 to 31, calibration February 1 to March 31, clean evaluation April 1 to June 14, injection from 2023-06-15 00:00. Windows are rolling and evaluated hourly.

Thresholds: 0.135 for 24-hour windows, 0.796 for 6-hour windows.

False alarms on clean April to June data: 36 of 1,776 evaluations for 24-hour windows (8 distinct days: 4 weekend days, 3 Mondays and 1 Friday; I did not investigate causes), and 7 of 1,794 for 6-hour windows. The threshold from a two-month calibration does not hold up for three more months.

| Injection from 06-15 | 24 h window, hours to first alert | 24 h share of windows alerting | 6 h window, hours to first alert |
|---|---|---|---|
| none (control) | 42 (false alarm) | 0.5% | none |
| 2x demand in the busiest zones | 31 | 16.7% | 293 |
| 1.3x demand everywhere | 22 | 55.2% | 29 |
| clock shifted by +1 h | never (max PSI 0.133, threshold 0.135) | 0% | 108 |

The 6-hour monitor already had 2 alerts in the week before injection, so its delays are not clean detections.

## Test 2: the 2020 shift
Same procedure on January to June 2020, reference January 8 to 31, calibration February 1 to 29 (thresholds 0.206 and 1.012). Total daily trips first dropped below 80% of the February mean for the same weekday on 2020-03-11. The 24-hour monitor first alerted on 2020-03-14 00:00 and the 6-hour monitor on 2020-03-14 12:00, about three days after the trip count fell below 80%. On 03-13, with trips at 54% of normal, daily max PSI was 0.199, just under the 0.206 threshold. Once it fired it stayed on: 90% of all 24-hour windows from March 1 to the end of June were above threshold, and daily max PSI rose to 3 to 4 by April. So it detects a collapse, but it is slower than a person watching total trips.

## What I'd say about it
- It catches large, sustained level shifts. 1.3x everywhere is found in about a day.
- It misses a one-hour clock shift at 24 hours (0.133 against 0.135; a different calibration period could flip it), because the monitored values are slot-relative and a one-hour shift only changes them at the edges of rush periods.
- The false alarm rate on clean data (36 of 1,776 hourly evaluations, mostly on weekends and the days next to them) is high enough that I would route it to a dashboard and a ticket and keep it off the on-call rotation.
- 6-hour windows are not usable with this construction.

## Not done
Threshold recalibration on a rolling clean window; a reference that models weekends separately, since the false alarms cluster there; more than one random injection per scenario, so the delays above are single runs with no interval.

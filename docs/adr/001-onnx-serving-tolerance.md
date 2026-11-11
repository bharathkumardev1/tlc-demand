# 001. ONNX serving: agreement with the trained model and the test tolerance

Status: accepted

## Context
The gateway calls a FastAPI service that runs the LightGBM model exported to ONNX. The converter does not carry categorical splits, so zone is a numeric feature in the exported model. Features are also float32 at the ONNX boundary.

## Measured (results/onnx_parity.json)
- WAPE on the evaluation set: 0.25483 with ONNX against 0.25484 with LightGBM.
- Absolute difference in predicted trips per zone-interval: median about 0, 99th percentile 0.043, 99.9th 0.459, max 2.311.
- Relative difference: 99th 0.0096, 99.9th 0.0523, max 0.1748.
- Rounding inputs to float32 and feeding LightGBM directly shifts predictions by up to 2.086, so most of the tail comes from thresholds close to a feature value, not from the ONNX runtime.

## Decision
Parity should be asserted in aggregate (WAPE within 0.001) and by a per-row bound at the 99th percentile (0.1 trips), not by elementwise equality. A few rows differ by more than 2 trips and that is expected.

tests/test_service.py asserts both bounds on a small synthetic model (99th percentile under 0.1 trips, summed absolute difference under 0.1% of summed predictions). scripts/check_onnx_parity.py measures the same quantities on the real June 2023 features and writes results/onnx_parity.json. The test bound on summed difference is on predictions, not on WAPE against actuals; the WAPE agreement above is the real-data check.

## Consequences
Anyone changing the feature set must rerun scripts/check_onnx_parity.py and update these numbers.

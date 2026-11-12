#!/usr/bin/env bash
# Train the serving model on the downloaded months, write the history snapshot, start the forecast service.
#   scripts/serve_local.sh data/raw [artifacts] [port]
# Then, in another terminal:
#   curl -s -X POST localhost:8000/v1/forecast -H 'Content-Type: application/json' -d '{"zone": 161}'
set -euo pipefail
raw=${1:?usage: serve_local.sh RAW_DIR [ARTIFACT_DIR] [PORT]}
art=${2:-artifacts}; port=${3:-8000}
python scripts/export_model.py "$raw" 2023 1 6 "$art"
python scripts/make_history.py "$raw" 2023 6 "$art/history.npz"
MODEL_PATH="$art/model.onnx" HISTORY_PATH="$art/history.npz" \
  exec python -m uvicorn --factory tlc_demand.service:create_app --port "$port"

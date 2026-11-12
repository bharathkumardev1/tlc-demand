#!/usr/bin/env bash
# Download the NYC TLC yellow-taxi files and the zone lookup used by the README quickstart.
#   scripts/fetch_tlc.sh data/raw            # 2023-01 to 2023-06
#   scripts/fetch_tlc.sh data/raw2020 2020 1 2   # year, first month, last month
set -euo pipefail
out=${1:?usage: fetch_tlc.sh OUT_DIR [YEAR FIRST_MONTH LAST_MONTH]}
year=${2:-2023}; first=${3:-1}; last=${4:-6}
base=https://d37ci6vzurychx.cloudfront.net
mkdir -p "$out"
for m in $(seq "$first" "$last"); do
  f=$(printf 'yellow_tripdata_%s-%02d.parquet' "$year" "$m")
  [ -s "$out/$f" ] || curl -fL --retry 3 -o "$out/$f" "$base/trip-data/$f"
done
[ -s "$out/taxi_zone_lookup.csv" ] || curl -fL --retry 3 -o "$out/taxi_zone_lookup.csv" "$base/misc/taxi_zone_lookup.csv"
ls -l "$out"

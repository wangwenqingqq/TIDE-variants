#!/usr/bin/env bash
# CPU-only guard/wrapper for the isolated frozen E1 SIFT HNSW semantic replay.
set -euo pipefail
ROOT=/workspace/experiments/tide_safe_c1_20260727
WORK="$ROOT/hnsw_fair_dynamic_knn"
BUNDLE="$ROOT/runs/e1gi_b_quantized_gts_bundles_v1_20260727/sift128_slice__sift128_slice__seed_20260727"
PY=/workspace/legacy_workspace/GTS/bench_env/bin/python
[[ "$(hostname)" == "CONFIGURE_ARCHIVE_HOST" ]] || { echo "Refusing non-8p host $(hostname)" >&2; exit 64; }
[[ -x "$PY" && -f "$WORK/src/replay_hnsw_dynamic_knn.py" && -d "$BUNDLE" ]] || { echo "missing runner/python/frozen bundle" >&2; exit 66; }
# Deliberately CPU-only. This invocation does not call CUDA, nvidia-smi, or any GPU program.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%Z)"
OUT="${1:-$WORK/runs/hnsw_sift128_e1trace_cpu_${STAMP}}"
case "$OUT" in "$WORK"/runs/*) ;; *) echo "Output must stay under $WORK/runs" >&2; exit 64;; esac
[[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 73; }
printf '%q ' "$PY" "$WORK/src/replay_hnsw_dynamic_knn.py" --bundle "$BUNDLE" --out "$OUT" --m 16 --ef-construction 200 --ef 4096 --seed 20260727 --wrapper-path "$0" > /tmp/hnsw_fair_cmd.$$
"$PY" "$WORK/src/replay_hnsw_dynamic_knn.py" --bundle "$BUNDLE" --out "$OUT" --m 16 --ef-construction 200 --ef 4096 --seed 20260727 --wrapper-path "$0"
mv /tmp/hnsw_fair_cmd.$$ "$OUT/execution_command.txt"
printf 'CPU_ONLY=1\nCUDA_VISIBLE_DEVICES=%q\nOMP_NUM_THREADS=%q\n' "$CUDA_VISIBLE_DEVICES" "$OMP_NUM_THREADS" > "$OUT/execution_environment.txt"
( cd "$OUT" && find . -maxdepth 1 -type f ! -name output_hashes.sha256 -printf '%P\0' | sort -z | xargs -0 sha256sum > output_hashes.sha256 )
echo "PASS wrapper output=$OUT"

#!/usr/bin/env bash
# CPU-only compilation: nvcc compiles but does not initialize or execute a GPU.
set -Eeuo pipefail
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_interval_bound_safe
NVCC=/usr/local/cuda-13.1/bin/nvcc
CUDA_INCLUDE=/usr/local/cuda-13.1/include
SRC="$ROOT/src/gts_safe_c2_interval_sift1m.cu"
OUT="$ROOT/bin/GTS_safe_c2_interval_sift1m"
LOG="$ROOT/logs/build_safe_c2_interval_$(date -u +%Y%m%dT%H%M%SZ).log"
[[ -x "$NVCC" && -f "$CUDA_INCLUDE/cuda_runtime.h" ]] || { echo 'pinned CUDA compiler/include unavailable' >&2; exit 70; }
{
  echo "started_utc=$(date -u -Is)"
  echo 'mode=CPU-only compile; no CUDA binary execution'
  "$NVCC" --version | tail -n 4
  "$NVCC" -std=c++17 -O3 --generate-code=arch=compute_120,code=[compute_120,sm_120] -I"$CUDA_INCLUDE" -I"$ROOT/include" "$SRC" -o "$OUT"
  echo "finished_utc=$(date -u -Is)"
  sha256sum "$SRC" "$ROOT/include/search_v2.cuh" "$ROOT/include/tree.cuh" "$ROOT/include/residual_pruning.cuh" "$OUT"
} 2>&1 | tee "$LOG"
printf '%s\n' "$LOG"

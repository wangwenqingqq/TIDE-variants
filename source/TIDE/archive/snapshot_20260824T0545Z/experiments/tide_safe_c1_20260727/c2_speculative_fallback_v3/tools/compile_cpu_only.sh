#!/usr/bin/env bash
# CPU/NVCC compilation only.  This script never launches the GTS binary,
# initializes a GPU, calls nvidia-smi, or writes outside the isolated v3 root.
set -Eeuo pipefail
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3
NVCC=/usr/local/cuda-13.1/bin/nvcc
CUDA_INCLUDE=/usr/local/cuda-13.1/include
SRC="$ROOT/src/gts_speculative_fallback_v3_sift1m.cu"
HEADER="$ROOT/include/search_v3.cuh"
OUT="$ROOT/bin/GTS_safe_c2_speculative_fallback_v3_sift1m"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/compile_cpu_only_v3_${STAMP}.log"
TMPDIR="$ROOT/.tmp_compile"
export TMPDIR
mkdir -p "$TMPDIR"
[[ -x "$NVCC" && -f "$CUDA_INCLUDE/cuda_runtime.h" && -f "$SRC" && -f "$HEADER" ]] || {
  echo 'pinned CUDA compiler/include/source unavailable' >&2
  exit 70
}
TMP_OUT="$OUT.next.$$"
trap 'rm -f "$TMP_OUT"' EXIT
{
  echo "started_utc=$(date -u -Is)"
  echo 'mode=CPU/NVCC compile only; no CUDA binary execution; no GPU-management command'
  "$NVCC" --version | tail -n 4
  "$NVCC" -std=c++17 -O3 --generate-code=arch=compute_120,code=[compute_120,sm_120] \
    -I"$CUDA_INCLUDE" -I"$ROOT/include" "$SRC" -o "$TMP_OUT"
  mv "$TMP_OUT" "$OUT"
  sha256sum "$SRC" "$HEADER" "$ROOT/include/tree.cuh" "$ROOT/include/residual_pruning.cuh" "$OUT"
  echo "finished_utc=$(date -u -Is)"
} 2>&1 | tee "$LOG"
printf '%s\n' "$LOG"

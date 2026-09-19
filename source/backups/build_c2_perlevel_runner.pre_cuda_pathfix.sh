#!/usr/bin/env bash
# CPU-only compile; it does not initialize or execute a CUDA device.
set -Eeuo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727/c2_gamma_heldout_scale"
SRC="$ROOT/src/gts_c2_perlevel_sift1m.cu"
OUT="$ROOT/bin/GTS_c2_perlevel_sift1m"
LOG="$ROOT/logs/build_c2_perlevel_sift1m_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$ROOT/bin" "$ROOT/logs"
{
  echo "started_utc=$(date -u -Is)"
  nvcc --version | tail -n 4
  nvcc -std=c++17 -O3 -arch=sm_120 -I"$ROOT/include" "$SRC" -o "$OUT"
  echo "finished_utc=$(date -u -Is)"
  sha256sum "$SRC" "$OUT"
} 2>&1 | tee "$LOG"
printf '%s\n' "$LOG"

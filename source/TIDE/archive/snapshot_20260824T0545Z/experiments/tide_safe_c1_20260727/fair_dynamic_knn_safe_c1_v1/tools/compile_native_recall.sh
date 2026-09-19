#!/usr/bin/env bash
# Compile-only helper for the native-GTS candidate-recall probe.  It does not launch CUDA code.
set -euo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1"
ARCH="$1"
if [[ ! "$ARCH" =~ ^sm_[0-9]+$ ]]; then
  echo "usage: $0 sm_<cc>" >&2
  exit 2
fi
NVCC="/usr/local/cuda-13.1/bin/nvcc"
CXX="/usr/bin/g++"
WRAPPER="$ROOT/runner/fair_safe_c1_native_recall_e1_runner.cu"
OUT="$ROOT/build/fair_safe_c1_native_recall_e1_runner.$ARCH.o"
[[ -x "$NVCC" && -x "$CXX" && -f "$WRAPPER" ]] || { echo "missing compiler/source" >&2; exit 2; }
exec "$NVCC" -std=c++17 -ccbin "$CXX" -O2 -c "-arch=$ARCH" \
  -I"$ROOT/src" -I"$ROOT/reference/include" "$WRAPPER" -o "$OUT"

#!/usr/bin/env bash
# Compile-only helper: it links no executable and launches no GPU code.
set -euo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1"
ARCH="$1"
if [[ ! "$ARCH" =~ ^sm_[0-9]+$ ]]; then
  echo "usage: $0 sm_<cc>  (example: $0 sm_120)" >&2
  exit 2
fi
NVCC="/usr/local/cuda-13.1/bin/nvcc"
CXX="/usr/bin/g++"
WRAPPER="$ROOT/runner/fair_safe_c1_delta_knn_e1_runner.cu"
OUT="$ROOT/build/fair_safe_c1_delta_knn_e1_runner.$ARCH.o"
[[ -x "$NVCC" && -x "$CXX" && -f "$WRAPPER" ]] || {
  echo "missing compiler or source" >&2
  exit 2
}
# Exactly one translation unit: wrapper includes the byte-identical v24 core.
exec "$NVCC" -std=c++17 -ccbin "$CXX" -O2 -c "-arch=$ARCH" \
  -I"$ROOT/src" -I"$ROOT/reference/include" "$WRAPPER" -o "$OUT"


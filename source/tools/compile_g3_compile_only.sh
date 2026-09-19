#!/usr/bin/env bash
# Final P0 compile-only closure check for isolated Safe-C1 G3.  This script
# never links an executable and never launches CUDA code, nvidia-smi, Nsight,
# or a benchmark.
set -euo pipefail
ROOT=${1:?usage: compile_g3_compile_only.sh /absolute/safe_c1_native_matrix_g3_v1}
NVCC=${NVCC:-/usr/local/cuda-13.1/bin/nvcc}
OUT="$ROOT/preflight/compile_p0"
mkdir -p "$OUT"
"$NVCC" --version >"$OUT/nvcc_version.txt"
"$NVCC" -std=c++17 -arch=compute_80 -code=compute_80 -c \
  -I"$ROOT/src" -I"$ROOT/reference/include" \
  "$ROOT/src/g3_safe_c1_native_matrix.cu" \
  -o "$OUT/g3_safe_c1_native_matrix.compute_80.o" \
  >"$OUT/nvcc_compile.log" 2>&1
printf 'COMPILE_ONLY_OK\n'

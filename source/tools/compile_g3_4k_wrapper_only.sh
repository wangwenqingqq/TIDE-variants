#!/usr/bin/env bash
# Compile-only closure; no link, CUDA runtime tool, GPU telemetry, or binary launch.
set -euo pipefail
umask 077
[[ "$#" -eq 1 ]] || { echo "usage: compile_g3_4k_wrapper_only.sh /absolute/v3-root" >&2; exit 69; }
ROOT="$1"
EXPECTED="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v3_controlled_4k_pilot"
[[ "$ROOT" == "$EXPECTED" && -d "$ROOT" && ! -L "$ROOT" ]] || { echo "unsafe root" >&2; exit 69; }
NVCC="/usr/local/cuda-13.1/bin/nvcc"
[[ -x "$NVCC" && -f "$ROOT/runner/g3_4k_pilot_wrapper.cu" && ! -L "$ROOT/runner/g3_4k_pilot_wrapper.cu" ]] || { echo "missing compiler/wrapper" >&2; exit 69; }
OUT="$ROOT/preflight/compile_wrapper_only"
mkdir -p "$OUT"
[[ ! -e "$OUT/g3_4k_pilot_wrapper.compute_80.o" ]] || { echo "prior object exists" >&2; exit 69; }
"$NVCC" --version >"$OUT/nvcc_version.txt"
"$NVCC" -std=c++17 -arch=compute_80 -code=compute_80 -c -I"$ROOT/src" -I"$ROOT/reference/include" "$ROOT/runner/g3_4k_pilot_wrapper.cu" -o "$OUT/g3_4k_pilot_wrapper.compute_80.o" >"$OUT/nvcc_compile.log" 2>&1
printf "COMPILE_ONLY_OK\n"

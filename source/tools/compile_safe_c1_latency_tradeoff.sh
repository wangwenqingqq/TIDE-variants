#!/usr/bin/env bash
# Compile/link only; this helper never launches CUDA code.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1"
ARCH="${1:?usage: $0 sm_<cc> [compile|all]}"
MODE="${2:-all}"
[[ "$ARCH" =~ ^sm_[0-9]+$ ]] || { echo "invalid arch: $ARCH" >&2; exit 2; }
[[ "$MODE" == "compile" || "$MODE" == "all" ]] || { echo "mode must be compile or all" >&2; exit 2; }
NVCC="/usr/local/cuda-13.1/bin/nvcc"
CXX="/usr/bin/g++"
SRC="$ROOT/runner/fair_safe_c1_latency_tradeoff_e1_runner.cu"
OBJ="$ROOT/build/fair_safe_c1_latency_tradeoff_e1_runner.$ARCH.o"
BIN="$ROOT/bin/fair_safe_c1_latency_tradeoff_e1_runner.$ARCH"
[[ -x "$NVCC" && -x "$CXX" && -f "$SRC" ]] || { echo "compiler/source missing" >&2; exit 2; }
TMP_OBJ="$OBJ.tmp.$$"
trap 'rm -f "$TMP_OBJ" "$BIN.tmp.$$"' EXIT
"$NVCC" -std=c++17 -ccbin "$CXX" -O2 -c "-arch=$ARCH" \
  -I"$ROOT/src" -I"$ROOT/reference/include" "$SRC" -o "$TMP_OBJ"
mv -f "$TMP_OBJ" "$OBJ"
if [[ "$MODE" == "all" ]]; then
  TMP_BIN="$BIN.tmp.$$"
  "$NVCC" "-arch=$ARCH" "$OBJ" -o "$TMP_BIN"
  mv -f "$TMP_BIN" "$BIN"
  chmod 700 "$BIN"
fi
printf 'compiled=%s\nobject=%s\nbinary=%s\n' "$SRC" "$OBJ" "$BIN"

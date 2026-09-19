#!/usr/bin/env bash
# Compile/link only; this helper never launches a CUDA workload.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1"
DIAG="$ROOT/diagnostics/fresh_alloc_telemetry_v1"
[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: $0 sm_<cc> [compile|all]" >&2; exit 2; }
ARCH="$1"
MODE=all
if [[ $# -eq 2 ]]; then MODE="$2"; fi
[[ "$ARCH" =~ ^sm_[0-9]+$ ]] || { echo "invalid arch" >&2; exit 2; }
[[ "$MODE" == "compile" || "$MODE" == "all" ]] || { echo "mode must be compile or all" >&2; exit 2; }
NVCC="/usr/local/cuda-13.1/bin/nvcc"
CXX="/usr/bin/g++"
SRC="$DIAG/runner/fair_safe_c1_fresh_alloc_telemetry_e1_runner.cu"
OBJ="$DIAG/build/fair_safe_c1_fresh_alloc_telemetry_e1_runner.$ARCH.o"
BIN="$DIAG/bin/fair_safe_c1_fresh_alloc_telemetry_e1_runner.$ARCH"
[[ -x "$NVCC" && -x "$CXX" && -f "$SRC" ]] || { echo "compiler/source missing" >&2; exit 2; }
TMP_OBJ="$OBJ.tmp.$$"
TMP_BIN="$BIN.tmp.$$"
cleanup() {
  /usr/bin/python3 - "$TMP_OBJ" "$TMP_BIN" <<'PY'
import os, sys
for raw in sys.argv[1:]:
    try:
        os.unlink(raw)
    except FileNotFoundError:
        pass
PY
}
trap cleanup EXIT
"$NVCC" -std=c++17 -ccbin "$CXX" -O2 -c "-arch=$ARCH" \
  -I"$DIAG/src" -I"$ROOT/src" -I"$ROOT/reference/include" "$SRC" -o "$TMP_OBJ"
mv "$TMP_OBJ" "$OBJ"
if [[ "$MODE" == "all" ]]; then
  "$NVCC" "-arch=$ARCH" "$OBJ" -o "$TMP_BIN"
  mv "$TMP_BIN" "$BIN"
  chmod 700 "$BIN"
fi
printf 'compiled=%s\nobject=%s\nbinary=%s\n' "$SRC" "$OBJ" "$BIN"

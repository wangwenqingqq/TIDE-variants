#!/usr/bin/env bash
# Static compile/link helper for the root-private silent-only diagnostic variant.
# This helper never executes the CUDA binary or launches a GPU workload.
set -euo pipefail
umask 077
ROOT="/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1"
DIAG="$ROOT/diagnostics/silent_only_v1"
if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 sm_<cc> [compile|all]" >&2
  exit 2
fi
ARCH="$1"
MODE="all"
if [[ "$#" -ge 2 ]]; then MODE="$2"; fi
[[ "$ARCH" =~ ^sm_[0-9]+$ ]] || { echo "invalid arch: $ARCH" >&2; exit 2; }
[[ "$MODE" == "compile" || "$MODE" == "all" ]] || { echo "mode must be compile or all" >&2; exit 2; }
NVCC="/usr/local/cuda-13.1/bin/nvcc"
CXX="/usr/bin/g++"
SRC="$DIAG/runner/fair_safe_c1_latency_silent_only_e1_runner.cu"
OBJ="$DIAG/build/fair_safe_c1_latency_silent_only_e1_runner.$ARCH.o"
BIN="$DIAG/bin/fair_safe_c1_latency_silent_only_e1_runner.$ARCH"
MANIFEST="$DIAG/provenance/silent_only_source_manifest.json"
RECEIPT="$DIAG/build/silent_only_static_build_receipt.env"
[[ -x "$NVCC" && -x "$CXX" && -f "$SRC" && -f "$MANIFEST" ]] || {
  echo "compiler/source/manifest missing" >&2; exit 2;
}
TMP_OBJ="$OBJ.tmp.$$"
TMP_BIN="$BIN.tmp.$$"
TMP_RECEIPT="$RECEIPT.tmp.$$"
trap 'rm -f "$TMP_OBJ" "$TMP_BIN" "$TMP_RECEIPT"' EXIT
"$NVCC" -std=c++17 -ccbin "$CXX" -O2 -c "-arch=$ARCH" \
  -I"$DIAG/src" -I"$ROOT/reference/include" "$SRC" -o "$TMP_OBJ"
chmod 600 "$TMP_OBJ"
mv -f "$TMP_OBJ" "$OBJ"
if [[ "$MODE" == "all" ]]; then
  "$NVCC" "-arch=$ARCH" "$OBJ" -o "$TMP_BIN"
  chmod 700 "$TMP_BIN"
  mv -f "$TMP_BIN" "$BIN"
fi
{
  printf 'schema=fair-safe-c1-silent-only-static-build-receipt-v1\n'
  printf 'diagnostic_variant=optimization_diagnostic_silent_only\n'
  printf 'publication_eligible=false\n'
  printf 'gpu_workload_launched=false\n'
  printf 'compile_mode=%s\n' "$MODE"
  printf 'arch=%s\n' "$ARCH"
  printf 'nvcc=%s\n' "$NVCC"
  printf 'cxx=%s\n' "$CXX"
  printf 'runner_source_sha256=%s\n' "$(sha256sum "$SRC" | awk '{print $1}')"
  printf 'header_source_sha256=%s\n' "$(sha256sum "$DIAG/src/g3_safe_search_v2_silent_only.cuh" | awk '{print $1}')"
  printf 'matrix_source_sha256=%s\n' "$(sha256sum "$DIAG/src/g3_safe_c1_native_matrix_silent_only.cu" | awk '{print $1}')"
  printf 'compile_helper_sha256=%s\n' "$(sha256sum "$0" | awk '{print $1}')"
  printf 'source_manifest_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'object_sha256=%s\n' "$(sha256sum "$OBJ" | awk '{print $1}')"
  if [[ "$MODE" == "all" ]]; then
    printf 'binary_sha256=%s\n' "$(sha256sum "$BIN" | awk '{print $1}')"
  fi
} > "$TMP_RECEIPT"
chmod 600 "$TMP_RECEIPT"
mv -f "$TMP_RECEIPT" "$RECEIPT"
printf 'static_compile_only=true\nobject=%s\nbinary=%s\nreceipt=%s\n' "$OBJ" "$BIN" "$RECEIPT"

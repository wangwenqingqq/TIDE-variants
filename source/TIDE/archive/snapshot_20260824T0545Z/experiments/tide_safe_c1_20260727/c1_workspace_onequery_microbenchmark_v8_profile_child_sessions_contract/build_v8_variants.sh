#!/usr/bin/env bash
# CPU/NVCC build only. This script never executes C1Microbench, nsys, or a GPU workload.
set -Eeuo pipefail
umask 077
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract'
WT="$ROOT/worktree"
CUDA_ROOT='/usr/local/cuda-13.1'
[[ "$(hostname)" == CONFIGURE_ARCHIVE_HOST ]] || { echo 'expected 8p host' >&2; exit 69; }
[[ -x "$CUDA_ROOT/bin/nvcc" ]] || { echo 'missing pinned nvcc' >&2; exit 64; }
[[ -f "$WT/src/c1_microbench.cu" && -f "$WT/include/update.cuh" ]] || { echo 'missing v8 worktree' >&2; exit 64; }
JOBS="${C1_BUILD_JOBS:-2}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'C1_BUILD_JOBS must be positive' >&2; exit 64; }
mkdir -p "$ROOT/builds" "$ROOT/build_logs"
# Native CPU-only parser used later by the strict semantic verifier.  It does
# not link CUDA and is compiled here solely to make large canonical artifacts
# auditable without executing any C1 binary.
CXX_BIN="${CXX:-c++}"
command -v "$CXX_BIN" >/dev/null 2>&1 || { echo 'missing C++ compiler for canonical inspector' >&2; exit 64; }
INSPECT_SRC="$ROOT/tools/canonical_inspect_v8.cpp"
INSPECT_BIN="$ROOT/builds/canonical_inspect_v8"
[[ -f "$INSPECT_SRC" && ! -L "$INSPECT_SRC" ]] || { echo 'missing canonical inspector source' >&2; exit 64; }
"$CXX_BIN" -std=c++17 -O3 -Wall -Wextra -Werror "$INSPECT_SRC" -o "$INSPECT_BIN" >"$ROOT/build_logs/build_canonical_inspect_v8.log" 2>&1
[[ -x "$INSPECT_BIN" && ! -L "$INSPECT_BIN" ]] || { echo 'canonical inspector build failed' >&2; exit 70; }
echo 'V8_CPU_CANONICAL_INSPECTOR_BUILT'
variants=(E_G_c1_off_reference P_G_workspace_only E_F_fastpath_only P_F_full_C1)
for variant in "${variants[@]}"; do
  case "$variant" in
    E_G_c1_off_reference) p=OFF; f=OFF ;;
    P_G_workspace_only) p=ON; f=OFF ;;
    E_F_fastpath_only) p=OFF; f=ON ;;
    P_F_full_C1) p=ON; f=ON ;;
  esac
  build="$ROOT/builds/$variant"
  CUDAToolkit_ROOT="$CUDA_ROOT" cmake -S "$WT" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DC1_PERSISTENT_WORKSPACE="$p" \
    -DC1_ONE_QUERY_FASTPATH="$f" >"$ROOT/build_logs/configure_${variant}.log" 2>&1
  cmake --build "$build" --target C1Microbench --parallel "$JOBS" >"$ROOT/build_logs/build_${variant}.log" 2>&1
  [[ -x "$build/bin/C1Microbench" && ! -L "$build/bin/C1Microbench" ]] || { echo "build failed: $variant" >&2; exit 70; }
  echo "V8_CPU_NVCC_BUILT $variant"
done
python3 -B "$ROOT/create_v8_build_manifest.py" --root "$ROOT"
echo 'V8_BUILD_COMPLETE_NO_CUDA_BINARY_EXECUTED'

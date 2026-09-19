#!/usr/bin/env bash
# CPU/NVCC compilation only.  This script never invokes C1Microbench or a GPU workload.
set -Eeuo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark"
WT="$ROOT/worktree"
[[ "$(hostname)" == "CONFIGURE_ARCHIVE_HOST" ]] || { echo "Refusing: expected 8p host context." >&2; exit 69; }
[[ -f "$WT/src/c1_microbench.cu" && -f "$WT/include/update.cuh" ]] || { echo "Missing isolated worktree." >&2; exit 64; }
CUDA_ROOT="/usr/local/cuda-13.1"
[[ -x "$CUDA_ROOT/bin/nvcc" ]] || { echo "Missing expected CUDA compiler at $CUDA_ROOT/bin/nvcc" >&2; exit 64; }
JOBS="${C1_BUILD_JOBS:-2}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "C1_BUILD_JOBS must be a positive integer." >&2; exit 64; }
mkdir -p "$ROOT/builds" "$ROOT/build_logs"
variants=(E_G_c1_off_reference P_G_workspace_only E_F_fastpath_only P_F_full_C1)
for variant in "${variants[@]}"; do
  case "$variant" in
    E_G_c1_off_reference) p=OFF; f=OFF ;;
    P_G_workspace_only)   p=ON;  f=OFF ;;
    E_F_fastpath_only)    p=OFF; f=ON  ;;
    P_F_full_C1)          p=ON;  f=ON  ;;
  esac
  build="$ROOT/builds/$variant"
  CUDAToolkit_ROOT="$CUDA_ROOT" cmake -S "$WT" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DC1_PERSISTENT_WORKSPACE="$p" \
    -DC1_ONE_QUERY_FASTPATH="$f" >"$ROOT/build_logs/configure_${variant}.log" 2>&1
  cmake --build "$build" --target C1Microbench --parallel "$JOBS" >"$ROOT/build_logs/build_${variant}.log" 2>&1
  [[ -x "$build/bin/C1Microbench" ]] || { echo "Missing binary for $variant" >&2; exit 70; }
  echo "BUILT_CPU_ONLY $variant $build/bin/C1Microbench"
done
python3 "$ROOT/create_c1_build_manifest.py" --root "$ROOT"
echo "DONE: binaries compiled; none executed."

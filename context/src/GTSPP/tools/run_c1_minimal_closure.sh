#!/usr/bin/env bash
# Run only with an explicit shared-GPU lease. This script intentionally refuses
# to guess an idle card or to touch existing GPU jobs.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${TIDE_GPU_UUID:?Set an allocated GPU UUID, e.g. GPU-...}"
: "${TIDE_GPU_LEASE:?Set the explicit lease/owner note before launching}"

GPU_INDEX=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
  awk -F', *' -v uuid="$TIDE_GPU_UUID" '$2 == uuid {print $1; exit}')
if [[ -z "$GPU_INDEX" ]]; then
  echo "Refusing run: TIDE_GPU_UUID=$TIDE_GPU_UUID is not present on this host." >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$TIDE_GPU_UUID"
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$ROOT/../runs/c1_minimal_gpu_${STAMP}"
mkdir -p "$RUN"
printf '%q ' "$0" "$@" > "$RUN/command.txt"
printf '\n' >> "$RUN/command.txt"
printf 'GPU_UUID=%s\nGPU_INDEX=%s\nLEASE=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$TIDE_GPU_UUID" "$GPU_INDEX" "$TIDE_GPU_LEASE" "$CUDA_VISIBLE_DEVICES" > "$RUN/lease.txt"

cmake -S "$ROOT" -B "$ROOT/build-c1-minimal" \
  -DCMAKE_CUDA_COMPILER="${TIDE_CUDA_COMPILER:-/usr/local/cuda-13.1/bin/nvcc}" \
  -DCMAKE_CUDA_ARCHITECTURES=120 2>&1 | tee "$RUN/cmake-configure.log"
cmake --build "$ROOT/build-c1-minimal" --target C1MinimalClosure -j "${TIDE_BUILD_JOBS:-4}" \
  2>&1 | tee "$RUN/cmake-build.log"

SEEDS=${TIDE_C1_SEEDS:-128}
OPS=${TIDE_C1_OPS:-400}
DELTA_CAPACITY=${TIDE_C1_DELTA_CAPACITY:-64}
K=${TIDE_C1_K:-3}
"$ROOT/bin/C1MinimalClosure" --seeds "$SEEDS" --ops "$OPS" \
  --delta-capacity "$DELTA_CAPACITY" --k "$K" --receipt "$RUN/summary.json" \
  2>&1 | tee "$RUN/program.log"

python3 "$ROOT/tools/validate_c1_minimal_receipt.py" \
  --summary "$RUN/summary.json" --source "$ROOT" --gpu-uuid "$TIDE_GPU_UUID" \
  --lease "$TIDE_GPU_LEASE" --command "$RUN/command.txt" --output "$RUN/receipt.json" \
  2>&1 | tee "$RUN/validator.log"

echo "PASS receipt: $RUN/receipt.json"

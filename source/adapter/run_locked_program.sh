#!/usr/bin/env bash
set -euo pipefail

if (($# < 1)); then
  echo "usage: $0 <python-script> [arguments...]" >&2
  exit 2
fi

readonly scratch=/workspace/TIDE/active/tide_surechembl_gate6_20260828/scratch/nvmolkit060_query/nvmolkit060_20260904_v1
readonly python=/workspace/echophys_20260903_structured_impulse_gh_video_continuation/.venv/bin/python
readonly expected_uuid=GPU-CONFIGURE-ARCHIVE-DEVICE
readonly global_lock=/tmp/tide_surechembl_gpu0123.lock
readonly gpu_lock=/tmp/tide_surechembl_gate6_gpu2.lock

exec 9>"$global_lock"
flock -n 9 || { echo "global campaign lock is busy" >&2; exit 31; }
exec 8>"$gpu_lock"
flock -n 8 || { echo "GPU2 lock is busy" >&2; exit 32; }
actual_uuid=$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$actual_uuid" == "$expected_uuid" ]] || exit 33
apps=$(nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
[[ -z "$apps" ]] || { echo "$apps" >&2; exit 34; }

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$scratch/site"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec taskset -c 127 "$python" "$@"

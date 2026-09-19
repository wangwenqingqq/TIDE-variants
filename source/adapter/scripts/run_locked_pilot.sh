#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 REMOTE_SCRATCH [PHYSICAL_GPU_INDEX]" >&2
  exit 2
fi

readonly scratch=$1
readonly canonical=/workspace/TIDE/active/tide_surechembl_gate6_20260828
readonly raw=${scratch}/raw/02_pilot_v2
readonly binary=${scratch}/build_release_v4/gpusim_direct_exact
readonly fp=${canonical}/data/prepared/snapshots/2026-08-25/fp_u64x4.bin
readonly ids=${canonical}/data/prepared/snapshots/2026-08-25/id_i64.bin
readonly queries=${canonical}/data/prepared/transition_queries/2026-08-18_to_2026-08-25/queries_u64x6.bin
readonly oracle=${canonical}/results/external/exact_id_oracle_latest_v2.csv
readonly rows=41126808
readonly gpu_index=${2:-2}
case "${gpu_index}" in
  2)
    readonly expected_uuid=GPU-CONFIGURE-ARCHIVE-DEVICE
    readonly gpu_lock=/tmp/tide_surechembl_gate6_gpu2.lock
    ;;
  3)
    readonly expected_uuid=GPU-CONFIGURE-ARCHIVE-DEVICE
    readonly gpu_lock=/tmp/tide_surechembl_gpusim_gpu3.lock
    ;;
  *)
    echo "unsupported physical GPU index: ${gpu_index}" >&2
    exit 4
    ;;
esac

mkdir -p "${raw}"
for output in pilot.csv pilot_summary.json; do
  if [[ -e "${raw}/${output}" ]]; then
    echo "refusing to replace ${raw}/${output}" >&2
    exit 3
  fi
done

exec 8>/tmp/tide_surechembl_gpu0123.lock
exec 9>"${gpu_lock}"
flock -n 8 || { echo "global GPU0-3 campaign lock is busy" >&2; exit 73; }
flock -n 9 || { echo "selected GPUSimilarity GPU lock is busy" >&2; exit 74; }

test "$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid --format=csv,noheader)" = "${expected_uuid}"
if pgrep -af "run_teacher_corpus_shard.py.*--physical-gpu-index ${gpu_index}" >&2; then
  echo "GPU ${gpu_index} has a persistent foreign shard launcher" >&2
  exit 76
fi
for idle_check in 1 2 3; do
  mapfile -t preexisting < <(
    nvidia-smi -i "${gpu_index}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
      sed '/^[[:space:]]*$/d'
  )
  if (( ${#preexisting[@]} != 0 )); then
    printf 'GPU %s has pre-existing compute PID(s): %s\n' "${gpu_index}" "${preexisting[*]}" >&2
    exit 75
  fi
  (( idle_check == 3 )) || sleep 10
done

{
  echo timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo hostname=$(hostname)
  echo user=$(id -un)
  echo cpu_affinity=127
  echo physical_gpu=${gpu_index}
  echo expected_uuid=${expected_uuid}
  nvidia-smi -i "${gpu_index}" --query-gpu=index,uuid,name,pci.bus_id,driver_version,pstate,compute_mode,memory.used,memory.total,utilization.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  free -h
  df -h "${scratch}"
  sha256sum "${binary}" "${fp}" "${ids}" "${queries}" "${oracle}"
} > "${raw}/pilot_preflight.txt"

export CUDA_VISIBLE_DEVICES=${gpu_index}
process_begin_ns=$(date +%s%N)
taskset -c 127 "${binary}" \
  "${fp}" "${ids}" "${queries}" "${raw}/pilot.csv" \
  --rows "${rows}" --query-limit 4 --warmup 2 --cap 10000 \
  --threshold-order 7/10,4/5 \
  > "${raw}/pilot.stdout.txt" \
  2> "${raw}/pilot.stderr.txt"
process_end_ns=$(date +%s%N)
python3 - "${process_begin_ns}" "${process_end_ns}" > "${raw}/pilot_process_wall.txt" <<'PY'
import sys
begin, end = map(int, sys.argv[1:])
print(f"process_wall_seconds={(end - begin) / 1e9:.9f}")
PY
peak_rss_kib=$(awk -F= '/^peak_rss_kib=/{print $2}' "${raw}/pilot.stderr.txt")
test -n "${peak_rss_kib}"
if (( peak_rss_kib > 32 * 1024 * 1024 )); then
  echo "host peak RSS exceeds 32 GiB: ${peak_rss_kib} KiB" >&2
  exit 78
fi

python3 "${scratch}/adapter/tools/summarize_exact.py" \
  --run "${raw}/pilot.csv" --oracle "${oracle}" \
  --rows "${rows}" --queries 4 --cap 10000 \
  --output "${raw}/pilot_summary.json" \
  > "${raw}/pilot_summary.stdout.txt"

nvidia-smi -i "${gpu_index}" --query-gpu=index,uuid,pstate,memory.used,memory.total,utilization.gpu,power.draw,clocks.current.sm,clocks.current.memory --format=csv,noheader \
  > "${raw}/pilot_gpu_after.csv"
sha256sum "${raw}/pilot.csv" "${raw}/pilot_summary.json" \
  > "${raw}/pilot_output.sha256.txt"
cat "${raw}/pilot_summary.json"

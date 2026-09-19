#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 REMOTE_SCRATCH" >&2
  exit 2
fi

readonly scratch=$1
readonly raw=${scratch}/raw/01_build_fixture_sanitizer
readonly binary=${scratch}/build_release_v4/gpusim_direct_exact
readonly fixture=${scratch}/fixture/data
readonly expected_uuid=GPU-CONFIGURE-ARCHIVE-DEVICE

mkdir -p "${raw}"
for output in fixture_release.csv fixture_memcheck.csv fixture_summary.json; do
  if [[ -e "${raw}/${output}" ]]; then
    echo "refusing to replace ${raw}/${output}" >&2
    exit 3
  fi
done

exec 8>/tmp/tide_surechembl_gpu0123.lock
exec 9>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 8 || { echo "global GPU0-3 campaign lock is busy" >&2; exit 73; }
flock -n 9 || { echo "Gate-6 GPU2 campaign lock is busy" >&2; exit 74; }

test "$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)" = "${expected_uuid}"
if pgrep -af 'run_teacher_corpus_shard.py.*--physical-gpu-index 2' >&2; then
  echo "GPU 2 has a persistent foreign shard launcher" >&2
  exit 76
fi
for idle_check in 1 2 3; do
  mapfile -t preexisting < <(
    nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
      sed '/^[[:space:]]*$/d'
  )
  if (( ${#preexisting[@]} != 0 )); then
    printf 'GPU 2 has pre-existing compute PID(s): %s\n' "${preexisting[*]}" >&2
    exit 75
  fi
  (( idle_check == 3 )) || sleep 10
done

{
  echo timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo hostname=$(hostname)
  echo user=$(id -un)
  echo cpu_affinity=127
  echo physical_gpu=2
  echo expected_uuid=${expected_uuid}
  echo CUDA_VISIBLE_DEVICES=2
  nvidia-smi -i 2 --query-gpu=index,uuid,name,pci.bus_id,driver_version,pstate,compute_mode,memory.used,memory.total,utilization.gpu,power.draw,clocks.current.sm,clocks.current.memory --format=csv,noheader
  sha256sum "${binary}" "${fixture}"/*
} > "${raw}/fixture_run_preflight.txt"

export CUDA_VISIBLE_DEVICES=2
taskset -c 127 "${binary}" \
  "${fixture}/fp_u64x4.bin" \
  "${fixture}/id_i64.bin" \
  "${fixture}/queries_u64x6.bin" \
  "${raw}/fixture_release.csv" \
  --rows 9 --query-limit 2 --warmup 2 --cap 100 \
  --threshold-order 7/10,4/5 \
  > "${raw}/fixture_release.stdout.txt" \
  2> "${raw}/fixture_release.stderr.txt"

taskset -c 127 compute-sanitizer --tool memcheck --error-exitcode=77 \
  "${binary}" \
  "${fixture}/fp_u64x4.bin" \
  "${fixture}/id_i64.bin" \
  "${fixture}/queries_u64x6.bin" \
  "${raw}/fixture_memcheck.csv" \
  --rows 9 --query-limit 2 --warmup 0 --cap 100 \
  --threshold-order 4/5,7/10 \
  > "${raw}/fixture_memcheck.stdout.txt" \
  2> "${raw}/fixture_memcheck.stderr.txt"

python3 "${scratch}/adapter/tools/summarize_exact.py" \
  --run "${raw}/fixture_release.csv" \
  --run "${raw}/fixture_memcheck.csv" \
  --oracle "${fixture}/oracle.csv" \
  --rows 9 --queries 2 --cap 100 \
  --output "${raw}/fixture_summary.json" \
  > "${raw}/fixture_summary.stdout.txt"

nvidia-smi -i 2 --query-gpu=index,uuid,pstate,memory.used,memory.total,utilization.gpu,power.draw,clocks.current.sm,clocks.current.memory --format=csv,noheader \
  > "${raw}/fixture_gpu_after.csv"
sha256sum "${raw}"/fixture_*.csv "${raw}/fixture_summary.json" \
  > "${raw}/fixture_output.sha256.txt"
cat "${raw}/fixture_summary.json"

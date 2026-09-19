#!/usr/bin/env bash
set -euo pipefail
WORK=$1
ROOT=/workspace/TIDE/active/tide_surechembl_gate6_20260828
PY=$ROOT/env/bin/python
UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
[[ $(hostname) == CONFIGURE_ARCHIVE_HOST && $(id -un) == archive_user ]]
[[ "$WORK" == "$ROOT"/scratch/svc_f0_* ]]
cd "$WORK"
mkdir raw results
exec 8>/tmp/tide_surechembl_gpu0123.lock
flock -n 8
exec 9>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 9
exec 7>/tmp/tide_surechembl_cpu_oracle.lock
flock -n 7
printf '%s\n' "shell_pid=$$" "work=$WORK" > raw/owner.txt
date -u +%FT%TZ > raw/start_utc.txt
sha256sum -c PAYLOAD_SHA256.txt > raw/payload_hash_check.txt
sha256sum "$ROOT/env/lib/python3.12/site-packages/FPSim2/FPSim2Cuda.py" > raw/native_before.sha256
who > raw/users_before.txt
ps -eo user,pid,ppid,pcpu,pmem,comm --sort=-pcpu > raw/processes_before.txt
free -b > raw/memory_before.txt
[[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -gt 16777216 ]]
[[ $(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader) == "$UUID" ]]
[[ -z $(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader) ]]
[[ $(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits) -gt 8192 ]]
export CUDA_VISIBLE_DEVICES=$UUID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR=$WORK/cupy_cache
mkdir "$CUPY_CACHE_DIR"

state() {
  nvidia-smi -i 2 --query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
}
state > raw/gpu_before.txt
run() {
  local label=$1; shift
  printf '%q ' "$@" > "raw/$label.command.txt"; printf '\n' >> "raw/$label.command.txt"
  date -u +%FT%TZ > "raw/$label.start.txt"
  set +e
  timeout --signal=TERM --kill-after=15s 600s nice -n 10 "$@" > "raw/$label.stdout.txt" 2> "raw/$label.stderr.txt"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "raw/$label.exit_code.txt"
  date -u +%FT%TZ > "raw/$label.end.txt"
  state > "raw/$label.gpu_after.txt"
  if [[ "$rc" -ne 0 ]]; then
    cat "raw/$label.stderr.txt"
    return "$rc"
  fi
}
run prepare "$PY" -B src/gate.py prepare --data data --output prepared
for label in normal memcheck synccheck; do
  [[ -z $(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader) ]]
  if [[ "$label" == normal ]]; then
    run "$label" "$PY" -B src/gate.py validate --data data --prepared prepared --output "results/$label"
  else
    run "$label" compute-sanitizer --tool "$label" --error-exitcode 86 --target-processes all \
      "$PY" -B src/gate.py validate --data data --prepared prepared --output "results/$label"
  fi
done
sha256sum "$ROOT/env/lib/python3.12/site-packages/FPSim2/FPSim2Cuda.py" > raw/native_after.sha256
cmp raw/native_before.sha256 raw/native_after.sha256
state > raw/gpu_after.txt
free -b > raw/memory_after.txt
ps -eo user,pid,ppid,pcpu,pmem,comm --sort=-pcpu > raw/processes_after.txt
date -u +%FT%TZ > raw/end_utc.txt
printf 'PASS\n' > raw/status.txt
printf 'SVC-F0 complete: %s\n' "$WORK"

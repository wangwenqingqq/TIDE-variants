#!/usr/bin/env bash
set -euo pipefail
WORK=$1
MODE=$2
ATTEMPT=${3:-a01}
ROOT=/workspace/TIDE/active/tide_surechembl_gate6_20260828
PY=$ROOT/env/bin/python
UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
[[ $(hostname) == CONFIGURE_ARCHIVE_HOST && $(id -un) == archive_user ]]
[[ "$WORK" == "$ROOT"/scratch/svc_q0_* && "$ATTEMPT" =~ ^a[0-9][0-9]$ ]]
cd "$WORK"
mkdir -p raw results
export PATH=/usr/local/cuda-13.1/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

state() {
  date -u +%FT%TZ
  nvidia-smi -i 2 --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  free -b
}

run() {
  local label=$1; shift
  [[ ! -e "raw/$label.command.txt" ]]
  printf '%q ' "$@" > "raw/$label.command.txt"; printf '\n' >> "raw/$label.command.txt"
  date -u +%FT%TZ > "raw/$label.start.txt"
  set +e
  timeout --signal=TERM --kill-after=15s 600s nice -n 10 "$@" > "raw/$label.stdout.txt" 2> "raw/$label.stderr.txt"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "raw/$label.exit_code.txt"
  date -u +%FT%TZ > "raw/$label.end.txt"
  if [[ "$rc" -ne 0 ]]; then cat "raw/$label.stderr.txt"; return "$rc"; fi
}

if [[ "$MODE" == build ]]; then
  mkdir build
  exec 7>/tmp/tide_surechembl_cpu_oracle.lock
  flock -n 7
  [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -gt 16777216 ]]
  sha256sum -c PAYLOAD_SHA256.txt > raw/build_payload_check.txt
  { hostname; id; uname -a; lscpu; nvcc --version; "$PY" --version;
    nvidia-smi -q -i 2; who; ps -eo user,pid,ppid,pcpu,pmem,comm --sort=-pcpu; } > raw/build_environment.txt
  printf 'shell_pid=%s\nwork=%s\n' "$$" "$WORK" > raw/build_owner.txt
  run build nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC -arch=sm_120 \
    --ptxas-options=-v src/svc_q0_bridge.cu -o build/libsvcq.so
  run sass cuobjdump --dump-sass build/libsvcq.so
  run cpu_input_audit "$PY" -B src/run_q0.py --mode audit --output results/cpu_input_audit
  sha256sum build/libsvcq.so keeper/tide_query_keeper.cu > raw/build_identity.sha256
  printf 'BUILD_AND_CPU_INPUT_AUDIT_PASS_NO_GPU_EXECUTION\n' > raw/build_status.txt
  exit 0
fi

[[ "$MODE" == gpu && -f raw/build_status.txt ]]
mkdir "raw/$ATTEMPT"
exec 8>/tmp/tide_surechembl_gpu0123.lock
flock -n 8
exec 9>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 9
exec 7>/tmp/tide_surechembl_cpu_oracle.lock
flock -n 7
state > "raw/$ATTEMPT/preflight.txt"
{ who; ps -eo user,pid,ppid,pcpu,pmem,comm --sort=-pcpu; } > "raw/$ATTEMPT/users_processes.txt"
printf 'shell_pid=%s\nwork=%s\n' "$$" "$WORK" > "raw/$ATTEMPT/owner.txt"
[[ $(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader) == "$UUID" ]]
if [[ -n $(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader) ]]; then
  printf 'DEFERRED_GPU_OCCUPIED_NO_GPU_EXECUTION\n' > "raw/$ATTEMPT/status.txt"
  cat "raw/$ATTEMPT/status.txt"
  exit 75
fi
[[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -gt 16777216 ]]
[[ $(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits) -gt 8192 ]]
[[ ! -f raw/TELEMETRY.jsonl && ! -f raw/CONTAMINATION.json ]]
export CUDA_VISIBLE_DEVICES=$UUID SVC_Q0_GPU_ADMITTED=1
export CUPY_CACHE_DIR=$WORK/cupy_cache
mkdir "$CUPY_CACHE_DIR"
sha256sum -c PAYLOAD_SHA256.txt > "raw/$ATTEMPT/payload_check.txt"
sha256sum -c raw/build_identity.sha256 > "raw/$ATTEMPT/build_identity_check.txt"
sha256sum "$ROOT/env/lib/python3.12/site-packages/FPSim2/FPSim2Cuda.py" > "raw/$ATTEMPT/native_before.sha256"
"$PY" -B src/monitor_q0.py "$WORK" "$$" &
MONITOR=$!
cleanup() {
  touch raw/MONITOR_STOP
  wait "$MONITOR" || true
  state > "raw/$ATTEMPT/postflight.txt"
}
trap cleanup EXIT

for engine in tide fpsim2; do
  for check in normal memcheck synccheck; do
    label="${ATTEMPT}_${engine}_${check}"
    if [[ "$check" == normal ]]; then
      run "$label" "$PY" -B src/run_q0.py --mode validate --engine "$engine" --output "results/$label"
    else
      run "$label" compute-sanitizer --tool "$check" --error-exitcode 86 --target-processes all \
        "$PY" -B src/run_q0.py --mode validate --engine "$engine" --output "results/$label"
      grep -q 'ERROR SUMMARY: 0 errors' "raw/$label.stdout.txt" "raw/$label.stderr.txt"
    fi
    [[ ! -f raw/CONTAMINATION.json ]]
  done
done
for pair in 0 1 2 3; do
  engines='tide fpsim2'
  if [[ "$pair" == 1 || "$pair" == 2 ]]; then engines='fpsim2 tide'; fi
  for engine in $engines; do
    label="${ATTEMPT}_pair${pair}_${engine}"
    [[ -z $(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader) ]]
    run "$label" "$PY" -B src/run_q0.py --mode measure --engine "$engine" --pair "$pair" --output "results/$label"
    [[ ! -f raw/CONTAMINATION.json ]]
    state > "raw/$label.postflight.txt"
  done
done
sha256sum "$ROOT/env/lib/python3.12/site-packages/FPSim2/FPSim2Cuda.py" > "raw/$ATTEMPT/native_after.sha256"
cmp "raw/$ATTEMPT/native_before.sha256" "raw/$ATTEMPT/native_after.sha256"
printf 'GPU_VALIDATION_AND_DIAGNOSTIC_PASS_ANALYSIS_REQUIRED\n' > "raw/$ATTEMPT/status.txt"

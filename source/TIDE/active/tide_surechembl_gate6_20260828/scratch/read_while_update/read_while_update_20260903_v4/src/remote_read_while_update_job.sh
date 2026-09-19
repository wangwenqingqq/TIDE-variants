#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 WORK ROOT DATA FIXTURE GPU GPU_UUID" >&2
  exit 64
fi

WORK=$1
ROOT=$2
DATA=$3
FIXTURE=$4
GPU=$5
GPU_UUID=$6
CUDA=/usr/local/cuda-13.1
BIN=$WORK/build/gate6_overlap_campaign_v1
ORACLE_BIN=$WORK/build/gate6_epoch_id_oracle_v1
FORMAL_ORACLE=$WORK/oracle/largest_epoch_oracle.csv
FIXTURE_ORACLE=$WORK/oracle/fixture_epoch_oracle.csv
EXPECTED_BODY_SHA=92027514dfbd14260b084e919d1d16a2237f9169b0afab97b0ec144cfd8ff264

mkdir -p "$WORK"/{build,audit,oracle,results,raw}
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/audit/build_start_utc.txt"
sha256sum "$WORK"/src/* "$WORK"/cards/* > "$WORK/audit/uploaded_hashes.txt"
"$CUDA/bin/nvcc" -std=c++17 -O3 -lineinfo -arch=sm_120a \
  -Xcompiler=-pthread,-Wall,-Wextra \
  "$WORK/src/gate6_overlap_campaign_v1.cu" -o "$BIN" \
  > "$WORK/audit/nvcc.stdout.txt" 2> "$WORK/audit/nvcc.stderr.txt"
g++ -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic \
  "$WORK/src/gate6_epoch_id_oracle_v1.cpp" -o "$ORACLE_BIN" \
  > "$WORK/audit/gxx.stdout.txt" 2> "$WORK/audit/gxx.stderr.txt"
if grep -Eqi 'warning:|error:' "$WORK/audit/nvcc.stderr.txt" \
    "$WORK/audit/gxx.stderr.txt"; then
  echo 'warning/error found in build logs' >&2
  exit 65
fi
"$CUDA/bin/cuobjdump" --dump-sass "$BIN" \
  > "$WORK/audit/overlap.full.sass" 2> "$WORK/audit/overlap.sass.stderr.txt"
"$CUDA/bin/cuobjdump" --dump-resource-usage "$BIN" \
  > "$WORK/audit/overlap.resources.txt" \
  2> "$WORK/audit/overlap.resources.stderr.txt"
python3 - "$WORK/audit/overlap.full.sass" \
  "$WORK/audit/overlap.words4.instruction_body.sass" \
  "$WORK/audit/overlap.resources.txt" "$EXPECTED_BODY_SHA" <<'PY'
import hashlib, re, sys

sass_path, body_path, resource_path, expected_hash = sys.argv[1:]
selected = False
body = []
for raw in open(sass_path, errors="replace"):
    function = re.match(r"\s*Function\s*:\s*(\S+)", raw)
    if function:
        selected = "exact_runs_kernelILi4" in function.group(1)
        continue
    if not selected:
        continue
    address = re.match(r"\s*/\*\s*[0-9a-fA-F]+\s*\*/\s*(.*)", raw)
    if not address:
        continue
    instruction = re.sub(
        r"\s*/\*\s*(?:0x)?[0-9a-fA-F]{8,}\s*\*/\s*$", "", address.group(1)
    ).strip()
    instruction = re.sub(r"\s+", " ", instruction)
    if instruction.endswith(";"):
        body.append(instruction)
text = "\n".join(body) + "\n"
open(body_path, "w").write(text)
actual = hashlib.sha256(text.encode()).hexdigest()
if len(body) != 128 or actual != expected_hash:
    raise SystemExit(f"WORDS4 instruction body changed: lines={len(body)} sha={actual}")

lines = open(resource_path).read().splitlines()
for index, line in enumerate(lines):
    if "exact_runs_kernelILi4" in line:
        match = re.search(
            r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+) CONSTANT\[0\]:(\d+)",
            lines[index + 1],
        )
        if not match:
            raise SystemExit("cannot parse WORDS4 resource line")
        values = tuple(map(int, match.groups()))
        if values != (34, 0, 0, 0, 952):
            raise SystemExit(f"WORDS4 resources changed: {values}")
        break
else:
    raise SystemExit("WORDS4 resource record not found")
print(f"WORDS4 instruction body and resources pass: {actual} {values}")
PY
sha256sum "$WORK"/build/* "$WORK/audit/overlap.words4.instruction_body.sass" \
  > "$WORK/audit/binary_and_body_hashes.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/audit/build_end_utc.txt"

exec 7>/tmp/tide_surechembl_cpu_oracle.lock
flock -n 7 || { echo 'CPU oracle lock busy' >&2; exit 75; }
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/oracle/start_utc.txt"
ps -eo user,pid,ppid,lstart,cmd --sort=pid > "$WORK/oracle/processes_before.txt"
fixture_start_ns=$(date +%s%N)
OMP_NUM_THREADS=32 "$ORACLE_BIN" "$FIXTURE" "$FIXTURE_ORACLE" 512 \
  > "$WORK/oracle/fixture.stdout.txt" 2> "$WORK/oracle/fixture.stderr.txt"
fixture_end_ns=$(date +%s%N)
largest_start_ns=$(date +%s%N)
OMP_NUM_THREADS=32 "$ORACLE_BIN" "$DATA" "$FORMAL_ORACLE" 512 \
  > "$WORK/oracle/largest.stdout.txt" 2> "$WORK/oracle/largest.stderr.txt"
largest_end_ns=$(date +%s%N)
python3 - "$fixture_start_ns" "$fixture_end_ns" \
  > "$WORK/oracle/fixture.wall_seconds.txt" <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1e9)
PY
python3 - "$largest_start_ns" "$largest_end_ns" \
  > "$WORK/oracle/largest.wall_seconds.txt" <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1e9)
PY
python3 - "$FIXTURE_ORACLE.summary.json" "$FORMAL_ORACLE.summary.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    if not json.load(open(path))["pass"]:
        raise SystemExit(f"oracle failed: {path}")
print("both epoch oracles pass")
PY
sha256sum "$WORK"/oracle/*.csv "$WORK"/oracle/*.json \
  > "$WORK/oracle/output_hashes.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/oracle/end_utc.txt"
flock -u 7

exec 8>/tmp/tide_surechembl_gpu0123.lock
exec 9>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 8 || { echo 'global GPU0-3 lock busy' >&2; exit 73; }
flock -n 9 || { echo 'Gate-6 GPU2 lock busy' >&2; exit 74; }
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/raw/start_utc.txt"
printf '%s host_pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
  > "$WORK/raw/lock_owner.txt"
hostname > "$WORK/raw/hostname.txt"
id > "$WORK/raw/identity.txt"
who > "$WORK/raw/users_initial.txt" || true
ps -eo user,pid,ppid,lstart,cmd --sort=pid > "$WORK/raw/processes_initial.txt"
nvidia-smi -L > "$WORK/raw/gpu_inventory.txt"
actual_uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$actual_uuid" == "$GPU_UUID" ]]

check_idle() {
  local name=$1
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader > "$WORK/raw/$name.compute_before.csv" 2>/dev/null || true
  [[ ! -s "$WORK/raw/$name.compute_before.csv" ]] || {
    echo "GPU2 became occupied before $name" >&2
    exit 76
  }
  nvidia-smi -i "$GPU" \
    --query-gpu=index,uuid,name,pstate,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader > "$WORK/raw/$name.gpu_before.csv"
}

record_after() {
  local name=$1
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader > "$WORK/raw/$name.compute_after.csv" 2>/dev/null || true
  nvidia-smi -i "$GPU" \
    --query-gpu=index,uuid,name,pstate,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader > "$WORK/raw/$name.gpu_after.csv"
  [[ ! -s "$WORK/raw/$name.compute_after.csv" ]] || {
    echo "GPU2 process remained after $name" >&2
    exit 77
  }
}

validate_output() {
  local output=$1
  python3 - "$output.summary.json" "$output.probe.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    if not json.load(open(path))["pass"]:
        raise SystemExit(f"failed JSON gate: {path}")
PY
}

run_one() {
  local name=$1 data=$2 oracle=$3 mode=$4 requests=$5 formal=$6 after=$7
  local output=$WORK/results/$name.csv
  local command=(
    "$BIN" --root "$data" --mode "$mode" --output "$output"
    --gpu "$GPU" --words 4 --query-limit 512 --oracle "$oracle"
    --reader-requests "$requests" --formal-overlap-gates "$formal"
    --update-after-completions "$after"
  )
  check_idle "$name"
  printf '%q ' "${command[@]}" > "$WORK/raw/$name.command.txt"
  printf '\n' >> "$WORK/raw/$name.command.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/raw/$name.start_utc.txt"
  set +e
  "${command[@]}" > "$WORK/raw/$name.stdout.txt" 2> "$WORK/raw/$name.stderr.txt"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$WORK/raw/$name.exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/raw/$name.end_utc.txt"
  record_after "$name"
  [[ "$rc" -eq 0 ]]
  [[ ! -s "$WORK/raw/$name.stderr.txt" ]]
  validate_output "$output"
  sha256sum "$output" "$output".* > "$WORK/raw/$name.output_hashes.txt"
  printf 'END %s rc=0 %s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "$WORK/raw/progress.log"
}

run_sanitizer() {
  local name=$1 tool=$2
  local output=$WORK/results/$name.csv
  local command=("$CUDA/bin/compute-sanitizer" --tool "$tool" --error-exitcode 99)
  if [[ "$tool" == memcheck ]]; then
    command+=(--leak-check full --report-api-errors all)
  fi
  command+=(
    "$BIN" --root "$FIXTURE" --mode overlap --output "$output"
    --gpu "$GPU" --words 4 --query-limit 512 --oracle "$FIXTURE_ORACLE"
    --reader-requests 64 --formal-overlap-gates 0
    --update-after-completions 4
  )
  check_idle "$name"
  printf '%q ' "${command[@]}" > "$WORK/raw/$name.command.txt"
  printf '\n' >> "$WORK/raw/$name.command.txt"
  set +e
  "${command[@]}" > "$WORK/raw/$name.stdout.txt" 2> "$WORK/raw/$name.stderr.txt"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$WORK/raw/$name.exit_code.txt"
  record_after "$name"
  [[ "$rc" -eq 0 ]]
  grep -Eq 'ERROR SUMMARY: 0 errors' \
    "$WORK/raw/$name.stdout.txt" "$WORK/raw/$name.stderr.txt"
  validate_output "$output"
  sha256sum "$output" "$output".* > "$WORK/raw/$name.output_hashes.txt"
}

run_one fixture_control "$FIXTURE" "$FIXTURE_ORACLE" control 64 0 0
run_one fixture_overlap "$FIXTURE" "$FIXTURE_ORACLE" overlap 64 0 4
run_sanitizer fixture_memcheck memcheck
run_sanitizer fixture_synccheck synccheck
run_one real_smoke_overlap "$DATA" "$FORMAL_ORACLE" overlap 4096 1 0

for pair in 1 2 3 4 5 6; do
  if (( pair % 2 == 1 )); then
    order=(control overlap)
  else
    order=(overlap control)
  fi
  for mode in "${order[@]}"; do
    run_one "pair${pair}_${mode}" "$DATA" "$FORMAL_ORACLE" "$mode" 4096 1 0
  done
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/raw/end_utc.txt"
ps -eo user,pid,ppid,lstart,cmd --sort=pid > "$WORK/raw/processes_final.txt"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "$WORK/raw/compute_final.csv" 2>/dev/null || true
[[ ! -s "$WORK/raw/compute_final.csv" ]]
find "$WORK/raw" "$WORK/results" -type f ! -name retained.sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$WORK/raw/retained.sha256.txt"
echo "REMOTE_READ_WHILE_UPDATE_JOB_PASS"

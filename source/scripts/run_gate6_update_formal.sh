#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/TIDE/active/tide_surechembl_gate6_20260828
RAW="$ROOT/results/raw/20260829T050000Z_gate6_update_formal_v1"
OUT="$ROOT/results/update/formal_surechembl_v1"
BIN="$ROOT/build/gate6_update_campaign_v1"
GPU=2
UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
TRANSITIONS=(
  2026-06-01_to_2026-06-15
  2026-06-15_to_2026-07-01
  2026-07-01_to_2026-07-17
  2026-07-17_to_2026-08-04
  2026-08-04_to_2026-08-18
  2026-08-18_to_2026-08-25
)

mkdir -p "$RAW" "$OUT"
if [[ -e "$RAW/start_utc.txt" ]]; then
  echo "refusing to overwrite existing update campaign: $RAW" >&2
  exit 3
fi

exec 8>/tmp/tide_surechembl_gpu0123.lock
exec 9>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 8 || { echo "global GPU0-3 lock busy" >&2; exit 73; }
flock -n 9 || { echo "Gate-6 GPU2 lock busy" >&2; exit 74; }

date -u +%Y-%m-%dT%H:%M:%SZ > "$RAW/start_utc.txt"
printf '%s host_pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "$RAW/lock_owner.txt"
hostname > "$RAW/hostname.txt"
id > "$RAW/identity.txt"
who > "$RAW/users_initial.txt" || true
ps -eo user,pid,ppid,lstart,cmd --sort=pid > "$RAW/processes_initial.txt"
nvidia-smi -L > "$RAW/gpu_inventory.txt"
nvidia-smi -q -i "$GPU" > "$RAW/gpu2_initial_q.txt"
/usr/local/cuda-13.1/bin/nvcc --version > "$RAW/nvcc_version.txt"
cp "$ROOT/UPDATE_EXECUTION_CARD_20260829.md" "$RAW/frozen_execution_card.md"
shasum -a 256 "$ROOT/UPDATE_EXECUTION_CARD_20260829.md" \
  "$ROOT/src/gate6_update_campaign_v1.cu" "$BIN" > "$RAW/source_binary_card_hashes.txt"

for transition in "${TRANSITIONS[@]}"; do
  prefix="$RAW/$transition"
  output="$OUT/$transition.csv"
  data="$ROOT/data/prepared/transition_roots/$transition"

  date -u +%Y-%m-%dT%H:%M:%SZ > "$prefix.start_utc.txt"
  ps -eo user,pid,ppid,lstart,cmd --sort=pid > "$prefix.processes_before.txt"
  nvidia-smi -i "$GPU" \
    --query-gpu=index,uuid,name,persistence_mode,compute_mode,pstate,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader > "$prefix.gpu_before.csv"
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader > "$prefix.compute_before.csv" 2>/dev/null || true
  actual_uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
  [[ "$actual_uuid" == "$UUID" ]] || { echo "GPU2 UUID changed" >&2; exit 75; }
  [[ ! -s "$prefix.compute_before.csv" ]] || { echo "GPU2 busy before $transition" >&2; exit 76; }

  command=(
    "$BIN"
    --root "$data"
    --mode update
    --output "$output"
    --gpu "$GPU"
    --words 4
    --update-warmups 16
    --update-requests 512
  )
  printf '%q ' "${command[@]}" > "$prefix.command.txt"
  printf '\n' >> "$prefix.command.txt"
  set +e
  "${command[@]}" > "$prefix.stdout.log" 2> "$prefix.stderr.log"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$prefix.exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$prefix.end_utc.txt"
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader > "$prefix.compute_after.csv" 2>/dev/null || true
  nvidia-smi -i "$GPU" \
    --query-gpu=index,uuid,name,persistence_mode,compute_mode,pstate,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader > "$prefix.gpu_after.csv"

  if [[ "$rc" -ne 0 ]]; then
    echo "update process failed: $transition rc=$rc" >&2
    exit "$rc"
  fi
  [[ ! -s "$prefix.stderr.log" ]] || { echo "non-empty stderr: $transition" >&2; exit 77; }
  [[ ! -s "$prefix.compute_after.csv" ]] || { echo "GPU2 process remained after $transition" >&2; exit 78; }
  python3 - "$output" <<'PY'
import csv, json, sys
path = sys.argv[1]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 512 or any(row["match"] != "1" for row in rows):
    raise SystemExit("formal output completeness/exactness failure")
with open(path + ".validation.json") as handle:
    validation = json.load(handle)
if not validation["pass"] or validation["retained_mismatches"] != 0:
    raise SystemExit("formal validation JSON failure")
PY
  shasum -a 256 "$output" "$output.setup.json" "$output.validation.json" \
    > "$prefix.output_hashes.txt"
  printf 'END %s rc=0 %s\n' "$transition" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RAW/progress.log"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$RAW/end_utc.txt"
find "$RAW" "$OUT" -maxdepth 1 -type f ! -name sha256.txt -print0 | \
  sort -z | xargs -0 shasum -a 256 > "$RAW/sha256.txt"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "$RAW/compute_final.csv" 2>/dev/null || true
[[ ! -s "$RAW/compute_final.csv" ]]

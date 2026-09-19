#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/TIDE/active/tide_surechembl_gate6_20260828
RAW="$ROOT/results/raw/20260829T042300Z_mechanism_ncu_v2"
OUT="$ROOT/results/mechanism_ncu/formal_surechembl_latest_v2"
BIN="$ROOT/build/gate6_query_campaign_sustained_unbounded_v1"
DATA="$ROOT/data/prepared/transition_roots/2026-08-18_to_2026-08-25"
NCU=/usr/local/cuda-13.1/bin/ncu
GPU=2
UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
METRICS=gpu__time_duration.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__inst_executed_op_branch.sum,smsp__inst_executed_op_generic_atom.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sectors_op_read.sum

mkdir -p "$RAW" "$OUT"
if [[ -e "$RAW/start_utc.txt" ]]; then
  echo "refusing to overwrite existing NCU campaign: $RAW" >&2
  exit 3
fi

# Serialize against the frozen Gate-6 GPU2 and GPU0-3 campaigns. Lock files are
# persistent rendezvous points; the advisory locks, not file existence, matter.
exec 8>/tmp/tide_surechembl_gpu0123.lock
exec 9>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 8 || { echo "global GPU0-3 lock busy" >&2; exit 73; }
flock -n 9 || { echo "Gate-6 GPU2 lock busy" >&2; exit 74; }

date -u +%Y-%m-%dT%H:%M:%SZ > "$RAW/start_utc.txt"
id > "$RAW/identity.txt"
who > "$RAW/users_initial.txt" || true
ps -eo user,pid,ppid,lstart,cmd --sort=pid > "$RAW/processes_initial.txt"
nvidia-smi -L > "$RAW/gpu_inventory.txt"
nvidia-smi -i "$GPU" --query-gpu=index,uuid,name,persistence_mode,compute_mode,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RAW/gpu2_initial.csv"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$RAW/gpu2_compute_apps_initial.csv" 2>/dev/null || true
sudo -n "$NCU" --version > "$RAW/ncu_version.txt" 2>&1
grep -E 'RmProfilingAdminOnly' /proc/driver/nvidia/params > "$RAW/driver_counter_policy.txt" || true

actual_uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$actual_uuid" == "$UUID" ]] || { echo "GPU2 UUID changed: $actual_uuid" >&2; exit 75; }
[[ ! -s "$RAW/gpu2_compute_apps_initial.csv" ]] || { echo "GPU2 has an external compute process" >&2; exit 76; }

run_variant() {
  local variant=$1
  local prefix="$RAW/$variant"
  local output="$OUT/$variant.csv"
  local report="$RAW/$variant"

  nvidia-smi -i "$GPU" --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$prefix.gpu_before.csv"
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$prefix.compute_apps_before.csv" 2>/dev/null || true
  [[ ! -s "$prefix.compute_apps_before.csv" ]] || { echo "GPU2 became busy before $variant" >&2; return 76; }

  printf '%q ' sudo -n "$NCU" --devices "$GPU" --replay-mode kernel --kernel-name-base demangled --kernel-name 'regex:.*exact_runs_kernel.*' --launch-skip 16 --launch-count 1 --section InstructionStats --section SourceCounters --metrics "$METRICS" --disable-extra-suffixes --force-overwrite --export "$report" "$BIN" --root "$DATA" --mode benchmark --output "$output" --gpu "$GPU" --words 4 --query-limit 1 --order "$variant" --threshold-order 70,80 > "$prefix.command.txt"
  printf '\n' >> "$prefix.command.txt"

  set +e
  sudo -n "$NCU" \
    --devices "$GPU" \
    --replay-mode kernel \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*exact_runs_kernel.*' \
    --launch-skip 16 \
    --launch-count 1 \
    --section InstructionStats \
    --section SourceCounters \
    --metrics "$METRICS" \
    --disable-extra-suffixes \
    --force-overwrite \
    --export "$report" \
    "$BIN" \
      --root "$DATA" \
      --mode benchmark \
      --output "$output" \
      --gpu "$GPU" \
      --words 4 \
      --query-limit 1 \
      --order "$variant" \
      --threshold-order 70,80 \
      > "$prefix.stdout.log" 2> "$prefix.stderr.log"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 && ! -e "$report.ncu-rep" ]]; then
    echo "NCU returned zero but produced no report" >> "$prefix.stderr.log"
    rc=77
  fi
  printf '%s\n' "$rc" > "$prefix.exit_code.txt"

  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$prefix.compute_apps_after.csv" 2>/dev/null || true
  nvidia-smi -i "$GPU" --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$prefix.gpu_after.csv"
  if [[ -e "$report.ncu-rep" ]]; then
    sudo -n "$NCU" --import "$report.ncu-rep" --page raw --csv --units base > "$prefix.raw.csv" 2> "$prefix.import_raw.stderr.log" || true
    sudo -n "$NCU" --import "$report.ncu-rep" --page source --print-source sass --csv --units base > "$prefix.source_sass.csv" 2> "$prefix.import_source.stderr.log" || true
  fi
  sudo -n chown -R archive_user:archive_user "$RAW" "$OUT"
  return "$rc"
}

final_rc=0
for variant in logical64 unbounded64; do
  if run_variant "$variant"; then
    :
  else
    rc=$?
    final_rc=$rc
    break
  fi
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$RAW/end_utc.txt"
find "$RAW" "$OUT" -maxdepth 1 -type f ! -name sha256.txt -print0 | sort -z | xargs -0 shasum -a 256 > "$RAW/sha256.txt"
sudo -n chown -R archive_user:archive_user "$RAW" "$OUT"
exit "$final_rc"

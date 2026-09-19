#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ROOT RAW_ID OUTPUT_LABEL" >&2
  exit 64
fi

root=$1
raw_id=$2
output_label=$3
binary="$root/build/gate6_query_campaign"
source_file="$root/src/gate6_query_campaign.cu"
raw="$root/results/raw/$raw_id"
output="$root/results/query_correctness/$output_label"
common_lock=/tmp/tide_surechembl_gpu0123.lock
gate6_lock=/tmp/tide_surechembl_gate6_gpu2.lock
gpu_uuid=GPU-CONFIGURE-ARCHIVE-DEVICE
transitions=(
  2026-06-01_to_2026-06-15
  2026-06-15_to_2026-07-01
  2026-07-01_to_2026-07-17
  2026-07-17_to_2026-08-04
  2026-08-04_to_2026-08-18
  2026-08-18_to_2026-08-25
)

if [[ -e $raw || -e $output ]]; then
  echo "refusing to reuse exact-sweep paths: $raw or $output" >&2
  exit 65
fi
mkdir -p "$raw" "$output"
exec 9>"$common_lock"
if ! flock -n 9; then
  echo shared_lock_busy > "$raw/blocker.txt"
  exit 75
fi
exec 8>"$gate6_lock"
if ! flock -n 8; then
  echo gate6_gpu2_lock_busy > "$raw/blocker.txt"
  exit 75
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$raw/start_utc.txt"
hostname > "$raw/hostname.txt"
printf 'pid=%s common_lock=%s gate6_lock=%s\n' \
  "$$" "$common_lock" "$gate6_lock" > "$raw/lock_owner.txt"
printf '%s\n' \
  'six frozen SureChEMBL transitions; 512 deterministic queries; exact rational 7/10 and 4/5; compacted, logical 1/4/16/64, and unbounded 1/4/16/64; correctness only, not timing evidence' \
  > "$raw/contract.txt"
sha256sum "$binary" "$source_file" "$0" \
  "$root/data/prepared/FLAT_INPUT_MANIFEST.json" \
  "$root/results/FLAT_INPUT_VALIDATION.json" > "$raw/input_hashes.txt"
nvidia-smi -L > "$raw/nvidia_smi_L.txt"
/usr/local/cuda-13.1/bin/nvcc --version > "$raw/nvcc_version.txt"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u \
  > "$raw/driver_version.txt"

overall=0
for transition in "${transitions[@]}"; do
  transition_root="$root/data/prepared/transition_roots/$transition"
  result="$output/$transition.json"
  echo "BEGIN $transition $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "$raw/progress.log"
  nvidia-smi \
    --query-gpu=index,uuid,name,pci.bus_id,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,compute_mode \
    --format=csv,noheader,nounits > "$raw/$transition.gpu_state_before.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$raw/$transition.compute_apps_before.csv" \
    || true
  if grep -Fq "$gpu_uuid" "$raw/$transition.compute_apps_before.csv"; then
    echo "$transition foreign_or_existing_gpu2_compute_process" \
      > "$raw/blocker.txt"
    printf '75\n' > "$raw/$transition.exit_code.txt"
    overall=75
    break
  fi
  printf '%s\n' \
    "$binary --root $transition_root --mode correctness --output $result --gpu 2 --words 4 --query-limit 512" \
    > "$raw/$transition.command.txt"
  sha256sum "$transition_root/LAYOUT_MANIFEST.json" \
    > "$raw/$transition.layout_hash.txt"
  "$binary" --root "$transition_root" --mode correctness \
    --output "$result" --gpu 2 --words 4 --query-limit 512 \
    > "$raw/$transition.stdout.log" \
    2> "$raw/$transition.stderr.log"
  rc=$?
  printf '%s\n' "$rc" > "$raw/$transition.exit_code.txt"
  if [[ -f $result ]]; then
    sha256sum "$result" "$result.samples.csv" \
      > "$raw/$transition.output_hashes.txt"
  fi
  nvidia-smi --query-gpu=index,uuid,pstate,utilization.gpu,memory.used,power.draw,clocks.sm \
    --format=csv,noheader,nounits > "$raw/$transition.gpu_state_after.csv"
  echo "END $transition rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "$raw/progress.log"
  if [[ $rc -ne 0 ]]; then
    overall=$rc
    break
  fi
done

python3 - "$output" "$raw/summary.json" "${transitions[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
transitions = sys.argv[3:]
records = []
sample_counts = {}
overflows = {}
for transition in transitions:
    path = output / f"{transition}.json"
    sample_path = output / f"{transition}.json.samples.csv"
    if not path.exists() or not sample_path.exists():
        continue
    record = json.loads(path.read_text())
    records.append({"transition": transition, **record})
    with sample_path.open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    sample_counts[transition] = len(samples)
    overflows[transition] = sum(int(row["overflow"]) for row in samples)
summary = {
    "experiment_id": "tide_20260828_gate6_exact_surechembl_formal",
    "planned_transitions": transitions,
    "completed_transitions": [record["transition"] for record in records],
    "sample_rows": sample_counts,
    "overflows": overflows,
    "all_surechembl_exact_component_pass": bool(records)
    and len(records) == len(transitions)
    and all(record["G6_EXACT_COMPONENT"] for record in records)
    and all(value == 0 for value in overflows.values()),
    "records": records,
    "aggregate_G6_EXACT": (
        "blocked: frozen ChEMBL 37 source-integrity contract failed; "
        "SureChEMBL component reported separately"
    ),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
(output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
summary_rc=$?
if [[ $summary_rc -ne 0 && $overall -eq 0 ]]; then
  overall=$summary_rc
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$raw/end_utc.txt"
printf '%s\n' "$overall" > "$raw/exit_code.txt"
exit "$overall"

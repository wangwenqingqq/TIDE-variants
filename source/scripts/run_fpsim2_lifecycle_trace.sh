#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ROOT TRACE_ID OUTPUT_LABEL" >&2
  exit 64
fi

root=$1
trace_id=$2
output_label=$3
python="$root/env/bin/python"
measure="$root/scripts/measure_fpsim2_lifecycle.py"
raw="$root/results/raw/$trace_id"
output="$root/results/fpsim2_lifecycle/$output_label"

if [[ -e $raw || -e $output ]]; then
  echo "refusing to reuse trace paths: $raw or $output" >&2
  exit 65
fi
mkdir -p "$raw" "$output"

transitions=(
  2026-06-01_to_2026-06-15
  2026-06-15_to_2026-07-01
  2026-07-01_to_2026-07-17
  2026-07-17_to_2026-08-04
  2026-08-04_to_2026-08-18
  2026-08-18_to_2026-08-25
)

date -u +%Y-%m-%dT%H:%M:%SZ > "$raw/start_utc.txt"
hostname > "$raw/hostname.txt"
printf '%s\n' \
  'six frozen transitions; actual FPSim2 0.7.4; one clean sequential process each; post-fingerprint append + complete popcount CSI creation + global sort/bin repair + reload; source copy excluded' \
  > "$raw/contract.txt"
sha256sum "$measure" "$0" "$root/results/TRANSITION_REALITY.json" \
  "$root/results/DELTA_ORDER_REPAIR.json" > "$raw/input_hashes.txt"
"$python" -m pip freeze > "$raw/pip_freeze.txt"
df -h "$root" > "$raw/disk_before.txt"

overall=0
for transition in "${transitions[@]}"; do
  echo "BEGIN $transition $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$raw/progress.log"
  "$python" "$measure" \
    --root "$root" \
    --transition "$transition" \
    --run-id "${output_label}_${transition}" \
    --output "$output/${transition}.json" \
    > "$raw/${transition}.stdout.log" \
    2> "$raw/${transition}.stderr.log"
  rc=$?
  printf '%s\n' "$rc" > "$raw/${transition}.exit_code.txt"
  echo "END $transition rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$raw/progress.log"
  if [[ $rc -ne 0 ]]; then
    overall=$rc
    break
  fi
done

"$python" - "$output" "$raw/summary.json" "${transitions[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
transitions = sys.argv[3:]
records = []
for transition in transitions:
    path = output / f"{transition}.json"
    if path.exists():
        records.append(json.loads(path.read_text()))
summary = {
    "planned_transitions": transitions,
    "completed_transitions": [record["transition"] for record in records],
    "all_query_ready_component_pass": bool(records)
    and len(records) == len(transitions)
    and all(record["query_ready_component_pass"] for record in records),
    "records": records,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY
summary_rc=$?
if [[ $summary_rc -ne 0 && $overall -eq 0 ]]; then
  overall=$summary_rc
fi

df -h "$root" > "$raw/disk_after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$raw/end_utc.txt"
printf '%s\n' "$overall" > "$raw/exit_code.txt"
exit "$overall"

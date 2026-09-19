#!/usr/bin/env bash
# Guarded single-run launcher for Safe-C1 G1A. Correctness only; no timing claim.
set -Eeuo pipefail

readonly ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_dynamic_gts_v1"
readonly EXPECTED_HOST="CONFIGURE_ARCHIVE_HOST"
readonly EXPECTED_UUID="GPU-CONFIGURE-ARCHIVE-DEVICE"
readonly PYTHON="/workspace/legacy_workspace/GTS/bench_env/bin/python"
readonly BIN="$ROOT/bin/GTS_safe_c1_g1"
readonly BUNDLE="$ROOT/bundles/g1a_witness_pre_rebuild_v1"
readonly AUDITOR="$ROOT/tools/audit_g1_static_contract.py"
readonly VALIDATOR="$ROOT/tools/verify_g1_witness_export.py"
readonly MAX_IDLE_MEMORY_MIB=256

usage() {
  cat <<'EOF'
Usage: run_g1a_guarded.sh --out /workspace/experiments/tide_safe_c1_20260727/runs/<new-run>

This launches exactly one G1A correctness gate on 8p physical GPU 0 only.
It refuses non-idle GPU0, never kills a process, and does not test range,
base deletion, rebuild, timing, or throughput.
EOF
}

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done
[[ -n "$OUT" ]] || { usage >&2; exit 64; }
[[ -x "$PYTHON" && -x "$BIN" && -x "$AUDITOR" && -x "$VALIDATOR" ]] || { echo "required runner artifact missing" >&2; exit 66; }
[[ -d "$BUNDLE" ]] || { echo "bundle missing" >&2; exit 66; }
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || { echo "refuse non-8p host $(hostname)" >&2; exit 77; }
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" != *a800* ]] || { echo "A800 forbidden" >&2; exit 77; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 70; }
command -v timeout >/dev/null || { echo "timeout unavailable" >&2; exit 70; }

OUT="$($PYTHON - "$OUT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
case "$OUT" in
  /workspace/experiments/tide_safe_c1_20260727/runs/*) ;;
  *) echo "output must be under isolated runs root" >&2; exit 64 ;;
esac
if [[ -e "$OUT" ]]; then
  [[ -d "$OUT" && -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "output must be new or empty" >&2; exit 73; }
fi
mkdir -p "$OUT/logs" "$OUT/runner_output" "$OUT/preflight"

# CPU-only gates happen before any GPU status query or CUDA binary call.
"$PYTHON" "$AUDITOR" --root "$ROOT" --out "$OUT/preflight/static_contract.json" >"$OUT/logs/static_audit.stdout.log" 2>"$OUT/logs/static_audit.stderr.log"
"$PYTHON" - "$BUNDLE" "$OUT/preflight/bundle_sha256.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
b=Path(sys.argv[1]); out=Path(sys.argv[2])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for q in iter(lambda:f.read(1<<20),b''): h.update(q)
 return h.hexdigest()
files=sorted(p for p in b.iterdir() if p.is_file())
assert {'pool.i16','queries.i16','trace.e1gtrc','metadata.json','witness_contract.json'} <= {p.name for p in files}
json.dump({'status':'PASS_CPU_ONLY_BUNDLE_AUDIT','gpu_used':False,'files':{p.name:sha(p) for p in files}},out.open('w'),indent=2,sort_keys=True)
PY

RUN_CARD="$OUT/run_card.json"
STATUS="PREPARING"
STAGE="cpu_preflight"
MESSAGE=""
GPU_UUID=""
RUNNER_EXIT="not_started"
VALIDATOR_EXIT="not_started"
write_card() {
  "$PYTHON" - "$RUN_CARD" <<'PY'
import json, os, sys
from datetime import datetime, timezone
p=sys.argv[1]; e=os.environ
obj={
 'schema':'safe-c1-g1a-run-card-v1','status':e['G1_STATUS'],'updated_at':datetime.now(timezone.utc).isoformat(),
 'scope':'G1A correctness only: top-k, frozen base, no base delete/rebuild/range/timing claim',
 'host':e['G1_HOST'],'gpu_policy':{'physical_gpu':0,'uuid':e.get('G1_UUID') or None,'cuda_visible_devices':'0','no_process_control':True,'max_idle_memory_mib':256},
 'stage':e['G1_STAGE'],'message':e.get('G1_MESSAGE',''),'runner_exit':e.get('G1_RUNNER_EXIT'),'validator_exit':e.get('G1_VALIDATOR_EXIT'),
 'inputs':{'binary':e['G1_BIN'],'bundle':e['G1_BUNDLE'],'static_contract':e['G1_STATIC']},
 'do_not_touch':['A800-1','A800-2','A800-3','GPU 7','any foreign process','original GTS archives']}
json.dump(obj,open(p,'w'),indent=2,sort_keys=True); open(p,'a').write('\n')
PY
}
export G1_HOST="$(hostname)" G1_BIN="$BIN" G1_BUNDLE="$BUNDLE" G1_STATIC="$OUT/preflight/static_contract.json"
export G1_STATUS="$STATUS" G1_STAGE="$STAGE" G1_MESSAGE="$MESSAGE" G1_UUID="$GPU_UUID" G1_RUNNER_EXIT="$RUNNER_EXIT" G1_VALIDATOR_EXIT="$VALIDATOR_EXIT"
write_card
finish_failure() {
  local message="$1"
  STATUS="FAILED"; MESSAGE="$message"
  export G1_STATUS="$STATUS" G1_STAGE="$STAGE" G1_MESSAGE="$MESSAGE" G1_UUID="$GPU_UUID" G1_RUNNER_EXIT="$RUNNER_EXIT" G1_VALIDATOR_EXIT="$VALIDATOR_EXIT"
  write_card
  echo "$message" >&2
  exit 2
}
trap 'rc=$?; if [[ $rc -ne 0 && "$STATUS" != "COMPLETE" && "$STATUS" != "FAILED" ]]; then STATUS="FAILED"; MESSAGE="unexpected launcher failure"; export G1_STATUS="$STATUS" G1_STAGE="$STAGE" G1_MESSAGE="$MESSAGE" G1_UUID="$GPU_UUID" G1_RUNNER_EXIT="$RUNNER_EXIT" G1_VALIDATOR_EXIT="$VALIDATOR_EXIT"; write_card || true; fi' EXIT

# Read-only GPU0 preflight.  No process is stopped or modified.
STAGE="gpu0_idle_preflight"
GPU_ROW="$(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)" || finish_failure "GPU0 telemetry query failed"
printf '%s\n' "$GPU_ROW" | tee "$OUT/logs/preflight_gpu0.csv"
IFS=',' read -r GPU_UUID GPU_MEM GPU_UTIL <<<"$GPU_ROW"
GPU_UUID="$(echo "$GPU_UUID" | xargs)"; GPU_MEM="$(echo "$GPU_MEM" | xargs)"; GPU_UTIL="$(echo "$GPU_UTIL" | xargs)"
[[ "$GPU_UUID" == "$EXPECTED_UUID" ]] || finish_failure "GPU0 UUID mismatch: $GPU_UUID"
[[ "$GPU_MEM" =~ ^[0-9]+$ && "$GPU_UTIL" =~ ^[0-9]+$ ]] || finish_failure "non-numeric GPU0 telemetry"
[[ "$GPU_MEM" -le "$MAX_IDLE_MEMORY_MIB" && "$GPU_UTIL" -eq 0 ]] || finish_failure "GPU0 is not idle (memory=${GPU_MEM}MiB util=${GPU_UTIL}%)"
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/preflight_compute_apps.csv" || true
if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,' "$OUT/logs/preflight_compute_apps.csv"; then
  finish_failure "GPU0 has active compute process(es); no action taken"
fi

STATUS="RUNNING"; STAGE="cuda_g1_runner"; MESSAGE="GPU0 passed idle guard"
export G1_STATUS="$STATUS" G1_STAGE="$STAGE" G1_MESSAGE="$MESSAGE" G1_UUID="$GPU_UUID" G1_RUNNER_EXIT="$RUNNER_EXIT" G1_VALIDATOR_EXIT="$VALIDATOR_EXIT"
write_card
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0
set +e
timeout 900 "$BIN" --bundle "$BUNDLE" --out "$OUT/runner_output/engine_results.jsonl" --summary "$OUT/runner_output/engine_summary.json" --leaf-capacity 2 >"$OUT/logs/runner.stdout.log" 2>"$OUT/logs/runner.stderr.log"
RUNNER_EXIT="$?"
set -e
[[ "$RUNNER_EXIT" == 0 ]] || finish_failure "G1 CUDA runner failed (exit=$RUNNER_EXIT)"

STAGE="cpu_independent_validation"
set +e
CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=none "$PYTHON" "$VALIDATOR" --bundle "$BUNDLE" --engine-jsonl "$OUT/runner_output/engine_results.jsonl" --out "$OUT/validation.json" >"$OUT/logs/validator.stdout.log" 2>"$OUT/logs/validator.stderr.log"
VALIDATOR_EXIT="$?"
set -e
[[ "$VALIDATOR_EXIT" == 0 ]] || finish_failure "independent G1 validator failed (exit=$VALIDATOR_EXIT)"

STAGE="postrun_read_only_check"
nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >"$OUT/logs/postrun_gpu0.csv" || true
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/postrun_compute_apps.csv" || true
STATUS="COMPLETE"; MESSAGE="PASS_G1A_INDEPENDENT_VALIDATION"
export G1_STATUS="$STATUS" G1_STAGE="$STAGE" G1_MESSAGE="$MESSAGE" G1_UUID="$GPU_UUID" G1_RUNNER_EXIT="$RUNNER_EXIT" G1_VALIDATOR_EXIT="$VALIDATOR_EXIT"
write_card
printf 'PASS_G1A_INDEPENDENT_VALIDATION out=%s\n' "$OUT"

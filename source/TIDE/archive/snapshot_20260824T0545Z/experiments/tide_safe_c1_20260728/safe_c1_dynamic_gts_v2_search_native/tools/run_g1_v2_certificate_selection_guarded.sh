#!/usr/bin/env bash
# Guarded one-shot v2 certificate-selection runner. DO NOT execute without explicit approval.
set -Eeuo pipefail

readonly ROOT="/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native"
readonly RUNS_ROOT="/workspace/experiments/tide_safe_c1_20260728/runs"
readonly V1_ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_dynamic_gts_v1"
readonly EXPECTED_HOST="CONFIGURE_ARCHIVE_HOST"
readonly EXPECTED_UUID="GPU-CONFIGURE-ARCHIVE-DEVICE"
readonly PYTHON="${PYTHON:-python3}"
readonly BIN="$ROOT/bin/GTS_safe_c1_g1_v2_search_native"
readonly AUDITOR="$ROOT/tools/audit_g1_static_contract.py"
readonly BUNDLE_PREFLIGHT="$ROOT/tools/preflight_g1_v2_bundle.py"
readonly FINALIZER="$ROOT/tools/finalize_g1_v2_candidate_selection.py"
readonly MAX_IDLE_MEMORY_MIB=256

usage() {
  cat <<'EOF'
Usage: run_g1_v2_certificate_selection_guarded.sh --bundle <fresh-v2-selection-bundle> --out <new-v2-run-root>

Runs exactly one new-v2 certificate-selection trace on 8p GPU0 after:
  1. CPU-only v2 static audit and bounded-tie selection-bundle preflight;
  2. read-only GPU0 UUID/idle/process checks.

It never kills or changes another process. It uses the new v2 binary only,
requires leaf-capacity=1, emits a terminal run card, writes a new
candidate_selection_v2.json, and atomically marks that fresh bundle READY only
if a same-leaf direct group of three validates. Certificate-delta is recorded
only and is not a selection gate. No timing/throughput/capacity result claim is
produced. Do not execute until explicit GPU0 authorization.
EOF
}

BUNDLE=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done
[[ -n "$BUNDLE" && -n "$OUT" ]] || { usage >&2; exit 64; }
[[ -d "$ROOT" && "$ROOT" != "$V1_ROOT" ]] || { echo "invalid v2 root boundary" >&2; exit 64; }
case "$ROOT" in "$V1_ROOT"|"$V1_ROOT"/*) echo "v1 root is forbidden" >&2; exit 64;; esac
case "$RUNS_ROOT" in "$V1_ROOT"|"$V1_ROOT"/*) echo "v1 run boundary is forbidden" >&2; exit 64;; esac
[[ -x "$BIN" && -x "$AUDITOR" && -x "$BUNDLE_PREFLIGHT" && -x "$FINALIZER" ]] || {
  echo "required v2 certificate-selection artifact missing" >&2; exit 66;
}
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || { echo "refuse non-8p host $(hostname)" >&2; exit 77; }
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" != *a800* ]] || { echo "A800 forbidden" >&2; exit 77; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 70; }
command -v timeout >/dev/null || { echo "timeout unavailable" >&2; exit 70; }

BUNDLE="$($PYTHON - "$BUNDLE" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT="$($PYTHON - "$OUT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
case "$BUNDLE" in "$ROOT"/bundles/*) ;; *) echo "bundle must be under new v2 bundles root" >&2; exit 64;; esac
case "$OUT" in "$RUNS_ROOT"/*) ;; *) echo "out must be under new v2 runs root" >&2; exit 64;; esac
[[ ! -e "$OUT" ]] || { echo "out must be a new path; no v1/v2 run reuse" >&2; exit 73; }
[[ ! -e "$BUNDLE/candidate_selection_v2.json" ]] || { echo "bundle already has candidate selection artifact; use a fresh selection bundle" >&2; exit 73; }

mkdir -p "$OUT/logs" "$OUT/preflight" "$OUT/runner_output"
STATUS="PREPARING"
STAGE="cpu_preflight"
MESSAGE=""
GPU_UUID=""
RUNNER_EXIT="not_started"
FINALIZER_EXIT="not_started"
write_card() {
  "$PYTHON" - "$OUT/run_card.json" <<'PY'
import json, os, sys
from datetime import datetime, timezone
p=sys.argv[1]; e=os.environ
obj={
 'schema':'safe-c1-g1-v2-certificate-selection-run-card',
 'status':e['G1V2_STATUS'],
 'updated_at':datetime.now(timezone.utc).isoformat(),
 'scope':'new-v2 certificate selection only; bounded static-prefix/dynamic-boundary witness; isolated leaf capacity one; same-leaf direct group of three required; certificate-delta record-only; no timing/throughput/capacity-result/all-input-tie claim',
 'host':e['G1V2_HOST'],
 'stage':e['G1V2_STAGE'],
 'message':e.get('G1V2_MESSAGE',''),
 'gpu_policy':{'physical_gpu':0,'uuid':e.get('G1V2_UUID') or None,'cuda_visible_devices':'0','read_only_idle_preflight':True,'no_process_control':True,'max_idle_memory_mib':256},
 'inputs':{'binary':e['G1V2_BIN'],'bundle':e['G1V2_BUNDLE'],'static_contract':e['G1V2_STATIC']},
 'runner_exit':e.get('G1V2_RUNNER_EXIT'),'finalizer_exit':e.get('G1V2_FINALIZER_EXIT'),
 'outputs':{'candidate_selection':e.get('G1V2_SELECTION_OUT','')},
 'forbidden_reuse':['v1 selection evidence','v1 bundle schema','v1 run output','foreign GPU process','all-input canonical (distance,stable_id) claim'],
}
json.dump(obj,open(p,'w'),indent=2,sort_keys=True);open(p,'a').write('\n')
PY
}
finish_failure() {
  STATUS="FAILED"; MESSAGE="$1"
  export G1V2_STATUS="$STATUS" G1V2_STAGE="$STAGE" G1V2_MESSAGE="$MESSAGE" G1V2_UUID="$GPU_UUID" G1V2_RUNNER_EXIT="$RUNNER_EXIT" G1V2_FINALIZER_EXIT="$FINALIZER_EXIT"
  write_card
  echo "$MESSAGE" >&2
  exit 2
}
export G1V2_HOST="$(hostname)" G1V2_BIN="$BIN" G1V2_BUNDLE="$BUNDLE" G1V2_STATIC="$OUT/preflight/static_contract_v2.json" G1V2_SELECTION_OUT="$BUNDLE/candidate_selection_v2.json"
export G1V2_STATUS="$STATUS" G1V2_STAGE="$STAGE" G1V2_MESSAGE="$MESSAGE" G1V2_UUID="$GPU_UUID" G1V2_RUNNER_EXIT="$RUNNER_EXIT" G1V2_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
trap 'rc=$?; if [[ $rc -ne 0 && "$STATUS" != "COMPLETE" && "$STATUS" != "FAILED" ]]; then STATUS="FAILED"; MESSAGE="unexpected guarded selection failure"; export G1V2_STATUS="$STATUS" G1V2_STAGE="$STAGE" G1V2_MESSAGE="$MESSAGE" G1V2_UUID="$GPU_UUID" G1V2_RUNNER_EXIT="$RUNNER_EXIT" G1V2_FINALIZER_EXIT="$FINALIZER_EXIT"; write_card || true; fi' EXIT

# CPU-only gates precede every GPU query or CUDA binary invocation.
"$PYTHON" "$AUDITOR" --root "$ROOT" --out "$OUT/preflight/static_contract_v2.json" \
  >"$OUT/logs/static_audit.stdout.log" 2>"$OUT/logs/static_audit.stderr.log" \
  || finish_failure "v2 static contract audit failed"
"$PYTHON" "$BUNDLE_PREFLIGHT" --root "$ROOT" --bundle "$BUNDLE" --mode certificate-selection \
  --out "$OUT/preflight/selection_bundle_preflight.json" \
  >"$OUT/logs/selection_bundle_preflight.stdout.log" 2>"$OUT/logs/selection_bundle_preflight.stderr.log" \
  || finish_failure "fresh v2 certificate-selection bundle preflight failed"

# Read-only GPU0 preflight. No kill, reset, process mutation, or shared-service action.
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

STATUS="RUNNING"; STAGE="new_v2_certificate_selection_runner"; MESSAGE="GPU0 passed read-only idle guard"
export G1V2_STATUS="$STATUS" G1V2_STAGE="$STAGE" G1V2_MESSAGE="$MESSAGE" G1V2_UUID="$GPU_UUID" G1V2_RUNNER_EXIT="$RUNNER_EXIT" G1V2_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0
set +e
timeout 900 "$BIN" --bundle "$BUNDLE" --out "$OUT/runner_output/engine_results.jsonl" \
  --summary "$OUT/runner_output/engine_summary.json" --leaf-capacity 1 \
  >"$OUT/logs/runner.stdout.log" 2>"$OUT/logs/runner.stderr.log"
RUNNER_EXIT="$?"
set -e
[[ "$RUNNER_EXIT" == 0 ]] || finish_failure "new v2 certificate-selection runner failed (exit=$RUNNER_EXIT)"

STATUS="FINALIZING"; STAGE="cpu_candidate_selection_validation"; MESSAGE="validate same-leaf direct group of three and record certificate-delta statistics"
export G1V2_STATUS="$STATUS" G1V2_STAGE="$STAGE" G1V2_MESSAGE="$MESSAGE" G1V2_UUID="$GPU_UUID" G1V2_RUNNER_EXIT="$RUNNER_EXIT" G1V2_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
set +e
"$PYTHON" "$FINALIZER" --root "$ROOT" --bundle "$BUNDLE" --run-root "$OUT" \
  --engine-jsonl "$OUT/runner_output/engine_results.jsonl" \
  --engine-summary "$OUT/runner_output/engine_summary.json" \
  --static-contract "$OUT/preflight/static_contract_v2.json" --binary "$BIN" \
  --out "$BUNDLE/candidate_selection_v2.json" \
  >"$OUT/logs/finalizer.stdout.log" 2>"$OUT/logs/finalizer.stderr.log"
FINALIZER_EXIT="$?"
set -e
[[ "$FINALIZER_EXIT" == 0 ]] || finish_failure "new v2 candidate-selection finalizer failed; bundle remains non-ready"

STAGE="postrun_read_only_check"
nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >"$OUT/logs/postrun_gpu0.csv" || true
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/postrun_compute_apps.csv" || true
STATUS="COMPLETE"; MESSAGE="PASS_G1_V2_CERTIFICATE_SELECTION_NO_PERFORMANCE_CLAIM"
export G1V2_STATUS="$STATUS" G1V2_STAGE="$STAGE" G1V2_MESSAGE="$MESSAGE" G1V2_UUID="$GPU_UUID" G1V2_RUNNER_EXIT="$RUNNER_EXIT" G1V2_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
printf 'PASS_G1_V2_CERTIFICATE_SELECTION out=%s bundle=%s\n' "$OUT" "$BUNDLE"

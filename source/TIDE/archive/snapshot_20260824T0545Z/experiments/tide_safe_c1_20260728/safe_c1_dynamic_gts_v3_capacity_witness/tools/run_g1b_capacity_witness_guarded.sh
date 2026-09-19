#!/usr/bin/env bash
# Guarded one-shot Safe-C1 G1B capacity-two witness runner.
# DO NOT execute without explicit approval. This script never kills/resets or
# otherwise controls another GPU process.
set -Eeuo pipefail

readonly ROOT='/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness'
readonly RUNS_ROOT='/workspace/experiments/tide_safe_c1_20260728/runs'
readonly V1_ROOT='/workspace/experiments/tide_safe_c1_20260727/safe_c1_dynamic_gts_v1'
readonly V2_ROOT='/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native'
readonly EXPECTED_HOST='CONFIGURE_ARCHIVE_HOST'
readonly EXPECTED_UUID='GPU-CONFIGURE-ARCHIVE-DEVICE'
readonly MAX_IDLE_MEMORY_MIB=256
readonly PYTHON="${PYTHON:-python3}"
readonly BIN="$ROOT/bin/GTS_safe_c1_g1b_v3_capacity_witness"
readonly AUDIT="$ROOT/tools/audit_g1b_static_contract.py"
readonly PREFLIGHT="$ROOT/tools/preflight_g1b_capacity_witness.py"
readonly FINALIZER="$ROOT/tools/finalize_g1b_capacity_witness.py"

usage() {
  cat <<'EOF'
Usage: run_g1b_capacity_witness_guarded.sh --execute --bundle <new-v3-bundle> --out <new-run-root>

This is the only GPU-capable G1B command. It requires explicit --execute, then:
  1. runs CPU-only static audit and exact bundle preflight;
  2. performs read-only GPU0 UUID/idle/process checks;
  3. runs exactly the strict capacity=2 trace;
  4. CPU-finalizes receipts and full-active exact oracle.

It does NOT benchmark, rebuild, handle range/base-delete, or establish C3.
No kill, reset, process mutation, or shared-service action is ever performed.
EOF
}

EXECUTE=0
BUNDLE=''
OUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done
[[ "$EXECUTE" == 1 ]] || { echo 'refusing GPU-capable run without explicit --execute' >&2; usage >&2; exit 64; }
[[ -n "$BUNDLE" && -n "$OUT" ]] || { usage >&2; exit 64; }
[[ -d "$ROOT" && "$ROOT" != "$V1_ROOT" && "$ROOT" != "$V2_ROOT" ]] || { echo 'invalid v3 root boundary' >&2; exit 64; }
[[ -x "$BIN" && -x "$AUDIT" && -x "$PREFLIGHT" && -x "$FINALIZER" ]] || { echo 'required v3 artifact missing' >&2; exit 66; }
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || { echo "refuse non-8p host $(hostname)" >&2; exit 77; }
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" != *a800* ]] || { echo 'A800 forbidden' >&2; exit 77; }
command -v timeout >/dev/null || { echo 'timeout unavailable' >&2; exit 70; }

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
case "$BUNDLE" in "$ROOT"/bundles/*) ;; *) echo 'bundle must be under new v3 bundles root' >&2; exit 64 ;; esac
case "$OUT" in "$RUNS_ROOT"/*) ;; *) echo 'out must be under fresh shared runs root' >&2; exit 64 ;; esac
[[ ! -e "$OUT" ]] || { echo 'out must be a fresh path; no v1/v2/v3 run reuse' >&2; exit 73; }
[[ ! -e "$BUNDLE/g1b_capacity_witness_v3.json" ]] || { echo 'bundle already finalized; no rerun/reuse' >&2; exit 73; }

mkdir -p "$OUT/logs" "$OUT/preflight" "$OUT/runner_output"
STATUS='PREPARING'; STAGE='cpu_preflight'; MESSAGE=''; GPU_UUID=''; RUNNER_EXIT='not_started'; FINALIZER_EXIT='not_started'
write_card() {
  "$PYTHON" - "$OUT/run_card.json" <<'PY'
import json, os, sys
from datetime import datetime, timezone
p=sys.argv[1]; e=os.environ
obj={
 'schema':'safe-c1-g1b-v3-capacity-witness-run-card',
 'status':e['G1B_STATUS'], 'updated_at':datetime.now(timezone.utc).isoformat(),
 'scope':'strict capacity-two receipt only; no performance/rebuild/range/base-delete/complete-C3/all-input-tie claim',
 'host':e['G1B_HOST'], 'stage':e['G1B_STAGE'], 'message':e.get('G1B_MESSAGE',''),
 'gpu_policy':{'physical_gpu':0,'uuid':e.get('G1B_UUID') or None,'cuda_visible_devices':'0',
               'read_only_idle_preflight':True,'no_process_control':True,'max_idle_memory_mib':256},
 'inputs':{'binary':e['G1B_BIN'],'bundle':e['G1B_BUNDLE'],'static_contract':e['G1B_STATIC']},
 'runner_exit':e.get('G1B_RUNNER_EXIT'),'finalizer_exit':e.get('G1B_FINALIZER_EXIT'),
 'outputs':{'capacity_witness_result':e['G1B_RESULT_OUT']},
 'forbidden_reuse':['A800-1','A800-2','A800-3','v1 source/bundle/run','v2 result as v3 result',
                    'foreign GPU process control','performance claim','rebuild/range/base-delete/complete-C3 claim'],
}
json.dump(obj,open(p,'w'),indent=2,sort_keys=True);open(p,'a').write('\n')
PY
}
finish_failure() {
  STATUS='FAILED'; MESSAGE="$1"
  export G1B_STATUS="$STATUS" G1B_STAGE="$STAGE" G1B_MESSAGE="$MESSAGE" G1B_UUID="$GPU_UUID" G1B_RUNNER_EXIT="$RUNNER_EXIT" G1B_FINALIZER_EXIT="$FINALIZER_EXIT"
  write_card
  echo "$MESSAGE" >&2
  exit 2
}
export G1B_HOST="$(hostname)" G1B_BIN="$BIN" G1B_BUNDLE="$BUNDLE" G1B_STATIC="$OUT/preflight/static_contract_v3.json" G1B_RESULT_OUT="$BUNDLE/g1b_capacity_witness_v3.json"
export G1B_STATUS="$STATUS" G1B_STAGE="$STAGE" G1B_MESSAGE="$MESSAGE" G1B_UUID="$GPU_UUID" G1B_RUNNER_EXIT="$RUNNER_EXIT" G1B_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
trap 'rc=$?; if [[ $rc -ne 0 && "$STATUS" != COMPLETE && "$STATUS" != FAILED ]]; then STATUS=FAILED; MESSAGE="unexpected guarded G1B failure"; export G1B_STATUS="$STATUS" G1B_STAGE="$STAGE" G1B_MESSAGE="$MESSAGE" G1B_UUID="$GPU_UUID" G1B_RUNNER_EXIT="$RUNNER_EXIT" G1B_FINALIZER_EXIT="$FINALIZER_EXIT"; write_card || true; fi' EXIT

# CPU-only gates must succeed before the script even checks for nvidia-smi.
"$PYTHON" "$AUDIT" --root "$ROOT" --out "$OUT/preflight/static_contract_v3.json" \
  >"$OUT/logs/static_audit.stdout.log" 2>"$OUT/logs/static_audit.stderr.log" \
  || finish_failure 'v3 static contract audit failed'
"$PYTHON" "$PREFLIGHT" --root "$ROOT" --bundle "$BUNDLE" --mode capacity-witness \
  --out "$OUT/preflight/capacity_bundle_preflight.json" \
  >"$OUT/logs/capacity_bundle_preflight.stdout.log" 2>"$OUT/logs/capacity_bundle_preflight.stderr.log" \
  || finish_failure 'fresh v3 capacity bundle preflight failed'

# Read-only GPU0 inspection begins only after every CPU contract passed.
command -v nvidia-smi >/dev/null || finish_failure 'nvidia-smi unavailable after CPU-only gates'
STAGE='gpu0_idle_preflight'
GPU_ROW="$(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)" || finish_failure 'GPU0 telemetry query failed'
printf '%s\n' "$GPU_ROW" | tee "$OUT/logs/preflight_gpu0.csv"
IFS=',' read -r GPU_UUID GPU_MEM GPU_UTIL <<<"$GPU_ROW"
GPU_UUID="$(echo "$GPU_UUID" | xargs)"; GPU_MEM="$(echo "$GPU_MEM" | xargs)"; GPU_UTIL="$(echo "$GPU_UTIL" | xargs)"
[[ "$GPU_UUID" == "$EXPECTED_UUID" ]] || finish_failure "GPU0 UUID mismatch: $GPU_UUID"
[[ "$GPU_MEM" =~ ^[0-9]+$ && "$GPU_UTIL" =~ ^[0-9]+$ ]] || finish_failure 'non-numeric GPU0 telemetry'
[[ "$GPU_MEM" -le "$MAX_IDLE_MEMORY_MIB" && "$GPU_UTIL" -eq 0 ]] || finish_failure "GPU0 not idle (memory=${GPU_MEM}MiB util=${GPU_UTIL}%)"
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/preflight_compute_apps.csv" || true
if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,' "$OUT/logs/preflight_compute_apps.csv"; then
  finish_failure 'GPU0 has active compute process(es); no action taken'
fi

STATUS='RUNNING'; STAGE='strict_g1b_capacity_witness_runner'; MESSAGE='GPU0 passed read-only idle guard'
export G1B_STATUS="$STATUS" G1B_STAGE="$STAGE" G1B_MESSAGE="$MESSAGE" G1B_UUID="$GPU_UUID" G1B_RUNNER_EXIT="$RUNNER_EXIT" G1B_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0
set +e
timeout 900 "$BIN" --bundle "$BUNDLE" --out "$OUT/runner_output/engine_results.jsonl" \
  --summary "$OUT/runner_output/engine_summary.json" \
  --capacity-witness-contract "$BUNDLE/g1b_capacity_contract.txt" --leaf-capacity 2 \
  >"$OUT/logs/runner.stdout.log" 2>"$OUT/logs/runner.stderr.log"
RUNNER_EXIT="$?"
set -e
[[ "$RUNNER_EXIT" == 0 ]] || finish_failure "v3 G1B runner failed (exit=$RUNNER_EXIT)"

STATUS='FINALIZING'; STAGE='independent_full_active_exact_oracle'; MESSAGE='verify capacity receipt and all three exact active sets'
export G1B_STATUS="$STATUS" G1B_STAGE="$STAGE" G1B_MESSAGE="$MESSAGE" G1B_UUID="$GPU_UUID" G1B_RUNNER_EXIT="$RUNNER_EXIT" G1B_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
set +e
"$PYTHON" "$FINALIZER" --root "$ROOT" --bundle "$BUNDLE" --run-root "$OUT" \
  --engine-jsonl "$OUT/runner_output/engine_results.jsonl" --engine-summary "$OUT/runner_output/engine_summary.json" \
  --static-contract "$OUT/preflight/static_contract_v3.json" --binary "$BIN" \
  --out "$BUNDLE/g1b_capacity_witness_v3.json" \
  >"$OUT/logs/finalizer.stdout.log" 2>"$OUT/logs/finalizer.stderr.log"
FINALIZER_EXIT="$?"
set -e
[[ "$FINALIZER_EXIT" == 0 ]] || finish_failure 'v3 G1B finalizer failed; bundle remains pending'

STAGE='postrun_read_only_check'
nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >"$OUT/logs/postrun_gpu0.csv" || true
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/postrun_compute_apps.csv" || true
STATUS='COMPLETE'; MESSAGE='PASS_G1B_CAPACITY_WITNESS_NO_PERFORMANCE_CLAIM'
export G1B_STATUS="$STATUS" G1B_STAGE="$STAGE" G1B_MESSAGE="$MESSAGE" G1B_UUID="$GPU_UUID" G1B_RUNNER_EXIT="$RUNNER_EXIT" G1B_FINALIZER_EXIT="$FINALIZER_EXIT"
write_card
printf 'PASS_G1B_CAPACITY_WITNESS out=%s bundle=%s\n' "$OUT" "$BUNDLE"

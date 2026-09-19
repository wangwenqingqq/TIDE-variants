#!/usr/bin/env bash
# Guarded one-shot G2 v4 strict rebuild witness.  Do not run without --execute.
# It never kills/resets/manipulates another process or shared service.
set -Eeuo pipefail
readonly ROOT='/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v4_rebuild_witness'
readonly RUNS_ROOT='/workspace/experiments/tide_safe_c1_20260728/runs'
readonly EXPECTED_HOST='CONFIGURE_ARCHIVE_HOST'
readonly EXPECTED_UUID='GPU-CONFIGURE-ARCHIVE-DEVICE'
readonly MAX_IDLE_MEMORY_MIB=256
readonly PYTHON="${PYTHON:-python3}"
readonly BIN="$ROOT/bin/GTS_safe_c1_g2_v4_rebuild_witness"
readonly AUDIT="$ROOT/tools/audit_g2_static_contract.py"
readonly PREFLIGHT="$ROOT/tools/preflight_g2_rebuild_witness.py"
readonly FINALIZER="$ROOT/tools/finalize_g2_rebuild_witness.py"
usage() { cat <<'EOF'
Usage: run_g2_rebuild_witness_guarded.sh --execute --bundle <new-v4-bundle> --out <fresh-run-root>

Only GPU-capable G2 command.  After explicit --execute it runs CPU static/bundle
checks, read-only GPU0 UUID/idle/process checks, exactly one strict 9-event
witness, then CPU-only independent finalization.  No benchmark, no timing
claim, no process control, no A800, and no reuse of a run/bundle result.
EOF
}
EXECUTE=0; BUNDLE=''; OUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done
[[ "$EXECUTE" == 1 ]] || { echo 'refusing GPU-capable G2 run without explicit --execute' >&2; exit 64; }
[[ -n "$BUNDLE" && -n "$OUT" ]] || { usage >&2; exit 64; }
[[ -x "$BIN" && -x "$AUDIT" && -x "$PREFLIGHT" && -x "$FINALIZER" ]] || { echo 'required v4 artifact missing' >&2; exit 66; }
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
case "$BUNDLE" in "$ROOT"/bundles/*) ;; *) echo 'bundle must be below v4 bundles root' >&2; exit 64;; esac
case "$OUT" in "$RUNS_ROOT"/*) ;; *) echo 'out must be below fresh shared runs root' >&2; exit 64;; esac
[[ ! -e "$OUT" ]] || { echo 'out must be a fresh run root' >&2; exit 73; }
[[ ! -e "$BUNDLE/engine_results.jsonl" && ! -e "$BUNDLE/g2_final.json" ]] || { echo 'bundle result reuse forbidden' >&2; exit 73; }
mkdir -p "$OUT/logs" "$OUT/preflight" "$OUT/runner_output" "$OUT/final"
STATUS='PREPARING'; STAGE='cpu_preflight'; MESSAGE=''; GPU_UUID=''; RUNNER_EXIT='not_started'; FINALIZER_EXIT='not_started'
write_card() {
  G2_STATUS="$STATUS" G2_STAGE="$STAGE" G2_MESSAGE="$MESSAGE" G2_UUID="$GPU_UUID" G2_RUNNER_EXIT="$RUNNER_EXIT" G2_FINALIZER_EXIT="$FINALIZER_EXIT" G2_HOST="$(hostname)" G2_BIN="$BIN" G2_BUNDLE="$BUNDLE" G2_OUT="$OUT" "$PYTHON" - "$OUT/run_card.json" <<'PY'
import json,os,sys
from datetime import datetime,timezone
e=os.environ
x={'schema':'safe-c1-g2-v4-rebuild-witness-run-card','status':e['G2_STATUS'],'updated_at':datetime.now(timezone.utc).isoformat(),'stage':e['G2_STAGE'],'message':e['G2_MESSAGE'],'scope':'one strict stable-ID immediate rebuild witness; no timing/performance/general rebuild/range/C2/all-input tie claim','host':e['G2_HOST'],'gpu_policy':{'physical_gpu':0,'uuid':e['G2_UUID'] or None,'cuda_visible_devices':'0','read_only_idle_preflight':True,'no_process_control':True,'max_idle_memory_mib':256},'inputs':{'binary':e['G2_BIN'],'bundle':e['G2_BUNDLE']},'outputs':{'run_root':e['G2_OUT']},'runner_exit':e['G2_RUNNER_EXIT'],'finalizer_exit':e['G2_FINALIZER_EXIT'],'forbidden':['A800-1','A800-2','A800-3','v2/v3 mutation or result reuse','foreign process control','performance/general rebuild/range/C2 claim']}
json.dump(x,open(sys.argv[1],'w'),indent=2,sort_keys=True);open(sys.argv[1],'a').write('\n')
PY
}
finish_failure() { STATUS='FAILED'; MESSAGE="$1"; write_card; echo "$MESSAGE" >&2; exit 2; }
write_card
trap 'rc=$?; if [[ $rc -ne 0 && "$STATUS" != COMPLETE && "$STATUS" != FAILED ]]; then STATUS=FAILED; MESSAGE="unexpected guarded G2 failure"; write_card || true; fi' EXIT
# CPU-only gates precede nvidia-smi and the runner.
"$PYTHON" "$AUDIT" --root "$ROOT" --out "$OUT/preflight/g2_static_audit.json" >"$OUT/logs/static_audit.stdout.log" 2>"$OUT/logs/static_audit.stderr.log" || finish_failure 'G2 static audit failed'
"$PYTHON" "$PREFLIGHT" --root "$ROOT" --bundle "$BUNDLE" --out "$OUT/preflight/g2_bundle_preflight.json" >"$OUT/logs/bundle_preflight.stdout.log" 2>"$OUT/logs/bundle_preflight.stderr.log" || finish_failure 'G2 bundle preflight failed'
command -v nvidia-smi >/dev/null || finish_failure 'nvidia-smi unavailable after CPU gates'
STAGE='gpu0_idle_preflight'
GPU_ROW="$(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)" || finish_failure 'GPU0 telemetry failed'
printf '%s\n' "$GPU_ROW" | tee "$OUT/logs/preflight_gpu0.csv"
IFS=',' read -r GPU_UUID GPU_MEM GPU_UTIL <<<"$GPU_ROW"; GPU_UUID="$(echo "$GPU_UUID"|xargs)"; GPU_MEM="$(echo "$GPU_MEM"|xargs)"; GPU_UTIL="$(echo "$GPU_UTIL"|xargs)"
[[ "$GPU_UUID" == "$EXPECTED_UUID" ]] || finish_failure "GPU0 UUID mismatch: $GPU_UUID"
[[ "$GPU_MEM" =~ ^[0-9]+$ && "$GPU_UTIL" =~ ^[0-9]+$ ]] || finish_failure 'non-numeric GPU telemetry'
[[ "$GPU_MEM" -le "$MAX_IDLE_MEMORY_MIB" && "$GPU_UTIL" -eq 0 ]] || finish_failure "GPU0 not idle (memory=${GPU_MEM}MiB util=${GPU_UTIL}%)"
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/preflight_compute_apps.csv" || true
if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,' "$OUT/logs/preflight_compute_apps.csv"; then finish_failure 'GPU0 has active compute processes; no action taken'; fi
STATUS='RUNNING'; STAGE='strict_g2_rebuild_witness_runner'; MESSAGE='GPU0 passed read-only idle guard'; write_card
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0
set +e
timeout 1200 "$BIN" --bundle "$BUNDLE" --out "$OUT/runner_output/engine_results.jsonl" --summary "$OUT/runner_output/engine_summary.json" --g2-rebuild-contract "$BUNDLE/g2_rebuild_contract.txt" --leaf-capacity 1 >"$OUT/logs/runner.stdout.log" 2>"$OUT/logs/runner.stderr.log"
RUNNER_EXIT="$?"; set -e
[[ "$RUNNER_EXIT" == 0 ]] || finish_failure "G2 runner failed (exit=$RUNNER_EXIT)"
STATUS='FINALIZING'; STAGE='independent_cpu_finalizer'; MESSAGE='validate exact E0/E1 IDs, rebuild barrier, active preservation, tiers, and oracle receipts'; write_card
set +e
"$PYTHON" "$FINALIZER" --root "$ROOT" --bundle "$BUNDLE" --preflight "$OUT/preflight/g2_bundle_preflight.json" --engine-results "$OUT/runner_output/engine_results.jsonl" --engine-summary "$OUT/runner_output/engine_summary.json" --out "$OUT/final/g2_final.json" >"$OUT/logs/finalizer.stdout.log" 2>"$OUT/logs/finalizer.stderr.log"
FINALIZER_EXIT="$?"; set -e
[[ "$FINALIZER_EXIT" == 0 ]] || finish_failure 'G2 independent finalizer failed'
STAGE='postrun_read_only_check'
nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >"$OUT/logs/postrun_gpu0.csv" || true
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/postrun_compute_apps.csv" || true
STATUS='COMPLETE'; MESSAGE='PASS_G2_REBUILD_WITNESS_NO_PERFORMANCE_CLAIM'; write_card
printf 'PASS_G2_REBUILD_WITNESS out=%s bundle=%s\n' "$OUT" "$BUNDLE"

#!/usr/bin/env bash
# Runs exactly one E1-GR CUDA Safe-C1 reference trace.
# Hard safety boundary: physical GPU 0 only; no A800 access; GPU 7 prohibited.
# This executes a correctness reference, not an optimized GTS dynamic benchmark.
set -Eeuo pipefail

ROOT="/workspace/experiments/tide_safe_c1_20260727"
SELF="$ROOT/safe_c1_cuda_reference"
BIN="$SELF/bin/safe_c1_cuda_reference"
VERIFY="$ROOT/e1g_adapter/verify_cuda_reference_export.py"
PY="/workspace/legacy_workspace/GTS/bench_env/bin/python"
RUNS_DEFAULT="$ROOT/runs"
BUNDLE=""
MODE=""
RUNS="$RUNS_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
[[ -n "$BUNDLE" && -n "$MODE" ]] || { echo "usage: $0 --bundle DIR --mode safe|buffer [--runs DIR]" >&2; exit 64; }
[[ "$MODE" == safe || "$MODE" == buffer ]] || { echo "--mode must be safe or buffer" >&2; exit 64; }
[[ -x "$BIN" && -f "$VERIFY" && -x "$PY" ]] || { echo "runner/verifier/python unavailable" >&2; exit 64; }
BUNDLE="$(readlink -f "$BUNDLE")"
[[ -f "$BUNDLE/pool.f32" && -f "$BUNDLE/queries.f32" && -f "$BUNDLE/metadata.json" && -f "$BUNDLE/trace.e1gtrc" ]] || {
  echo "bundle lacks Safe-C1 CUDA contract files: $BUNDLE" >&2; exit 64;
}

# Caller input cannot override the only approved device.
export CUDA_VISIBLE_DEVICES=0
[[ "$CUDA_VISIBLE_DEVICES" == 0 ]] || { echo "GPU policy violation" >&2; exit 64; }
STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%Z)"
LABEL="$(basename "$BUNDLE")"
RUN_DIR="$RUNS/e1gr_cuda_safe_c1_reference_${LABEL}_${MODE}_${STAMP}"
LOG="$RUN_DIR/logs"
mkdir -p "$LOG"
START="$(date -Is)"
RESULTS="$RUN_DIR/engine_results.jsonl"
SUMMARY="$RUN_DIR/engine_summary.json"
VERIFY_JSON="$RUN_DIR/independent_verify.json"

fail_card() {
  local stage="$1" message="$2"
  "$PY" - "$RUN_DIR/run_card.json" "$RUN_DIR" "$START" "$stage" "$message" "$MODE" "$BUNDLE" <<'PY'
import json, sys
from datetime import datetime, timezone
out, run_dir, started, stage, message, mode, bundle = sys.argv[1:]
obj = {
  "experiment_id": "E1-GR", "status": "FAILED_OR_BLOCKED", "stage": stage,
  "reason": message, "started_at": started,
  "finished_at": datetime.now(timezone.utc).isoformat(), "run_dir": run_dir,
  "bundle": bundle, "mode": mode, "cuda_visible_devices": "0",
  "physical_gpu_allowed": [0], "forbidden_gpu": [7],
  "scope": "CUDA Safe-C1 reference correctness only; not GTS dynamic integration",
  "a800_hosts_used": []
}
open(out, "w").write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
PY
}
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$RUN_DIR/run_card.json" ]]; then fail_card "shell" "command failed (see logs)"; fi; exit $rc' EXIT

snapshot() {
  local phase="$1"
  nvidia-smi -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version \
    --format=csv,noheader,nounits > "$LOG/${phase}_gpu0.csv"
  nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$LOG/${phase}_compute_apps.csv" 2>&1 || true
}
assert_idle() {
  local phase="$1" row index uuid mem util
  row="$(head -n1 "$LOG/${phase}_gpu0.csv")"
  index="$(awk -F, '{gsub(/^ +| +$/, "", $1); print $1}' <<< "$row")"
  uuid="$(awk -F, '{gsub(/^ +| +$/, "", $2); print $2}' <<< "$row")"
  mem="$(awk -F, '{gsub(/^ +| +$/, "", $4); print $4}' <<< "$row")"
  util="$(awk -F, '{gsub(/^ +| +$/, "", $5); print $5}' <<< "$row")"
  [[ "$index" == 0 && "$uuid" == GPU-* && "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || {
    fail_card "${phase}_guard" "cannot parse physical GPU0 snapshot"; exit 1; }
  (( mem <= 256 && util == 0 )) || { fail_card "${phase}_guard" "GPU0 not idle: mem=${mem}MiB util=${util}%"; exit 1; }
  if grep -Fq "$uuid" "$LOG/${phase}_compute_apps.csv"; then
    fail_card "${phase}_guard" "GPU0 has compute process; refusing shared-process interference"; exit 1
  fi
  printf '%s\n' "$uuid" > "$LOG/${phase}_gpu0_uuid.txt"
}

snapshot preflight
assert_idle preflight
snapshot prerun
assert_idle prerun
GPU_UUID="$(cat "$LOG/prerun_gpu0_uuid.txt")"
printf '%q ' "$BIN" --bundle "$BUNDLE" --trace "$BUNDLE/trace.e1gtrc" --out "$RESULTS" --summary "$SUMMARY" --mode "$MODE" --leaves 16 --leaf-capacity 64 > "$LOG/command.txt"
printf '\n' >> "$LOG/command.txt"
"$BIN" --bundle "$BUNDLE" --trace "$BUNDLE/trace.e1gtrc" --out "$RESULTS" --summary "$SUMMARY" \
  --mode "$MODE" --leaves 16 --leaf-capacity 64 > "$LOG/runner.stdout.log" 2> "$LOG/runner.stderr.log"
"$PY" "$VERIFY" --prepared-bundle "$BUNDLE" --engine-jsonl "$RESULTS" --out "$VERIFY_JSON" \
  > "$LOG/verifier.stdout.log" 2> "$LOG/verifier.stderr.log"
"$PY" - "$VERIFY_JSON" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get("pass"):
    raise SystemExit("independent verifier reported failure")
PY
snapshot postrun

"$PY" - "$RUN_DIR/run_card.json" "$RUN_DIR" "$START" "$BUNDLE" "$MODE" "$GPU_UUID" "$BIN" "$SELF/run_one.sh" "$RESULTS" "$SUMMARY" "$VERIFY_JSON" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
(out, run_dir, started, bundle, mode, uuid, binary, script, results, summary, verify) = sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
metadata=json.load(open(os.path.join(bundle,'metadata.json')))
verification=json.load(open(verify))
engine=json.load(open(summary))
obj={
  "experiment_id":"E1-GR", "status":"PASS_REFERENCE_CORRECTNESS",
  "scope":"CUDA Safe-C1 reference correctness only; not GTS dynamic integration and not a latency result",
  "started_at":started,"finished_at":datetime.now(timezone.utc).isoformat(),
  "host":os.uname().nodename,"run_dir":run_dir,"bundle":bundle,"mode":mode,
  "cuda_visible_devices":"0","physical_gpu":{"index":0,"uuid":uuid},
  "forbidden_gpu":[7],"a800_hosts_used":[],"gpu_used":[0],
  "safety":{"preflight_and_prerun_idle_guards_passed":True,"no_process_control":True,
            "legacy_incremental_updater_used":False,"archived_gts_dynamic_code_used":False},
  "contract":{"schema":metadata.get("schema"),"trace_counts":metadata.get("trace_counts"),"manifest_sha256":sha(os.path.join(bundle,'manifest.json'))},
  "artifacts":{"binary":binary,"binary_sha256":sha(binary),"script":script,"script_sha256":sha(script),
               "engine_results":results,"engine_summary":summary,"independent_verify":verify,
               "verify_pass":verification.get("pass"),"engine_status":engine.get("status")}
}
open(out,'w').write(json.dumps(obj, indent=2, sort_keys=True)+'\n')
PY
trap - EXIT
printf '%s\n' "$RUN_DIR"

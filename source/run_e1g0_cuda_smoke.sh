#!/usr/bin/env bash
# E1-G0 infrastructure-only CUDA runtime smoke.
# It never invokes the GTS archive, Safe-C1 updater, or any benchmark workload.
# It is deliberately pinned to physical GPU 0 and refuses to run if GPU 0 is occupied.
set -Eeuo pipefail

EXP_ROOT="/workspace/experiments/tide_safe_c1_20260727"
PYTHON="/workspace/legacy_workspace/GTS/bench_env/bin/python"
CUDA_HOME="/usr/local/cuda-13.1"
NVCC="$CUDA_HOME/bin/nvcc"
SRC="$EXP_ROOT/harness/e1g0_cuda_smoke.cu"
OUT_ROOT="${1:-$EXP_ROOT/runs}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%Z)"
RUN_DIR="$OUT_ROOT/e1g0_cuda_runtime_smoke_${STAMP}"
LOG_DIR="$RUN_DIR/logs"
BIN_DIR="$RUN_DIR/bin"
RESULT_JSON="$LOG_DIR/result.json"
STATUS_JSON="$RUN_DIR/run_card.json"
STARTED_AT="$(date -Is)"
mkdir -p "$LOG_DIR" "$BIN_DIR"

# Hard boundary. No caller-provided setting can select another GPU.
export CUDA_VISIBLE_DEVICES=0

write_failure() {
  local stage="$1"
  local message="$2"
  "$PYTHON" - "$STATUS_JSON" "$RUN_DIR" "$STARTED_AT" "$stage" "$message" <<'PY'
import json, sys
out, run_dir, started_at, stage, message = sys.argv[1:]
obj = {
  "experiment_id": "E1-G0",
  "scope": "CUDA runtime infrastructure smoke only; not GTS evidence",
  "status": "BLOCKED_OR_FAILED_BEFORE_GTS",
  "started_at": started_at,
  "finished_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
  "run_dir": run_dir,
  "cuda_visible_devices": "0",
  "forbidden_gpus": [7],
  "stage": stage,
  "reason": message,
  "gpu_used": []
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

fail() {
  write_failure "$1" "$2"
  echo "E1-G0 blocked/failed at $1: $2" >&2
  exit 1
}

snapshot_gpu0() {
  local label="$1"
  nvidia-smi -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version \
    --format=csv,noheader,nounits > "$LOG_DIR/${label}_gpu0.csv" \
    || fail "${label}_snapshot" "Unable to read physical GPU 0 state"
  # With no compute process, nvidia-smi can return no rows or a textual no-process notice.
  nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$LOG_DIR/${label}_compute_apps.csv" 2>&1 || true
}

assert_gpu0_idle() {
  local label="$1"
  local row gpu_index gpu_uuid mem_mib util_pct
  row="$(head -n1 "$LOG_DIR/${label}_gpu0.csv" || true)"
  [[ -n "$row" ]] || fail "${label}_guard" "No GPU 0 state row"
  gpu_index="$(awk -F, '{gsub(/^ +| +$/, "", $1); print $1}' <<<"$row")"
  gpu_uuid="$(awk -F, '{gsub(/^ +| +$/, "", $2); print $2}' <<<"$row")"
  mem_mib="$(awk -F, '{gsub(/^ +| +$/, "", $4); print $4}' <<<"$row")"
  util_pct="$(awk -F, '{gsub(/^ +| +$/, "", $5); print $5}' <<<"$row")"
  [[ "$gpu_index" == "0" ]] || fail "${label}_guard" "Expected physical GPU 0, got $gpu_index"
  [[ "$gpu_uuid" == GPU-* ]] || fail "${label}_guard" "Invalid GPU 0 UUID: $gpu_uuid"
  [[ "$mem_mib" =~ ^[0-9]+$ && "$util_pct" =~ ^[0-9]+$ ]] || fail "${label}_guard" "Unparseable GPU0 memory/utilization"
  # <=256 MiB admits only driver bookkeeping; any compute process is an unconditional stop.
  (( mem_mib <= 256 )) || fail "${label}_guard" "GPU 0 memory is ${mem_mib} MiB (>256 MiB safety threshold)"
  (( util_pct == 0 )) || fail "${label}_guard" "GPU 0 utilization is ${util_pct}% (must be 0)"
  if grep -Fq "$gpu_uuid" "$LOG_DIR/${label}_compute_apps.csv"; then
    fail "${label}_guard" "GPU 0 has an active compute process; no shared-process interference allowed"
  fi
  printf '%s\n' "$gpu_uuid" > "$LOG_DIR/${label}_gpu0_uuid.txt"
}

[[ -x "$PYTHON" ]] || fail "environment" "Pinned Python is unavailable"
[[ -x "$NVCC" ]] || fail "environment" "Pinned CUDA toolkit compiler is unavailable: $NVCC"
[[ -f "$SRC" ]] || fail "source" "Smoke source is unavailable: $SRC"
[[ "${CUDA_VISIBLE_DEVICES}" == "0" ]] || fail "gpu_policy" "CUDA_VISIBLE_DEVICES must be exactly 0"

snapshot_gpu0 "prebuild"
assert_gpu0_idle "prebuild"

if ! "$NVCC" -O2 -std=c++17 -arch=sm_120 "$SRC" -o "$BIN_DIR/e1g0_cuda_smoke" \
     > "$LOG_DIR/build.log" 2>&1; then
  fail "build" "nvcc build failed; see logs/build.log"
fi

# Check again directly before the only CUDA runtime call.
snapshot_gpu0 "prerun"
assert_gpu0_idle "prerun"
GPU0_UUID="$(cat "$LOG_DIR/prerun_gpu0_uuid.txt")"

if ! "$BIN_DIR/e1g0_cuda_smoke" > "$RESULT_JSON" 2> "$LOG_DIR/runtime.stderr.log"; then
  snapshot_gpu0 "postrun" || true
  fail "runtime" "CUDA smoke binary failed; see logs/result.json and runtime.stderr.log"
fi
snapshot_gpu0 "postrun"

"$PYTHON" - "$STATUS_JSON" "$RESULT_JSON" "$RUN_DIR" "$STARTED_AT" "$GPU0_UUID" "$SRC" "$BIN_DIR/e1g0_cuda_smoke" "$0" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
(out, result_path, run_dir, started_at, gpu_uuid, source, binary, script) = sys.argv[1:]
with open(result_path, encoding="utf-8") as f:
    result = json.load(f)
if result.get("status") != "PASS":
    raise SystemExit("runtime result JSON does not report PASS")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
obj = {
    "experiment_id": "E1-G0",
    "scope": "CUDA runtime infrastructure smoke only; not GTS evidence",
    "status": "PASS_INFRASTRUCTURE_ONLY",
    "started_at": started_at,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "host": os.uname().nodename,
    "run_dir": run_dir,
    "cuda_visible_devices": "0",
    "physical_gpu": {"index": 0, "uuid": gpu_uuid},
    "forbidden_gpus": [7],
    "gpu_used": [0],
    "safety": {
        "prebuild_and_prerun_idle_guards_passed": True,
        "no_process_control": True,
        "no_archive_or_updater_invoked": True
    },
    "artifacts": {
        "cuda_toolkit": "/usr/local/cuda-13.1", "cuda_arch": "sm_120",
        "source": source, "source_sha256": sha(source),
        "script": script, "script_sha256": sha(script),
        "binary": binary, "binary_sha256": sha(binary),
        "runtime_result": result_path,
        "gpu_prebuild": os.path.join(run_dir, "logs/prebuild_gpu0.csv"),
        "gpu_prerun": os.path.join(run_dir, "logs/prerun_gpu0.csv"),
        "gpu_postrun": os.path.join(run_dir, "logs/postrun_gpu0.csv")
    },
    "runtime_result": result
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo "$RUN_DIR"

#!/usr/bin/env bash
# Guarded E1-GI-B launcher: Safe-C1 CUDA/GTS trace correctness gate.
#
# This launcher is intentionally a TEMPLATE until a runner and an independent
# validator satisfying the contracts below are supplied. It never edits either
# original GTS archive. It writes only below the isolated experiment root.
#
# Runner contract (called with CUDA_VISIBLE_DEVICES=0 only):
#   runner --bundle BUNDLE --out RUNNER_OUTPUT
#   - exports per-query stable-ID top-k rankings/distances and complete range-ID
#     sets for the trace in BUNDLE;
#   - must use the isolated source/build only, never an original archive;
#   - must not bypass CUDA_VISIBLE_DEVICES or select another physical GPU.
#
# Independent validator contract (called CPU-only with CUDA_VISIBLE_DEVICES=-1):
#   validator --bundle BUNDLE --candidate RUNNER_OUTPUT --out VALIDATION_JSON
#   VALIDATION_JSON must contain exactly these correctness counters at minimum:
#   {
#     "status": "PASS",
#     "validator": "independent_exact_oracle",
#     "topk_ranking_mismatches": 0,
#     "range_missing_ids": 0,
#     "range_extra_ids": 0
#   }
# A nonzero counter or missing field is a failed E1-GI-B run, never a pass.
set -Eeuo pipefail

readonly EXP_ROOT="/workspace/experiments/tide_safe_c1_20260727"
readonly RUNS_ROOT="$EXP_ROOT/runs"
readonly PYTHON="/workspace/legacy_workspace/GTS/bench_env/bin/python"
readonly EXPECTED_HOST="CONFIGURE_ARCHIVE_HOST"
readonly MAX_IDLE_MEMORY_MIB=256

usage() {
  cat <<'USAGE'
Usage:
  run_e1gi_b_safe_gts_trace.sh \
    --runner /absolute/path/to/e1gi_runner[.py] \
    --bundle /absolute/path/to/trace_bundle \
    --validator /absolute/path/to/independent_validator[.py] \
    --out /workspace/experiments/tide_safe_c1_20260727/runs/<new-run-dir> \
    [--timeout-seconds 1800]

All supplied runner, bundle, and validator paths must resolve below the isolated
experiment root. --out must resolve below its runs/ directory and must be new
or empty. The launcher hard-pins CUDA_VISIBLE_DEVICES=0 and refuses GPU 0 if it
has an active compute process, nonzero utilization, or >256 MiB used memory.
GPU 7 is forbidden. No A800 host is accepted.
USAGE
}

RUNNER=""
BUNDLE=""
VALIDATOR=""
OUT_DIR=""
TIMEOUT_SECONDS=1800
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner) RUNNER="${2:-}"; shift 2 ;;
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --validator) VALIDATOR="${2:-}"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

[[ -n "$RUNNER" && -n "$BUNDLE" && -n "$VALIDATOR" && -n "$OUT_DIR" ]] || {
  usage >&2
  exit 64
}
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid timeout" >&2; exit 64; }
[[ -x "$PYTHON" ]] || { echo "pinned Python unavailable: $PYTHON" >&2; exit 70; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 70; }
command -v timeout >/dev/null || { echo "timeout unavailable" >&2; exit 70; }

canonical_path() {
  "$PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

RUNNER="$(canonical_path "$RUNNER")"
BUNDLE="$(canonical_path "$BUNDLE")"
VALIDATOR="$(canonical_path "$VALIDATOR")"
OUT_DIR="$(canonical_path "$OUT_DIR")"

require_under() {
  local path="$1" root="$2" label="$3"
  case "$path" in
    "$root"/*) ;;
    *) echo "$label must resolve below $root: $path" >&2; exit 64 ;;
  esac
}
require_under "$RUNNER" "$EXP_ROOT" "runner"
require_under "$BUNDLE" "$EXP_ROOT" "bundle"
require_under "$VALIDATOR" "$EXP_ROOT" "validator"
require_under "$OUT_DIR" "$RUNS_ROOT" "out"
[[ -f "$RUNNER" ]] || { echo "runner is not a file: $RUNNER" >&2; exit 66; }
[[ -e "$BUNDLE" ]] || { echo "bundle missing: $BUNDLE" >&2; exit 66; }
[[ -f "$VALIDATOR" ]] || { echo "validator is not a file: $VALIDATOR" >&2; exit 66; }
[[ "$RUNNER" != "$VALIDATOR" ]] || { echo "runner and independent validator must differ" >&2; exit 64; }
if [[ "$RUNNER" != *.py && ! -x "$RUNNER" ]]; then
  echo "non-Python runner must be executable: $RUNNER" >&2; exit 66
fi
if [[ "$VALIDATOR" != *.py && ! -x "$VALIDATOR" ]]; then
  echo "non-Python validator must be executable: $VALIDATOR" >&2; exit 66
fi
if [[ -e "$OUT_DIR" ]]; then
  if [[ ! -d "$OUT_DIR" ]] || [[ -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "--out must be absent or an empty directory: $OUT_DIR" >&2; exit 73
  fi
fi

HOST="$(hostname)"
LOWER_HOST="$(tr '[:upper:]' '[:lower:]' <<<"$HOST")"
[[ "$LOWER_HOST" != *a800* ]] || { echo "A800 hosts are forbidden: $HOST" >&2; exit 77; }
[[ "$HOST" == "$EXPECTED_HOST" ]] || {
  echo "This E1-GI-B template is locked to 8p host $EXPECTED_HOST, got $HOST" >&2; exit 77
}

# Hard GPU boundary: the runner only sees logical device 0, mapped to physical GPU 0.
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_VISIBLE_DEVICES=0
[[ "$CUDA_VISIBLE_DEVICES" == "0" ]] || { echo "GPU policy breach" >&2; exit 77; }

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/runner_output"
readonly LOG_DIR="$OUT_DIR/logs"
readonly CARD="$OUT_DIR/run_card.json"
readonly RUNNER_OUT="$OUT_DIR/runner_output"
readonly VALIDATION_JSON="$OUT_DIR/validation.json"
readonly STARTED_AT="$(date -Is)"

hash_path() {
  "$PYTHON" - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys
p = Path(sys.argv[1])
h = hashlib.sha256()
if p.is_file():
    h.update(b"F\0")
    h.update(p.name.encode())
    h.update(b"\0")
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
elif p.is_dir():
    h.update(b"D\0")
    for child in sorted((x for x in p.rglob("*") if x.is_file()), key=lambda x: x.relative_to(p).as_posix()):
        rel = child.relative_to(p).as_posix().encode()
        h.update(rel); h.update(b"\0")
        with child.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
else:
    raise SystemExit(f"cannot hash {p}")
print(h.hexdigest())
PY
}

RUNNER_SHA="$(hash_path "$RUNNER")"
BUNDLE_SHA="$(hash_path "$BUNDLE")"
VALIDATOR_SHA="$(hash_path "$VALIDATOR")"
SOURCE_MANIFEST="$EXP_ROOT/source_manifest.sha256"
SOURCE_MANIFEST_SHA="$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"

# State variables are deliberately written into every card, including failures.
RUNNER_EXIT="not_started"
VALIDATOR_EXIT="not_started"
GPU_USED_JSON='[]'
GPU0_UUID=""
STAGE="setup"
MESSAGE=""

write_card() {
  local status="$1"
  "$PYTHON" - "$CARD" "$status" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
card, status = sys.argv[1:]
env = os.environ

def get(name, default=""):
    return env.get(name, default)

def file_hash(path):
    p = os.fspath(path)
    if not os.path.isfile(p): return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
obj = {
  "experiment_id": "E1-GI-B",
  "title": "Safe-C1 CUDA/GTS trace correctness gate",
  "scope": "Correctness gate only; not a performance claim. Pass requires an independent exact oracle.",
  "status": status,
  "started_at": get("E1GI_STARTED_AT"),
  "finished_or_updated_at": datetime.now(timezone.utc).isoformat(),
  "host": {"hostname": get("E1GI_HOST"), "expected_8p_hostname": "CONFIGURE_ARCHIVE_HOST"},
  "do_not_touch": [
      "/workspace/legacy_workspace/GTS (original archive)",
      "/workspace/project/GTS (second original archive)",
      "GPU 7",
      "A800-1, A800-2, A800-3",
      "any other user process or service"
  ],
  "resource_policy": {
      "cuda_visible_devices": "0",
      "physical_gpu_if_runner_started": 0,
      "gpu0_uuid": get("E1GI_GPU0_UUID") or None,
      "forbidden_gpus": [7],
      "a800_hosts_forbidden": True,
      "idle_guard_memory_limit_mib": 256,
      "no_process_control": True
  },
  "inputs": {
      "runner": {"path": get("E1GI_RUNNER"), "tree_sha256": get("E1GI_RUNNER_SHA")},
      "bundle": {"path": get("E1GI_BUNDLE"), "tree_sha256": get("E1GI_BUNDLE_SHA")},
      "validator": {"path": get("E1GI_VALIDATOR"), "tree_sha256": get("E1GI_VALIDATOR_SHA"), "mode": "CUDA_VISIBLE_DEVICES=-1"},
      "source_manifest": {"path": get("E1GI_SOURCE_MANIFEST"), "sha256": get("E1GI_SOURCE_MANIFEST_SHA")}
  },
  "outputs": {
      "runner_output": get("E1GI_RUNNER_OUT"),
      "validation_json": get("E1GI_VALIDATION_JSON"),
      "validator_output_sha256": file_hash(get("E1GI_VALIDATION_JSON")),
      "logs": get("E1GI_LOG_DIR")
  },
  "execution": {
      "timeout_seconds": int(get("E1GI_TIMEOUT", "0")),
      "runner_exit": get("E1GI_RUNNER_EXIT"),
      "validator_exit": get("E1GI_VALIDATOR_EXIT"),
      "gpu_used": json.loads(get("E1GI_GPU_USED_JSON", "[]")),
      "stage": get("E1GI_STAGE"),
      "message": get("E1GI_MESSAGE")
  },
  "required_validator_schema": {
      "status": "PASS",
      "validator": "independent_exact_oracle",
      "topk_ranking_mismatches": 0,
      "range_missing_ids": 0,
      "range_extra_ids": 0
  }
}
with open(card, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

export E1GI_STARTED_AT="$STARTED_AT" E1GI_HOST="$HOST" E1GI_RUNNER="$RUNNER" E1GI_BUNDLE="$BUNDLE"
export E1GI_VALIDATOR="$VALIDATOR" E1GI_RUNNER_SHA="$RUNNER_SHA" E1GI_BUNDLE_SHA="$BUNDLE_SHA"
export E1GI_VALIDATOR_SHA="$VALIDATOR_SHA" E1GI_SOURCE_MANIFEST="$SOURCE_MANIFEST"
export E1GI_SOURCE_MANIFEST_SHA="$SOURCE_MANIFEST_SHA" E1GI_RUNNER_OUT="$RUNNER_OUT"
export E1GI_VALIDATION_JSON="$VALIDATION_JSON" E1GI_LOG_DIR="$LOG_DIR" E1GI_TIMEOUT="$TIMEOUT_SECONDS"

fail() {
  STAGE="$1"; MESSAGE="$2"
  export E1GI_STAGE="$STAGE" E1GI_MESSAGE="$MESSAGE" E1GI_RUNNER_EXIT="$RUNNER_EXIT"
  export E1GI_VALIDATOR_EXIT="$VALIDATOR_EXIT" E1GI_GPU_USED_JSON="$GPU_USED_JSON" E1GI_GPU0_UUID="$GPU0_UUID"
  write_card "FAILED_OR_BLOCKED"
  echo "E1-GI-B blocked/failed at $STAGE: $MESSAGE" >&2
  exit 1
}

snapshot_gpu0() {
  local label="$1"
  nvidia-smi -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version \
    --format=csv,noheader,nounits > "$LOG_DIR/${label}_gpu0.csv" \
    || fail "${label}_snapshot" "unable to read physical GPU 0 state"
  nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits > "$LOG_DIR/${label}_compute_apps.csv" 2>&1 || true
}

assert_gpu0_idle() {
  local label="$1" row gpu_index gpu_uuid mem_mib util_pct
  row="$(head -n1 "$LOG_DIR/${label}_gpu0.csv" || true)"
  [[ -n "$row" ]] || fail "${label}_guard" "no GPU 0 state row"
  gpu_index="$(awk -F, '{gsub(/^ +| +$/, "", $1); print $1}' <<<"$row")"
  gpu_uuid="$(awk -F, '{gsub(/^ +| +$/, "", $2); print $2}' <<<"$row")"
  mem_mib="$(awk -F, '{gsub(/^ +| +$/, "", $4); print $4}' <<<"$row")"
  util_pct="$(awk -F, '{gsub(/^ +| +$/, "", $5); print $5}' <<<"$row")"
  [[ "$gpu_index" == "0" && "$gpu_uuid" == GPU-* ]] || fail "${label}_guard" "unexpected GPU0 identity"
  [[ "$mem_mib" =~ ^[0-9]+$ && "$util_pct" =~ ^[0-9]+$ ]] || fail "${label}_guard" "unparseable GPU0 state"
  (( mem_mib <= MAX_IDLE_MEMORY_MIB )) || fail "${label}_guard" "GPU0 memory ${mem_mib} MiB exceeds safe idle threshold"
  (( util_pct == 0 )) || fail "${label}_guard" "GPU0 utilization ${util_pct}% is not idle"
  if grep -Fq "$gpu_uuid" "$LOG_DIR/${label}_compute_apps.csv"; then
    fail "${label}_guard" "GPU0 has an active compute process; no interference is allowed"
  fi
  GPU0_UUID="$gpu_uuid"
  printf '%s\n' "$gpu_uuid" > "$LOG_DIR/${label}_gpu0_uuid.txt"
}

invoke_runner() {
  if [[ "$RUNNER" == *.py ]]; then
    timeout --preserve-status "${TIMEOUT_SECONDS}s" env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0 \
      "$PYTHON" "$RUNNER" --bundle "$BUNDLE" --out "$RUNNER_OUT"
  else
    timeout --preserve-status "${TIMEOUT_SECONDS}s" env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0 \
      "$RUNNER" --bundle "$BUNDLE" --out "$RUNNER_OUT"
  fi
}

invoke_validator() {
  if [[ "$VALIDATOR" == *.py ]]; then
    timeout --preserve-status "${TIMEOUT_SECONDS}s" env CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=none \
      "$PYTHON" "$VALIDATOR" --bundle "$BUNDLE" --candidate "$RUNNER_OUT" --out "$VALIDATION_JSON"
  else
    timeout --preserve-status "${TIMEOUT_SECONDS}s" env CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=none \
      "$VALIDATOR" --bundle "$BUNDLE" --candidate "$RUNNER_OUT" --out "$VALIDATION_JSON"
  fi
}

validate_schema() {
  "$PYTHON" - "$VALIDATION_JSON" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    x = json.load(f)
required = {
    "status": "PASS",
    "validator": "independent_exact_oracle",
    "topk_ranking_mismatches": 0,
    "range_missing_ids": 0,
    "range_extra_ids": 0,
}
for k, expected in required.items():
    if k not in x:
        raise SystemExit(f"validator JSON missing required key: {k}")
    if x[k] != expected:
        raise SystemExit(f"validator JSON {k}={x[k]!r}, expected {expected!r}")
PY
}

# Preflight guard before the runner can be launched.
snapshot_gpu0 "preflight"
assert_gpu0_idle "preflight"
STAGE="preflight_complete"; MESSAGE="GPU0 idle guard passed; runner not started"
export E1GI_STAGE="$STAGE" E1GI_MESSAGE="$MESSAGE" E1GI_RUNNER_EXIT="$RUNNER_EXIT"
export E1GI_VALIDATOR_EXIT="$VALIDATOR_EXIT" E1GI_GPU_USED_JSON="$GPU_USED_JSON" E1GI_GPU0_UUID="$GPU0_UUID"
write_card "PREFLIGHT_COMPLETE_RUNNER_NOT_STARTED"

# Second guard immediately before the only CUDA runner invocation.
snapshot_gpu0 "prerun"
assert_gpu0_idle "prerun"
STAGE="runner"; MESSAGE="runner started with CUDA_VISIBLE_DEVICES=0"
RUNNER_EXIT="running"; GPU_USED_JSON='[0]'
export E1GI_STAGE="$STAGE" E1GI_MESSAGE="$MESSAGE" E1GI_RUNNER_EXIT="$RUNNER_EXIT"
export E1GI_VALIDATOR_EXIT="$VALIDATOR_EXIT" E1GI_GPU_USED_JSON="$GPU_USED_JSON" E1GI_GPU0_UUID="$GPU0_UUID"
write_card "RUNNER_STARTED"

set +e
invoke_runner > "$LOG_DIR/runner.stdout.log" 2> "$LOG_DIR/runner.stderr.log"
RUNNER_RC=$?
set -e
RUNNER_EXIT="$RUNNER_RC"
snapshot_gpu0 "postrun"
if (( RUNNER_RC != 0 )); then
  fail "runner" "runner exited $RUNNER_RC; see logs/runner.stdout.log and logs/runner.stderr.log"
fi

STAGE="validator"; MESSAGE="CPU-only independent validator started"
VALIDATOR_EXIT="running"
export E1GI_STAGE="$STAGE" E1GI_MESSAGE="$MESSAGE" E1GI_RUNNER_EXIT="$RUNNER_EXIT"
export E1GI_VALIDATOR_EXIT="$VALIDATOR_EXIT" E1GI_GPU_USED_JSON="$GPU_USED_JSON" E1GI_GPU0_UUID="$GPU0_UUID"
write_card "VALIDATOR_STARTED"

set +e
invoke_validator > "$LOG_DIR/validator.stdout.log" 2> "$LOG_DIR/validator.stderr.log"
VALIDATOR_RC=$?
set -e
VALIDATOR_EXIT="$VALIDATOR_RC"
if (( VALIDATOR_RC != 0 )); then
  fail "validator" "validator exited $VALIDATOR_RC; see logs/validator.stdout.log and logs/validator.stderr.log"
fi
if ! validate_schema > "$LOG_DIR/validator_schema_check.log" 2>&1; then
  fail "validator_schema" "validator output did not prove zero mismatches; see logs/validator_schema_check.log"
fi

STAGE="complete"; MESSAGE="runner and independent exact-oracle validator passed"
export E1GI_STAGE="$STAGE" E1GI_MESSAGE="$MESSAGE" E1GI_RUNNER_EXIT="$RUNNER_EXIT"
export E1GI_VALIDATOR_EXIT="$VALIDATOR_EXIT" E1GI_GPU_USED_JSON="$GPU_USED_JSON" E1GI_GPU0_UUID="$GPU0_UUID"
write_card "PASS_VALIDATED_E1_GI_B"
printf '%s\n' "$OUT_DIR"

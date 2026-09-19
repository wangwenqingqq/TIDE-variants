#!/bin/bash
# Future fail-closed outer guard for Safe-C2 v3.  --verify-source-only is the
# only presently usable mode; --execute is intentionally gated and is not run
# by static checks.  This script never kills or modifies pre-existing processes.
set -Eeuo pipefail
PATH=/usr/bin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH BASH_ENV ENV CDPATH

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3
HOST_REQUIRED=CONFIGURE_ARCHIVE_HOST
PYTHON=/usr/bin/python3
BINARY="$ROOT/bin/GTS_safe_c2_speculative_fallback_v3_sift1m"
PLAN="$ROOT/protocols/c2_v3_execution_plan_v1.json"
PINS="$ROOT/provenance/c2_v3_execution_pins_v1.json"
PIN_VERIFY="$ROOT/tools/verify_c2_v3_execution_pins.py"
MANIFEST_TOOL="$ROOT/tools/c2_v3_execution_manifest.py"
ARTIFACT_VERIFY="$ROOT/tools/verify_c2_v3_execution_artifacts.py"
LAUNCHER="$ROOT/tools/launch_c2_v3_execution.sh"
WORKLOAD="$ROOT/inputs/sift_learn_compact10k_v1/workload_manifest.json"
LOCK="$ROOT/.locks/c2_v3_physical_gpu0.lock"
GPU0_MAX_MEMORY_USED_MIB=256
GPU0_IDLE_POLL_ATTEMPTS=15
GPU0_IDLE_POLL_SECONDS=2

OUT="${C2_V3_RUN_OUT:-}"
GPU0_UUID=""
GPU0_PCI=""
CURRENT_CLAIMED_STAGE=""
EXECUTE_MODE=0
SUCCESS=0
OUTER_EXIT_RECORDED=0
CHILD_WRAPPER_PID=""
CHILD_SESSION_PID=""
CHILD_PGID=""
CHILD_SID=""
CHILD_STAGE=""
CHILD_PHASE=""
CHILD_READY_FILE=""
CHILD_GO_FILE=""
CHILD_ABORT_FILE=""

usage() {
  cat <<'USAGE'
Usage:
  run_c2_v3_execution_guard.sh --verify-source-only
  (only via the pinned human launcher) run_c2_v3_execution_guard.sh --execute

--verify-source-only hashes/verifies only CPU-readable trust inputs.  It does
not call nvidia-smi and cannot execute the CUDA binary.
--execute remains a future explicit opt-in path and consumes an irrevocable
root workflow/stage ledger before output creation or GPU telemetry.
USAGE
}

die() {
  echo "SAFE-C2-V3 outer guard FAIL: $*" >&2
  exit 64
}

trim() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

normalize_nonblank_lines() {
  local raw=$1
  printf '%s\n' "$raw" | /usr/bin/awk '{ sub(/\r$/, ""); if ($0 ~ /[^[:space:]]/) print }'
}

require_exactly_one_nonblank_line() {
  local label=$1 raw=$2 clean
  clean="$(normalize_nonblank_lines "$raw")"
  [[ -n "$clean" ]] || die "$label returned no nonblank row"
  [[ "$clean" != *$'\n'* ]] || die "$label returned more than one nonblank row"
  printf '%s' "$clean"
}

reject_nvidia_error_text() {
  local label=$1 text=$2
  case "$text" in
    *"No devices were found"*|*"Failed to initialize NVML"*|*"NVIDIA-SMI has failed"*|*"Unknown Error"*)
      die "$label returned NVIDIA error text: $text" ;;
  esac
}

sha256_of() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || die "not a direct regular file: $path"
  /usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}'
}

safe_python() {
  /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C "$PYTHON" "$@"
}

verify_source_only() {
  if [[ -n "${1:-}" ]]; then
    safe_python "$PIN_VERIFY" --source-only --pins "$PINS" --expected-pins-sha "$1"
  else
    safe_python "$PIN_VERIFY" --source-only --pins "$PINS"
  fi
}

verify_launcher_chain() {
  [[ "${C2_V3_TRUST_LAUNCHER:-}" == "SAFE_C2_V3_LAUNCHER_V1" ]] || die 'missing launcher trust marker'
  [[ "${C2_V3_EXECUTE_ACK:-}" == "I_CONFIRM_ONE_IRREVOCABLE_C2_V3_WORKFLOW" ]] || die 'missing explicit irrevocable-workflow acknowledgement'
  [[ "${C2_V3_TRUST_LAUNCHER_PATH:-}" == "$LAUNCHER" ]] || die 'launcher path marker mismatch'
  [[ "$(/usr/bin/readlink -f "$0")" == "$ROOT/tools/run_c2_v3_execution_guard.sh" ]] || die 'guard invoked through unexpected path'
  [[ "$(/usr/bin/readlink -f "$LAUNCHER")" == "$LAUNCHER" && ! -L "$LAUNCHER" ]] || die 'launcher is not a direct canonical file'
  [[ "${C2_V3_TRUST_GUARD_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || die 'guard SHA marker malformed'
  [[ "${C2_V3_TRUST_PINS_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || die 'PINS SHA marker malformed'
  [[ "${C2_V3_TRUST_PINS_VERIFY_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || die 'PINS verifier SHA marker malformed'
  [[ "$(sha256_of "$ROOT/tools/run_c2_v3_execution_guard.sh")" == "$C2_V3_TRUST_GUARD_SHA" ]] || die 'outer guard SHA mismatch'
  [[ "$(sha256_of "$PINS")" == "$C2_V3_TRUST_PINS_SHA" ]] || die 'PINS SHA mismatch'
  [[ "$(sha256_of "$PIN_VERIFY")" == "$C2_V3_TRUST_PINS_VERIFY_SHA" ]] || die 'PINS verifier SHA mismatch'
  verify_source_only "$C2_V3_TRUST_PINS_SHA" >/dev/null
}

physical_nvidia_smi() {
  [[ -x /usr/bin/nvidia-smi ]] || die '/usr/bin/nvidia-smi is unavailable'
  /usr/bin/env -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES /usr/bin/nvidia-smi "$@"
}

record_telemetry_json() {
  local label=$1 status_raw=$2 apps_raw=$3 name=$4 driver=$5 temp=$6 sm_clock=$7 mem_clock=$8 power_draw=$9 power_limit=${10} pstate=${11} total=${12} used=${13} util=${14} strict=${15}
  local json="$OUT/telemetry/gpu0_${label}.json"
  safe_python - "$json" "$label" "$GPU0_UUID" "$GPU0_PCI" "$status_raw" "$apps_raw" "$name" "$driver" "$temp" "$sm_clock" "$mem_clock" "$power_draw" "$power_limit" "$pstate" "$total" "$used" "$util" "$strict" <<'PY'
import hashlib, json, pathlib, sys
out, label, uuid, pci, status_raw, apps_raw, name, driver, temp, sm_clock, mem_clock, power_draw, power_limit, pstate, total, used, util, strict = sys.argv[1:]
def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
payload = {
    "schema": "safe-c2-v3-gpu0-telemetry-v1",
    "label": label,
    "physical_gpu_index": 0,
    "physical_gpu_uuid": uuid,
    "pci_bus_id": pci,
    "status_raw_path": status_raw,
    "status_raw_sha256": digest(status_raw),
    "compute_apps_raw_path": apps_raw,
    "compute_apps_raw_sha256": digest(apps_raw),
    "gpu_name": name,
    "driver_version": driver,
    "temperature_c": float(temp),
    "sm_clock_mhz": float(sm_clock),
    "mem_clock_mhz": float(mem_clock),
    "power_draw_w": float(power_draw),
    "power_limit_w": float(power_limit),
    "pstate": pstate,
    "memory_total_mib": int(total),
    "memory_used_mib": int(used),
    "utilization_percent": int(util),
    "compute_processes": "none",
    "strict_idle_required": strict == "true",
}
pathlib.Path(out).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
  safe_python "$MANIFEST_TOOL" record-telemetry --run "$OUT" --label "$label" --file "$json" >/dev/null
}

# Requires one physical GPU0 row, exact UUID/PCI continuity, and explicitly no
# compute applications.  On strict=true, memory/utilization must additionally
# be idle.  This function is reachable only from --execute.
snapshot_gpu0() {
  local label=$1 strict=$2
  local status_raw="$OUT/telemetry/gpu0_${label}.status.csv"
  local apps_raw="$OUT/telemetry/gpu0_${label}.compute_apps.csv"
  local status line apps clean index uuid pci name driver temp sm_clock mem_clock power_draw power_limit pstate total used util extra float_re
  float_re='^[0-9]+([.][0-9]+)?$'
  if ! status="$(physical_nvidia_smi --id=0 --query-gpu=index,uuid,pci.bus_id,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,pstate,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
    printf '%s\n' "$status" >"$status_raw"
    die "cannot query physical GPU0 status: $label"
  fi
  printf '%s\n' "$status" >"$status_raw"
  line="$(require_exactly_one_nonblank_line "GPU0 status $label" "$status")"
  reject_nvidia_error_text "GPU0 status $label" "$line"
  IFS=, read -r index uuid pci name driver temp sm_clock mem_clock power_draw power_limit pstate total used util extra <<<"$line"
  index="$(trim "$index")"; uuid="$(trim "$uuid")"; pci="$(trim "$pci")"; name="$(trim "$name")"; driver="$(trim "$driver")"
  temp="$(trim "$temp")"; sm_clock="$(trim "$sm_clock")"; mem_clock="$(trim "$mem_clock")"; power_draw="$(trim "$power_draw")"; power_limit="$(trim "$power_limit")"; pstate="$(trim "$pstate")"
  total="$(trim "$total")"; used="$(trim "$used")"; util="$(trim "$util")"
  [[ -z "${extra:-}" && "$index" == 0 && "$uuid" =~ ^GPU-[A-Za-z0-9-]+$ && -n "$pci" && -n "$name" && "$driver" =~ ^[0-9]+([.][0-9]+)+$ && "$pstate" =~ ^P[0-9]+$ ]] || die "malformed physical GPU0 status: $line"
  [[ "$total" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$total" -gt 0 && "$temp" =~ $float_re && "$sm_clock" =~ $float_re && "$mem_clock" =~ $float_re && "$power_draw" =~ $float_re && "$power_limit" =~ $float_re ]] || die "non-numeric physical GPU0 status: $line"
  if [[ -z "$GPU0_UUID" ]]; then
    GPU0_UUID="$uuid"; GPU0_PCI="$pci"
  else
    [[ "$uuid" == "$GPU0_UUID" && "$pci" == "$GPU0_PCI" ]] || die "physical GPU0 UUID/PCI changed during guarded run"
  fi
  if ! apps="$(physical_nvidia_smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)"; then
    printf '%s\n' "$apps" >"$apps_raw"
    die "cannot query physical GPU0 compute apps: $label"
  fi
  printf '%s\n' "$apps" >"$apps_raw"
  clean="$(normalize_nonblank_lines "$apps")"
  reject_nvidia_error_text "GPU0 compute apps $label" "$clean"
  case "$clean" in
    ''|'No running processes found'|'No running compute processes found') ;;
    *) die "physical GPU0 has active/unknown compute application state ($label): $clean" ;;
  esac
  if [[ "$strict" == true ]]; then
    [[ "$used" -le "$GPU0_MAX_MEMORY_USED_MIB" && "$util" -eq 0 ]] || return 1
  fi
  record_telemetry_json "$label" "$status_raw" "$apps_raw" "$name" "$driver" "$temp" "$sm_clock" "$mem_clock" "$power_draw" "$power_limit" "$pstate" "$total" "$used" "$util" "$strict"
  return 0
}

wait_for_strict_idle() {
  local label=$1 attempt
  for ((attempt=1; attempt<=GPU0_IDLE_POLL_ATTEMPTS; ++attempt)); do
    if snapshot_gpu0 "$label" true; then
      return 0
    fi
    if [[ "$attempt" -lt "$GPU0_IDLE_POLL_ATTEMPTS" ]]; then /usr/bin/sleep "$GPU0_IDLE_POLL_SECONDS"; fi
  done
  die "physical GPU0 failed strict-idle requirement before $label"
}

write_runtime_environment() {
  local env_file="$OUT/runtime_child_environment.json"
  safe_python - "$env_file" "$GPU0_UUID" <<'PY'
import json, pathlib, sys
path, uuid = sys.argv[1:]
payload = {
  "schema": "safe-c2-v3-child-environment-v1",
  "environment": {
    "PATH": "/usr/bin:/bin",
    "HOME": "/root",
    "LANG": "C",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": uuid,
    "NVIDIA_VISIBLE_DEVICES": uuid,
    "LD_LIBRARY_PATH": "/usr/local/cuda-13.1/lib64",
  },
}
pathlib.Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
  safe_python "$MANIFEST_TOOL" record-runtime-binding --run "$OUT" --gpu-uuid "$GPU0_UUID" --pci-bus-id "$GPU0_PCI" --environment-file "$env_file" >/dev/null
}

clear_child_tracking() {
  CHILD_WRAPPER_PID=""; CHILD_SESSION_PID=""; CHILD_PGID=""; CHILD_SID=""
  CHILD_STAGE=""; CHILD_PHASE=""; CHILD_READY_FILE=""; CHILD_GO_FILE=""; CHILD_ABORT_FILE=""
}

verify_own_child_session() {
  local expected_phase=$1 line pid pgid sid cmd exe
  [[ -n "$CHILD_SESSION_PID" && "$CHILD_SESSION_PID" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$CHILD_SESSION_PID" 2>/dev/null || return 1
  line="$(/usr/bin/ps -o pid=,pgid=,sid= -p "$CHILD_SESSION_PID" 2>/dev/null | /usr/bin/awk 'NF {print $1 "," $2 "," $3}')"
  IFS=, read -r pid pgid sid <<<"$line"
  [[ "$pid" == "$CHILD_SESSION_PID" && "$pgid" == "$CHILD_SESSION_PID" && "$sid" == "$CHILD_SESSION_PID" ]] || return 1
  [[ "$CHILD_PGID" == "$pgid" && "$CHILD_SID" == "$sid" ]] || return 1
  if [[ "$expected_phase" == bootstrap ]]; then
    cmd="$(/usr/bin/tr '\0' ' ' < "/proc/$CHILD_SESSION_PID/cmdline" 2>/dev/null || true)"
    [[ "$cmd" == *safe-c2-v3-bootstrap* && "$cmd" == *"$BINARY"* && "$cmd" == *"$CHILD_READY_FILE"* ]] || return 1
  else
    exe="$(/usr/bin/readlink -f "/proc/$CHILD_SESSION_PID/exe" 2>/dev/null || true)"
    [[ "$exe" == "$BINARY" ]] || return 1
  fi
  return 0
}

adopt_own_child_session_from_ready() {
  local -a lines=()
  local pid pgid sid attempt
  for ((attempt=1; attempt<=100; ++attempt)); do
    if [[ -s "$CHILD_READY_FILE" ]]; then
      mapfile -t lines < "$CHILD_READY_FILE" || true
      if [[ "${#lines[@]}" -eq 3 && "${lines[0]}" =~ ^pid=[0-9]+$ && "${lines[1]}" =~ ^pgid=[0-9]+$ && "${lines[2]}" =~ ^sid=[0-9]+$ ]]; then
        pid="${lines[0]#pid=}"; pgid="${lines[1]#pgid=}"; sid="${lines[2]#sid=}"
        CHILD_SESSION_PID="$pid"; CHILD_PGID="$pgid"; CHILD_SID="$sid"; CHILD_PHASE=bootstrap
        if verify_own_child_session bootstrap; then return 0; fi
      fi
    fi
    if [[ -n "$CHILD_WRAPPER_PID" ]] && ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null; then
      return 1
    fi
    /usr/bin/sleep 0.05
  done
  return 1
}

confirm_own_binary_session_or_exit() {
  local attempt
  for ((attempt=1; attempt<=100; ++attempt)); do
    if ! kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then return 0; fi
    if verify_own_child_session binary; then CHILD_PHASE=binary; return 0; fi
    /usr/bin/sleep 0.05
  done
  return 1
}

write_cleanup_provenance() {
  local reason=$1 verified=$2 action=$3 alive=$4 path
  [[ -n "$OUT" && -d "$OUT" && -n "$CHILD_STAGE" ]] || return 0
  path="$OUT/own_child_cleanup_${CHILD_STAGE}.json"
  safe_python - "$path" "$reason" "$CHILD_STAGE" "$CHILD_SESSION_PID" "$CHILD_PGID" "$CHILD_SID" "$verified" "$action" "$alive" <<'PY'
import json, pathlib, sys
path, reason, stage, pid, pgid, sid, verified, action, alive = sys.argv[1:]
payload = {
  "schema": "safe-c2-v3-own-child-cleanup-v1",
  "reason": reason,
  "stage": stage,
  "pid": int(pid) if pid.isdigit() else None,
  "pgid": int(pgid) if pgid.isdigit() else None,
  "sid": int(sid) if sid.isdigit() else None,
  "verified_owned_session": verified == "true",
  "action": action,
  "still_alive_after_cleanup": alive == "true",
}
pathlib.Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
  safe_python "$MANIFEST_TOOL" record-cleanup --run "$OUT" --stage "$CHILD_STAGE" --file "$path" >/dev/null 2>&1 || true
}

own_wrapper_is_verified() {
  local exe cmd
  [[ -n "$CHILD_WRAPPER_PID" && "$CHILD_WRAPPER_PID" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null || return 1
  exe="$(/usr/bin/readlink -f "/proc/$CHILD_WRAPPER_PID/exe" 2>/dev/null || true)"
  cmd="$(/usr/bin/tr '\0' ' ' < "/proc/$CHILD_WRAPPER_PID/cmdline" 2>/dev/null || true)"
  [[ ( "$exe" == /usr/bin/setsid || "$exe" == /usr/bin/bash || "$exe" == /bin/bash ) && "$cmd" == *safe-c2-v3-bootstrap* && "$cmd" == *"$CHILD_READY_FILE"* && "$cmd" == *"$CHILD_GO_FILE"* && "$cmd" == *"$CHILD_ABORT_FILE"* && "$cmd" == *"$BINARY"* ]]
}

abort_pre_adoption_and_wait() {
  local action=$1 attempt
  [[ -z "$CHILD_ABORT_FILE" ]] || : > "$CHILD_ABORT_FILE" 2>/dev/null || true
  # Before GO exists the bootstrap is guaranteed never to exec the CUDA binary;
  # an abort file is therefore the primary safe rollback.  The wrapper PID is
  # only signaled after /proc verifies it is our exact setsid-controlled bootstrap PID.
  for ((attempt=1; attempt<=100; ++attempt)); do
    [[ -n "$CHILD_WRAPPER_PID" ]] && ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null && return 0
    /usr/bin/sleep 0.05
  done
  if own_wrapper_is_verified; then
    kill -TERM "$CHILD_WRAPPER_PID" 2>/dev/null || true
    for ((attempt=1; attempt<=100; ++attempt)); do
      ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null && return 0
      /usr/bin/sleep 0.05
    done
    kill -KILL "$CHILD_WRAPPER_PID" 2>/dev/null || true
    return 0
  fi
  return 1
}

cleanup_own_child() {
  local reason=$1 verified=false action=NO_LIVE_CHILD alive=false attempt preadopt=false
  # This covers the launch->ready/adopt race.  Abort is created even if the
  # session PID has not yet appeared; without GO that bootstrap cannot enter
  # the binary.  It never signals an unverified process group.
  if [[ -z "$CHILD_SESSION_PID" || "$CHILD_PHASE" == bootstrap ]]; then
    preadopt=true
    if abort_pre_adoption_and_wait; then
      action=ABORT_PRE_ADOPTION_THEN_OWN_WRAPPER_WAIT_OR_SIGNAL
    else
      action=ABORT_PRE_ADOPTION_REFUSED_UNVERIFIED_WRAPPER_SIGNAL
    fi
  fi
  if [[ -n "$CHILD_SESSION_PID" ]] && kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then
    if verify_own_child_session binary || verify_own_child_session bootstrap; then
      verified=true
      [[ -z "$CHILD_ABORT_FILE" ]] || : > "$CHILD_ABORT_FILE" 2>/dev/null || true
      kill -TERM -- "-$CHILD_PGID" 2>/dev/null || true
      action=TERM_OWN_VERIFIED_PROCESS_GROUP
      for ((attempt=1; attempt<=100; ++attempt)); do
        kill -0 "$CHILD_SESSION_PID" 2>/dev/null || break
        /usr/bin/sleep 0.05
      done
      if kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then
        kill -KILL -- "-$CHILD_PGID" 2>/dev/null || true
        action=TERM_THEN_KILL_OWN_VERIFIED_PROCESS_GROUP
        for ((attempt=1; attempt<=100; ++attempt)); do
          kill -0 "$CHILD_SESSION_PID" 2>/dev/null || break
          /usr/bin/sleep 0.05
        done
      fi
    elif [[ "$preadopt" != true ]]; then
      action=REFUSED_UNVERIFIED_PROCESS_GROUP
    fi
  fi
  if [[ -n "$CHILD_SESSION_PID" ]] && kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then alive=true; fi
  if [[ -n "$CHILD_WRAPPER_PID" ]] && ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null; then
    wait "$CHILD_WRAPPER_PID" 2>/dev/null || true
  fi
  write_cleanup_provenance "$reason" "$verified" "$action" "$alive"
  clear_child_tracking
}


mark_current_stage_failed() {
  local reason=$1
  if [[ -n "$CURRENT_CLAIMED_STAGE" ]]; then
    safe_python "$MANIFEST_TOOL" stage-fail --run "$OUT" --plan "$PLAN" --pins "$PINS" --stage "$CURRENT_CLAIMED_STAGE" --reason "$reason" >/dev/null 2>&1 || true
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$EXECUTE_MODE" -eq 1 && "$SUCCESS" -ne 1 ]]; then
    # Never target a name/global GPU process: only a session proven from this
    # guard's ready handshake may receive a process-group signal.
    cleanup_own_child "outer_guard_exit_${rc}"
    mark_current_stage_failed "outer_guard_exit_${rc}"
    if [[ -n "$OUT" && -d "$OUT" && -n "$GPU0_UUID" && "$OUTER_EXIT_RECORDED" -eq 0 ]]; then
      snapshot_gpu0 outer_exit false >/dev/null 2>&1 || true
    fi
  fi
  exit "$rc"
}
trap on_exit EXIT
# Convert asynchronous control signals into an EXIT path so the exact owned
# child cleanup above runs before the guard terminates.
trap 'exit 130' INT
trap 'exit 143' TERM HUP

run_stage() {
  local stage=$1 mode runner_stage ids timed gamma=""
  local -a command
  local stdout_log stderr_log verifier_json child rc
  ids="$(safe_python - "$WORKLOAD" "$stage" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m['stages'][sys.argv[2]]['ids_path'])
PY
)"
  timed="$(case "$stage" in calibration) echo 1 ;; validation) echo 3 ;; sealed_test) echo 5 ;; *) exit 64 ;; esac)"
  mode=calibrate; runner_stage=calibration
  if [[ "$stage" != calibration ]]; then
    mode=evaluate
    runner_stage="$( [[ "$stage" == validation ]] && echo validation || echo test )"
    gamma="$OUT/calibration/final_v3_speculative_gamma_vector.txt"
    # This consumes validation/test before their output mkdir or telemetry.
    safe_python "$MANIFEST_TOOL" claim-stage --run "$OUT" --plan "$PLAN" --pins "$PINS" --stage "$stage" --gamma-file "$gamma" >/dev/null
    CURRENT_CLAIMED_STAGE="$stage"
  fi
  wait_for_strict_idle "${stage}_pre_strict_idle"
  command=("$BINARY" --mode "$mode" --base-fvecs /workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs --query-fvecs "$ROOT/inputs/sift_learn_compact10k_v1/sift_learn_clean10k.fvecs" --groundtruth-ivecs "$ROOT/inputs/sift_learn_compact10k_v1/exact_oracle_v1/learn_clean10k_gt_fp32_top100.ivecs" --query-ids "$ids" --out "$OUT/$stage" --stage "$runner_stage" --workload-manifest "$WORKLOAD")
  if [[ -n "$gamma" ]]; then command+=(--gamma-vector-file "$gamma"); fi
  command+=(--warmup-reps 1 --timed-reps "$timed")
  safe_python "$MANIFEST_TOOL" stage-start --run "$OUT" --stage "$stage" --output "$OUT/$stage" --gpu-uuid "$GPU0_UUID" --pci-bus-id "$GPU0_PCI" --argv "${command[@]}" >/dev/null
  /usr/bin/mkdir -p "$OUT/logs"
  stdout_log="$OUT/logs/${stage}.stdout.log"
  stderr_log="$OUT/logs/${stage}.stderr.log"
  : >"$stdout_log"; : >"$stderr_log"
  # Bootstrap in a new session, prove its PID/PGID/SID is ours, then allow a
  # direct `exec /usr/bin/env -i ... $BINARY` only after that proof.  GNU
  # `setsid --wait` preserves the direct binary's return code for the guard.
  CHILD_STAGE="$stage"
  CHILD_READY_FILE="$OUT/logs/${stage}.child_ready"
  CHILD_GO_FILE="$OUT/logs/${stage}.child_go"
  CHILD_ABORT_FILE="$OUT/logs/${stage}.child_abort"
  /usr/bin/rm -f "$CHILD_READY_FILE" "$CHILD_GO_FILE" "$CHILD_ABORT_FILE"
  set +e
  /usr/bin/setsid --wait /bin/bash -c '
set -Eeuo pipefail
ready=$1; go=$2; abort=$3; binary=$4; uuid=$5; stage=$6; shift 6
pgid="$(/usr/bin/ps -o pgid= -p "$$" | /usr/bin/tr -d "[:space:]")"
sid="$(/usr/bin/ps -o sid= -p "$$" | /usr/bin/tr -d "[:space:]")"
printf "pid=%s\npgid=%s\nsid=%s\n" "$$" "$pgid" "$sid" > "$ready"
while [[ ! -e "$go" ]]; do
  [[ ! -e "$abort" ]] || exit 125
  /usr/bin/sleep 0.05
done
[[ ! -e "$abort" ]] || exit 125
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" NVIDIA_VISIBLE_DEVICES="$uuid" LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 C2_V3_GUARDED=1 C2_V3_STAGE="$stage" "$binary" "$@"
' safe-c2-v3-bootstrap "$CHILD_READY_FILE" "$CHILD_GO_FILE" "$CHILD_ABORT_FILE" "$BINARY" "$GPU0_UUID" "$stage" "${command[@]:1}" >"$stdout_log" 2>"$stderr_log" &
  CHILD_WRAPPER_PID=$!
  set -e
  adopt_own_child_session_from_ready || die "cannot verify owned setsid bootstrap session before direct binary exec"
  : > "$CHILD_GO_FILE"
  confirm_own_binary_session_or_exit || die "cannot verify owned direct binary session after bootstrap exec"
  set +e
  wait "$CHILD_WRAPPER_PID"
  rc=$?
  set -e
  safe_python "$MANIFEST_TOOL" record-child --run "$OUT" --stage "$stage" --pid "$CHILD_SESSION_PID" --returncode "$rc" >/dev/null
  clear_child_tracking
  [[ "$rc" -eq 0 ]] || die "pinned C++ binary returned nonzero for $stage: $rc"
  snapshot_gpu0 "${stage}_post" false
  verifier_json="$OUT/${stage}_artifact_verifier.json"
  safe_python "$ARTIFACT_VERIFY" --stage "$stage" --run-dir "$OUT" >"$verifier_json"
  safe_python "$MANIFEST_TOOL" stage-complete --run "$OUT" --stage "$stage" --verification "$verifier_json" --stdout-log "$stdout_log" --stderr-log "$stderr_log" >/dev/null
  CURRENT_CLAIMED_STAGE=""
}

execute() {
  EXECUTE_MODE=1
  verify_launcher_chain
  [[ "$(/usr/bin/hostname)" == "$HOST_REQUIRED" ]] || die "host must be $HOST_REQUIRED"
  [[ -n "$OUT" ]] || die 'launcher did not provide a fresh run output path'
  [[ "$OUT" == "$ROOT"/runs/c2_v3_exec_* ]] || die 'run output path is outside canonical v3 runs root'
  [[ ! -e "$OUT" && ! -L "$OUT" ]] || die 'run output already exists or is a symlink'
  /usr/bin/mkdir -p "$ROOT/.locks" "$ROOT/runs"
  exec 9>"$LOCK"
  /usr/bin/flock -n 9 || die 'another v3 guarded GPU0 workflow holds the lock'
  # The ledger claims are intentionally before run/output mkdir and before any
  # GPU management command or telemetry, so a preflight/partial failure cannot retry.
  safe_python "$MANIFEST_TOOL" claim-workflow --run "$OUT" --plan "$PLAN" --pins "$PINS" --workload-manifest "$WORKLOAD" >/dev/null
  safe_python "$MANIFEST_TOOL" claim-stage --run "$OUT" --plan "$PLAN" --pins "$PINS" --stage calibration >/dev/null
  CURRENT_CLAIMED_STAGE=calibration
  safe_python "$MANIFEST_TOOL" init-run --run "$OUT" --plan "$PLAN" --pins "$PINS" --guard "$ROOT/tools/run_c2_v3_execution_guard.sh" --launcher "$LAUNCHER" --host "$HOST_REQUIRED" --workload-manifest "$WORKLOAD" >/dev/null
  /usr/bin/mkdir -p "$OUT/telemetry"
  wait_for_strict_idle outer_pre_strict_idle
  write_runtime_environment
  run_stage calibration
  run_stage validation
  run_stage sealed_test
  snapshot_gpu0 outer_exit false
  OUTER_EXIT_RECORDED=1
  all_verifier="$OUT/all_stages_artifact_verifier.json"
  safe_python "$ARTIFACT_VERIFY" --all-stages --run-dir "$OUT" >"$all_verifier"
  safe_python "$MANIFEST_TOOL" finalize --run "$OUT" --verification "$all_verifier" >/dev/null
  SUCCESS=1
  printf 'SAFE-C2-V3 guarded workflow COMPLETE: %s\n' "$OUT"
}

case "${1:-}" in
  --verify-source-only)
    [[ "$(/usr/bin/hostname)" == "$HOST_REQUIRED" ]] || die "host must be $HOST_REQUIRED"
    verify_source_only
    ;;
  --execute)
    execute
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac

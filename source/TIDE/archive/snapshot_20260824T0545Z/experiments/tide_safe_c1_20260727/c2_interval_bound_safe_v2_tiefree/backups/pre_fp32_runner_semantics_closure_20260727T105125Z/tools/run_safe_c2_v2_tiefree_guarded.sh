#!/usr/bin/env bash
# Fail-closed launcher for corrected-after-submit Safe-C2 v2 tie-free only.
# --verify-static is CPU-only. --execute is opt-in and must be explicitly
# approved after a shared-GPU review. Safe-C2 is not the submitted legacy C2.
set -Eeuo pipefail

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_interval_bound_safe_v2_tiefree
HOST_REQUIRED=CONFIGURE_ARCHIVE_HOST
PYTHON=/workspace/legacy_workspace/GTS/bench_env/bin/python
BINARY="$ROOT/bin/GTS_safe_c2_v2_tiefree_sift1m"
SRC="$ROOT/src/gts_safe_c2_v2_tiefree_sift1m.cu"
PROTOCOL="$ROOT/protocols/safe_c2_interval_sift1m_v2_tiefree.json"
PREFLIGHT="$ROOT/preflight/sift1m_tiefree_v2_20260727T102548Z/preflight_safe_c2_tiefree_v2.json"
SOURCE_MANIFEST="$ROOT/source_manifest.sha256"
AUDIT="$ROOT/tools/audit_safe_c2_v2_tiefree_static.py"
TIEFREE_VERIFY="$ROOT/tools/verify_tiefree_v2.py"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERY=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
GT=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs
CAL_IDS="$ROOT/preflight/sift1m_tiefree_v2_20260727T102548Z/calibration.ids"
VAL_IDS="$ROOT/preflight/sift1m_tiefree_v2_20260727T102548Z/validation.ids"
TEST_IDS="$ROOT/preflight/sift1m_tiefree_v2_20260727T102548Z/test.ids"
LOCK="$ROOT/.locks/safe_c2_v2_tiefree_physical_gpu0.lock"
GPU0_MAX_MEMORY_USED_MIB=256
GPU0_IDLE_POLL_ATTEMPTS=15
GPU0_IDLE_POLL_SECONDS=2
GPU0_LAST_TOTAL_MIB=""
GPU0_LAST_MEMORY_MIB=""
GPU0_LAST_UTILIZATION_PERCENT=""

PRE=""
OUT=""
GPU0_UUID=""
LOCK_HELD=0
CHILD_LAUNCHER_PID=""
CHILD_SESSION_PID=""
CHILD_PGID=""
CHILD_READY_FILE=""
CHILD_GO_FILE=""
CHILD_ABORT_FILE=""
CHILD_CLEANUP_FAILED=0

usage() {
  cat <<'USAGE'
Usage:
  run_safe_c2_guarded.sh --verify-static
  SAFE_C2_V2_TIEFREE_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE \
    run_safe_c2_guarded.sh --execute

--execute is the only GPU path. It runs the immutable sequence:
calibration -> validation only if calibration passes -> sealed test only if validation passes.
It refuses non-8p hosts, nonempty or malformed GPU0 state, stale static artifacts,
unverified input hashes, pre-existing lock/run directory, or missing acknowledgement.
It never kills, restarts, or otherwise touches any pre-existing process. Safe-C2 v2 is
tie-free corrected-after-submit and must never be labeled as a run of submitted legacy C2.
USAGE
}

die() {
  echo "Safe-C2 v2 tie-free GPU0 guard: $*" >&2
  exit 64
}

trim() {
  local x=$1
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf '%s' "$x"
}

normalize_nonblank_lines() {
  local raw=$1
  printf '%s\n' "$raw" | awk '{ sub(/\r$/, ""); if ($0 ~ /[^[:space:]]/) print }'
}

require_exactly_one_nonblank_line() {
  local label=$1 raw=$2 clean
  clean="$(normalize_nonblank_lines "$raw")"
  [[ -n "$clean" ]] || die "$label returned no nonblank row"
  [[ "$clean" != *$'\n'* ]] || die "$label returned more than one nonblank row: $clean"
  printf '%s' "$clean"
}

reject_nvidia_error_text() {
  local label=$1 text=$2
  case "$text" in
    *"No devices were found"*|*"Failed to initialize NVML"*|*"NVIDIA-SMI has failed"*|*"Unknown Error"*)
      die "$label returned NVIDIA error text: $text" ;;
  esac
}

static_check() {
  local out=$1
  local tie_json="$(dirname "$out")/$(basename "$out" .log)_tiefree_reverification.json"
  mkdir -p "$(dirname "$out")"
  {
    echo "static_check_utc=$(date -u -Is)"
    echo "host=$(hostname)"
    sha256sum -c "$SOURCE_MANIFEST"
    "$AUDIT"
    "$PYTHON" "$TIEFREE_VERIFY" --protocol "$PROTOCOL" --preflight "$PREFLIGHT" --out "$tie_json"
    "$PYTHON" - "$PROTOCOL" "$PREFLIGHT" "$SRC" "$BINARY" <<'PY'
import json, pathlib, sys
protocol, preflight, source, binary = map(pathlib.Path, sys.argv[1:])
p = json.loads(protocol.read_text()); q = json.loads(preflight.read_text())
assert p['schema'] == 'safe-c2-corrected-after-submit-tiefree-protocol-v2'
assert p['implementation_identity']['corrected_after_submit'] is True
assert p['implementation_identity']['not_submitted_artifact'] is True
assert p['implementation_identity']['v2_tiefree'] is True
assert p['timing_protocol']['heldout_pairing']['required'] is True
assert q['schema'] == 'safe-c2-corrected-after-submit-tiefree-preflight-v2'
s = source.read_text()
for required in ('certify_and_outward_repair_intervals', 'run_paired_abba_stage', 'per_query_no_regression',
                 'write_baseline_invalid_diagnostic', 'safe-c2-tiefree-baseline-invalid-diagnostic-v2'):
    assert required in s
assert binary.is_file() and binary.stat().st_size > 0
print('Safe-C2 v2 tie-free corrected-after-submit static contract: PASS')
PY
  } >"$out" 2>&1
}

# This is the execute-path CPU preflight. It recomputes actual raw hashes and,
# before any nvidia-smi call or GPU lock, proves every selected v2 ID remains
# tie-free and each stage is exactly frozen-v1 membership minus its exclusions.
verify_frozen_data_and_tiefree() {
  local out=$1
  local json_out="${out%.env}.json"
  "$PYTHON" "$TIEFREE_VERIFY" --protocol "$PROTOCOL" --preflight "$PREFLIGHT" --out "$json_out"
  "$PYTHON" - "$json_out" "$out" <<'PY'
import json, pathlib, sys
jpath, outpath = map(pathlib.Path, sys.argv[1:])
j = json.loads(jpath.read_text())
assert j['schema'] == 'safe-c2-tiefree-reverification-v2'
assert j['status'] == 'PASS' and j['cuda_used'] is False
lines = [
    'data_hash_verification=PASS',
    'tie_free_reverification=PASS',
    'tie_free_reverification_json=' + str(jpath),
    'protocol_sha256_actual=' + j['protocol_sha256'],
    'preflight_sha256_actual=' + j['preflight_sha256'],
]
for name, digest in j['raw_actual_sha256'].items():
    lines.append(f'{name}_actual_sha256={digest}')
for stage, count in j['selected_counts'].items():
    lines.append(f'{stage}_selected_count={count}')
for stage, count in j['excluded_counts'].items():
    lines.append(f'{stage}_excluded_ambiguous_count={count}')
for stage, count in j['recomputed_ambiguous_counts'].items():
    lines.append(f'{stage}_recomputed_ambiguous_count={count}')
lines.append('selected_tie_or_mismatch_count=' + str(j['selected_tie_or_mismatch_count']))
outpath.write_text('\n'.join(lines) + '\n')
print('Safe-C2 v2 frozen data/tie-free verification PASS:', outpath)
PY
}

physical_nvidia_smi() {
  env -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES nvidia-smi "$@"
}

resolve_physical_gpu0_uuid() {
  local raw line index uuid extra
  if ! raw="$(physical_nvidia_smi --id=0 --query-gpu=index,uuid --format=csv,noheader,nounits 2>&1)"; then
    printf '%s\n' "$raw" > "$PRE/gpu0_uuid_resolve_raw.txt"
    die 'cannot resolve physical GPU0 UUID'
  fi
  printf '%s\n' "$raw" > "$PRE/gpu0_uuid_resolve_raw.txt"
  line="$(require_exactly_one_nonblank_line 'GPU0 UUID query' "$raw")"
  reject_nvidia_error_text 'GPU0 UUID query' "$line"
  IFS=, read -r index uuid extra <<< "$line"
  index="$(trim "$index")"; uuid="$(trim "$uuid")"
  [[ -z "${extra:-}" ]] || die "unexpected GPU0 UUID CSV fields: $line"
  [[ "$index" == 0 && "$uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "malformed GPU0 UUID row: $line"
  GPU0_UUID="$uuid"
  {
    echo 'physical_gpu_index=0'
    echo "physical_gpu_uuid=$GPU0_UUID"
    echo 'gpu0_uuid_query_exactly_one_row=PASS'
  } > "$PRE/gpu0_uuid_resolve.env"
}

# Strictly parses physical GPU0 identity and an explicit no-compute-app state.
# It records utilization/memory but intentionally does not classify transient
# post-phase telemetry as a failure; strict idle is enforced before the next phase.
gpu0_snapshot_identity_and_no_compute() {
  local label=$1 dir=$2
  local status status_line apps apps_clean
  local index uuid total mem util extra
  if ! status="$(physical_nvidia_smi --id=0 --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
    printf '%s\n' "$status" > "$dir/gpu0_status_${label}.txt"
    die "cannot query GPU0 status ($label)"
  fi
  printf '%s\n' "$status" > "$dir/gpu0_status_${label}.txt"
  status_line="$(require_exactly_one_nonblank_line "GPU0 status ($label)" "$status")"
  reject_nvidia_error_text "GPU0 status ($label)" "$status_line"
  IFS=, read -r index uuid total mem util extra <<< "$status_line"
  index="$(trim "$index")"; uuid="$(trim "$uuid")"; total="$(trim "$total")"
  mem="$(trim "$mem")"; util="$(trim "$util")"
  [[ -z "${extra:-}" ]] || die "unexpected GPU0 status CSV fields ($label): $status_line"
  [[ "$index" == 0 ]] || die "GPU0 status did not return physical index 0 ($label): $status_line"
  [[ "$uuid" == "$GPU0_UUID" ]] || die "GPU0 UUID mismatch ($label): expected $GPU0_UUID got $uuid"
  [[ "$total" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || die "non-numeric GPU0 state ($label): $status_line"
  [[ "$total" -gt 0 ]] || die "non-positive GPU0 total memory ($label): $status_line"

  if ! apps="$(physical_nvidia_smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)"; then
    printf '%s\n' "$apps" > "$dir/gpu0_compute_apps_${label}.txt"
    die "cannot query GPU0 compute applications ($label)"
  fi
  printf '%s\n' "$apps" > "$dir/gpu0_compute_apps_${label}.txt"
  apps_clean="$(normalize_nonblank_lines "$apps")"
  reject_nvidia_error_text "GPU0 compute application query ($label)" "$apps_clean"
  case "$apps_clean" in
    '') ;;
    'No running processes found'|'No running compute processes found') ;;
    *) die "GPU0 compute application query returned active or unknown text ($label): $apps_clean" ;;
  esac
  GPU0_LAST_TOTAL_MIB="$total"
  GPU0_LAST_MEMORY_MIB="$mem"
  GPU0_LAST_UTILIZATION_PERCENT="$util"
  {
    echo 'physical_gpu_index=0'
    echo "physical_gpu_uuid=$uuid"
    echo 'gpu_status_exactly_one_row=PASS'
    echo "memory_total_mib=$total"
    echo "memory_used_mib=$mem"
    echo "utilization_percent=$util"
    echo 'compute_processes=none'
    echo 'compute_application_query=explicit_empty_or_no_running_processes'
  } > "$dir/gpu0_identity_compute_check_${label}.env"
}

gpu0_snapshot_and_require_idle() {
  local label=$1 dir=$2
  gpu0_snapshot_identity_and_no_compute "$label" "$dir"
  [[ "$GPU0_LAST_MEMORY_MIB" -le "$GPU0_MAX_MEMORY_USED_MIB" ]] ||
      return 1
  [[ "$GPU0_LAST_UTILIZATION_PERCENT" -eq 0 ]] || return 1
  {
    echo 'strict_idle=PASS'
    echo "memory_used_mib=$GPU0_LAST_MEMORY_MIB"
    echo "utilization_percent=$GPU0_LAST_UTILIZATION_PERCENT"
    echo "memory_limit_mib=$GPU0_MAX_MEMORY_USED_MIB"
  } > "$dir/gpu0_strict_idle_check_${label}.env"
}

wait_for_gpu0_strict_idle() {
  local label=$1 dir=$2 attempt
  local poll_log="$dir/gpu0_strict_idle_poll_${label}.log"
  : > "$poll_log"
  for ((attempt=1; attempt<=GPU0_IDLE_POLL_ATTEMPTS; ++attempt)); do
    if gpu0_snapshot_and_require_idle "${label}_attempt_$(printf '%02d' "$attempt")" "$dir"; then
      {
        echo 'strict_idle=PASS'
        echo "attempt=$attempt"
        echo "max_attempts=$GPU0_IDLE_POLL_ATTEMPTS"
        echo "poll_seconds=$GPU0_IDLE_POLL_SECONDS"
        echo "memory_used_mib=$GPU0_LAST_MEMORY_MIB"
        echo "utilization_percent=$GPU0_LAST_UTILIZATION_PERCENT"
      } > "$dir/gpu0_strict_idle_check_${label}.env"
      return 0
    fi
    printf 'attempt=%s memory_used_mib=%s utilization_percent=%s compute_processes=none identity=PASS\n' \
      "$attempt" "$GPU0_LAST_MEMORY_MIB" "$GPU0_LAST_UTILIZATION_PERCENT" >> "$poll_log"
    if [[ "$attempt" -lt "$GPU0_IDLE_POLL_ATTEMPTS" ]]; then sleep "$GPU0_IDLE_POLL_SECONDS"; fi
  done
  die "GPU0 did not recover strict idle within ${GPU0_IDLE_POLL_ATTEMPTS}x${GPU0_IDLE_POLL_SECONDS}s before $label"
}

summary_gate() {
  local summary=$1 expected=$2 expected_pairs=$3
  "$PYTHON" - "$summary" "$expected" "$expected_pairs" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); expected = sys.argv[2]; expected_pairs = int(sys.argv[3])
d = json.loads(p.read_text())
assert d['schema'] == 'safe-c2-corrected-after-submit-tiefree-run-v2'
assert d['status'] == expected, d['status']
assert d['corrected_after_submit'] is True
assert d['v2_tiefree'] is True
assert d['archived_incremental_updater_used'] is False
assert d['dynamic_exact_smoke_used'] is False
c = d['interval_certificate']
assert c['post_reverify_passed'] is True
assert c['corrected_after_submit'] is True
assert d['per_query_no_regression_gate']['passed'] is True
pairing = d['timing_comparison']
if expected_pairs == 0:
    assert pairing['eligible_for_speed_comparison'] is False
else:
    assert pairing['eligible_for_speed_comparison'] is True
    assert len(pairing['timed_pair_order']) == expected_pairs
    assert pairing['shared_warmup_pairs'] >= 1
print('Safe-C2 summary gate PASS:', expected)
PY
}

clear_child_tracking() {
  CHILD_LAUNCHER_PID=""
  CHILD_SESSION_PID=""
  CHILD_PGID=""
  CHILD_READY_FILE=""
  CHILD_GO_FILE=""
  CHILD_ABORT_FILE=""
}

adopt_child_session_from_ready() {
  local -a lines=()
  local pid pgid sid actual
  [[ -n "$CHILD_READY_FILE" && -s "$CHILD_READY_FILE" ]] || return 1
  mapfile -t lines < "$CHILD_READY_FILE" || return 1
  [[ "${#lines[@]}" -eq 3 ]] || return 1
  [[ "${lines[0]}" =~ ^pid=[0-9]+$ ]] || return 1
  [[ "${lines[1]}" =~ ^pgid=[0-9]+$ ]] || return 1
  [[ "${lines[2]}" =~ ^sid=[0-9]+$ ]] || return 1
  pid="${lines[0]#pid=}"
  pgid="${lines[1]#pgid=}"
  sid="${lines[2]#sid=}"
  [[ "$pid" == "$pgid" && "$pid" == "$sid" ]] || return 1
  actual="$(ps -o pid=,pgid=,sid= -p "$pid" 2>/dev/null | awk 'NF {print $1 "," $2 "," $3}' || true)"
  [[ "$actual" == "$pid,$pgid,$sid" ]] || return 1
  CHILD_SESSION_PID="$pid"
  CHILD_PGID="$pgid"
  return 0
}

isolated_group_state() {
  local pgid=$1 table
  table="$(ps -eo pid=,pgid=,sid= 2>/dev/null || true)"
  printf '%s\n' "$table" | awk -v p="$pgid" '
    $2 == p { found=1; if ($3 != p) bad=1 }
    END {
      if (!found) print "empty"
      else if (bad) print "unsafe"
      else print "verified"
    }'
}

request_child_abort() {
  if [[ -n "$CHILD_ABORT_FILE" ]]; then
    : > "$CHILD_ABORT_FILE" 2>/dev/null || true
  fi
}

wait_for_launcher_exit() {
  local pid=$1 i stat
  for ((i=0; i<100; ++i)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e
      wait "$pid" 2>/dev/null
      set -e
      return 0
    fi
    # A completed child can remain a zombie until this shell calls wait; kill -0
    # still succeeds for a zombie, so detect/reap it rather than retaining a lock.
    stat="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    stat="$(trim "$stat")"
    if [[ "$stat" == Z* ]]; then
      set +e
      wait "$pid" 2>/dev/null
      set -e
      return 0
    fi
    sleep 0.05
  done
  return 1
}

# The launched wrapper is in a fresh setsid session. On EXIT/TERM/INT/HUP, this
# sends TERM to only that verified child process group, waits/reaps its setsid
# launcher, and only then permits lock release. An unverifiable live group
# leaves the lock in place rather than risking a signal to another process.
terminate_child_group() {
  local state=""
  request_child_abort

  if [[ -z "$CHILD_PGID" && -n "$CHILD_READY_FILE" ]]; then
    local i
    for ((i=0; i<20; ++i)); do
      if adopt_child_session_from_ready; then break; fi
      sleep 0.05
    done
  fi

  if [[ -n "$CHILD_PGID" ]]; then
    state="$(isolated_group_state "$CHILD_PGID")"
    case "$state" in
      verified)
        echo "Safe-C2 v2 tie-free GPU0 guard: terminating only Safe-C2 child process group $CHILD_PGID" >&2
        kill -TERM -- "-$CHILD_PGID" 2>/dev/null || true
        local j
        for ((j=0; j<100; ++j)); do
          [[ "$(isolated_group_state "$CHILD_PGID")" == empty ]] && break
          sleep 0.05
        done
        if [[ "$(isolated_group_state "$CHILD_PGID")" != empty ]]; then
          kill -KILL -- "-$CHILD_PGID" 2>/dev/null || true
        fi
        ;;
      empty)
        ;;
      *)
        echo "Safe-C2 v2 tie-free GPU0 guard: refusing to signal unverifiable process group $CHILD_PGID; lock will be retained if it remains live" >&2
        CHILD_CLEANUP_FAILED=1
        ;;
    esac
  fi

  if [[ -n "$CHILD_LAUNCHER_PID" ]]; then
    if ! wait_for_launcher_exit "$CHILD_LAUNCHER_PID"; then
      # The launcher is the exact setsid --wait process started by this guard.
      # It is not a shared GPU process. TERM is used only after an abort marker.
      kill -TERM "$CHILD_LAUNCHER_PID" 2>/dev/null || true
      if ! wait_for_launcher_exit "$CHILD_LAUNCHER_PID"; then
        echo "Safe-C2 v2 tie-free GPU0 guard: setsid launcher did not exit; lock will be retained" >&2
        CHILD_CLEANUP_FAILED=1
      fi
    fi
  fi

  # The setsid --wait launcher reaps any killed group leader. Recheck after
  # that wait; an extant process group means cleanup is incomplete, so preserve
  # the lock instead of claiming the GPU is free.
  if [[ -n "$CHILD_PGID" && "$(isolated_group_state "$CHILD_PGID")" != empty ]]; then
    echo "Safe-C2 v2 tie-free GPU0 guard: child process group $CHILD_PGID remains after TERM/KILL/reap; lock will be retained" >&2
    CHILD_CLEANUP_FAILED=1
  fi

  if [[ "$CHILD_CLEANUP_FAILED" -eq 0 ]]; then
    clear_child_tracking
  fi
}

release_lock() {
  if [[ "$LOCK_HELD" -eq 1 ]]; then
    rmdir "$LOCK" 2>/dev/null || {
      echo "Safe-C2 v2 tie-free GPU0 guard: unable to remove owned lock $LOCK; retaining it" >&2
      return 1
    }
    LOCK_HELD=0
  fi
  return 0
}

cleanup_on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  terminate_child_group || true
  if [[ "$CHILD_CLEANUP_FAILED" -eq 0 ]]; then
    release_lock || true
  else
    echo "Safe-C2 v2 tie-free GPU0 guard: retaining $LOCK because child session could not be proven reaped" >&2
  fi
  exit "$rc"
}

on_signal() {
  local sig=$1 rc=$2
  echo "Safe-C2 v2 tie-free GPU0 guard: received $sig; aborting/reaping only its own child session before lock release" >&2
  exit "$rc"
}

launch_session_guarded_command() {
  local phase_dir=$1
  shift
  CHILD_READY_FILE="$phase_dir/child_session.ready"
  CHILD_GO_FILE="$phase_dir/child_session.go"
  CHILD_ABORT_FILE="$phase_dir/child_session.abort"
  [[ ! -e "$CHILD_READY_FILE" && ! -e "$CHILD_GO_FILE" && ! -e "$CHILD_ABORT_FILE" ]] || die "phase child-session marker already exists: $phase_dir"

  # The wrapper cannot exec timeout/the CUDA binary until this guard validates
  # its fresh session/PGID and writes the GO marker.
  setsid --wait /bin/bash -c '
set -Eeuo pipefail
ready=$1
go=$2
abort=$3
shift 3
pid=$$
pgid=$(ps -o pgid= -p "$pid" | tr -d "[:space:]")
sid=$(ps -o sid= -p "$pid" | tr -d "[:space:]")
[[ "$pid" == "$pgid" && "$pid" == "$sid" ]] || exit 125
umask 077
printf "pid=%s\npgid=%s\nsid=%s\n" "$pid" "$pgid" "$sid" > "$ready"
while [[ ! -e "$go" ]]; do
  [[ ! -e "$abort" ]] || exit 125
  sleep 0.05
done
[[ ! -e "$abort" ]] || exit 125
exec timeout --foreground 90m "$@"
' safe-c2-session-wrapper "$CHILD_READY_FILE" "$CHILD_GO_FILE" "$CHILD_ABORT_FILE" "$@" > "$phase_dir/runner_console.log" 2>&1 &
  CHILD_LAUNCHER_PID=$!

  local i rc
  for ((i=0; i<100; ++i)); do
    if adopt_child_session_from_ready; then
      : > "$CHILD_GO_FILE"
      {
        echo "setsid_launcher_pid=$CHILD_LAUNCHER_PID"
        echo "child_session_pid=$CHILD_SESSION_PID"
        echo "child_process_group=$CHILD_PGID"
        echo 'child_session_verified_before_cuda_exec=PASS'
      } > "$phase_dir/child_process_group.env"
      return 0
    fi
    if ! kill -0 "$CHILD_LAUNCHER_PID" 2>/dev/null; then
      set +e
      wait "$CHILD_LAUNCHER_PID"
      rc=$?
      set -e
      clear_child_tracking
      die "setsid launcher exited before verified child session (exit=$rc); CUDA binary was never authorized"
    fi
    sleep 0.05
  done

  request_child_abort
  terminate_child_group
  die 'timed out waiting for verified child session; CUDA binary was never authorized'
}

run_phase() {
  local phase=$1 mode=$2 ids=$3 timed=$4 gamma=${5:-}
  local phase_dir="$OUT/$phase"
  local rc
  [[ ! -e "$phase_dir" ]] || die "pre-existing phase output refuses overwrite: $phase_dir"
  mkdir "$phase_dir"
  # Before every CUDA workload, bounded polling must recover strict physical-GPU0
  # idle state. This is the only timing that may block phase start on 1% telemetry.
  wait_for_gpu0_strict_idle "immediately_pre_${phase}" "$phase_dir"
  local -a cmd=("$BINARY" --mode "$mode" --base-fvecs "$BASE" --query-fvecs "$QUERY" --groundtruth-ivecs "$GT" --query-ids "$ids" --out "$phase_dir" --stage "$phase" --warmup-reps 1 --timed-reps "$timed")
  if [[ "$mode" == evaluate ]]; then cmd+=(--gamma-vector-file "$gamma"); fi
  printf '%q ' "${cmd[@]}" > "$phase_dir/command.shlex"
  printf '\n' >> "$phase_dir/command.shlex"

  launch_session_guarded_command "$phase_dir" "${cmd[@]}"
  set +e
  wait "$CHILD_LAUNCHER_PID"
  rc=$?
  set -e
  printf 'runner_exit_code=%s\n' "$rc" > "$phase_dir/launcher_result.env"
  clear_child_tracking

  # Post-phase is identity/no-compute telemetry only. It records a transient
  # 1% reading but does not erase a completed runner result; next phase polls
  # back to strict 0%/<=256MiB before authorizing its CUDA workload.
  gpu0_snapshot_identity_and_no_compute "after_${phase}" "$phase_dir"
  return "$rc"
}

[[ $# -eq 1 ]] || { usage; exit 64; }
case "$1" in
  --verify-static)
    [[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
    static_check "$ROOT/logs/static_guard_latest.log"
    echo 'Safe-C2 v2 tie-free CPU-only static verification PASS; no GPU query or CUDA binary execution.'
    exit 0 ;;
  --execute) ;;
  *) usage; exit 64 ;;
esac

[[ -v SAFE_C2_V2_TIEFREE_GPU0_APPROVED && "$SAFE_C2_V2_TIEFREE_GPU0_APPROVED" == "I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE" ]] || die 'set SAFE_C2_V2_TIEFREE_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE after shared-GPU review'
[[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || die 'unset CUDA_VISIBLE_DEVICES; guard maps physical GPU0 by UUID'
[[ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]] || die 'unset NVIDIA_VISIBLE_DEVICES; guard maps physical GPU0 by UUID'
[[ -x "$BINARY" && -r "$SRC" && -r "$PROTOCOL" && -r "$PREFLIGHT" && -r "$SOURCE_MANIFEST" && -x "$TIEFREE_VERIFY" ]] || die 'missing Safe-C2 artifact'
for p in "$BASE" "$QUERY" "$GT" "$CAL_IDS" "$VAL_IDS" "$TEST_IDS"; do [[ -r "$p" ]] || die "missing input $p"; done
command -v nvidia-smi >/dev/null || die 'nvidia-smi unavailable'
command -v timeout >/dev/null || die 'GNU timeout unavailable'
command -v setsid >/dev/null || die 'setsid unavailable'

PRE="$ROOT/logs/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
[[ ! -e "$PRE" ]] || die "preflight directory already exists; refusing overwrite: $PRE"
mkdir "$PRE"
static_check "$PRE/static_audit.log"
# This is intentionally before every nvidia-smi call and before the GPU0 lock.
verify_frozen_data_and_tiefree "$PRE/data_hash_verification.env"

resolve_physical_gpu0_uuid
wait_for_gpu0_strict_idle before_lock "$PRE"
mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then
  die "Safe-C2 v2 tie-free physical-GPU0 lock exists: $LOCK"
fi
LOCK_HELD=1
trap cleanup_on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

RUN_ID="safe_c2_v2_tiefree_sift1m_heldout_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/$RUN_ID"
[[ ! -e "$OUT" ]] || die "run directory already exists; refusing overwrite: $OUT"
mkdir "$OUT"
cp "$PRE/static_audit.log" "$OUT/static_audit.log"
cp "$PRE/data_hash_verification.env" "$OUT/data_hash_verification.env"
cp "$PRE"/gpu0_*_before_lock* "$OUT/" 2>/dev/null || true
{
  echo "run_id=$RUN_ID"
  echo "utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo 'implementation_identity=Safe-C2 v2 tie-free corrected-after-submit; not submitted legacy C2'
  echo 'physical_gpu_index=0'
  echo "physical_gpu_uuid=$GPU0_UUID"
  echo "CUDA_VISIBLE_DEVICES=$GPU0_UUID"
  echo "NVIDIA_VISIBLE_DEVICES=$GPU0_UUID"
  echo "protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')"
  echo "source_manifest_sha256=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"
  echo "runner_binary_sha256=$(sha256sum "$BINARY" | awk '{print $1}')"
  echo "runner_source_sha256=$(sha256sum "$SRC" | awk '{print $1}')"
  echo 'sequence=calibration_correctness_only_then_validation_ABBA_if_pass_then_sealed_test_ABBA_if_validation_pass'; echo 'tie_free_v2=true'
  echo 'gpu0_idle_policy=exact_identity_and_no_compute_every_inspection;strict_idle_utilization_percent=0_and_memory_used_mib<=256_before_each_phase_via_bounded_poll;post_phase_records_telemetry_only'
  echo 'session_cleanup_policy=setsid_child_group;TERM_then_reap_wait_before_lock_release'
  cat "$PRE/data_hash_verification.env"
} > "$OUT/run_manifest.env"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
export NVIDIA_VISIBLE_DEVICES="$GPU0_UUID"

CAL_RC=99
VAL_RC=not_run
TEST_RC=not_run
if run_phase calibration calibrate "$CAL_IDS" 1; then
  CAL_RC=0
  if summary_gate "$OUT/calibration/summary.json" PASS_SAFE_C2_CALIBRATION 0; then
    if run_phase validation evaluate "$VAL_IDS" 3 "$OUT/calibration/final_safe_gamma_vector.txt"; then
      VAL_RC=0
      if summary_gate "$OUT/validation/summary.json" PASS_SAFE_C2_HELDOUT_STAGE 3; then
        if run_phase test evaluate "$TEST_IDS" 5 "$OUT/calibration/final_safe_gamma_vector.txt"; then
          TEST_RC=0
          if ! summary_gate "$OUT/test/summary.json" PASS_SAFE_C2_HELDOUT_STAGE 5; then TEST_RC=4; fi
        else
          TEST_RC=$?
        fi
      else
        VAL_RC=4
      fi
    else
      VAL_RC=$?
    fi
  else
    CAL_RC=4
  fi
else
  CAL_RC=$?
fi
{
  echo "calibration_exit_code=$CAL_RC"
  echo "validation_exit_code=$VAL_RC"
  echo "test_exit_code=$TEST_RC"
} > "$OUT/launcher_result.env"

# The phase-level strict postchecks are authoritative; this final snapshot is
# retained only as a textual record and is still fail-closed on malformed state.
gpu0_snapshot_identity_and_no_compute after_all "$OUT"
if [[ "$CAL_RC" == 0 && "$VAL_RC" == 0 && "$TEST_RC" == 0 ]]; then
  echo "Safe-C2 v2 tie-free full held-out sequence PASS: $OUT"
  exit 0
fi
if [[ "$CAL_RC" != 0 ]]; then
  echo "Safe-C2 v2 tie-free stopped after calibration (no held-out evaluation): $OUT" >&2
elif [[ "$VAL_RC" != 0 ]]; then
  echo "Safe-C2 v2 tie-free validation did not pass; sealed test was not run: $OUT" >&2
else
  echo "Safe-C2 v2 tie-free sealed test did not pass: $OUT" >&2
fi
exit 3

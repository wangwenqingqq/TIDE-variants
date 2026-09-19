#!/usr/bin/env bash
# V7 profile outer guard. Only this process may launch nsys/C1Microbench for the
# explicitly unmeasured E_G/P_F profile pass; no public before/after bypass exists.
set -uo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_manifest_repair'
HOST='CONFIGURE_ARCHIVE_HOST'
GUARD="$ROOT/run_c1_profile_guard_v7.sh"
LAUNCHER="$ROOT/launch_c1_profile_v7.sh"
PLAN="$ROOT/run_c1_profile_primary_v7.sh"
PLAN_JSON="$ROOT/profile_execution_plan_v7.json"
PINS="$ROOT/hardened_static_pins_v7.json"
PIN_VERIFY="$ROOT/verify_v7_pins.py"
MANIFEST="$ROOT/c1_v7_profile_manifest.py"
VERIFIER="$ROOT/verify_c1_profile_artifacts_v7.py"
PROFILE_MANIFEST_FIXTURE="$ROOT/tools/test_c1_v7_profile_manifest_fixtures.py"
PROFILE_TERMINAL_FIXTURE="$ROOT/tools/test_c1_v7_profile_terminal_fixture.py"
PYTHON='/usr/bin/python3'
NSYS='/usr/local/bin/nsys'
SMI='/usr/bin/nvidia-smi'
SHA256='/usr/bin/sha256sum'
FLOCK='/usr/bin/flock'
SETSID='/usr/bin/setsid'
REALPATH='/usr/bin/realpath'
AWK='/usr/bin/awk'
TR='/usr/bin/tr'
GREP='/usr/bin/grep'
HEAD='/usr/bin/head'
SED='/usr/bin/sed'
SLEEP='/usr/bin/sleep'
LOCK="$ROOT/.c1_v7_gpu0.lock"
ATTEMPTS=15
POLL_SECONDS=2

GPU_UUID=''
SESSION_UUID=''
# Session fields are filled only after the controlled setsid --wait bootstrap
# proves exact PID=PGID=SID ownership and expected command identity.
CHILD_WRAPPER_PID=''
CHILD_SESSION_PID=''
CHILD_PGID=''
CHILD_SID=''
CHILD_PHASE=''
CHILD_READY_FILE=''
CHILD_GO_FILE=''
CHILD_ABORT_FILE=''
CHILD_LABEL=''
CHILD_OUT=''
CHILD_EXPECTED_KIND=''
CHILD_EXPECTED_EXECUTABLE=''
CHILD_EXPECTED_COMMAND=''
CHILD_VERIFIED=0
CHILD_BOOTSTRAP_VERIFIED=0
CPU_ONLY_FIXTURE_ACTIVE=0
# This flag is deliberately backed by a cleanup-time on-disk manifest check.
# PREPARED can be atomically written before any following shell assignment.
MANIFEST_READY=0
SESSION_READY=0
FINAL_STATUS='PREPARING'
FINAL_REASON='not started'
PROFILE_OUT=''
RUN_OUT=''

mark_failed_if_manifest() {
  if ensure_manifest_ready_from_disk 2>/dev/null; then
    FINAL_STATUS='FAILED'
    FINAL_REASON="$1"
  fi
}
die() { mark_failed_if_manifest "blocked: $*"; printf 'C1-V7-PROFILE-GUARD BLOCKED: %s\n' "$*" >&2; exit 69; }
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
safe_label() { printf '%s' "$1" | "$TR" -cs 'A-Za-z0-9._-' '_'; }
is_under() { [[ "$1" == "$2/"* ]]; }
require_regular_raw() {
  local p="$1" resolved
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  resolved="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$resolved" == "$p" ]]
}
read_env() { local key="$1"; if [[ -v "$key" ]]; then printenv "$key"; fi; }

manifest_exists() {
  [[ -n "$PROFILE_OUT" && -d "$PROFILE_OUT" && -f "$PROFILE_OUT/profile_run_manifest_v7.json" && ! -L "$PROFILE_OUT/profile_run_manifest_v7.json" ]]
}

ensure_manifest_ready_from_disk() {
  # This is intentionally an on-disk condition, not only a shell flag: init
  # atomically writes PREPARED before the next shell instruction can run.
  if manifest_exists; then
    MANIFEST_READY=1
    return 0
  fi
  return 1
}

clear_child_tracking() {
  CHILD_WRAPPER_PID=''; CHILD_SESSION_PID=''; CHILD_PGID=''; CHILD_SID=''
  CHILD_PHASE=''; CHILD_READY_FILE=''; CHILD_GO_FILE=''; CHILD_ABORT_FILE=''
  CHILD_LABEL=''; CHILD_OUT=''; CHILD_EXPECTED_KIND=''
  CHILD_EXPECTED_EXECUTABLE=''; CHILD_EXPECTED_COMMAND=''; CHILD_VERIFIED=0; CHILD_BOOTSTRAP_VERIFIED=0
}

write_child_session_record() {
  local reason="$1" verified="$2" action="$3" alive="$4" exit_code="${5:-}"
  local record_dir record
  [[ -n "$CHILD_LABEL" && -n "$CHILD_OUT" && -d "$CHILD_OUT" ]] || return 0
  record_dir="$CHILD_OUT/child_session_provenance"
  mkdir -p -- "$record_dir" || return 1
  record="$record_dir/owned_session_cleanup_v7.json"
  reason="$reason" verified="$verified" action="$action" alive="$alive" exit_code="$exit_code" \
    child_label="$CHILD_LABEL" child_kind="$CHILD_EXPECTED_KIND" child_exec="$CHILD_EXPECTED_EXECUTABLE" child_cmd="$CHILD_EXPECTED_COMMAND" \
    child_wrapper="$CHILD_WRAPPER_PID" child_pid="$CHILD_SESSION_PID" child_pgid="$CHILD_PGID" child_sid="$CHILD_SID" child_phase="$CHILD_PHASE" \
    "$PYTHON" -B -I - "$record" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
if p.exists() and p.is_symlink(): raise SystemExit('child session record symlink')
payload={
 "schema":"gtspp-c1-v7-profile-owned-child-cleanup-v1",
 "updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "label":os.environ["child_label"],"expected_kind":os.environ["child_kind"],
 "expected_executable":os.environ["child_exec"],"expected_command_path":os.environ["child_cmd"],
 "wrapper_pid":int(os.environ["child_wrapper"]) if os.environ["child_wrapper"].isdigit() else None,
 "session_pid":int(os.environ["child_pid"]) if os.environ["child_pid"].isdigit() else None,
 "pgid":int(os.environ["child_pgid"]) if os.environ["child_pgid"].isdigit() else None,
 "sid":int(os.environ["child_sid"]) if os.environ["child_sid"].isdigit() else None,
 "phase":os.environ["child_phase"],"reason":os.environ["reason"],
 "verified_owned_session":os.environ["verified"]=="true",
 "action":os.environ["action"],"still_alive_after_cleanup":os.environ["alive"]=="true",
 "child_returncode":int(os.environ["exit_code"]) if os.environ["exit_code"].lstrip("-").isdigit() else None,
}
tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(p)
PY
}

verify_own_child_session() {
  local phase="$1" line pid pgid sid exe cmd
  [[ "$CHILD_SESSION_PID" =~ ^[0-9]+$ && "$CHILD_WRAPPER_PID" =~ ^[0-9]+$ ]] || return 1
  # GNU setsid --wait retains one direct child whose PID is the new session
  # leader; do not adopt a merely similarly named process.
  [[ "$CHILD_SESSION_PID" == "$CHILD_WRAPPER_PID" ]] || return 1
  kill -0 "$CHILD_SESSION_PID" 2>/dev/null || return 1
  line="$(/usr/bin/ps -o pid=,pgid=,sid= -p "$CHILD_SESSION_PID" 2>/dev/null | "$AWK" 'NF {print $1 "," $2 "," $3}')"
  IFS=, read -r pid pgid sid <<<"$line"
  [[ "$pid" == "$CHILD_SESSION_PID" && "$pgid" == "$CHILD_SESSION_PID" && "$sid" == "$CHILD_SESSION_PID" ]] || return 1
  [[ "$CHILD_PGID" == "$pgid" && "$CHILD_SID" == "$sid" ]] || return 1
  exe="$(/usr/bin/readlink -f "/proc/$CHILD_SESSION_PID/exe" 2>/dev/null || true)"
  cmd="$("$TR" '\0' ' ' < "/proc/$CHILD_SESSION_PID/cmdline" 2>/dev/null || true)"
  if [[ "$phase" == bootstrap ]]; then
    [[ "$exe" == /usr/bin/bash || "$exe" == /bin/bash ]] || return 1
    [[ "$cmd" == *C1-V7-PROFILE-OWNED-BOOTSTRAP* && "$cmd" == *"$CHILD_READY_FILE"* && "$cmd" == *"$CHILD_GO_FILE"* && "$cmd" == *"$CHILD_ABORT_FILE"* && "$cmd" == *"$CHILD_EXPECTED_EXECUTABLE"* && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]] || return 1
    return 0
  fi
  case "$CHILD_EXPECTED_KIND" in
    profile_nsys)
      [[ "$exe" == "$CHILD_EXPECTED_EXECUTABLE" && "$cmd" == *profile* && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]] ;;
    cpu_fixture)
      [[ "$exe" == /usr/bin/sleep && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]] ;;
    *) return 1 ;;
  esac
}

adopt_own_child_session_from_ready() {
  local -a lines=()
  local pid pgid sid attempt
  for ((attempt=1; attempt<=100; ++attempt)); do
    if [[ -s "$CHILD_READY_FILE" && ! -L "$CHILD_READY_FILE" ]]; then
      mapfile -t lines < "$CHILD_READY_FILE" || true
      if [[ "${#lines[@]}" -eq 3 && "${lines[0]}" =~ ^pid=[0-9]+$ && "${lines[1]}" =~ ^pgid=[0-9]+$ && "${lines[2]}" =~ ^sid=[0-9]+$ ]]; then
        pid="${lines[0]#pid=}"; pgid="${lines[1]#pgid=}"; sid="${lines[2]#sid=}"
        CHILD_SESSION_PID="$pid"; CHILD_PGID="$pgid"; CHILD_SID="$sid"; CHILD_PHASE='bootstrap'
        if verify_own_child_session bootstrap; then CHILD_BOOTSTRAP_VERIFIED=1; return 0; fi
      fi
    fi
    if [[ -n "$CHILD_WRAPPER_PID" ]] && ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null; then return 1; fi
    "$SLEEP" 0.05
  done
  return 1
}

confirm_own_child_exec_session() {
  local attempt
  for ((attempt=1; attempt<=100; ++attempt)); do
    kill -0 "$CHILD_SESSION_PID" 2>/dev/null || return 1
    if verify_own_child_session direct; then
      CHILD_PHASE='direct'; CHILD_VERIFIED=1
      return 0
    fi
    "$SLEEP" 0.05
  done
  return 1
}

own_wrapper_is_verified() {
  local line pid pgid sid exe cmd
  [[ "$CHILD_WRAPPER_PID" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null || return 1
  line="$(/usr/bin/ps -o pid=,pgid=,sid= -p "$CHILD_WRAPPER_PID" 2>/dev/null | "$AWK" 'NF {print $1 "," $2 "," $3}')"
  IFS=, read -r pid pgid sid <<<"$line"
  [[ "$pid" == "$CHILD_WRAPPER_PID" && "$pgid" == "$CHILD_WRAPPER_PID" && "$sid" == "$CHILD_WRAPPER_PID" ]] || return 1
  exe="$(/usr/bin/readlink -f "/proc/$CHILD_WRAPPER_PID/exe" 2>/dev/null || true)"
  cmd="$("$TR" '\0' ' ' < "/proc/$CHILD_WRAPPER_PID/cmdline" 2>/dev/null || true)"
  [[ ( "$exe" == /usr/bin/bash || "$exe" == /bin/bash || "$exe" == /usr/bin/setsid ) && "$cmd" == *C1-V7-PROFILE-OWNED-BOOTSTRAP* && "$cmd" == *"$CHILD_READY_FILE"* && "$cmd" == *"$CHILD_GO_FILE"* && "$cmd" == *"$CHILD_ABORT_FILE"* && "$cmd" == *"$CHILD_EXPECTED_EXECUTABLE"* && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]]
}

abort_pre_adoption_and_wait() {
  local attempt
  [[ -z "$CHILD_ABORT_FILE" ]] || : > "$CHILD_ABORT_FILE" 2>/dev/null || true
  for ((attempt=1; attempt<=100; ++attempt)); do
    [[ -n "$CHILD_WRAPPER_PID" ]] && ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null && return 0
    "$SLEEP" 0.05
  done
  # A direct, separately verified bootstrap may be signalled. An unverified
  # PID is never signalled; the guard instead retains its flock until its own
  # direct setsid --wait child exits.
  if own_wrapper_is_verified; then
    kill -TERM "$CHILD_WRAPPER_PID" 2>/dev/null || true
    for ((attempt=1; attempt<=100; ++attempt)); do
      ! kill -0 "$CHILD_WRAPPER_PID" 2>/dev/null && return 0
      "$SLEEP" 0.05
    done
    if own_wrapper_is_verified; then kill -KILL "$CHILD_WRAPPER_PID" 2>/dev/null || true; fi
    return 0
  fi
  return 1
}

cleanup_own_child() {
  local reason="$1" verified=false action='NO_LIVE_CHILD' alive=false attempt
  [[ "$CHILD_BOOTSTRAP_VERIFIED" == 1 || "$CHILD_VERIFIED" == 1 ]] && verified=true
  if [[ -z "$CHILD_SESSION_PID" || "$CHILD_PHASE" == bootstrap ]]; then
    if abort_pre_adoption_and_wait; then
      action='ABORT_PRE_ADOPTION_THEN_WAIT_OR_VERIFIED_WRAPPER_SIGNAL'
    else
      action='ABORT_PRE_ADOPTION_REFUSED_UNVERIFIED_SIGNAL_HOLDING_LOCK'
    fi
  fi
  if [[ -n "$CHILD_SESSION_PID" ]] && kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then
    if verify_own_child_session direct || verify_own_child_session bootstrap; then
      verified=true
      [[ -z "$CHILD_ABORT_FILE" ]] || : > "$CHILD_ABORT_FILE" 2>/dev/null || true
      kill -TERM -- "-$CHILD_PGID" 2>/dev/null || true
      action='TERM_OWN_VERIFIED_PROCESS_GROUP'
      for ((attempt=1; attempt<=100; ++attempt)); do
        kill -0 "$CHILD_SESSION_PID" 2>/dev/null || break
        "$SLEEP" 0.05
      done
      if kill -0 "$CHILD_SESSION_PID" 2>/dev/null && ( verify_own_child_session direct || verify_own_child_session bootstrap ); then
        kill -KILL -- "-$CHILD_PGID" 2>/dev/null || true
        action='TERM_THEN_KILL_OWN_VERIFIED_PROCESS_GROUP'
      fi
    else
      action='REFUSED_UNVERIFIED_PROCESS_GROUP_HOLDING_LOCK'
    fi
  fi
  # setsid --wait is intentionally waited even after an ownership failure:
  # this prevents fd 9/flock from being released while a possible child session
  # can still exist. wait targets only this shell's direct child.
  if [[ -n "$CHILD_WRAPPER_PID" ]]; then
    wait "$CHILD_WRAPPER_PID" 2>/dev/null || true
  fi
  # GNU setsid --wait should already make the wrapper wait sufficient. If
  # that invariant ever fails, do not unlock while the recorded session leader
  # is still observable: wait without signalling an unverified process.
  if [[ -n "$CHILD_SESSION_PID" ]] && kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then
    action="${action}_HOLD_LOCK_UNTIL_SESSION_EXIT"
    while kill -0 "$CHILD_SESSION_PID" 2>/dev/null; do "$SLEEP" 0.10; done
  fi
  if [[ -n "$CHILD_SESSION_PID" ]] && kill -0 "$CHILD_SESSION_PID" 2>/dev/null; then alive=true; fi
  if ! write_child_session_record "$reason" "$verified" "$action" "$alive"; then return 1; fi
  clear_child_tracking
  [[ "$alive" == false ]]
}

spawn_owned_child_session() {
  local label="$1" out="$2" stdout_log="$3" stderr_log="$4" kind="$5" expected_exe="$6" expected_command="$7"
  shift 7
  [[ "${1:-}" == -- ]] || return 1
  shift
  [[ "$expected_exe" == /* && "$expected_command" == /* && "$out" == "$PROFILE_OUT/"* && -d "$out" ]] || return 1
  CHILD_LABEL="$label"; CHILD_OUT="$out"; CHILD_EXPECTED_KIND="$kind"; CHILD_EXPECTED_EXECUTABLE="$expected_exe"; CHILD_EXPECTED_COMMAND="$expected_command"
  local session_dir key
  session_dir="$PROFILE_OUT/child_sessions"; key="$(safe_label "$label")"
  mkdir -p -- "$session_dir" || return 1
  CHILD_READY_FILE="$session_dir/${key}.ready"; CHILD_GO_FILE="$session_dir/${key}.go"; CHILD_ABORT_FILE="$session_dir/${key}.abort"
  rm -f -- "$CHILD_READY_FILE" "$CHILD_GO_FILE" "$CHILD_ABORT_FILE"
  "$SETSID" --wait /bin/bash -c '
set -uo pipefail
ready=$1; go=$2; abort=$3; expected_exe=$4; expected_command=$5; shift 5
pgid="$(/usr/bin/ps -o pgid= -p "$$" | /usr/bin/tr -d "[:space:]")"
sid="$(/usr/bin/ps -o sid= -p "$$" | /usr/bin/tr -d "[:space:]")"
tmp="${ready}.$$.tmp"
umask 077
printf "pid=%s\npgid=%s\nsid=%s\n" "$$" "$pgid" "$sid" > "$tmp"
/usr/bin/mv -f -- "$tmp" "$ready"
while [[ ! -e "$go" ]]; do
  [[ ! -e "$abort" ]] || exit 125
  /usr/bin/sleep 0.05
done
[[ ! -e "$abort" ]] || exit 125
exec "$@"
' C1-V7-PROFILE-OWNED-BOOTSTRAP "$CHILD_READY_FILE" "$CHILD_GO_FILE" "$CHILD_ABORT_FILE" "$expected_exe" "$expected_command" "$@" >"$stdout_log" 2>"$stderr_log" &
  CHILD_WRAPPER_PID=$!
  adopt_own_child_session_from_ready
}

allow_and_verify_owned_child_exec() {
  [[ "$CHILD_PHASE" == bootstrap && -n "$CHILD_GO_FILE" ]] || return 1
  : > "$CHILD_GO_FILE" || return 1
  confirm_own_child_exec_session
}

wait_owned_child_session() {
  local rc
  [[ "$CHILD_VERIFIED" == 1 && "$CHILD_WRAPPER_PID" =~ ^[0-9]+$ ]] || return 69
  if wait "$CHILD_WRAPPER_PID"; then rc=0; else rc=$?; fi
  write_child_session_record "direct child exited" true 'WAITED_FOR_VERIFIED_OWNED_SESSION' false "$rc" || return 69
  clear_child_tracking
  return "$rc"
}

write_guard_card() {
  local status="$1" reason="$2"
  [[ -n "$PROFILE_OUT" && -d "$PROFILE_OUT" ]] || return 0
  status="$status" reason="$reason" GPU_UUID="$SESSION_UUID" "$PYTHON" -B -I - "$PROFILE_OUT/profile_guard_card_v7.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
old={}
if p.exists():
    if p.is_symlink(): raise SystemExit('profile guard card symlink')
    old=json.loads(p.read_text())
old.update({"schema":"gtspp-c1-v7-profile-guard-card-v1","updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "status":os.environ["status"],"reason":os.environ["reason"],"host":os.uname().nodename,
 "physical_gpu_index":0,"physical_gpu_uuid":os.environ.get("GPU_UUID") or None,"execution_mode":"PROFILE",
 "scope":"Explicitly unmeasured E_G/P_F Nsight allocation profile only; C1 query-only, C2 mode=0, no C3 mutations, never latency data.",
 "snapshots_dir":str(p.parent/"gpu_snapshots"),"manifest":str(p.parent/"profile_run_manifest_v7.json")})
tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(old,indent=2,sort_keys=True)+'\n');tmp.replace(p)
PY
}
manifest_event() { "$PYTHON" -B -I "$MANIFEST" event --out "$PROFILE_OUT" --kind "$1" --label "$2" --variant "$3" --detail "$4"; }
manifest_final() { "$PYTHON" -B -I "$MANIFEST" final --out "$PROFILE_OUT" --status "$1" --reason "$2"; }
verify_source_only() {
  require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$MANIFEST" && require_regular_raw "$VERIFIER" && require_regular_raw "$PLAN" && require_regular_raw "$PLAN_JSON" && require_regular_raw "$PROFILE_MANIFEST_FIXTURE" && require_regular_raw "$PROFILE_TERMINAL_FIXTURE" || return 1
  "$PYTHON" -B -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --static-source || return 1
  # Both executable fixtures are CPU-only and run before any nvidia-smi/nsys path.
  "$PYTHON" -B -I "$PROFILE_MANIFEST_FIXTURE" --root "$ROOT" || return 1
  "$PYTHON" -B -I "$PROFILE_TERMINAL_FIXTURE" --root "$ROOT" || return 1
  "$PYTHON" -B -I - "$PLAN_JSON" <<'PY' || return 1
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
def need(x,m):
    if not x: raise SystemExit(m)
plan=json.loads(p.read_text())
reports=["cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_mem_size_sum","um_sum"]
need(plan.get("schema")=="gtspp-c1-v7-profile-execution-plan-v1","schema")
need(plan.get("schedule")==[{"replicate":1,"variant":"E_G_c1_off_reference"},{"replicate":1,"variant":"P_F_full_C1"}],"schedule")
need(plan.get("binary_mode")=={"radius":500.0,"required_argument":"--profile-only","required_c2_residual_mode":0,"required_run_mode":"UNMEASURED_NSYS_PROFILE_DO_NOT_USE","trace_operations":1024},"binary mode")
nsys=plan.get("nsys",{})
need(nsys.get("executable")=="/usr/local/bin/nsys","nsys executable")
need(nsys.get("profile_arguments")==["profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true"],"profile args")
need(nsys.get("stats_arguments")==["stats","--force-export=true","--force-overwrite=true","--format","csv"],"stats args")
need(nsys.get("reports")==reports,"fixed reports")
need(nsys.get("artifact_layout")=={"nsys_rep":"nsys/profile.nsys-rep","sqlite":"nsys/profile.sqlite","stats_stdout":"nsys/stats_stdout.log","csv_reports":["nsys/reports/profile_"+x+".csv" for x in reports]},"artifact layout")
need(plan.get("um_zero_byte_contract")=={"allowed_report":"nsys/reports/profile_um_sum.csv","nonempty_reports":["nsys/reports/profile_"+x+".csv" for x in reports[:-1]],"zero_byte_requires":{"stats_stdout":"nsys/stats_stdout.log","processed_marker":"/um_sum.py] to [{absolute profile_um_sum.csv path}]... PROCESSED (EMPTY RESULTS)","no_page_fault_marker":"does not contain CUDA Unified Memory CPU page faults data."}},"UM zero-byte contract")
child=plan.get("child_environment",{})
need(child.get("env_i") is True and child.get("uuid_bound")==["CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES"],"env-I/UUID binding")
PY
  local expected actual
  expected=$'1\tE_G_c1_off_reference\n1\tP_F_full_C1'
  actual="$("$PLAN" --emit-primary-profile-plan)" || return 1
  [[ "$actual" == "$expected" ]]
}
verify_variant_material() { "$PYTHON" -B -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --variant "$1"; }

snapshot_full() {
  local label="$1" expected="$2" idle="$3" dir tag raw tele apps0 appsall line index uuid name used util apps
  dir="$PROFILE_OUT/gpu_snapshots"; mkdir -p -- "$dir" || return 2
  tag="$(safe_label "$label")"
  raw="$dir/${tag}_safety_gpu0.csv"; tele="$dir/${tag}_telemetry_gpu0.csv"; apps0="$dir/${tag}_compute_gpu0.csv"; appsall="$dir/${tag}_compute_all_visible.csv"
  "$SMI" -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits >"$raw" 2>&1 || return 2
  "$SMI" -i 0 --query-gpu=timestamp,index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit,pstate --format=csv,noheader,nounits >"$tele" 2>&1 || return 2
  "$SMI" -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >"$apps0" 2>&1 || return 2
  "$SMI" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >"$appsall" 2>&1 || return 2
  [[ "$("$GREP" -cve '^[[:space:]]*$' "$raw")" == 1 ]] || return 2
  line="$("$HEAD" -n1 "$raw" | "$TR" -d '\r')"; IFS=',' read -r index uuid name used util _ <<<"$line"
  index="$(printf '%s' "$index" | "$AWK" '{$1=$1;print}')"; uuid="$(printf '%s' "$uuid" | "$AWK" '{$1=$1;print}')"; name="$(printf '%s' "$name" | "$AWK" '{$1=$1;print}')"; used="$(printf '%s' "$used" | "$AWK" '{$1=$1;print}')"; util="$(printf '%s' "$util" | "$AWK" '{$1=$1;print}')"
  [[ "$index" == 0 && "$uuid" == GPU-* && "$name" == *'RTX PRO 6000'* ]] || return 2
  [[ -z "$expected" || "$uuid" == "$expected" ]] || return 2
  [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 2
  apps="$("$SED" '/^[[:space:]]*$/d' "$apps0")"
  if [[ -n "$apps" ]] && ! printf '%s\n' "$apps" | "$GREP" -Eq '^(No running compute processes found|No running processes found)$'; then return 3; fi
  GPU_UUID="$uuid"
  if [[ "$idle" == YES ]]; then (( used <= 256 && util == 0 )) || return 1; fi
  return 0
}
wait_idle() {
  local label="$1" expected="$2" dir tag log n rc
  dir="$PROFILE_OUT/gpu_snapshots"; mkdir -p -- "$dir" || die 'cannot create profile telemetry directory'
  tag="$(safe_label "$label")";log="$dir/${tag}_strict_idle_poll.log";: >"$log" || die 'cannot create profile idle-poll log'
  for ((n=1;n<=ATTEMPTS;n++)); do
    snapshot_full "${label}_attempt_$(printf '%02d' "$n")" "$expected" YES;rc=$?
    if [[ "$rc" -eq 0 ]]; then
      printf 'strict_idle=PASS attempt=%s uuid=%s\n' "$n" "$GPU_UUID" >>"$log"
      printf '%s\t%s\n' "$GPU_UUID" "$dir/${tag}_attempt_$(printf '%02d' "$n")_telemetry_gpu0.csv"
      return 0
    fi
    printf 'strict_idle=NOT_YET rc=%s attempt=%s\n' "$rc" "$n" >>"$log"
    [[ "$rc" -eq 1 ]] || die "GPU0 identity/telemetry/compute-app validation failed at $label"
    (( n < ATTEMPTS )) && "$SLEEP" "$POLL_SECONDS"
  done
  die 'GPU0 did not become strictly idle; no process was touched'
}
profile_completion_contract() {
  "$PYTHON" -B -I - "$1" "$2" "$3" <<'PY'
import json,math,pathlib,sys
out=pathlib.Path(sys.argv[1]);variant=sys.argv[2];expected_uuid=sys.argv[3]
def need(x,m):
 if not x:raise SystemExit(m)
gates={'E_G_c1_off_reference':(0,0),'P_F_full_C1':(1,1)}
card=json.loads((out/'run_card.json').read_text());done=json.loads((out/'completion.json').read_text())
need(card.get('schema')=='gtspp-c1-microbench-run-card-v7' and done.get('schema')=='gtspp-c1-microbench-completion-v7','schema')
need(card.get('run_mode')=='UNMEASURED_NSYS_PROFILE_DO_NOT_USE' and done.get('run_mode')=='UNMEASURED_NSYS_PROFILE_DO_NOT_USE' and card.get('profile_only') is True,'not profile-only')
need(card.get('requested_variant')==variant==card.get('compiled_variant')==done.get('compiled_variant'),'variant')
need(card.get('replicate')==1 and card.get('trace_limit')==1024 and card.get('C2_residual_mode')==0,'fixed profile contract')
need(math.isclose(float(card.get('radius')),500.0,rel_tol=0,abs_tol=0),'radius')
need((card.get('C1_PERSISTENT_WORKSPACE'),card.get('C1_ONE_QUERY_FASTPATH'))==gates[variant],'gates')
need(isinstance(card.get('visible_cuda_device_pci_bus_id'),str) and card.get('visible_cuda_device_pci_bus_id'),'PCI')
need(isinstance(card.get('visible_cuda_device_uuid'),str) and card.get('visible_cuda_device_uuid')==expected_uuid and expected_uuid.startswith('GPU-'),'CUDA UUID/session')
need(done.get('tree_invariance_pass') is True,'tree invariance')
PY
}
run_profile_variant() {
  local rep="$1" variant="$2" bin cache out label idle pre_uuid pre_snapshot rc prefix report_prefix repfile sqlite nsys_real
  [[ "$rep" == 1 ]] || die 'profile replicate must be one'
  case "$variant" in E_G_c1_off_reference|P_F_full_C1) ;; *) die 'profile variant not primary';; esac
  bin="$ROOT/builds/$variant/bin/C1Microbench";cache="$ROOT/builds/$variant/CMakeCache.txt";out="$PROFILE_OUT/$variant";label="profile/$variant"
  require_regular_raw "$bin" && require_regular_raw "$cache" || die "invalid profile binary/cache $variant"
  [[ ! -e "$out" && ! -L "$out" ]] || die "profile output exists $variant"
  mkdir -p -- "$out/nsys/reports" || die 'cannot create profile variant output'
  verify_variant_material "$variant" || die "runtime pin check failed $variant"
  idle="$(wait_idle "pre_${label}" "$SESSION_UUID")" || die "profile idle precheck failed $variant"
  IFS=$'\t' read -r pre_uuid pre_snapshot <<<"$idle";GPU_UUID="$pre_uuid"
  [[ "$pre_uuid" == "$SESSION_UUID" ]] || die "profile UUID continuity $variant"
  manifest_event profile_variant_prelaunch_pass "$label" "$variant" "pins+idle telemetry=$pre_snapshot" || die 'profile manifest prelaunch'
  "$PYTHON" -B -I "$MANIFEST" child --out "$PROFILE_OUT" --rep 1 --variant "$variant" --binary "$bin" --cmake-cache "$cache" --variant-out "$out" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" --uuid "$SESSION_UUID" || die 'profile child manifest binding'
  prefix="$out/nsys/profile"
  nsys_real="$("$REALPATH" -e -- "$NSYS")" || die 'cannot canonicalize fixed nsys executable'
  require_regular_raw "$nsys_real" || die 'canonical nsys executable is not a regular file'
  clear_child_tracking
  # The profile binary remains blocked before GO until this guard proves the
  # exact new session leader and the post-exec nsys command/binary binding.
  if ! spawn_owned_child_session "$label" "$out" "$out/stdout.log" "$out/stderr.log" profile_nsys "$nsys_real" "$bin" -- \
    /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' LD_LIBRARY_PATH='/usr/local/cuda-13.1/lib64' \
    CUDA_VISIBLE_DEVICES="$SESSION_UUID" NVIDIA_VISIBLE_DEVICES="$SESSION_UUID" CUDA_DEVICE_ORDER='PCI_BUS_ID' C1_EXECUTION_MODE='PROFILE' \
    "$NSYS" profile --trace=cuda --cuda-memory-usage=true --force-overwrite=true --output "$prefix" -- \
    "$bin" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" --radius 500 --out "$out" --replicate 1 --variant "$variant" --profile-only; then
    FINAL_STATUS='FAILED'; FINAL_REASON="cannot verify pre-GO owned profile session $variant"; exit 69
  fi
  manifest_event profile_owned_child_bootstrap_verified "$label" "$variant" "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; nsys/CUDA binary blocked on GO" || {
    FINAL_STATUS='FAILED'; FINAL_REASON="cannot record pre-GO profile ownership $variant"; exit 69; }
  if ! allow_and_verify_owned_child_exec; then
    FINAL_STATUS='FAILED'; FINAL_REASON="cannot verify post-GO nsys/binary owned session $variant"; exit 69
  fi
  manifest_event profile_owned_child_exec_verified "$label" "$variant" "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; direct canonical nsys=$nsys_real command binds $bin" || {
    FINAL_STATUS='FAILED'; FINAL_REASON="cannot record direct profile ownership $variant"; exit 69; }
  if wait_owned_child_session; then rc=0; else rc=$?; fi
  if ! snapshot_full "post_${label}" "$SESSION_UUID" NO; then FINAL_STATUS='TELEMETRY_INCOMPLETE';FINAL_REASON="profile post telemetry $variant";exit 69;fi
  manifest_event profile_variant_postlaunch_telemetry "$label" "$variant" 'post telemetry captured and UUID-bound' || die 'profile manifest postlaunch'
  [[ "$rc" -eq 0 ]] || { FINAL_STATUS='ENGINE_FAILED';FINAL_REASON="nsys/binary rc=$rc $variant";exit "$rc"; }
  profile_completion_contract "$out" "$variant" "$SESSION_UUID" || { FINAL_STATUS='CONTRACT_FAILED';FINAL_REASON="profile completion $variant";exit 69; }
  repfile="$out/nsys/profile.nsys-rep";sqlite="$out/nsys/profile.sqlite";report_prefix="$out/nsys/reports/profile"
  require_regular_raw "$repfile" || { FINAL_STATUS='PROFILE_ARTIFACT_FAILED';FINAL_REASON="missing rep $variant";exit 69; }
  "$NSYS" stats --force-export=true --force-overwrite=true --format csv --output "$report_prefix" --report cuda_api_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report um_sum "$repfile" >"$out/nsys/stats_stdout.log" 2>"$out/nsys/stats_stderr.log" || { FINAL_STATUS='PROFILE_ARTIFACT_FAILED';FINAL_REASON="nsys stats $variant";exit 69; }
  require_regular_raw "$repfile" && require_regular_raw "$sqlite" && require_regular_raw "$out/nsys/stats_stdout.log" && require_regular_raw "$out/nsys/reports/profile_cuda_api_sum.csv" && require_regular_raw "$out/nsys/reports/profile_cuda_gpu_mem_time_sum.csv" && require_regular_raw "$out/nsys/reports/profile_cuda_gpu_mem_size_sum.csv" && require_regular_raw "$out/nsys/reports/profile_um_sum.csv" || { FINAL_STATUS='FAILED';FINAL_REASON="fixed stats artifact layout $variant";exit 69; }
  "$PYTHON" -B -I "$MANIFEST" artifacts --out "$PROFILE_OUT" --variant "$variant" --rep 1 --repfile "$repfile" --sqlite "$sqlite" --stats-stdout "$out/nsys/stats_stdout.log" --report "$out/nsys/reports/profile_cuda_api_sum.csv" --report "$out/nsys/reports/profile_cuda_gpu_mem_time_sum.csv" --report "$out/nsys/reports/profile_cuda_gpu_mem_size_sum.csv" --report "$out/nsys/reports/profile_um_sum.csv" || die 'profile artifact manifest binding'
  manifest_event profile_nsys_stats_complete "$label" "$variant" 'fixed four CSV reports hash-bound; a zero-byte um_sum is only an explicit no-UM-events diagnostic and never allocation evidence' || die 'profile stats event'
  manifest_event profile_variant_completion_contract_pass "$label" "$variant" 'profile-only completion contract passed' || die 'profile completion event'
}
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "$CHILD_WRAPPER_PID" || -n "$CHILD_SESSION_PID" ]]; then
    if ! cleanup_own_child "profile_outer_guard_exit_$rc"; then
      rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='profile owned-child cleanup/provenance incomplete; flock retained until direct wrapper exit'
    fi
  fi
  # The atomic PREPARED write can win a signal race against MANIFEST_READY=1.
  # Discovering the regular manifest here makes every such failure terminal.
  if ensure_manifest_ready_from_disk; then
    if [[ "$rc" -ne 0 || "$FINAL_STATUS" != COMPLETE ]]; then
      [[ -n "$FINAL_REASON" && "$FINAL_REASON" != 'not started' ]] || FINAL_REASON='guard exited before profile completion'
      FINAL_STATUS='FAILED'
    fi
    if [[ "$SESSION_READY" == 1 ]]; then
      if ! snapshot_full outer_exit "$SESSION_UUID" NO; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='profile outer-exit telemetry missing'; fi
      if ! rm -f -- "$PROFILE_OUT/.c1_profile_guard_session_v7.json"; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='profile session removal failed'; fi
    fi
    if ! manifest_final "$FINAL_STATUS" "$FINAL_REASON"; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='profile manifest finalization failed'; fi
    if ! write_guard_card "$FINAL_STATUS" "$FINAL_REASON"; then
      rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='profile guard card finalization failed'
      manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
    fi
    if [[ "$rc" -eq 0 && "$FINAL_STATUS" == COMPLETE && "$CPU_ONLY_FIXTURE_ACTIVE" != 1 ]]; then
      if ! verify_source_only; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='profile postrun pin/CPU fixture verification failed'
      elif ! "$PYTHON" -B -I "$VERIFIER" --profile-root "$PROFILE_OUT" --pins "$PINS"; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='strict profile verifier failed'
      fi
      if [[ "$rc" -ne 0 ]]; then
        manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
        write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true
      fi
    fi
    if [[ "$rc" -ne 0 || "$FINAL_STATUS" != COMPLETE ]]; then
      FINAL_STATUS='FAILED'
      manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
      write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true
    fi
  fi
  # cleanup_own_child waits the direct setsid --wait wrapper before unlock, so
  # an unverified possible session cannot outlive this guard's flock.
  "$FLOCK" -u 9 || true
  exit "$rc"
}
signal_handler() {
  FINAL_STATUS='FAILED'
  FINAL_REASON='signal received; EXIT cleanup will only target a verified owned profile session'
  exit 130
}

run_cpu_only_fixture_mode() {
  local mode="$1" fixture_out="$2" fixture_variant
  [[ "${C1_V7_CPU_ONLY_FIXTURE:-}" == YES ]] || { echo 'C1-V7 profile fixture mode requires explicit CPU-only marker' >&2; exit 64; }
  [[ "$fixture_out" == /tmp/* && -d "$fixture_out" && -f "$fixture_out/profile_run_manifest_v7.json" && ! -L "$fixture_out/profile_run_manifest_v7.json" ]] || {
    echo 'C1-V7 profile fixture requires a prepared regular manifest under /tmp' >&2; exit 64; }
  PROFILE_OUT="$fixture_out"; RUN_OUT="$PROFILE_OUT"; SESSION_READY=0; SESSION_UUID=''; GPU_UUID=''
  CPU_ONLY_FIXTURE_ACTIVE=1; MANIFEST_READY=0; FINAL_STATUS='PREPARING'; FINAL_REASON='CPU-only profile fixture started before manifest-ready assignment'
  exec 9>>"$fixture_out/.fixture_guard.lock"; "$FLOCK" -n 9 || { echo 'fixture lock unavailable' >&2; exit 69; }
  trap cleanup EXIT; trap signal_handler INT TERM HUP
  fixture_variant="$fixture_out/E_G_c1_off_reference"
  mkdir -p -- "$fixture_variant" || { FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot create first-profile-variant directory'; exit 69; }
  clear_child_tracking
  if ! spawn_owned_child_session 'fixture/profile/E_G_c1_off_reference' "$fixture_variant" "$fixture_variant/stdout.log" "$fixture_variant/stderr.log" cpu_fixture /usr/bin/sleep /usr/bin/sleep -- /usr/bin/sleep 1; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot verify profile CPU bootstrap ownership'; exit 69
  fi
  manifest_event fixture_owned_bootstrap_verified 'fixture/profile/E_G_c1_off_reference' E_G_c1_off_reference "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; no GO yet" || {
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot record owned profile bootstrap'; exit 69; }
  if [[ "$mode" == --fixture-first-variant-midfailure ]]; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture first-profile-variant mid-failure before GO with MANIFEST_READY=0'
    exit 69
  fi
  [[ "$mode" == --fixture-owned-session ]] || { FINAL_STATUS='FAILED'; FINAL_REASON='unknown CPU profile fixture mode'; exit 64; }
  if ! allow_and_verify_owned_child_exec; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot verify direct profile CPU session'; exit 69
  fi
  manifest_event fixture_owned_exec_verified 'fixture/profile/E_G_c1_off_reference' E_G_c1_off_reference "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; /usr/bin/sleep direct exec verified" || {
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot record direct profile CPU ownership'; exit 69; }
  if ! wait_owned_child_session; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture profile CPU child exited nonzero'; exit 69
  fi
  FINAL_STATUS='COMPLETE'; FINAL_REASON='CPU-only profile owned-session fixture passed with MANIFEST_READY=0'
  exit 0
}

if [[ "$#" -gt 0 && "$1" == --fixture-terminal-failure ]];then
  shift;[[ "$#" -eq 1 ]] || { echo 'Usage: run_c1_profile_guard_v7.sh --fixture-terminal-failure ABS_PROFILE_OUT' >&2; exit 64; }
  PROFILE_OUT="$1";RUN_OUT="$PROFILE_OUT";MANIFEST_READY=0;SESSION_READY=0;FINAL_STATUS='PREPARING';FINAL_REASON='fixture midpoint before manifest-ready assignment'
  [[ "$PROFILE_OUT" == /tmp/* && -d "$PROFILE_OUT" && -f "$PROFILE_OUT/profile_run_manifest_v7.json" && ! -L "$PROFILE_OUT/profile_run_manifest_v7.json" ]] || { echo 'C1-V7-PROFILE-GUARD fixture requires prepared regular manifest under /tmp' >&2; exit 64; }
  exec 9>>"$PROFILE_OUT/.fixture_guard.lock"; "$FLOCK" -n 9 || { echo 'fixture lock unavailable' >&2; exit 69; }
  trap cleanup EXIT
  die 'fixture injected midpoint failure after PREPARED before MANIFEST_READY'
fi
if [[ "$#" -gt 0 && ( "$1" == --fixture-owned-session || "$1" == --fixture-first-variant-midfailure ) ]]; then
  mode="$1"; shift; [[ "$#" -eq 1 ]] || { echo 'Usage: profile fixture mode ABS_TMP_OUT' >&2; exit 64; }
  run_cpu_only_fixture_mode "$mode" "$1"
fi
if [[ "$#" -gt 0 && "$1" == --verify-source-only ]];then shift;[[ "$#" -eq 0 ]] || die 'extra source-only arguments';verify_source_only;exit $?;fi
[[ "$#" -eq 0 ]] || die 'unsupported profile guard subcommand'
C1_V7_PROFILE_TRUST_ROOT="$(read_env C1_V7_PROFILE_TRUST_ROOT)";C1_V7_PROFILE_LAUNCHER="$(read_env C1_V7_PROFILE_LAUNCHER)";C1_V7_PROFILE_TRUST_GUARD_SHA="$(read_env C1_V7_PROFILE_TRUST_GUARD_SHA)";C1_V7_PROFILE_TRUST_PINS_SHA="$(read_env C1_V7_PROFILE_TRUST_PINS_SHA)";C1_V7_PROFILE_TRUST_PIN_VERIFY_SHA="$(read_env C1_V7_PROFILE_TRUST_PIN_VERIFY_SHA)";C1_ALLOW_GPU0="$(read_env C1_ALLOW_GPU0)";RUN_OUT="$(read_env C1_PROFILE_OUT)"
[[ "$0" == "$GUARD" && "$C1_V7_PROFILE_TRUST_ROOT" == YES && "$C1_V7_PROFILE_LAUNCHER" == "$LAUNCHER" ]] || die 'reviewed profile trust-root required'
require_regular_raw "$LAUNCHER" && require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$MANIFEST" && require_regular_raw "$VERIFIER" && require_regular_raw "$PLAN" && require_regular_raw "$PLAN_JSON" && [[ -x "$NSYS" ]] || die 'profile runtime file missing/symlink'
[[ "$(sha "$GUARD")" == "$C1_V7_PROFILE_TRUST_GUARD_SHA" && "$(sha "$PINS")" == "$C1_V7_PROFILE_TRUST_PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$C1_V7_PROFILE_TRUST_PIN_VERIFY_SHA" ]] || die 'profile trust-root bootstrap hash mismatch'
[[ "$C1_ALLOW_GPU0" == YES && "$(hostname)" == "$HOST" ]] || die 'profile GPU0/host constraint failed'
[[ -n "$RUN_OUT" && "$RUN_OUT" == /* ]] || die 'absolute profile launcher output required'
BASE="$ROOT/profiles";OUT="$("$REALPATH" -m -- "$RUN_OUT")" || die 'bad profile output';is_under "$OUT" "$BASE" && [[ "$(basename "$OUT")" == c1_v7_profile_* && ! -e "$OUT" && ! -L "$OUT" ]] || die 'unsafe profile output'
PROFILE_OUT="$OUT";export PROFILE_OUT
exec 9>>"$LOCK";"$FLOCK" -n 9 || die 'another v7 GPU0 session holds lock'
trap cleanup EXIT;trap signal_handler INT TERM HUP
verify_source_only || die 'initial profile source/pin verification failed'
mkdir -p -- "$BASE" && mkdir -- "$PROFILE_OUT" || die 'cannot create profile root'
"$PYTHON" -B -I "$MANIFEST" init --out "$PROFILE_OUT" --root "$ROOT" --pins "$PINS" --launcher "$LAUNCHER" --guard "$GUARD" -- "$GUARD" || die 'profile manifest init'
MANIFEST_READY=1
idle="$(wait_idle outer_prelaunch '')" || die 'profile outer idle precheck';IFS=$'\t' read -r SESSION_UUID snapshot <<<"$idle";GPU_UUID="$SESSION_UUID";[[ "$SESSION_UUID" == GPU-* ]] || die 'profile outer UUID'
"$PYTHON" -B -I "$MANIFEST" gpu --out "$PROFILE_OUT" --uuid "$SESSION_UUID" --snapshot "$snapshot" || die 'profile manifest GPU'
"$PYTHON" -B -I - "$PROFILE_OUT/.c1_profile_guard_session_v7.json" "$SESSION_UUID" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.write_text(json.dumps({'schema':'gtspp-c1-v7-profile-session-v1','physical_gpu_index':0,'physical_gpu_uuid':sys.argv[2]})+'\n')
PY
chmod 0600 "$PROFILE_OUT/.c1_profile_guard_session_v7.json" || die 'profile session chmod'
SESSION_READY=1
write_guard_card ENGINE_RUNNING 'outer profile guard pins/provenance/idle passed' || die 'initial profile guard card'
manifest_event outer_prelaunch_pass '' '' "strict idle uuid=$SESSION_UUID" || die 'profile manifest outer prelaunch'
expected_plan=('1:E_G_c1_off_reference' '1:P_F_full_C1');count=0
while IFS=$'\t' read -r rep variant;do
  [[ -n "$rep" && -n "$variant" && "$count" -lt 2 ]] || die 'malformed/overlong profile plan'
  expected="${expected_plan[$count]}";[[ "$rep:$variant" == "$expected" ]] || die "profile plan mismatch got $rep:$variant expected $expected"
  run_profile_variant "$rep" "$variant";count=$((count+1))
done < <("$PLAN" --emit-primary-profile-plan)
[[ "$count" -eq 2 ]] || die 'profile plan count'
manifest_event profile_schedule_verified '' '' 'exact fixed E_G/P_F profile schedule verified before every outer-owned nsys launch' || die 'profile schedule event'
verify_source_only || die 'profile postengine pin check'
if ! snapshot_full outer_postrun "$SESSION_UUID" NO;then FINAL_STATUS='TELEMETRY_INCOMPLETE';FINAL_REASON='profile outer-postrun telemetry missing';exit 69;fi
FINAL_STATUS='COMPLETE';FINAL_REASON='outer-owned unmeasured profile execution and fixed Nsight artifact collection passed'
exit 0

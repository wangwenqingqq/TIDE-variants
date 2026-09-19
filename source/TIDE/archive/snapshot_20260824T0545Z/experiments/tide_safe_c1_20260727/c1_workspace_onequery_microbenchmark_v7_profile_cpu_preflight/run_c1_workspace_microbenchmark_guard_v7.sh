#!/usr/bin/env bash
# V7 outer guard: only this process launches C1Microbench.  There are no
# public before/after helper subcommands, so an engine cannot bypass trust,
# host, lock, pin, UUID, or telemetry checks.
set -uo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_cpu_preflight'
HOST='CONFIGURE_ARCHIVE_HOST'
GUARD="$ROOT/run_c1_workspace_microbenchmark_guard_v7.sh"
LAUNCHER="$ROOT/launch_c1_v7.sh"
PLAN="$ROOT/run_c1_four_variants_v7.sh"
PLAN_JSON="$ROOT/measured_execution_plan_v7.json"
PINS="$ROOT/hardened_static_pins_v7.json"
PIN_VERIFY="$ROOT/verify_v7_pins.py"
MANIFEST="$ROOT/c1_v7_manifest.py"
SEMANTIC="$ROOT/verify_c1_semantics_v7.py"
ANALYZER="$ROOT/analyze_c1_microbenchmark_v7.py"
FINALIZER="$ROOT/finalize_c1_evidence_v7.py"
PYTHON='/usr/bin/python3'
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
# A child is never targeted merely because it is the shell's background PID.
# The fields below are populated only after a ready/go/abort bootstrap proves
# that the actual session leader is this guard's direct setsid --wait child.
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
# Manifest readiness is deliberately redundant with an on-disk check in EXIT
# cleanup. A signal may arrive after atomic init writes PREPARED but before a
# shell assignment can set this flag.
MANIFEST_READY=0
SESSION_READY=0
FINAL_STATUS='PREPARING'
FINAL_REASON='not started'
RUN_OUT=''

die() { printf 'C1-V7-GUARD BLOCKED: %s\n' "$*" >&2; exit 69; }
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
safe_label() { printf '%s' "$1" | "$TR" -cs 'A-Za-z0-9._-' '_'; }
is_under() { [[ "$1" == "$2/"* ]]; }
require_regular_raw() {
  local p="$1" resolved
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  resolved="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$resolved" == "$p" ]]
}
read_env() {
  local key="$1"
  if [[ -v "$key" ]]; then
    printenv "$key"
  fi
}

manifest_exists() {
  [[ -n "$C1_RUN_OUT" && -d "$C1_RUN_OUT" && -f "$C1_RUN_OUT/run_manifest_v7.json" && ! -L "$C1_RUN_OUT/run_manifest_v7.json" ]]
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
 "schema":"gtspp-c1-v7-owned-child-cleanup-v1",
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
    [[ "$cmd" == *C1-V7-MEASURED-OWNED-BOOTSTRAP* && "$cmd" == *"$CHILD_READY_FILE"* && "$cmd" == *"$CHILD_GO_FILE"* && "$cmd" == *"$CHILD_ABORT_FILE"* && "$cmd" == *"$CHILD_EXPECTED_EXECUTABLE"* && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]] || return 1
    return 0
  fi
  case "$CHILD_EXPECTED_KIND" in
    measured_binary)
      [[ "$exe" == "$CHILD_EXPECTED_EXECUTABLE" && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]] ;;
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
  [[ ( "$exe" == /usr/bin/bash || "$exe" == /bin/bash || "$exe" == /usr/bin/setsid ) && "$cmd" == *C1-V7-MEASURED-OWNED-BOOTSTRAP* && "$cmd" == *"$CHILD_READY_FILE"* && "$cmd" == *"$CHILD_GO_FILE"* && "$cmd" == *"$CHILD_ABORT_FILE"* && "$cmd" == *"$CHILD_EXPECTED_EXECUTABLE"* && "$cmd" == *"$CHILD_EXPECTED_COMMAND"* ]]
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
  [[ "$expected_exe" == /* && "$expected_command" == /* && "$out" == "$C1_RUN_OUT/"* && -d "$out" ]] || return 1
  CHILD_LABEL="$label"; CHILD_OUT="$out"; CHILD_EXPECTED_KIND="$kind"; CHILD_EXPECTED_EXECUTABLE="$expected_exe"; CHILD_EXPECTED_COMMAND="$expected_command"
  local session_dir key
  session_dir="$C1_RUN_OUT/child_sessions"; key="$(safe_label "$label")"
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
' C1-V7-MEASURED-OWNED-BOOTSTRAP "$CHILD_READY_FILE" "$CHILD_GO_FILE" "$CHILD_ABORT_FILE" "$expected_exe" "$expected_command" "$@" >"$stdout_log" 2>"$stderr_log" &
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
  [[ -n "$C1_RUN_OUT" && -d "$C1_RUN_OUT" ]] || return 0
  status="$status" reason="$reason" GPU_UUID="$SESSION_UUID" "$PYTHON" -B -I - "$C1_RUN_OUT/guard_run_card_v7.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
old={}
if p.exists():
    try: old=json.loads(p.read_text())
    except Exception as exc: raise SystemExit(f"invalid prior guard card: {exc}")
old.update({"schema":"gtspp-c1-v7-guard-card-v2",
 "updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "status":os.environ["status"],"reason":os.environ["reason"],
 "host":os.uname().nodename,"physical_gpu_index":0,
 "physical_gpu_uuid":os.environ.get("GPU_UUID") or None,
 "execution_mode":"MEASURED",
 "scope":"C1 only: query-only SIFT1M, C2 mode=0, no C3 mutations",
 "snapshots_dir":str(p.parent/"gpu_snapshots"),
 "manifest":str(p.parent/"run_manifest_v7.json")})
tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(old,indent=2,sort_keys=True)+"\n"); tmp.replace(p)
PY
}
manifest_event() { "$PYTHON" -B -I "$MANIFEST" event --out "$C1_RUN_OUT" --kind "$1" --label "$2" --variant "$3" --detail "$4"; }
manifest_final() { "$PYTHON" -B -I "$MANIFEST" final --out "$C1_RUN_OUT" --status "$1" --reason "$2"; }
verify_source_only() {
  require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$MANIFEST" || return 1
  "$PYTHON" -B -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --static-source
}
verify_variant_material() { "$PYTHON" -B -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --variant "$1"; }

snapshot_full() {
  local label="$1" expected="$2" idle="$3" dir tag raw tele apps0 appsall line index uuid name used util apps
  dir="$C1_RUN_OUT/gpu_snapshots"; mkdir -p -- "$dir" || return 2
  tag="$(safe_label "$label")"
  raw="$dir/$tag"_safety_gpu0.csv
  tele="$dir/$tag"_telemetry_gpu0.csv
  apps0="$dir/$tag"_compute_gpu0.csv
  appsall="$dir/$tag"_compute_all_visible.csv
  "$SMI" -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits >"$raw" 2>&1 || return 2
  "$SMI" -i 0 --query-gpu=timestamp,index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit,pstate --format=csv,noheader,nounits >"$tele" 2>&1 || return 2
  "$SMI" -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >"$apps0" 2>&1 || return 2
  "$SMI" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >"$appsall" 2>&1 || return 2
  [[ "$("$GREP" -cve '^[[:space:]]*$' "$raw")" == 1 ]] || return 2
  line="$("$HEAD" -n1 "$raw" | "$TR" -d '\r')"
  IFS=',' read -r index uuid name used util _ <<<"$line"
  index="$(printf '%s' "$index" | "$AWK" '{$1=$1;print}')"
  uuid="$(printf '%s' "$uuid" | "$AWK" '{$1=$1;print}')"
  name="$(printf '%s' "$name" | "$AWK" '{$1=$1;print}')"
  used="$(printf '%s' "$used" | "$AWK" '{$1=$1;print}')"
  util="$(printf '%s' "$util" | "$AWK" '{$1=$1;print}')"
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
  dir="$C1_RUN_OUT/gpu_snapshots"; mkdir -p -- "$dir" || die 'cannot create telemetry directory'
  tag="$(safe_label "$label")"; log="$dir/$tag"_strict_idle_poll.log; : >"$log" || die 'cannot create idle-poll log'
  for ((n=1;n<=ATTEMPTS;n++)); do
    snapshot_full "$label"_attempt_"$(printf '%02d' "$n")" "$expected" YES; rc=$?
    if [[ "$rc" -eq 0 ]]; then
      printf 'strict_idle=PASS attempt=%s uuid=%s\n' "$n" "$GPU_UUID" >>"$log"
      printf '%s\t%s\n' "$GPU_UUID" "$dir/$tag"_attempt_"$(printf '%02d' "$n")"_telemetry_gpu0.csv
      return 0
    fi
    printf 'strict_idle=NOT_YET rc=%s attempt=%s\n' "$rc" "$n" >>"$log"
    [[ "$rc" -eq 1 ]] || die "GPU0 identity/telemetry/compute-app validation failed at $label"
    (( n < ATTEMPTS )) && "$SLEEP" "$POLL_SECONDS"
  done
  die "GPU0 did not become strictly idle; no process was touched"
}
completion_contract() {
  "$PYTHON" -B -I - "$1" "$2" "$3" "$4" <<'PY'
import json,math,pathlib,sys
out=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; rep=int(sys.argv[3]); expected_uuid=sys.argv[4]
def need(x,m):
    if not x: raise SystemExit(m)
card=json.loads((out/"run_card.json").read_text()); done=json.loads((out/"completion.json").read_text())
gates={"E_G_c1_off_reference":(0,0),"P_G_workspace_only":(1,0),"E_F_fastpath_only":(0,1),"P_F_full_C1":(1,1)}
need(variant in gates,"unknown variant")
need(card.get("schema")=="gtspp-c1-microbench-run-card-v7","run-card schema")
need(done.get("schema")=="gtspp-c1-microbench-completion-v7","completion schema")
need(card.get("run_mode")=="MEASURED_PROTOCOL" and done.get("run_mode")=="MEASURED_PROTOCOL","not measured")
need(card.get("profile_only") is False,"profile output")
need(card.get("requested_variant")==variant==card.get("compiled_variant")==done.get("compiled_variant"),"variant mismatch")
need(card.get("replicate")==rep,"replicate mismatch")
need((card.get("C1_PERSISTENT_WORKSPACE"),card.get("C1_ONE_QUERY_FASTPATH"))==gates[variant],"gate mismatch")
need(card.get("C2_residual_mode")==0,"C2 mode")
need(math.isclose(float(card.get("radius")),500.0,rel_tol=0.0,abs_tol=0.0),"radius")
need(card.get("trace_limit")==1024 and card.get("cuda_runtime_version",0)>0 and card.get("cuda_driver_version",0)>0,"runtime provenance")
need(str(card.get("visible_cuda_device_name","")).startswith("NVIDIA RTX PRO 6000"),"device provenance")
need(isinstance(card.get("visible_cuda_device_pci_bus_id"),str) and card.get("visible_cuda_device_pci_bus_id"),"device PCI provenance")
need(isinstance(card.get("visible_cuda_device_uuid"),str) and card.get("visible_cuda_device_uuid")==expected_uuid and expected_uuid.startswith("GPU-"),"device UUID/session provenance")
need(done.get("tree_invariance_pass") is True,"tree invariance")
PY
}
run_variant() {
  local rep="$1" variant="$2" bin out label idle pre_uuid pre_snapshot rc
  [[ "$rep" =~ ^[1-5]$ ]] || die 'bad replicate'
  case "$variant" in E_G_c1_off_reference|P_G_workspace_only|E_F_fastpath_only|P_F_full_C1) ;; *) die 'bad variant';; esac
  bin="$ROOT/builds/$variant/bin/C1Microbench"; out="$C1_RUN_OUT/rep$rep/$variant"; label="rep$rep/$variant"
  require_regular_raw "$bin" || die "invalid binary $variant"
  [[ ! -e "$out" && ! -L "$out" ]] || die "output exists $label"
  mkdir -p -- "$(dirname "$out")" && mkdir -- "$out" || die 'cannot create variant output'
  verify_variant_material "$variant" || die "runtime pin check failed $variant"
  idle="$(wait_idle pre_"$label" "$SESSION_UUID")" || die "idle precheck failed $label"
  IFS=$'\t' read -r pre_uuid pre_snapshot <<<"$idle"
  GPU_UUID="$pre_uuid"
  [[ "$pre_uuid" == "$SESSION_UUID" && "$GPU_UUID" == "$SESSION_UUID" ]] || die "UUID continuity failure $label"
  manifest_event variant_prelaunch_pass "$label" "$variant" "pins+idle telemetry=$pre_snapshot" || die 'manifest prelaunch'
  "$PYTHON" -B -I "$MANIFEST" child --out "$C1_RUN_OUT" --rep "$rep" --variant "$variant" --binary "$bin" --variant-out "$out" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" --uuid "$SESSION_UUID" || die 'manifest child binding'
  clear_child_tracking
  # Bootstrap is a new, waitable session that cannot exec the CUDA binary
  # until its observed PID=PGID=SID and expected command are verified.
  if ! spawn_owned_child_session "$label" "$out" "$out/stdout.log" "$out/stderr.log" measured_binary "$bin" "$bin" -- \
    /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' LD_LIBRARY_PATH='/usr/local/cuda-13.1/lib64' \
    CUDA_VISIBLE_DEVICES="$SESSION_UUID" NVIDIA_VISIBLE_DEVICES="$SESSION_UUID" CUDA_DEVICE_ORDER='PCI_BUS_ID' C1_EXECUTION_MODE='MEASURED' \
    "$bin" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" \
    --radius 500 --out "$out" --replicate "$rep" --variant "$variant"; then
    FINAL_STATUS='CHILD_OWNERSHIP_FAILED'; FINAL_REASON="cannot verify pre-GO owned session $label"; exit 69
  fi
  manifest_event owned_child_bootstrap_verified "$label" "$variant" "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; binary is still blocked on GO" || {
    FINAL_STATUS='CHILD_OWNERSHIP_FAILED'; FINAL_REASON="cannot record pre-GO ownership $label"; exit 69; }
  if ! allow_and_verify_owned_child_exec; then
    FINAL_STATUS='CHILD_OWNERSHIP_FAILED'; FINAL_REASON="cannot verify post-GO direct binary session $label"; exit 69
  fi
  manifest_event owned_child_exec_verified "$label" "$variant" "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; direct executable=$bin" || {
    FINAL_STATUS='CHILD_OWNERSHIP_FAILED'; FINAL_REASON="cannot record direct ownership $label"; exit 69; }
  if wait_owned_child_session; then rc=0; else rc=$?; fi
  if ! snapshot_full post_"$label" "$SESSION_UUID" NO; then FINAL_STATUS='TELEMETRY_INCOMPLETE'; FINAL_REASON="post telemetry $label"; exit 69; fi
  manifest_event variant_postlaunch_telemetry "$label" "$variant" 'post telemetry captured and UUID-bound' || die 'manifest postlaunch'
  [[ "$rc" -eq 0 ]] || { FINAL_STATUS='ENGINE_FAILED'; FINAL_REASON="binary rc=$rc $label"; exit "$rc"; }
  completion_contract "$out" "$variant" "$rep" "$SESSION_UUID" || { FINAL_STATUS='CONTRACT_FAILED'; FINAL_REASON="completion $label"; exit 69; }
  manifest_event variant_completion_contract_pass "$label" "$variant" 'measured completion contract passed' || die 'manifest contract'
}
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "$CHILD_WRAPPER_PID" || -n "$CHILD_SESSION_PID" ]]; then
    if ! cleanup_own_child "outer_guard_exit_$rc"; then
      rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='owned child cleanup/provenance incomplete; flock retained until direct wrapper exit'
    fi
  fi
  # Atomic init may have completed immediately before a signal. Never rely on
  # MANIFEST_READY alone: discovering a regular manifest here closes that race.
  if ensure_manifest_ready_from_disk; then
    if [[ "$rc" -ne 0 || "$FINAL_STATUS" != COMPLETE ]]; then
      [[ -n "$FINAL_REASON" && "$FINAL_REASON" != 'not started' ]] || FINAL_REASON='guard exited before measured completion'
      FINAL_STATUS='FAILED'
    fi
    if [[ "$SESSION_READY" == 1 ]]; then
      if ! snapshot_full outer_exit "$SESSION_UUID" NO; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='outer-exit telemetry missing'; fi
      if ! rm -f -- "$C1_RUN_OUT/.c1_guard_session_v7.json"; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='session removal failed'; fi
    fi
    # Write the manifest first so every prepared root has a terminal state even
    # if the human-readable card writer itself later fails.
    if ! manifest_final "$FINAL_STATUS" "$FINAL_REASON"; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='manifest finalization failed'; fi
    if ! write_guard_card "$FINAL_STATUS" "$FINAL_REASON"; then
      rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='guard card finalization failed'
      manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
    fi
    if [[ "$rc" -eq 0 && "$FINAL_STATUS" == COMPLETE && "$CPU_ONLY_FIXTURE_ACTIVE" != 1 ]]; then
      if ! verify_source_only; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='postrun pin verification failed'
      elif ! "$PYTHON" -B -I "$FINALIZER" --run-root "$C1_RUN_OUT" --pins "$PINS"; then rc=69; FINAL_STATUS='FAILED'; FINAL_REASON='final evidence attestation failed'
      fi
      if [[ "$rc" -ne 0 ]]; then
        manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
        write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true
      fi
    fi
    # A failed card/finalizer retry must never leave a stale COMPLETE manifest.
    if [[ "$rc" -ne 0 || "$FINAL_STATUS" != COMPLETE ]]; then
      FINAL_STATUS='FAILED'
      manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
      write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true
    fi
  fi
  # cleanup_own_child waits the direct setsid --wait wrapper before this point;
  # therefore fd 9 cannot be released while an unverified possible session lives.
  "$FLOCK" -u 9 || true
  exit "$rc"
}
signal_handler() {
  FINAL_STATUS='FAILED'
  FINAL_REASON='signal received; EXIT cleanup will only target a verified owned session'
  exit 130
}

run_cpu_only_fixture_mode() {
  local mode="$1" fixture_out="$2" fixture_variant
  [[ "${C1_V7_CPU_ONLY_FIXTURE:-}" == YES ]] || { echo 'C1-V7 fixture mode requires explicit CPU-only marker' >&2; exit 64; }
  [[ "$fixture_out" == /tmp/* && -d "$fixture_out" && -f "$fixture_out/run_manifest_v7.json" && ! -L "$fixture_out/run_manifest_v7.json" ]] || {
    echo 'C1-V7 fixture requires a prepared regular manifest under /tmp' >&2; exit 64; }
  C1_RUN_OUT="$fixture_out"; RUN_OUT="$fixture_out"; SESSION_READY=0; SESSION_UUID=''; GPU_UUID=''
  # Deliberately leave this zero after PREPARED exists: the EXIT path must find
  # the on-disk manifest rather than depend on a post-init shell assignment.
  CPU_ONLY_FIXTURE_ACTIVE=1; MANIFEST_READY=0; FINAL_STATUS='PREPARING'; FINAL_REASON='CPU-only fixture started before manifest-ready assignment'
  exec 9>>"$fixture_out/.fixture_guard.lock"; "$FLOCK" -n 9 || { echo 'fixture lock unavailable' >&2; exit 69; }
  trap cleanup EXIT; trap signal_handler INT TERM HUP
  fixture_variant="$fixture_out/rep1/E_G_c1_off_reference"
  mkdir -p -- "$fixture_variant" || { FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot create first-variant directory'; exit 69; }
  clear_child_tracking
  if ! spawn_owned_child_session 'fixture/rep1/E_G_c1_off_reference' "$fixture_variant" "$fixture_variant/stdout.log" "$fixture_variant/stderr.log" cpu_fixture /usr/bin/sleep /usr/bin/sleep -- /usr/bin/sleep 1; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot verify CPU bootstrap ownership'; exit 69
  fi
  manifest_event fixture_owned_bootstrap_verified 'fixture/rep1/E_G_c1_off_reference' E_G_c1_off_reference "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; no GO yet" || {
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot record owned bootstrap'; exit 69; }
  if [[ "$mode" == --fixture-first-variant-midfailure ]]; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture first-variant mid-failure before GO with MANIFEST_READY=0'
    exit 69
  fi
  [[ "$mode" == --fixture-owned-session ]] || { FINAL_STATUS='FAILED'; FINAL_REASON='unknown CPU fixture mode'; exit 64; }
  if ! allow_and_verify_owned_child_exec; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot verify direct CPU session'; exit 69
  fi
  manifest_event fixture_owned_exec_verified 'fixture/rep1/E_G_c1_off_reference' E_G_c1_off_reference "pid=$CHILD_SESSION_PID pgid=$CHILD_PGID sid=$CHILD_SID; /usr/bin/sleep direct exec verified" || {
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture cannot record direct CPU ownership'; exit 69; }
  if ! wait_owned_child_session; then
    FINAL_STATUS='FAILED'; FINAL_REASON='fixture CPU child exited nonzero'; exit 69
  fi
  FINAL_STATUS='COMPLETE'; FINAL_REASON='CPU-only owned-session fixture passed with MANIFEST_READY=0'
  exit 0
}

if [[ "$#" -gt 0 && ( "$1" == --fixture-owned-session || "$1" == --fixture-first-variant-midfailure ) ]]; then
  mode="$1"; shift; [[ "$#" -eq 1 ]] || { echo 'Usage: measured fixture mode ABS_TMP_OUT' >&2; exit 64; }
  run_cpu_only_fixture_mode "$mode" "$1"
fi
if [[ "$#" -gt 0 && "$1" == --verify-source-only ]]; then shift; [[ "$#" -eq 0 ]] || die 'extra source-only arguments'; verify_source_only; exit $?; fi
[[ "$#" -eq 0 ]] || die 'unsupported guard subcommand'
C1_V7_TRUST_ROOT="$(read_env C1_V7_TRUST_ROOT)"
C1_V7_LAUNCHER="$(read_env C1_V7_LAUNCHER)"
C1_V7_TRUST_GUARD_SHA="$(read_env C1_V7_TRUST_GUARD_SHA)"
C1_V7_TRUST_PINS_SHA="$(read_env C1_V7_TRUST_PINS_SHA)"
C1_V7_TRUST_PIN_VERIFY_SHA="$(read_env C1_V7_TRUST_PIN_VERIFY_SHA)"
C1_ALLOW_GPU0="$(read_env C1_ALLOW_GPU0)"
RUN_OUT="$(read_env C1_RUN_OUT)"
[[ "$0" == "$GUARD" && "$C1_V7_TRUST_ROOT" == YES && "$C1_V7_LAUNCHER" == "$LAUNCHER" ]] || die 'reviewed trust-root required'
require_regular_raw "$LAUNCHER" && require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$PLAN" && require_regular_raw "$PLAN_JSON" && require_regular_raw "$SEMANTIC" && require_regular_raw "$ANALYZER" && require_regular_raw "$FINALIZER" || die 'runtime file missing/symlink'
[[ "$(sha "$GUARD")" == "$C1_V7_TRUST_GUARD_SHA" && "$(sha "$PINS")" == "$C1_V7_TRUST_PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$C1_V7_TRUST_PIN_VERIFY_SHA" ]] || die 'trust-root bootstrap hash mismatch'
[[ "$C1_ALLOW_GPU0" == YES && "$(hostname)" == "$HOST" ]] || die 'GPU0/host constraint failed'
[[ -n "$RUN_OUT" && "$RUN_OUT" == /* ]] || die 'absolute launcher output required'
BASE="$ROOT/runs"; OUT="$("$REALPATH" -m -- "$RUN_OUT")" || die 'bad output'
is_under "$OUT" "$BASE" && [[ "$(basename "$OUT")" == c1_v7_measured_* && ! -e "$OUT" && ! -L "$OUT" ]] || die 'unsafe output'
C1_RUN_OUT="$OUT"; export C1_RUN_OUT
exec 9>>"$LOCK"; "$FLOCK" -n 9 || die 'another v7 GPU0 session holds lock'
trap cleanup EXIT; trap signal_handler INT TERM HUP
verify_source_only || die 'initial source/pin verification failed'
mkdir -p -- "$BASE" && mkdir -- "$C1_RUN_OUT" || die 'cannot create run root'
"$PYTHON" -B -I "$MANIFEST" init --out "$C1_RUN_OUT" --root "$ROOT" --pins "$PINS" --launcher "$LAUNCHER" --mode MEASURED -- "$GUARD" || die 'manifest init'
MANIFEST_READY=1
"$PYTHON" -B -I "$MANIFEST" bind-plan --out "$C1_RUN_OUT" --execution-plan "$PLAN_JSON" || die 'manifest plan binding'
idle="$(wait_idle outer_prelaunch '')" || die 'outer idle precheck'
IFS=$'\t' read -r SESSION_UUID snapshot <<<"$idle"
GPU_UUID="$SESSION_UUID"
[[ "$SESSION_UUID" == GPU-* && "$SESSION_UUID" == "$GPU_UUID" ]] || die 'outer UUID'
"$PYTHON" -B -I "$MANIFEST" gpu --out "$C1_RUN_OUT" --uuid "$SESSION_UUID" --snapshot "$snapshot" || die 'manifest GPU'
"$PYTHON" -B -I - "$C1_RUN_OUT/.c1_guard_session_v7.json" "$SESSION_UUID" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({"schema":"gtspp-c1-v7-session-v2","physical_gpu_index":0,"physical_gpu_uuid":sys.argv[2]})+"\n")
PY
chmod 0600 "$C1_RUN_OUT/.c1_guard_session_v7.json" || die 'session chmod'
SESSION_READY=1
write_guard_card ENGINE_RUNNING 'outer guard pins/provenance/idle passed' || die 'initial guard card'
manifest_event outer_prelaunch_pass '' '' "strict idle uuid=$SESSION_UUID" || die 'manifest outer prelaunch'
expected_plan=(
 '1:E_G_c1_off_reference' '1:P_G_workspace_only' '1:E_F_fastpath_only' '1:P_F_full_C1'
 '2:P_G_workspace_only' '2:E_F_fastpath_only' '2:P_F_full_C1' '2:E_G_c1_off_reference'
 '3:E_F_fastpath_only' '3:P_F_full_C1' '3:E_G_c1_off_reference' '3:P_G_workspace_only'
 '4:P_F_full_C1' '4:E_G_c1_off_reference' '4:P_G_workspace_only' '4:E_F_fastpath_only'
 '5:E_G_c1_off_reference' '5:E_F_fastpath_only' '5:P_G_workspace_only' '5:P_F_full_C1'
)
count=0
while IFS=$'\t' read -r rep variant; do
  [[ -n "$rep" && -n "$variant" && "$count" -lt 20 ]] || die 'malformed/overlong plan'
  expected="${expected_plan[$count]}"
  [[ "$rep:$variant" == "$expected" ]] || die "plan tuple mismatch: got $rep:$variant expected $expected"
  run_variant "$rep" "$variant"; count=$((count+1))
done < <("$PLAN" --emit-measured-plan)
[[ "$count" -eq 20 ]] || die 'plan count'
manifest_event execution_schedule_verified '' '' 'exact fixed 20-tuple Latin schedule verified before every outer-owned launch' || die 'manifest schedule event'
verify_source_only || die 'postengine pin check'
if ! snapshot_full outer_postrun "$SESSION_UUID" NO; then FINAL_STATUS='TELEMETRY_INCOMPLETE'; FINAL_REASON='outer-postrun telemetry missing'; exit 69; fi
"$PYTHON" -B -I "$SEMANTIC" --run-root "$C1_RUN_OUT" --pins "$PINS" || { FINAL_STATUS='SEMANTIC_VERIFICATION_FAILED'; FINAL_REASON='strict verifier'; exit 69; }
"$PYTHON" -B -I "$ANALYZER" --run-root "$C1_RUN_OUT" --pins "$PINS" || { FINAL_STATUS='ANALYSIS_FAILED'; FINAL_REASON='analyzer'; exit 69; }
manifest_event engine_semantic_and_analysis_pass '' '' 'strict semantic and measured-only analysis passed' || die 'manifest semantic'
FINAL_STATUS='COMPLETE'; FINAL_REASON='outer-owned measured execution and CPU evidence passed'
exit 0


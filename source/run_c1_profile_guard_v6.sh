#!/usr/bin/env bash
# V6 profile outer guard. Only this process may launch nsys/C1Microbench for the
# explicitly unmeasured E_G/P_F profile pass; no public before/after bypass exists.
set -uo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot'
HOST='CONFIGURE_ARCHIVE_HOST'
GUARD="$ROOT/run_c1_profile_guard_v6.sh"
LAUNCHER="$ROOT/launch_c1_profile_v6.sh"
PLAN="$ROOT/run_c1_profile_primary_v6.sh"
PLAN_JSON="$ROOT/profile_execution_plan_v6.json"
PINS="$ROOT/hardened_static_pins_v6.json"
PIN_VERIFY="$ROOT/verify_v6_pins.py"
MANIFEST="$ROOT/c1_v6_profile_manifest.py"
VERIFIER="$ROOT/verify_c1_profile_artifacts_v6.py"
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
LOCK="$ROOT/.c1_v6_gpu0.lock"
ATTEMPTS=15
POLL_SECONDS=2

GPU_UUID=''
SESSION_UUID=''
CHILD_PID=''
SESSION_READY=0
FINAL_STATUS='PREPARING'
FINAL_REASON='not started'
PROFILE_OUT=''
RUN_OUT=''

die() { printf 'C1-V6-PROFILE-GUARD BLOCKED: %s\n' "$*" >&2; exit 69; }
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

write_guard_card() {
  local status="$1" reason="$2"
  [[ -n "$PROFILE_OUT" && -d "$PROFILE_OUT" ]] || return 0
  status="$status" reason="$reason" GPU_UUID="$SESSION_UUID" "$PYTHON" -B -I - "$PROFILE_OUT/profile_guard_card_v6.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
old={}
if p.exists():
    if p.is_symlink(): raise SystemExit('profile guard card symlink')
    old=json.loads(p.read_text())
old.update({"schema":"gtspp-c1-v6-profile-guard-card-v1","updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "status":os.environ["status"],"reason":os.environ["reason"],"host":os.uname().nodename,
 "physical_gpu_index":0,"physical_gpu_uuid":os.environ.get("GPU_UUID") or None,"execution_mode":"PROFILE",
 "scope":"Explicitly unmeasured E_G/P_F Nsight allocation profile only; C1 query-only, C2 mode=0, no C3 mutations, never latency data.",
 "snapshots_dir":str(p.parent/"gpu_snapshots"),"manifest":str(p.parent/"profile_run_manifest_v6.json")})
tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(old,indent=2,sort_keys=True)+'\n');tmp.replace(p)
PY
}
manifest_event() { "$PYTHON" -B -I "$MANIFEST" event --out "$PROFILE_OUT" --kind "$1" --label "$2" --variant "$3" --detail "$4"; }
manifest_final() { "$PYTHON" -B -I "$MANIFEST" final --out "$PROFILE_OUT" --status "$1" --reason "$2"; }
verify_source_only() {
  require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$MANIFEST" && require_regular_raw "$VERIFIER" && require_regular_raw "$PLAN" && require_regular_raw "$PLAN_JSON" || return 1
  "$PYTHON" -B -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --static-source || return 1
  "$PYTHON" -B -I - "$PLAN_JSON" <<'PY' || return 1
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
def need(x,m):
    if not x: raise SystemExit(m)
plan=json.loads(p.read_text())
reports=["cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_mem_size_sum","um_sum"]
need(plan.get("schema")=="gtspp-c1-v6-profile-execution-plan-v1","schema")
need(plan.get("schedule")==[{"replicate":1,"variant":"E_G_c1_off_reference"},{"replicate":1,"variant":"P_F_full_C1"}],"schedule")
need(plan.get("binary_mode")=={"radius":500.0,"required_argument":"--profile-only","required_c2_residual_mode":0,"required_run_mode":"UNMEASURED_NSYS_PROFILE_DO_NOT_USE","trace_operations":1024},"binary mode")
nsys=plan.get("nsys",{})
need(nsys.get("executable")=="/usr/local/bin/nsys","nsys executable")
need(nsys.get("profile_arguments")==["profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true"],"profile args")
need(nsys.get("stats_arguments")==["stats","--force-export=true","--force-overwrite=true","--format","csv"],"stats args")
need(nsys.get("reports")==reports,"fixed reports")
need(nsys.get("artifact_layout")=={"nsys_rep":"nsys/profile.nsys-rep","sqlite":"nsys/profile.sqlite","csv_reports":["nsys/reports/profile_"+x+".csv" for x in reports]},"artifact layout")
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
need(card.get('schema')=='gtspp-c1-microbench-run-card-v6' and done.get('schema')=='gtspp-c1-microbench-completion-v6','schema')
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
  local rep="$1" variant="$2" bin cache out label idle pre_uuid pre_snapshot rc prefix report_prefix repfile sqlite
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
  CHILD_PID=''
  "$SETSID" /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' LD_LIBRARY_PATH='/usr/local/cuda-13.1/lib64' \
    CUDA_VISIBLE_DEVICES="$SESSION_UUID" NVIDIA_VISIBLE_DEVICES="$SESSION_UUID" CUDA_DEVICE_ORDER='PCI_BUS_ID' C1_EXECUTION_MODE='PROFILE' \
    "$NSYS" profile --trace=cuda --cuda-memory-usage=true --force-overwrite=true --output "$prefix" -- \
    "$bin" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" --radius 500 --out "$out" --replicate 1 --variant "$variant" --profile-only >"$out/stdout.log" 2>"$out/stderr.log" &
  CHILD_PID=$!;wait "$CHILD_PID";rc=$?;CHILD_PID=''
  if ! snapshot_full "post_${label}" "$SESSION_UUID" NO; then FINAL_STATUS='TELEMETRY_INCOMPLETE';FINAL_REASON="profile post telemetry $variant";exit 69;fi
  manifest_event profile_variant_postlaunch_telemetry "$label" "$variant" 'post telemetry captured and UUID-bound' || die 'profile manifest postlaunch'
  [[ "$rc" -eq 0 ]] || { FINAL_STATUS='ENGINE_FAILED';FINAL_REASON="nsys/binary rc=$rc $variant";exit "$rc"; }
  profile_completion_contract "$out" "$variant" "$SESSION_UUID" || { FINAL_STATUS='CONTRACT_FAILED';FINAL_REASON="profile completion $variant";exit 69; }
  repfile="$out/nsys/profile.nsys-rep";sqlite="$out/nsys/profile.sqlite";report_prefix="$out/nsys/reports/profile"
  require_regular_raw "$repfile" || { FINAL_STATUS='PROFILE_ARTIFACT_FAILED';FINAL_REASON="missing rep $variant";exit 69; }
  "$NSYS" stats --force-export=true --force-overwrite=true --format csv --output "$report_prefix" --report cuda_api_sum --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum --report um_sum "$repfile" >"$out/nsys/stats_stdout.log" 2>"$out/nsys/stats_stderr.log" || { FINAL_STATUS='PROFILE_ARTIFACT_FAILED';FINAL_REASON="nsys stats $variant";exit 69; }
  require_regular_raw "$sqlite" && require_regular_raw "$out/nsys/reports/profile_cuda_api_sum.csv" && require_regular_raw "$out/nsys/reports/profile_cuda_gpu_mem_time_sum.csv" && require_regular_raw "$out/nsys/reports/profile_cuda_gpu_mem_size_sum.csv" && require_regular_raw "$out/nsys/reports/profile_um_sum.csv" || { FINAL_STATUS='PROFILE_ARTIFACT_FAILED';FINAL_REASON="fixed stats reports $variant";exit 69; }
  "$PYTHON" -B -I "$MANIFEST" artifacts --out "$PROFILE_OUT" --variant "$variant" --rep 1 --sqlite "$sqlite" --report "$out/nsys/reports/profile_cuda_api_sum.csv" --report "$out/nsys/reports/profile_cuda_gpu_mem_time_sum.csv" --report "$out/nsys/reports/profile_cuda_gpu_mem_size_sum.csv" --report "$out/nsys/reports/profile_um_sum.csv" || die 'profile artifact manifest binding'
  manifest_event profile_nsys_stats_complete "$label" "$variant" 'fixed four CSV reports generated and hash-bound' || die 'profile stats event'
  manifest_event profile_variant_completion_contract_pass "$label" "$variant" 'profile-only completion contract passed' || die 'profile completion event'
}
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null;then
    kill -TERM -- "-$CHILD_PID" 2>/dev/null || true;wait "$CHILD_PID" 2>/dev/null || true
    rc=69;FINAL_STATUS='INTERRUPTED';FINAL_REASON='only profile-guard-owned child group terminated'
  fi
  if [[ "$SESSION_READY" == 1 ]];then
    if ! snapshot_full outer_exit "$SESSION_UUID" NO;then rc=69;FINAL_STATUS='TELEMETRY_INCOMPLETE';FINAL_REASON='profile outer-exit telemetry missing';fi
    if ! rm -f -- "$PROFILE_OUT/.c1_profile_guard_session_v6.json";then rc=69;FINAL_STATUS='SESSION_CLEANUP_INCOMPLETE';FINAL_REASON='profile session removal failed';fi
    if ! write_guard_card "$FINAL_STATUS" "$FINAL_REASON";then rc=69;FINAL_STATUS='GUARD_CARD_INCOMPLETE';FINAL_REASON='profile guard card finalization failed';fi
    if ! manifest_final "$FINAL_STATUS" "$FINAL_REASON";then rc=69;FINAL_STATUS='MANIFEST_INCOMPLETE';FINAL_REASON='profile manifest finalization failed';fi
    if [[ "$rc" -eq 0 && "$FINAL_STATUS" == COMPLETE ]];then
      if ! verify_source_only;then rc=69;FINAL_STATUS='POSTRUN_PIN_VERIFICATION_FAILED';FINAL_REASON='profile postrun pin verification failed'
      elif ! "$PYTHON" -B -I "$VERIFIER" --profile-root "$PROFILE_OUT" --pins "$PINS";then rc=69;FINAL_STATUS='PROFILE_VERIFICATION_FAILED';FINAL_REASON='strict profile verifier failed'
      fi
      if [[ "$rc" -ne 0 ]];then write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true;manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true;fi
    fi
  fi
  "$FLOCK" -u 9 || true
  exit "$rc"
}
signal_handler() { FINAL_STATUS='INTERRUPTED';FINAL_REASON='signal; only profile guard-owned child is terminable';[[ -n "$CHILD_PID" ]] && kill -TERM -- "-$CHILD_PID" 2>/dev/null || true;exit 130; }

if [[ "$#" -gt 0 && "$1" == --verify-source-only ]];then shift;[[ "$#" -eq 0 ]] || die 'extra source-only arguments';verify_source_only;exit $?;fi
[[ "$#" -eq 0 ]] || die 'unsupported profile guard subcommand'
C1_V6_PROFILE_TRUST_ROOT="$(read_env C1_V6_PROFILE_TRUST_ROOT)";C1_V6_PROFILE_LAUNCHER="$(read_env C1_V6_PROFILE_LAUNCHER)";C1_V6_PROFILE_TRUST_GUARD_SHA="$(read_env C1_V6_PROFILE_TRUST_GUARD_SHA)";C1_V6_PROFILE_TRUST_PINS_SHA="$(read_env C1_V6_PROFILE_TRUST_PINS_SHA)";C1_V6_PROFILE_TRUST_PIN_VERIFY_SHA="$(read_env C1_V6_PROFILE_TRUST_PIN_VERIFY_SHA)";C1_ALLOW_GPU0="$(read_env C1_ALLOW_GPU0)";RUN_OUT="$(read_env C1_PROFILE_OUT)"
[[ "$0" == "$GUARD" && "$C1_V6_PROFILE_TRUST_ROOT" == YES && "$C1_V6_PROFILE_LAUNCHER" == "$LAUNCHER" ]] || die 'reviewed profile trust-root required'
require_regular_raw "$LAUNCHER" && require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$MANIFEST" && require_regular_raw "$VERIFIER" && require_regular_raw "$PLAN" && require_regular_raw "$PLAN_JSON" && [[ -x "$NSYS" ]] || die 'profile runtime file missing/symlink'
[[ "$(sha "$GUARD")" == "$C1_V6_PROFILE_TRUST_GUARD_SHA" && "$(sha "$PINS")" == "$C1_V6_PROFILE_TRUST_PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$C1_V6_PROFILE_TRUST_PIN_VERIFY_SHA" ]] || die 'profile trust-root bootstrap hash mismatch'
[[ "$C1_ALLOW_GPU0" == YES && "$(hostname)" == "$HOST" ]] || die 'profile GPU0/host constraint failed'
[[ -n "$RUN_OUT" && "$RUN_OUT" == /* ]] || die 'absolute profile launcher output required'
BASE="$ROOT/profiles";OUT="$("$REALPATH" -m -- "$RUN_OUT")" || die 'bad profile output';is_under "$OUT" "$BASE" && [[ "$(basename "$OUT")" == c1_v6_profile_* && ! -e "$OUT" && ! -L "$OUT" ]] || die 'unsafe profile output'
PROFILE_OUT="$OUT";export PROFILE_OUT
exec 9>>"$LOCK";"$FLOCK" -n 9 || die 'another v6 GPU0 session holds lock'
trap cleanup EXIT;trap signal_handler INT TERM HUP
verify_source_only || die 'initial profile source/pin verification failed'
mkdir -p -- "$BASE" && mkdir -- "$PROFILE_OUT" || die 'cannot create profile root'
"$PYTHON" -B -I "$MANIFEST" init --out "$PROFILE_OUT" --root "$ROOT" --pins "$PINS" --launcher "$LAUNCHER" --guard "$GUARD" -- "$GUARD" || die 'profile manifest init'
idle="$(wait_idle outer_prelaunch '')" || die 'profile outer idle precheck';IFS=$'\t' read -r SESSION_UUID snapshot <<<"$idle";GPU_UUID="$SESSION_UUID";[[ "$SESSION_UUID" == GPU-* ]] || die 'profile outer UUID'
"$PYTHON" -B -I "$MANIFEST" gpu --out "$PROFILE_OUT" --uuid "$SESSION_UUID" --snapshot "$snapshot" || die 'profile manifest GPU'
"$PYTHON" -B -I - "$PROFILE_OUT/.c1_profile_guard_session_v6.json" "$SESSION_UUID" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.write_text(json.dumps({'schema':'gtspp-c1-v6-profile-session-v1','physical_gpu_index':0,'physical_gpu_uuid':sys.argv[2]})+'\n')
PY
chmod 0600 "$PROFILE_OUT/.c1_profile_guard_session_v6.json" || die 'profile session chmod'
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

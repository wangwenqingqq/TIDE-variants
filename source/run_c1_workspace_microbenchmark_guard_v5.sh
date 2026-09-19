#!/usr/bin/env bash
# V5 outer guard: only this process launches C1Microbench.  There are no
# public before/after helper subcommands, so an engine cannot bypass trust,
# host, lock, pin, UUID, or telemetry checks.
set -uo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v5_formal'
HOST='CONFIGURE_ARCHIVE_HOST'
GUARD="$ROOT/run_c1_workspace_microbenchmark_guard_v5.sh"
LAUNCHER="$ROOT/launch_c1_v5.sh"
PLAN="$ROOT/run_c1_four_variants_v5.sh"
PLAN_JSON="$ROOT/measured_execution_plan_v5.json"
PINS="$ROOT/hardened_static_pins_v5.json"
PIN_VERIFY="$ROOT/verify_v5_pins.py"
MANIFEST="$ROOT/c1_v5_manifest.py"
SEMANTIC="$ROOT/verify_c1_semantics_v5.py"
ANALYZER="$ROOT/analyze_c1_microbenchmark_v5.py"
FINALIZER="$ROOT/finalize_c1_evidence_v5.py"
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
LOCK="$ROOT/.c1_v5_gpu0.lock"
ATTEMPTS=15
POLL_SECONDS=2

GPU_UUID=''
SESSION_UUID=''
CHILD_PID=''
SESSION_READY=0
FINAL_STATUS='PREPARING'
FINAL_REASON='not started'
RUN_OUT=''

die() { printf 'C1-V5-GUARD BLOCKED: %s\n' "$*" >&2; exit 69; }
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

write_guard_card() {
  local status="$1" reason="$2"
  [[ -n "$C1_RUN_OUT" && -d "$C1_RUN_OUT" ]] || return 0
  status="$status" reason="$reason" GPU_UUID="$SESSION_UUID" "$PYTHON" -I - "$C1_RUN_OUT/guard_run_card_v5.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
old={}
if p.exists():
    try: old=json.loads(p.read_text())
    except Exception as exc: raise SystemExit(f"invalid prior guard card: {exc}")
old.update({"schema":"gtspp-c1-v5-guard-card-v2",
 "updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "status":os.environ["status"],"reason":os.environ["reason"],
 "host":os.uname().nodename,"physical_gpu_index":0,
 "physical_gpu_uuid":os.environ.get("GPU_UUID") or None,
 "execution_mode":"MEASURED",
 "scope":"C1 only: query-only SIFT1M, C2 mode=0, no C3 mutations",
 "snapshots_dir":str(p.parent/"gpu_snapshots"),
 "manifest":str(p.parent/"run_manifest_v5.json")})
tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(old,indent=2,sort_keys=True)+"\n"); tmp.replace(p)
PY
}
manifest_event() { "$PYTHON" -I "$MANIFEST" event --out "$C1_RUN_OUT" --kind "$1" --label "$2" --variant "$3" --detail "$4"; }
manifest_final() { "$PYTHON" -I "$MANIFEST" final --out "$C1_RUN_OUT" --status "$1" --reason "$2"; }
verify_source_only() {
  require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$MANIFEST" || return 1
  "$PYTHON" -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --static-source
}
verify_variant_material() { "$PYTHON" -I "$PIN_VERIFY" --root "$ROOT" --pins "$PINS" --variant "$1"; }

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
  "$PYTHON" -I - "$1" "$2" "$3" "$4" <<'PY'
import json,math,pathlib,sys
out=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; rep=int(sys.argv[3]); expected_uuid=sys.argv[4]
def need(x,m):
    if not x: raise SystemExit(m)
card=json.loads((out/"run_card.json").read_text()); done=json.loads((out/"completion.json").read_text())
gates={"E_G_c1_off_reference":(0,0),"P_G_workspace_only":(1,0),"E_F_fastpath_only":(0,1),"P_F_full_C1":(1,1)}
need(variant in gates,"unknown variant")
need(card.get("schema")=="gtspp-c1-microbench-run-card-v5","run-card schema")
need(done.get("schema")=="gtspp-c1-microbench-completion-v5","completion schema")
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
  "$PYTHON" -I "$MANIFEST" child --out "$C1_RUN_OUT" --rep "$rep" --variant "$variant" --binary "$bin" --variant-out "$out" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" --uuid "$SESSION_UUID" || die 'manifest child binding'
  CHILD_PID=''
  "$SETSID" /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' LD_LIBRARY_PATH='/usr/local/cuda-13.1/lib64' \
    CUDA_VISIBLE_DEVICES="$SESSION_UUID" NVIDIA_VISIBLE_DEVICES="$SESSION_UUID" CUDA_DEVICE_ORDER='PCI_BUS_ID' C1_EXECUTION_MODE='MEASURED' \
    "$bin" --base '/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt' --trace "$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt" \
    --radius 500 --out "$out" --replicate "$rep" --variant "$variant" >"$out/stdout.log" 2>"$out/stderr.log" &
  CHILD_PID=$!; wait "$CHILD_PID"; rc=$?; CHILD_PID=''
  if ! snapshot_full post_"$label" "$SESSION_UUID" NO; then FINAL_STATUS='TELEMETRY_INCOMPLETE'; FINAL_REASON="post telemetry $label"; exit 69; fi
  manifest_event variant_postlaunch_telemetry "$label" "$variant" 'post telemetry captured and UUID-bound' || die 'manifest postlaunch'
  [[ "$rc" -eq 0 ]] || { FINAL_STATUS='ENGINE_FAILED'; FINAL_REASON="binary rc=$rc $label"; exit "$rc"; }
  completion_contract "$out" "$variant" "$rep" "$SESSION_UUID" || { FINAL_STATUS='CONTRACT_FAILED'; FINAL_REASON="completion $label"; exit 69; }
  manifest_event variant_completion_contract_pass "$label" "$variant" 'measured completion contract passed' || die 'manifest contract'
}
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM -- "-$CHILD_PID" 2>/dev/null || true; wait "$CHILD_PID" 2>/dev/null || true
    rc=69; FINAL_STATUS='INTERRUPTED'; FINAL_REASON='only guard-owned C1 child group terminated'
  fi
  if [[ "$SESSION_READY" == 1 ]]; then
    if ! snapshot_full outer_exit "$SESSION_UUID" NO; then rc=69; FINAL_STATUS='TELEMETRY_INCOMPLETE'; FINAL_REASON='outer-exit telemetry missing'; fi
    if ! rm -f -- "$C1_RUN_OUT/.c1_guard_session_v5.json"; then rc=69; FINAL_STATUS='SESSION_CLEANUP_INCOMPLETE'; FINAL_REASON='session removal failed'; fi
    if ! write_guard_card "$FINAL_STATUS" "$FINAL_REASON"; then rc=69; FINAL_STATUS='GUARD_CARD_INCOMPLETE'; FINAL_REASON='final guard card failed'; fi
    if ! manifest_final "$FINAL_STATUS" "$FINAL_REASON"; then rc=69; FINAL_STATUS='MANIFEST_INCOMPLETE'; FINAL_REASON='manifest finalization failed'; fi
    # If any finalization step failed, retry both artifacts with an explicit
    # non-COMPLETE state; no stale COMPLETE card may remain usable.
    if [[ "$rc" -ne 0 ]]; then
      write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true
      manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true
    fi
    if [[ "$rc" -eq 0 && "$FINAL_STATUS" == COMPLETE ]]; then
      if ! verify_source_only; then rc=69; FINAL_STATUS='POSTRUN_PIN_VERIFICATION_FAILED'; FINAL_REASON='postrun pin verification failed'
      elif ! "$PYTHON" -I "$FINALIZER" --run-root "$C1_RUN_OUT" --pins "$PINS"; then rc=69; FINAL_STATUS='FINAL_EVIDENCE_FAILED'; FINAL_REASON='final evidence attestation failed'
      fi
      if [[ "$rc" -ne 0 ]]; then write_guard_card "$FINAL_STATUS" "$FINAL_REASON" || true; manifest_final "$FINAL_STATUS" "$FINAL_REASON" || true; fi
    fi
  fi
  "$FLOCK" -u 9 || true
  exit "$rc"
}
signal_handler() { FINAL_STATUS='INTERRUPTED'; FINAL_REASON='signal; only guard-owned child is terminable'; [[ -n "$CHILD_PID" ]] && kill -TERM -- "-$CHILD_PID" 2>/dev/null || true; exit 130; }

if [[ "$#" -gt 0 && "$1" == --verify-source-only ]]; then shift; [[ "$#" -eq 0 ]] || die 'extra source-only arguments'; verify_source_only; exit $?; fi
[[ "$#" -eq 0 ]] || die 'unsupported guard subcommand'
C1_V5_TRUST_ROOT="$(read_env C1_V5_TRUST_ROOT)"
C1_V5_LAUNCHER="$(read_env C1_V5_LAUNCHER)"
C1_V5_TRUST_GUARD_SHA="$(read_env C1_V5_TRUST_GUARD_SHA)"
C1_V5_TRUST_PINS_SHA="$(read_env C1_V5_TRUST_PINS_SHA)"
C1_V5_TRUST_PIN_VERIFY_SHA="$(read_env C1_V5_TRUST_PIN_VERIFY_SHA)"
C1_ALLOW_GPU0="$(read_env C1_ALLOW_GPU0)"
RUN_OUT="$(read_env C1_RUN_OUT)"
[[ "$0" == "$GUARD" && "$C1_V5_TRUST_ROOT" == YES && "$C1_V5_LAUNCHER" == "$LAUNCHER" ]] || die 'reviewed trust-root required'
require_regular_raw "$LAUNCHER" && require_regular_raw "$PINS" && require_regular_raw "$PIN_VERIFY" && require_regular_raw "$PLAN" && require_regular_raw "$PLAN_JSON" && require_regular_raw "$SEMANTIC" && require_regular_raw "$ANALYZER" && require_regular_raw "$FINALIZER" || die 'runtime file missing/symlink'
[[ "$(sha "$GUARD")" == "$C1_V5_TRUST_GUARD_SHA" && "$(sha "$PINS")" == "$C1_V5_TRUST_PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$C1_V5_TRUST_PIN_VERIFY_SHA" ]] || die 'trust-root bootstrap hash mismatch'
[[ "$C1_ALLOW_GPU0" == YES && "$(hostname)" == "$HOST" ]] || die 'GPU0/host constraint failed'
[[ -n "$RUN_OUT" && "$RUN_OUT" == /* ]] || die 'absolute launcher output required'
BASE="$ROOT/runs"; OUT="$("$REALPATH" -m -- "$RUN_OUT")" || die 'bad output'
is_under "$OUT" "$BASE" && [[ "$(basename "$OUT")" == c1_v5_measured_* && ! -e "$OUT" && ! -L "$OUT" ]] || die 'unsafe output'
C1_RUN_OUT="$OUT"; export C1_RUN_OUT
exec 9>>"$LOCK"; "$FLOCK" -n 9 || die 'another v5 GPU0 session holds lock'
trap cleanup EXIT; trap signal_handler INT TERM HUP
verify_source_only || die 'initial source/pin verification failed'
mkdir -p -- "$BASE" && mkdir -- "$C1_RUN_OUT" || die 'cannot create run root'
"$PYTHON" -I "$MANIFEST" init --out "$C1_RUN_OUT" --root "$ROOT" --pins "$PINS" --launcher "$LAUNCHER" --mode MEASURED -- "$GUARD" || die 'manifest init'
"$PYTHON" -I "$MANIFEST" bind-plan --out "$C1_RUN_OUT" --execution-plan "$PLAN_JSON" || die 'manifest plan binding'
idle="$(wait_idle outer_prelaunch '')" || die 'outer idle precheck'
IFS=$'\t' read -r SESSION_UUID snapshot <<<"$idle"
GPU_UUID="$SESSION_UUID"
[[ "$SESSION_UUID" == GPU-* && "$SESSION_UUID" == "$GPU_UUID" ]] || die 'outer UUID'
"$PYTHON" -I "$MANIFEST" gpu --out "$C1_RUN_OUT" --uuid "$SESSION_UUID" --snapshot "$snapshot" || die 'manifest GPU'
"$PYTHON" -I - "$C1_RUN_OUT/.c1_guard_session_v5.json" "$SESSION_UUID" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({"schema":"gtspp-c1-v5-session-v2","physical_gpu_index":0,"physical_gpu_uuid":sys.argv[2]})+"\n")
PY
chmod 0600 "$C1_RUN_OUT/.c1_guard_session_v5.json" || die 'session chmod'
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
"$PYTHON" -I "$SEMANTIC" --run-root "$C1_RUN_OUT" --pins "$PINS" || { FINAL_STATUS='SEMANTIC_VERIFICATION_FAILED'; FINAL_REASON='strict verifier'; exit 69; }
"$PYTHON" -I "$ANALYZER" --run-root "$C1_RUN_OUT" --pins "$PINS" || { FINAL_STATUS='ANALYSIS_FAILED'; FINAL_REASON='analyzer'; exit 69; }
manifest_event engine_semantic_and_analysis_pass '' '' 'strict semantic and measured-only analysis passed' || die 'manifest semantic'
FINAL_STATUS='COMPLETE'; FINAL_REASON='outer-owned measured execution and CPU evidence passed'
exit 0


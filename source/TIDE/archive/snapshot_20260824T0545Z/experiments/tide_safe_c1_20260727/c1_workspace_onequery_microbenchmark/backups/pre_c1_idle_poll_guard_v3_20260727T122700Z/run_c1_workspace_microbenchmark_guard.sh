#!/usr/bin/env bash
# Hardened outer/helper guard for the isolated C1 workspace-reuse + qnum=1 benchmark.
# --verify-source-only is source/provenance-only: it never invokes nvidia-smi or a binary.
# --verify-static additionally runs parser --help with CUDA visibility disabled.
set -uo pipefail
umask 077

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark'
EXPECTED_HOST='CONFIGURE_ARCHIVE_HOST'
ENGINE="$ROOT/run_c1_four_variants.sh"
PINS="$ROOT/hardened_static_pins_v2.json"
EXPECTED_PINS_SHA='9914f593e5095bd6b46e43d7895c18ff73dbee822a427b8d87def4ea0603f38e'
LOCK_FILE="$ROOT/.c1_gpu0_guard.lock"
PYTHON="$(command -v python3 || true)"
SMI="$(command -v nvidia-smi || true)"

die() { printf 'C1-GUARD BLOCKED: %s\n' "$*" >&2; exit 69; }
sha256() { sha256sum -- "$1" | awk '{print $1}'; }
trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
safe_label() { printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_'; }
is_under() { [[ "$1" == "$2/"* ]]; }

[[ -n "$PYTHON" && -x "$PYTHON" ]] || die 'python3 unavailable for static safety verification'

verify_source_only() {
  [[ -f "$PINS" && ! -L "$PINS" ]] || die "pins file missing/symlink: $PINS"
  local actual
  actual="$(sha256 "$PINS")" || die 'cannot hash static pins'
  [[ "$actual" == "$EXPECTED_PINS_SHA" ]] || die "static pins SHA mismatch: $actual"
  "$PYTHON" - "$ROOT" "$PINS" <<'PY'
import hashlib, json, pathlib, re, sys
root=pathlib.Path(sys.argv[1]).resolve()
pin_path=pathlib.Path(sys.argv[2]).resolve()
p=json.loads(pin_path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def check(entry, what):
    path=pathlib.Path(entry['path']).resolve()
    got=sha(path)
    if got != entry['sha256']:
        raise SystemExit(f'{what} sha mismatch: {path}: expected {entry["sha256"]}, got {got}')
assert p['schema']=='gtspp-c1-hardened-static-pins-v2'
protocol=pathlib.Path(p['protocol']['path']).resolve()
check(p['protocol'],'protocol')
check(p['gpu_uuid_safety_supersession'],'GPU UUID safety supersession')
d=json.loads(protocol.read_text())
assert d['schema']=='gtspp-c1-workspace-onequery-microbenchmark-protocol-v1'
assert d['status']=='PRE_REGISTERED_NOT_EXECUTED'
settings=d['data_and_input']['fixed_engine_settings']
assert settings['process_type']==2 and settings['qnum_per_engine_call']==1
assert settings['C2_residual_pruning_mode']==0
assert d['data_and_input']['radius']==500.0
safety_path=pathlib.Path(p['gpu_uuid_safety_supersession']['path']).resolve()
safety=json.loads(safety_path.read_text())
assert safety['schema']=='gtspp-c1-gpu-uuid-safety-supersession-v1'
assert safety['status']=='CPU_ONLY_SAFETY_SUPERSESSION_PRE_GPU_EXECUTION'
assert safety['supersedes_only']['protocol_path']==str(protocol)
assert safety['supersedes_only']['protocol_sha256']==p['protocol']['sha256']
assert safety['supersedes_only']['v1_clause_location']=='gpu0_safety_guard.device'
assert safety['replacement_safety_binding']['visibility'].startswith('Set CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES to that discovered physical GPU0 UUID')
assert 'never the numeric string 0' in safety['replacement_safety_binding']['visibility']
assert safety['explicit_non_changes']['radius']=='Unchanged: 500.0.'
assert safety['explicit_non_changes']['c1_semantics']=='Unchanged: workspace reuse plus qnum=1 fast path.'
assert safety['explicit_non_changes']['c2'].endswith('mode=0.')
assert safety['explicit_non_changes']['c3'].startswith('Unchanged and out of scope')
check(p['implementation_correction'],'ephemeral workspace implementation correction')
correction_path=pathlib.Path(p['implementation_correction']['path']).resolve()
correction=json.loads(correction_path.read_text())
assert correction['schema']=='gtspp-c1-ephemeral-workspace-implementation-correction-v1'
assert correction['status']=='CPU_NVCC_REBUILD_AND_SOURCE_VALIDATION_COMPLETE_NO_POSTFIX_GPU_EXECUTION'
assert correction['scope']['frozen_protocol_modified'] is False
assert correction['scope']['archive_sources_modified'] is False
assert correction['scope']['run_directories_modified'] is False
assert correction['scope']['frozen_protocol_sha256']==p['protocol']['sha256']
assert correction['minimal_source_change']['before_sha256']==sha(root/'backups/update.cuh.pre_ephemeral_workspace_null_fix')
assert correction['minimal_source_change']['after_sha256']==sha(root/'worktree/include/update.cuh')
assert correction['minimal_source_change']['inserted_statements']==['update_result_id_ws = nullptr;','update_result_dis_ws = nullptr;']
rebuild=correction['cpu_nvcc_rebuild']
assert rebuild['manifest_sha256']==p['source_build_manifest']['sha256']
assert sha(pathlib.Path(rebuild['build_wrapper_log']))==rebuild['build_wrapper_log_sha256']
validation=correction['source_only_validation']
assert validation['status']=='PASS_NO_GPU_QUERY_OR_BINARY_EXECUTION'
assert sha(pathlib.Path(validation['report_path']))==validation['report_sha256']
for name,evidence in correction['failed_smoke_preserved']['evidence'].items():
    assert sha(pathlib.Path(evidence['path']))==evidence['sha256'], name
for key,entry in p['inputs'].items(): check(entry, f'input:{key}')
check(p['source_build_manifest'],'worktree build manifest')
check(p['worktree_origin_manifest'],'worktree origin manifest')
check(p['execution_engine'],'isolated execution engine')
for entry in p['worktree_source_files']: check(entry, 'worktree source')
for entry in p['variant_binaries']:
    check(entry, 'variant binary')
    assert entry['name'] in {'E_G_c1_off_reference','P_G_workspace_only','E_F_fastpath_only','P_F_full_C1'}
build=json.loads(pathlib.Path(p['source_build_manifest']['path']).read_text())
assert len(build['variants'])==4
for item in build['variants']:
    e=next(x for x in p['variant_binaries'] if x['name']==item['name'])
    assert item['binary_sha256']==e['sha256']
update=(root/'worktree/include/update.cuh').read_text()
release=update[update.index('static inline void releaseC1QueryWorkspace'):update.index('static inline void ensureRebuildInsertWorkspace')]
ensure=update[update.index('void ensureUpdateSearchWorkspace'):update.index('static inline void ensureDeletePrefixValid')]
free_id=release.index('if (update_result_id_ws != nullptr) CHECK(C1_CUDA_FREE(update_result_id_ws));')
free_dis=release.index('if (update_result_dis_ws != nullptr) CHECK(C1_CUDA_FREE(update_result_dis_ws));')
null_id=release.index('update_result_id_ws = nullptr;')
null_dis=release.index('update_result_dis_ws = nullptr;')
cap_zero=release.index('update_result_ws_cap = 0;')
assert free_id < free_dis < null_id < null_dis < cap_zero
assert 'if (update_result_ws_cap < max_result_slots_needed)' in ensure
assert ensure.index('if (update_result_ws_cap < max_result_slots_needed)') < ensure.index('CHECK(C1_CUDA_MALLOC_MANAGED(&update_result_id_ws')
assert ensure.index('if (update_result_ws_cap < max_result_slots_needed)') < ensure.index('CHECK(C1_CUDA_MALLOC_MANAGED(&update_result_dis_ws')
source=(root/'worktree/src/c1_microbench.cu').read_text()
assert re.search(r'C2_residual_mode[^\n]*:\s*0', source)
assert re.search(r'upload_rp_constants\s*\([^;]*,\s*0\s*,\s*0\s*\)', source, re.S)
assert 'if (in_size != 0) die("query-only C1 runner observed nonzero insert buffer")' in source
assert 'no C3 mutation' in source
assert 'cfg.replicate < 1' in source
for flag in ('--base','--trace','--radius','--out','--replicate','--variant'):
    assert f'arg == "{flag}"' in source
engine=pathlib.Path(p['execution_engine']['path']).read_text()
assert 'run_variant 1 E_G_c1_off_reference smoke "$smoke_ops"' in engine
assert 'run_variant 0 E_G_c1_off_reference smoke' not in engine
print(json.dumps({'pass':True,'scope':'C1 only; C2 mode=0; no C3 mutation','pins_sha256':sha(pin_path),'sources':len(p['worktree_source_files']),'binaries':len(p['variant_binaries'])},sort_keys=True))
PY
  [[ $? -eq 0 ]] || return 1
}

verify_static() {
  verify_source_only || return $?
  local variant bin help rc
  for variant in E_G_c1_off_reference P_G_workspace_only E_F_fastpath_only P_F_full_C1; do
    bin="$ROOT/builds/$variant/bin/C1Microbench"
    help="$(env CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES='' "$bin" --help 2>&1)"
    rc=$?
    [[ "$rc" -eq 0 ]] || { printf 'CPU-only parser --help failed for %s: %s
' "$variant" "$help" >&2; return 1; }
    [[ "$help" == *'--base PATH --trace PATH --radius R --out DIR --replicate N --variant NAME'* ]] || { printf 'Unexpected parser help for %s
' "$variant" >&2; return 1; }
  done
  printf 'C1_PARSER_HELP_CPU_ONLY_PASS variants=4 separate_flag_form=verified
'
}

verify_variant_pin() {
  local variant="$1"
  "$PYTHON" - "$PINS" "$variant" <<'PY'
import hashlib,json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
v=next((x for x in p['variant_binaries'] if x['name']==sys.argv[2]),None)
assert v is not None
h=hashlib.sha256(pathlib.Path(v['path']).read_bytes()).hexdigest()
assert h==v['sha256'], (v['path'],h,v['sha256'])
print(v['sha256'])
PY
  [[ $? -eq 0 ]] || die "variant binary pin failed: $variant"
}

write_guard_card() {
  local status="$1" reason="$2"
  [[ -n "${C1_RUN_OUT:-}" && -d "${C1_RUN_OUT:-}" ]] || return 0
  export ROOT C1_RUN_OUT EXPECTED_HOST GPU_UUID EXPECTED_PINS_SHA status reason
  "$PYTHON" - "$C1_RUN_OUT/guard_run_card.json" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
old={}
try: old=json.loads(p.read_text())
except Exception: pass
old.update({
 'schema':'gtspp-c1-hardened-guard-run-card-v1',
 'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':os.environ.get('status'),
 'reason':os.environ.get('reason'),
 'host':os.uname().nodename,
 'physical_gpu_index':0,
 'physical_gpu_uuid':os.environ.get('GPU_UUID') or None,
 'cuda_visibility_policy':'GPU UUID only; numeric CUDA_VISIBLE_DEVICES mapping is forbidden',
 'pins_sha256':os.environ.get('EXPECTED_PINS_SHA'),
 'scope':'C1 workspace reuse + qnum=1 fast path only; C2 residual mode=0; no C3 mutations',
 'snapshots_dir':str(p.parent/'gpu_snapshots'),
 'session_file':str(p.parent/'.c1_guard_session.json'),
})
t=p.with_suffix('.tmp')
t.write_text(json.dumps(old,indent=2,sort_keys=True)+'\n')
t.replace(p)
PY
}

snapshot_idle() {
  local label="$1" expected_uuid="$2" dir="$C1_RUN_OUT/gpu_snapshots"
  local tag raw apps line index uuid name used util driver
  [[ -n "$SMI" && -x "$SMI" ]] || die 'nvidia-smi unavailable; GPU0 guard fails closed'
  mkdir -p -- "$dir" || die 'cannot create snapshot directory'
  tag="$(safe_label "$label")"
  raw="$dir/${tag}_gpu.csv"; apps="$dir/${tag}_compute.csv"
  "$SMI" -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits >"$raw" 2>&1
  [[ $? -eq 0 ]] || die "GPU0 snapshot query failed at $label"
  "$SMI" -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >"$apps" 2>&1
  [[ $? -eq 0 ]] || die "GPU0 compute-app query failed at $label"
  [[ "$(grep -cve '^[[:space:]]*$' "$raw")" == 1 ]] || die "GPU0 identity query returned an ambiguous number of lines at $label"
  line="$(head -n1 "$raw" | tr -d '\r')"
  IFS=',' read -r index uuid name used util driver <<<"$line"
  index="$(printf '%s' "$index" | trim)"; uuid="$(printf '%s' "$uuid" | trim)"
  name="$(printf '%s' "$name" | trim)"; used="$(printf '%s' "$used" | trim)"; util="$(printf '%s' "$util" | trim)"
  [[ "$index" == '0' ]] || die "GPU physical-index mapping failed at $label: '$index'"
  [[ "$uuid" == GPU-* ]] || die "GPU0 UUID unavailable/ambiguous at $label: '$uuid'"
  [[ "$name" == *'RTX PRO 6000'* ]] || die "GPU0 is not reviewed RTX PRO 6000 target at $label: '$name'"
  [[ -z "$expected_uuid" || "$uuid" == "$expected_uuid" ]] || die "physical GPU0 UUID changed at $label: expected $expected_uuid got $uuid"
  [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || die "GPU0 state unparsable at $label"
  (( used <= 256 && util == 0 )) || die "GPU0 is not idle at $label: memory=${used}MiB util=${util}%"
  local apps_nonempty
  apps_nonempty="$(sed '/^[[:space:]]*$/d' "$apps")"
  if [[ -n "$apps_nonempty" ]]; then
    if ! printf '%s\n' "$apps_nonempty" | grep -Eq '^(No running compute processes found|No running processes found)$'; then
      die "GPU0 compute-app state is nonempty/ambiguous at $label; no process will be interrupted"
    fi
  fi
  GPU_UUID="$uuid"
  export GPU_UUID
  printf '%s\n' "$uuid"
}

snapshot_post() {
  local label="$1" dir="${C1_RUN_OUT:-}/gpu_snapshots"
  [[ -n "${C1_RUN_OUT:-}" && -d "$C1_RUN_OUT" && -n "$SMI" && -x "$SMI" ]] || return 0
  mkdir -p -- "$dir" 2>/dev/null || return 0
  local tag raw apps
  tag="$(safe_label "$label")"
  raw="$dir/${tag}_gpu.csv"; apps="$dir/${tag}_compute.csv"
  "$SMI" -i 0 --query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits >"$raw" 2>&1 || true
  "$SMI" -i 0 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >"$apps" 2>&1 || true
}

check_session() {
  [[ -n "${C1_RUN_OUT:-}" && -n "${C1_GUARD_SESSION_FILE:-}" && -n "${C1_GUARD_SESSION_NONCE:-}" ]] || die 'internal guard session variables absent'
  [[ -f "$C1_GUARD_SESSION_FILE" && ! -L "$C1_GUARD_SESSION_FILE" ]] || die 'guard session file absent/symlink'
  [[ "$(readlink -f "$C1_GUARD_SESSION_FILE")" == "$C1_RUN_OUT/.c1_guard_session.json" ]] || die 'session file path invalid'
  "$PYTHON" - "$C1_GUARD_SESSION_FILE" "$C1_GUARD_SESSION_NONCE" "$EXPECTED_PINS_SHA" <<'PY'
import json,os,sys
x=json.load(open(sys.argv[1]))
assert x['nonce']==sys.argv[2]
assert x['pins_sha256']==sys.argv[3]
assert x['host']==os.uname().nodename
assert x['physical_gpu_index']==0
assert isinstance(x['physical_gpu_uuid'],str) and x['physical_gpu_uuid'].startswith('GPU-')
PY
  [[ $? -eq 0 ]] || die 'guard session validation failed'
}

helper_before() {
  local label="${1:-}" variant="${2:-}"
  [[ -n "$label" && -n "$variant" ]] || die 'before-launch requires label and variant'
  check_session
  verify_variant_pin "$variant"
  local expected
  expected="$("$PYTHON" - "$C1_GUARD_SESSION_FILE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['physical_gpu_uuid'])
PY
)"
  snapshot_idle "pre_$label" "$expected" >/dev/null
  write_guard_card 'GPU_LAUNCH_PRECHECK_PASS' "GPU0 UUID and idle/compute checks passed immediately before $label"
}

helper_after() {
  local label="${1:-}"
  [[ -n "$label" ]] || die 'after-launch requires label'
  check_session
  snapshot_post "post_$label"
  write_guard_card 'GPU_LAUNCH_POST_SNAPSHOT' "Post-launch GPU0 snapshot recorded for $label"
}

if [[ "${1:-}" == '--verify-source-only' ]]; then
  shift
  [[ $# -eq 0 ]] || die '--verify-source-only accepts no extra arguments'
  verify_source_only || exit $?
  exit 0
fi
if [[ "${1:-}" == '--verify-static' ]]; then
  shift
  [[ $# -eq 0 ]] || die '--verify-static accepts no extra arguments'
  verify_static || exit $?
  exit 0
fi
if [[ "${1:-}" == '--before-launch' ]]; then
  shift
  helper_before "$@"
  exit 0
fi
if [[ "${1:-}" == '--after-launch' ]]; then
  shift
  helper_after "$@"
  exit 0
fi

[[ "${C1_ALLOW_GPU0:-}" == 'YES' ]] || die 'set C1_ALLOW_GPU0=YES only after review'
[[ "${C1_MICROBENCH_ENGINE:-}" == "$ENGINE" ]] || die 'engine must be exact isolated four-variant runner'
[[ -x "$ENGINE" && ! -L "$ENGINE" ]] || die 'reviewed isolated engine missing/symlink'
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || die "wrong host: $(hostname)"
[[ -n "${C1_RUN_OUT:-}" && "$C1_RUN_OUT" == /* ]] || die 'C1_RUN_OUT must be an absolute isolated path'

ROOT_REAL="$(readlink -f "$ROOT")" || die 'cannot resolve isolated root'
RUNS_REAL="$ROOT_REAL/runs"
mkdir -p -- "$RUNS_REAL" || die 'cannot create isolated runs directory'
OUT_REAL="$(realpath -m -- "$C1_RUN_OUT")" || die 'cannot normalize C1_RUN_OUT'
is_under "$OUT_REAL" "$RUNS_REAL" || die 'C1_RUN_OUT must be below isolated runs directory'
[[ ! -e "$OUT_REAL" && ! -L "$OUT_REAL" ]] || die 'output exists; no overwrite or append is allowed'
C1_RUN_OUT="$OUT_REAL"
export C1_RUN_OUT

LOCK_HELD=0
CHILD_PID=''
SESSION_READY=0
GPU_UUID=''
FINAL_STATUS='PREPARING'
FINAL_REASON=''
exec 9>>"$LOCK_FILE"
flock -n 9 || die 'another guarded C1 GPU0 session holds the exclusive lock'
LOCK_HELD=1

cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "$CHILD_PID" ]]; then
    if kill -0 "$CHILD_PID" 2>/dev/null; then kill -TERM -- "-$CHILD_PID" 2>/dev/null || true; fi
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  if [[ "$SESSION_READY" == 1 ]]; then
    snapshot_post 'outer_exit'
    write_guard_card "$FINAL_STATUS" "$FINAL_REASON"
    rm -f -- "$C1_RUN_OUT/.c1_guard_session.json"
  fi
  if [[ "$LOCK_HELD" == 1 ]]; then flock -u 9 || true; fi
  exit "$rc"
}
on_signal() {
  FINAL_STATUS='INTERRUPTED'
  FINAL_REASON='Signal received; guarded child process group is terminated/reaped; no retry.'
  if [[ -n "$CHILD_PID" ]]; then kill -TERM -- "-$CHILD_PID" 2>/dev/null || true; fi
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM HUP

verify_static || die 'static pin verification failed'
[[ ! -e "$C1_RUN_OUT" && ! -L "$C1_RUN_OUT" ]] || die 'output appeared during static verification; refusing overwrite'
mkdir -- "$C1_RUN_OUT" || die 'cannot create fresh isolated output'
GPU_UUID="$(snapshot_idle 'outer_prelaunch' '')" || die 'outer GPU0 idle check failed'
NONCE="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
[[ "$NONCE" =~ ^[0-9a-f]{32}$ ]] || die 'cannot generate session nonce'
export GPU_UUID
"$PYTHON" - "$C1_RUN_OUT/.c1_guard_session.json" "$NONCE" "$GPU_UUID" "$EXPECTED_PINS_SHA" <<'PY'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
x={'schema':'gtspp-c1-guard-session-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'host':os.uname().nodename,'nonce':sys.argv[2],'physical_gpu_index':0,
   'physical_gpu_uuid':sys.argv[3],'pins_sha256':sys.argv[4],
   'scope':'C1 only: workspace reuse + qnum=1 fast path; C2 mode=0; no C3 mutation'}
p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
PY
chmod 0600 "$C1_RUN_OUT/.c1_guard_session.json"
SESSION_READY=1
FINAL_STATUS='ENGINE_RUNNING'
FINAL_REASON='Static pins, physical GPU0 UUID mapping and prelaunch idle check passed.'
write_guard_card "$FINAL_STATUS" "$FINAL_REASON"

command -v setsid >/dev/null 2>&1 || die 'setsid unavailable; cannot safely reap an engine process group'
env C1_ALLOW_GPU0=YES \
  C1_RUN_OUT="$C1_RUN_OUT" \
  C1_GUARD_SCRIPT="$ROOT/run_c1_workspace_microbenchmark_guard.sh" \
  C1_GUARD_SESSION_FILE="$C1_RUN_OUT/.c1_guard_session.json" \
  C1_GUARD_SESSION_NONCE="$NONCE" \
  C1_GPU_UUID="$GPU_UUID" \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  NVIDIA_VISIBLE_DEVICES="$GPU_UUID" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  setsid "$ENGINE" "$@" &
CHILD_PID=$!
wait "$CHILD_PID"
ENGINE_RC=$?
CHILD_PID=''
snapshot_post 'outer_postrun'
if [[ "$ENGINE_RC" -eq 0 ]]; then
  FINAL_STATUS='COMPLETE'
  FINAL_REASON='Engine completed; post-run GPU0 snapshot recorded.'
else
  FINAL_STATUS='ENGINE_FAILED'
  FINAL_REASON="Engine exited $ENGINE_RC; post-run GPU0 snapshot recorded; no retry."
fi
write_guard_card "$FINAL_STATUS" "$FINAL_REASON"
exit "$ENGINE_RC"

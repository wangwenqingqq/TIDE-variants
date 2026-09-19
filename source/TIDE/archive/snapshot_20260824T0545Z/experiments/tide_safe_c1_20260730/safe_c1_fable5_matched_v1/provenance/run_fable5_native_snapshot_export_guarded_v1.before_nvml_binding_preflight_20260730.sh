#!/usr/bin/env bash
# Guarded Fable5 native E0 frozen-tree snapshot export (v1).
# Default behavior is refusal.  One permitted GPU action builds an immutable
# E0 tree and exports frozen metadata. It never replays policies, updates, or
# dynamic queries; it never measures time and never terminates a process.
# The snapshot exporter manifest pins this narrow scope.
set -euo pipefail

readonly ROOT='/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1'
readonly REQUIRED_GPU_ORDINAL='1'
readonly REQUIRED_GPU_UUID='GPU-CONFIGURE-ARCHIVE-DEVICE'
readonly MAX_IDLE_MEMORY_MIB='1024'
readonly PYTHON_BIN='python3'
readonly EXPORTER_SOURCE="$ROOT/src/fable5_native_snapshot_exporter_v1.cu"
readonly CORE_SOURCE="$ROOT/src/safe_c1_dynamic_gts.cu"
readonly MATCHED_RUNNER_SOURCE="$ROOT/src/fable5_matched_gpu_runner_v1.cu"
readonly STATIC_AUDIT="$ROOT/tools/audit_fable5_native_snapshot_export_static_v1.py"

EXECUTE=0; RUN_NAME=''; APPROVAL_ID=''; APPROVAL_EPOCH=''; TTL=''
GPU_ORDINAL=''; GPU_UUID=''; BUNDLE=''; HEADER_TRACE=''; HEADER_TRACE_SHA256=''
MANIFEST=''; MANIFEST_SHA256=''; BINARY=''; BINARY_SHA256=''
EXPORTER_SOURCE_SHA256=''; CORE_SOURCE_SHA256=''; MATCHED_RUNNER_SOURCE_SHA256=''
POOL_SHA256=''; QUERIES_SHA256=''; INITIAL_BASE_SHA256=''; STABLE_MAP_SHA256=''

usage() {
  cat <<'EOF'
Usage (default is a no-action refusal):
  run_fable5_native_snapshot_export_guarded_v1.sh --execute \
    --run-name NAME --approval-id ID --approval-issued-epoch UNIX_SECONDS --ttl 1..120 \
    --gpu-ordinal 1 --gpu-uuid GPU-CONFIGURE-ARCHIVE-DEVICE \
    --bundle BUNDLE --header-trace BUNDLE/trace.g2trc --header-trace-sha256 SHA256 \
    --manifest MANIFEST.json --manifest-sha256 SHA256 \
    --binary GTS_fable5_native_snapshot_exporter_v1 --binary-sha256 SHA256 \
    --exporter-source-sha256 SHA256 --core-source-sha256 SHA256 \
    --matched-runner-source-sha256 SHA256 \
    --pool-sha256 SHA256 --queries-sha256 SHA256 \
    --initial-base-sha256 SHA256 --stable-map-sha256 SHA256
EOF
}
fail() { printf 'FABLE5_SNAPSHOT_GUARD_FAIL: %s\n' "$*" >&2; exit 2; }
need_value() { (( $# >= 2 )) || fail "missing value after $1"; printf '%s' "$2"; }
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
valid_sha256() { [[ "$1" =~ ^[0-9a-fA-F]{64}$ ]]; }

while (( $# > 0 )); do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --run-name) RUN_NAME="$(need_value "$@")"; shift 2 ;;
    --approval-id) APPROVAL_ID="$(need_value "$@")"; shift 2 ;;
    --approval-issued-epoch) APPROVAL_EPOCH="$(need_value "$@")"; shift 2 ;;
    --ttl) TTL="$(need_value "$@")"; shift 2 ;;
    --gpu-ordinal) GPU_ORDINAL="$(need_value "$@")"; shift 2 ;;
    --gpu-uuid) GPU_UUID="$(need_value "$@")"; shift 2 ;;
    --bundle) BUNDLE="$(need_value "$@")"; shift 2 ;;
    --header-trace) HEADER_TRACE="$(need_value "$@")"; shift 2 ;;
    --header-trace-sha256) HEADER_TRACE_SHA256="$(need_value "$@")"; shift 2 ;;
    --manifest) MANIFEST="$(need_value "$@")"; shift 2 ;;
    --manifest-sha256) MANIFEST_SHA256="$(need_value "$@")"; shift 2 ;;
    --binary) BINARY="$(need_value "$@")"; shift 2 ;;
    --binary-sha256) BINARY_SHA256="$(need_value "$@")"; shift 2 ;;
    --exporter-source-sha256) EXPORTER_SOURCE_SHA256="$(need_value "$@")"; shift 2 ;;
    --core-source-sha256) CORE_SOURCE_SHA256="$(need_value "$@")"; shift 2 ;;
    --matched-runner-source-sha256) MATCHED_RUNNER_SOURCE_SHA256="$(need_value "$@")"; shift 2 ;;
    --pool-sha256) POOL_SHA256="$(need_value "$@")"; shift 2 ;;
    --queries-sha256) QUERIES_SHA256="$(need_value "$@")"; shift 2 ;;
    --initial-base-sha256) INITIAL_BASE_SHA256="$(need_value "$@")"; shift 2 ;;
    --stable-map-sha256) STABLE_MAP_SHA256="$(need_value "$@")"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

# No mutation, Python import, NVML, visibility change, or binary execution
# before this explicit user-approval gate.
[[ "$EXECUTE" == 1 ]] || { printf 'FABLE5_SNAPSHOT_GUARD_REFUSED: add --execute after fresh approval; no action taken\n' >&2; exit 64; }
[[ -d "$ROOT" && -f "$EXPORTER_SOURCE" && -f "$CORE_SOURCE" && -f "$MATCHED_RUNNER_SOURCE" && -f "$STATIC_AUDIT" ]] || fail 'isolated root incomplete'
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$ ]] || fail 'invalid run name'
[[ "$APPROVAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{2,120}$ ]] || fail 'invalid approval ID'
[[ "$APPROVAL_EPOCH" =~ ^[0-9]{10,}$ ]] || fail 'approval-issued-epoch must be UNIX seconds'
[[ "$TTL" =~ ^[0-9]+$ ]] && (( TTL >= 1 && TTL <= 120 )) || fail 'TTL must be 1..120 seconds'
now_epoch="$(date +%s)"; age=$(( now_epoch - APPROVAL_EPOCH ))
(( age >= 0 && age <= TTL )) || fail "approval expired or future (age=$age ttl=$TTL)"
[[ "$GPU_ORDINAL" == "$REQUIRED_GPU_ORDINAL" && "$GPU_UUID" == "$REQUIRED_GPU_UUID" ]] || fail 'GPU identity not approved GPU1'
for digest in "$HEADER_TRACE_SHA256" "$MANIFEST_SHA256" "$BINARY_SHA256" "$EXPORTER_SOURCE_SHA256" "$CORE_SOURCE_SHA256" "$MATCHED_RUNNER_SOURCE_SHA256" "$POOL_SHA256" "$QUERIES_SHA256" "$INITIAL_BASE_SHA256" "$STABLE_MAP_SHA256"; do valid_sha256 "$digest" || fail 'malformed SHA256'; done
[[ -d "$BUNDLE" && "$HEADER_TRACE" == "$BUNDLE/trace.g2trc" && -f "$HEADER_TRACE" ]] || fail 'header trace must be bundle-local trace.g2trc'
[[ -f "$BUNDLE/pool.i16" && -f "$BUNDLE/queries.i16" && -f "$BUNDLE/initial_base_stable_ids.i32" && -f "$BUNDLE/stable_id_to_pool_row.i32" ]] || fail 'bundle input set incomplete'
[[ -f "$MANIFEST" && -x "$BINARY" ]] || fail 'manifest or exporter binary missing'
[[ "$(sha256_of "$HEADER_TRACE")" == "$HEADER_TRACE_SHA256" ]] || fail 'header trace hash mismatch'
[[ "$(sha256_of "$MANIFEST")" == "$MANIFEST_SHA256" && "$(sha256_of "$BINARY")" == "$BINARY_SHA256" ]] || fail 'manifest or binary hash mismatch'
[[ "$(sha256_of "$EXPORTER_SOURCE")" == "$EXPORTER_SOURCE_SHA256" && "$(sha256_of "$CORE_SOURCE")" == "$CORE_SOURCE_SHA256" && "$(sha256_of "$MATCHED_RUNNER_SOURCE")" == "$MATCHED_RUNNER_SOURCE_SHA256" ]] || fail 'source hash mismatch'
[[ "$(sha256_of "$BUNDLE/pool.i16")" == "$POOL_SHA256" && "$(sha256_of "$BUNDLE/queries.i16")" == "$QUERIES_SHA256" && "$(sha256_of "$BUNDLE/initial_base_stable_ids.i32")" == "$INITIAL_BASE_SHA256" && "$(sha256_of "$BUNDLE/stable_id_to_pool_row.i32")" == "$STABLE_MAP_SHA256" ]] || fail 'payload hash mismatch'

# CPU-only header/layout preflight comes before RUN_DIR/mkdir and NVML.
INPUT_PREFLIGHT_JSON=''
if ! INPUT_PREFLIGHT_JSON="$("$PYTHON_BIN" - "$ROOT" "$BUNDLE" "$HEADER_TRACE" "$HEADER_TRACE_SHA256" <<'PY'
import hashlib,json,struct,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True); bundle=Path(sys.argv[2]).resolve(strict=True); trace=Path(sys.argv[3]).resolve(strict=True); trace_sha=sys.argv[4]
if root!=Path('/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1') or bundle.parent!=(root/'bundles').resolve() or trace!=bundle/'trace.g2trc': raise SystemExit('wrong_isolated_input_path')
raw=trace.read_bytes()
if len(raw)<48: raise SystemExit('truncated_trace')
h=struct.unpack_from('<8sI6IfQ',raw,0)
if h[:8]!=(b'E1GTRC02',2,32,4096,2048,6144,3,10) or len(raw)!=48+h[9]*12: raise SystemExit('unexpected_immutable_header_or_size')
if list(struct.unpack('<6144i',(bundle/'stable_id_to_pool_row.i32').read_bytes()))!=list(range(6144)): raise SystemExit('nonidentity_stable_map')
if list(struct.unpack('<4096i',(bundle/'initial_base_stable_ids.i32').read_bytes()))!=list(range(4096)): raise SystemExit('nonidentity_initial_base')
if hashlib.sha256(raw).hexdigest()!=trace_sha: raise SystemExit('header_trace_hash_mismatch')
print(json.dumps({'status':'PASS_SNAPSHOT_INPUT_PRE_NVML','header':{'dimension':32,'base_n':4096,'pool_n':6144,'query_n':3,'k':10,'event_count':h[9]},'stable_id_layout':'identity_full_immutable_pool_seeded_stable_ids_v5'},sort_keys=True))
PY
)"; then
  printf '%s\n' "$INPUT_PREFLIGHT_JSON" >&2; fail 'input preflight failed before run-directory/NVML'
fi

RUN_DIR="$ROOT/runs/$RUN_NAME"
[[ ! -e "$RUN_DIR" ]] || fail 'refusing existing run directory'
mkdir -p "$RUN_DIR/logs"
printf '%s\n' "$INPUT_PREFLIGHT_JSON" >"$RUN_DIR/input_preflight.json"

# The snapshot exporter manifest is static-audited before the first NVML import.
"$PYTHON_BIN" "$STATIC_AUDIT" --root "$ROOT" --out "$RUN_DIR/static_audit.json" >"$RUN_DIR/logs/static_audit.stdout.log" 2>"$RUN_DIR/logs/static_audit.stderr.log" || fail 'snapshot static audit failed'
"$PYTHON_BIN" - "$RUN_DIR/run_card.json" "$RUN_NAME" "$APPROVAL_ID" "$TTL" "$GPU_ORDINAL" "$GPU_UUID" "$HEADER_TRACE_SHA256" "$MANIFEST_SHA256" "$BINARY_SHA256" "$EXPORTER_SOURCE_SHA256" "$CORE_SOURCE_SHA256" "$MATCHED_RUNNER_SOURCE_SHA256" "$POOL_SHA256" "$QUERIES_SHA256" "$INITIAL_BASE_SHA256" "$STABLE_MAP_SHA256" <<'PY'
import json,sys
a=sys.argv[1:]
keys=('path','run_name','approval_id','ttl','ordinal','uuid','header','manifest','binary','exporter','core','runner','pool','queries','initial','mapping')
v=dict(zip(keys,a))
doc={'schema':'fable5-native-snapshot-guard-run-card-v1','status':'PRE_NVML_INPUT_AND_STATIC_AUDITS_PASSED','run_name':v['run_name'],'approval_id':v['approval_id'],'ttl_seconds':int(v['ttl']),'gpu_ordinal':int(v['ordinal']),'gpu_uuid':v['uuid'],'hashes':{k:v[k] for k in ('header','manifest','binary','exporter','core','runner','pool','queries','initial','mapping')},'scope':'one E0 frozen-tree export only; no update/query/policy replay/timing/C2/C3/deployment'}
with open(v['path'],'w') as f: json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')
PY

now_epoch="$(date +%s)"; age=$(( now_epoch - APPROVAL_EPOCH ))
(( age >= 0 && age <= TTL )) || fail "approval expired before NVML (age=$age ttl=$TTL)"
# NVML only: check identity/idle state. No process is altered.
"$PYTHON_BIN" - "$GPU_ORDINAL" "$GPU_UUID" "$MAX_IDLE_MEMORY_MIB" >"$RUN_DIR/logs/nvml_preflight.json" <<'PY'
import json,sys
try: import pynvml
except Exception as e: raise SystemExit('NVML Python binding unavailable: '+repr(e))
ordinal,expected,maxm=int(sys.argv[1]),sys.argv[2],int(sys.argv[3]); pynvml.nvmlInit()
try:
 h=pynvml.nvmlDeviceGetHandleByIndex(ordinal); uuid=pynvml.nvmlDeviceGetUUID(h); uuid=uuid.decode() if isinstance(uuid,bytes) else uuid
 m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); procs=[]
 for n in ('nvmlDeviceGetComputeRunningProcesses_v3','nvmlDeviceGetComputeRunningProcesses'):
  f=getattr(pynvml,n,None)
  if f is not None:
   try: procs=f(h)
   except pynvml.NVMLError_NotSupported: procs=[]
   break
 used=m.used//(1024*1024)
 if uuid!=expected: raise SystemExit('GPU UUID mismatch: '+str(uuid))
 if procs: raise SystemExit('GPU has active compute processes; no action taken')
 if used>maxm or u.gpu!=0: raise SystemExit('GPU is not idle; no action taken')
 print(json.dumps({'status':'PASS_NVML_GPU1_IDLE','ordinal':ordinal,'uuid':uuid,'memory_used_mib':used,'utilization_gpu_percent':u.gpu,'compute_process_count':len(procs)},sort_keys=True))
finally: pynvml.nvmlShutdown()
PY

SNAPSHOT="$RUN_DIR/native_E0_frozen_snapshot.json"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ORDINAL"
"$BINARY" --bundle "$BUNDLE" --header-trace "$HEADER_TRACE" --out "$SNAPSHOT" --trace-sha256 "$HEADER_TRACE_SHA256" --exporter-source-sha256 "$EXPORTER_SOURCE_SHA256" --core-source-sha256 "$CORE_SOURCE_SHA256" --matched-runner-source-sha256 "$MATCHED_RUNNER_SOURCE_SHA256" --pool-sha256 "$POOL_SHA256" --queries-sha256 "$QUERIES_SHA256" --initial-base-sha256 "$INITIAL_BASE_SHA256" --stable-map-sha256 "$STABLE_MAP_SHA256" >"$RUN_DIR/logs/exporter.stdout.log" 2>"$RUN_DIR/logs/exporter.stderr.log"

SNAPSHOT_SHA256="$("$PYTHON_BIN" - "$SNAPSHOT" "$HEADER_TRACE_SHA256" "$POOL_SHA256" "$QUERIES_SHA256" "$INITIAL_BASE_SHA256" "$STABLE_MAP_SHA256" "$EXPORTER_SOURCE_SHA256" "$CORE_SOURCE_SHA256" "$MATCHED_RUNNER_SOURCE_SHA256" <<'PY'
import hashlib,json,sys
p,trace,pool,queries,initial,mapping,exporter,core,runner=sys.argv[1:]
x=json.load(open(p))
if x.get('schema')!='fable5-native-frozen-snapshot-v1' or x.get('status')!='NATIVE_E0_CAPTURED_PENDING_CPU_SELECTOR': raise SystemExit('bad_snapshot_schema_or_status')
if x.get('residual_pruning',{}).get('mode')!=0 or x.get('residual_pruning',{}).get('query_executed') is not False: raise SystemExit('snapshot_not_baseline_mode_zero')
if x.get('c2_used') is not False or x.get('c3_used') is not False: raise SystemExit('snapshot_must_machine_record_c2_c3_unused')
i=x.get('immutable_input',{}); e={'header_trace_sha256':trace,'pool_i16_sha256':pool,'queries_i16_sha256':queries,'initial_base_stable_ids_i32_sha256':initial,'stable_id_to_pool_row_i32_sha256':mapping,'exporter_source_sha256':exporter,'core_source_sha256':core,'matched_runner_source_sha256':runner}
if any(i.get(k)!=v for k,v in e.items()): raise SystemExit('snapshot_immutable_hash_binding_mismatch')
if not isinstance(x.get('tree_height'),int) or x['tree_height']<=1 or x.get('fanout')!=10: raise SystemExit('snapshot_tree_shape_invalid')
print(hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
)"
"$PYTHON_BIN" - "$RUN_DIR/run_card.json" "$SNAPSHOT" "$SNAPSHOT_SHA256" <<'PY'
import json,sys
p,s,h=sys.argv[1:]; x=json.load(open(p)); x['status']='PASS_FABLE5_NATIVE_E0_SNAPSHOT_EXPORT_PENDING_CPU_SELECTOR'; x['snapshot']={'path':s,'sha256':h}
with open(p,'w') as f: json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
PY
printf 'PASS_FABLE5_NATIVE_E0_SNAPSHOT_EXPORT_PENDING_CPU_SELECTOR\n'

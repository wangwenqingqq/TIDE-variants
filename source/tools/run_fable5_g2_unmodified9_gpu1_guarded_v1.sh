#!/usr/bin/env bash
# Guarded GPU1 execution for the unmodified G2 9-op Safe-C1 correctness witness.
#
# This is not a benchmark. It runs the existing no-timing matched runner only
# after byte/ABI/oracle/static gates, a fresh approval, and a read-only NVML
# GPU1 identity/idle check. It never terminates or alters another process.
set -euo pipefail

readonly ROOT='/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1'
readonly REQUIRED_GPU_ORDINAL='1'
readonly REQUIRED_GPU_UUID='GPU-CONFIGURE-ARCHIVE-DEVICE'
readonly MAX_IDLE_MEMORY_MIB='1024'
readonly PYTHON_BIN='python3'
readonly NVML_PYTHON_BIN='/workspace/project/livsyn/.venv/bin/python'
readonly SOURCE="$ROOT/src/fable5_matched_gpu_runner_v1.cu"
readonly BINARY_DEFAULT="$ROOT/build/GTS_fable5_matched_v1"
readonly VALIDATOR="$ROOT/tools/validate_fable5_g2_unmodified9_gpu_receipts_v1.py"
readonly STATIC_AUDIT="$ROOT/tools/audit_fable5_g2_unmodified9_gpu_static_v1.py"
readonly TRACE_PREFLIGHT="$ROOT/tools/preflight_fable5_g2_unmodified9_trace_v1.py"
readonly TRACE_PREFLIGHT_CONTRACT="$ROOT/manifests/fable5_g2_unmodified9_trace_preflight_v1.json"
readonly SELF="$ROOT/tools/run_fable5_g2_unmodified9_gpu1_guarded_v1.sh"

EXECUTE=0
RUN_NAME=''
APPROVAL_ID=''
APPROVAL_EPOCH=''
TTL=''
GPU_ORDINAL=''
GPU_UUID=''
BUNDLE=''
TRACE=''
TRACE_SHA256=''
MANIFEST=''
MANIFEST_SHA256=''
BINARY=''
BINARY_SHA256=''
SOURCE_SHA256=''
LEAF_CAPACITY=''

usage() {
  cat <<'EOF'
Usage (default refuses without touching NVML/CUDA):
  run_fable5_g2_unmodified9_gpu1_guarded_v1.sh --execute \
    --run-name NAME --approval-id ID --approval-issued-epoch UNIX_SECONDS --ttl 1..120 \
    --gpu-ordinal 1 --gpu-uuid GPU-CONFIGURE-ARCHIVE-DEVICE \
    --bundle BUNDLE --trace BUNDLE/trace.g2trc --trace-sha256 SHA256 \
    --manifest MANIFEST.json --manifest-sha256 SHA256 \
    --binary GTS_fable5_matched_v1 --binary-sha256 SHA256 \
    --source-sha256 SHA256 --leaf-capacity 1
EOF
}

fail() { printf 'FABLE5_G2U9_GUARD_FAIL: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -eq 2 ]] || fail "missing value after $1"; printf '%s' "$2"; }
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
valid_sha256() { [[ "$1" =~ ^[0-9a-fA-F]{64}$ ]]; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --run-name) RUN_NAME="$(need_value "$1" "${2-}")"; shift 2 ;;
    --approval-id) APPROVAL_ID="$(need_value "$1" "${2-}")"; shift 2 ;;
    --approval-issued-epoch) APPROVAL_EPOCH="$(need_value "$1" "${2-}")"; shift 2 ;;
    --ttl) TTL="$(need_value "$1" "${2-}")"; shift 2 ;;
    --gpu-ordinal) GPU_ORDINAL="$(need_value "$1" "${2-}")"; shift 2 ;;
    --gpu-uuid) GPU_UUID="$(need_value "$1" "${2-}")"; shift 2 ;;
    --bundle) BUNDLE="$(need_value "$1" "${2-}")"; shift 2 ;;
    --trace) TRACE="$(need_value "$1" "${2-}")"; shift 2 ;;
    --trace-sha256) TRACE_SHA256="$(need_value "$1" "${2-}")"; shift 2 ;;
    --manifest) MANIFEST="$(need_value "$1" "${2-}")"; shift 2 ;;
    --manifest-sha256) MANIFEST_SHA256="$(need_value "$1" "${2-}")"; shift 2 ;;
    --binary) BINARY="$(need_value "$1" "${2-}")"; shift 2 ;;
    --binary-sha256) BINARY_SHA256="$(need_value "$1" "${2-}")"; shift 2 ;;
    --source-sha256) SOURCE_SHA256="$(need_value "$1" "${2-}")"; shift 2 ;;
    --leaf-capacity) LEAF_CAPACITY="$(need_value "$1" "${2-}")"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

# This refusal gate deliberately precedes path resolution, Python, file writes,
# GPU visibility changes, and NVML imports.
[[ "$EXECUTE" == 1 ]] || { printf 'FABLE5_G2U9_GUARD_REFUSED: add --execute after fresh approval; no action taken\n' >&2; exit 64; }

[[ -d "$ROOT" && -f "$SOURCE" && -f "$VALIDATOR" && -f "$STATIC_AUDIT" && -f "$TRACE_PREFLIGHT" && -f "$TRACE_PREFLIGHT_CONTRACT" && -f "$SELF" ]] || fail 'isolated root is incomplete'
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$ ]] || fail 'invalid run name'
[[ "$APPROVAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{2,120}$ ]] || fail 'invalid approval ID'
[[ "$APPROVAL_EPOCH" =~ ^[0-9]{10,}$ ]] || fail 'approval-issued-epoch must be UNIX seconds'
[[ "$TTL" =~ ^[0-9]+$ ]] && (( TTL >= 1 && TTL <= 120 )) || fail 'TTL must be 1..120 seconds'
now_epoch="$(date +%s)"
age=$(( now_epoch - APPROVAL_EPOCH ))
(( age >= 0 && age <= TTL )) || fail "approval expired or from the future (age=${age}s ttl=${TTL}s)"
[[ "$GPU_ORDINAL" == "$REQUIRED_GPU_ORDINAL" ]] || fail 'guard is restricted to GPU ordinal 1'
[[ "$GPU_UUID" == "$REQUIRED_GPU_UUID" ]] || fail 'GPU UUID does not match approved GPU1'
[[ "$LEAF_CAPACITY" == '1' ]] || fail 'unmodified G2 9-op contract requires leaf capacity 1'
for item in "$TRACE_SHA256" "$MANIFEST_SHA256" "$BINARY_SHA256" "$SOURCE_SHA256"; do valid_sha256 "$item" || fail 'malformed SHA256'; done
[[ -d "$BUNDLE" && -f "$BUNDLE/pool.i16" && -f "$BUNDLE/queries.i16" && -f "$BUNDLE/stable_id_to_pool_row.i32" && -f "$BUNDLE/initial_base_stable_ids.i32" ]] || fail 'bundle input set is incomplete'
[[ -f "$TRACE" && -f "$MANIFEST" && -x "$BINARY" ]] || fail 'trace, manifest, or executable missing'
[[ "$(sha256_of "$TRACE")" == "$TRACE_SHA256" ]] || fail 'trace hash mismatch'
[[ "$(sha256_of "$MANIFEST")" == "$MANIFEST_SHA256" ]] || fail 'manifest hash mismatch'
[[ "$(sha256_of "$SOURCE")" == "$SOURCE_SHA256" ]] || fail 'runner source hash mismatch'
[[ "$(sha256_of "$BINARY")" == "$BINARY_SHA256" ]] || fail 'runner binary hash mismatch'
[[ "$BINARY" == "$BINARY_DEFAULT" ]] || fail 'binary path must be canonical'

# Manifest binds the pre-existing input, native E0 snapshot, no-timing runner,
# CPU build receipt, and every safety tool before trace preflight/run dir/NVML.
MANIFEST_PRECHECK_JSON=''
if ! MANIFEST_PRECHECK_JSON="$("$PYTHON_BIN" - "$ROOT" "$MANIFEST" "$BUNDLE" "$TRACE" "$TRACE_SHA256" "$BINARY" "$BINARY_SHA256" "$SOURCE_SHA256" "$SELF" "$VALIDATOR" "$STATIC_AUDIT" "$TRACE_PREFLIGHT" "$TRACE_PREFLIGHT_CONTRACT" <<'PY'
import hashlib,json,sys
from pathlib import Path
(root_s, manifest_s, bundle_s, trace_s, trace_sha, binary_s, binary_sha, source_sha,
 guard_s, validator_s, audit_s, preflight_s, contract_s) = sys.argv[1:]
root=Path(root_s).resolve(strict=True); manifest=Path(manifest_s).resolve(strict=True)
bundle=Path(bundle_s).resolve(strict=True); trace=Path(trace_s).resolve(strict=True)
binary=Path(binary_s).resolve(strict=True); guard=Path(guard_s).resolve(strict=True)
validator=Path(validator_s).resolve(strict=True); audit=Path(audit_s).resolve(strict=True)
preflight=Path(preflight_s).resolve(strict=True); contract=Path(contract_s).resolve(strict=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def under(rel):
 if not isinstance(rel,str) or not rel or Path(rel).is_absolute(): raise SystemExit('manifest_relative_path_invalid')
 p=(root/rel).resolve(strict=True)
 try:p.relative_to(root)
 except ValueError:raise SystemExit('manifest_path_escapes_root')
 return p
m=json.load(open(manifest))
if m.get('schema')!='fable5-g2-unmodified9-gpu-runner-v1' or m.get('status')!='CPU_BUILT_PRE_GPU': raise SystemExit('wrong_manifest_schema_or_status')
if m.get('execution_boundary')!={'compiled':True,'cuda_binary_executed':False,'gpu_used':False,'nvml_queried':False}: raise SystemExit('manifest_execution_boundary')
v=m.get('variant',{})
if v.get('id')!='unmodified_g2_l2_correctness_witness_v1' or v.get('input_modification')!='none' or 'performance' not in v.get('prohibited_claims',[]): raise SystemExit('manifest_variant')
r=m.get('runner',{})
if under(r.get('source'))!=root/'src/fable5_matched_gpu_runner_v1.cu' or r.get('source_sha256')!=source_sha: raise SystemExit('manifest_runner_source')
if under(r.get('binary'))!=binary or r.get('binary_sha256')!=binary_sha or sha(binary)!=binary_sha: raise SystemExit('manifest_runner_binary')
receipt=under(r.get('cpu_build_receipt'))
if r.get('cpu_build_receipt_sha256')!=sha(receipt): raise SystemExit('manifest_build_receipt_hash')
rd=json.load(open(receipt))
if rd.get('status')!='PASS_CPU_BUILD_ONLY_NOT_EXECUTED' or rd.get('source_sha256')!=source_sha or rd.get('binary_sha256')!=binary_sha or rd.get('cuda_binary_executed') is not False or rd.get('gpu_used') is not False: raise SystemExit('manifest_build_receipt_semantics')
i=m.get('input',{})
if under(i.get('bundle'))!=bundle or under(i.get('trace'))!=trace or i.get('trace_sha256')!=trace_sha or sha(trace)!=trace_sha: raise SystemExit('manifest_input')
if under(i.get('preflight_contract'))!=contract or i.get('preflight_contract_sha256')!=sha(contract): raise SystemExit('manifest_contract')
snap=under(i.get('native_e0_snapshot'))
if i.get('native_e0_snapshot_sha256')!=sha(snap): raise SystemExit('manifest_snapshot')
tools=m.get('tools',{})
expected={'guard':guard,'receipt_validator':validator,'static_audit':audit,'trace_preflight':preflight}
for name,path in expected.items():
 rel=tools.get(name)
 if under(rel)!=path or tools.get(name+'_sha256')!=sha(path): raise SystemExit('manifest_tool_'+name)
print(json.dumps({'status':'PASS_FABLE5_G2U9_MANIFEST_PRECHECK','variant':v['id'],'bundle':str(bundle),'trace_sha256':trace_sha},sort_keys=True))
PY
)"; then
  printf '%s\n' "$MANIFEST_PRECHECK_JSON" >&2
  fail 'manifest precheck failed before trace preflight/run-directory/NVML'
fi

TRACE_PREFLIGHT_JSON=''
if ! TRACE_PREFLIGHT_JSON="$("$PYTHON_BIN" "$TRACE_PREFLIGHT" \
  --root "$ROOT" --bundle "$BUNDLE" --trace "$TRACE" --trace-sha256 "$TRACE_SHA256" \
  --contract "$TRACE_PREFLIGHT_CONTRACT")"; then
  printf '%s\n' "$TRACE_PREFLIGHT_JSON" >&2
  fail 'unmodified G2 trace preflight failed before run-directory/NVML'
fi

[[ -x "$NVML_PYTHON_BIN" ]] || fail 'pinned Python NVML interpreter unavailable'
"$NVML_PYTHON_BIN" -c 'import pynvml' >/dev/null 2>&1 || fail 'pinned Python NVML binding unavailable'

RUN_DIR="$ROOT/runs/$RUN_NAME"
[[ ! -e "$RUN_DIR" ]] || fail 'refusing to reuse an existing run directory'
mkdir -p "$RUN_DIR/safe_c1" "$RUN_DIR/buffer_only" "$RUN_DIR/logs"
printf '%s\n' "$MANIFEST_PRECHECK_JSON" > "$RUN_DIR/manifest_precheck.json"
printf '%s\n' "$TRACE_PREFLIGHT_JSON" > "$RUN_DIR/trace_preflight.json"

# Static audit is deliberately after read-only preflight but before the first
# NVML initialization or CUDA visibility change.
"$PYTHON_BIN" "$STATIC_AUDIT" --root "$ROOT" --out "$RUN_DIR/static_audit.json" >"$RUN_DIR/logs/static_audit.stdout.log" 2>"$RUN_DIR/logs/static_audit.stderr.log" || fail 'static audit failed'

"$PYTHON_BIN" - "$RUN_DIR/run_card.json" "$RUN_NAME" "$APPROVAL_ID" "$TTL" "$GPU_ORDINAL" "$GPU_UUID" "$TRACE_SHA256" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$BINARY_SHA256" <<'PY'
import json,sys
(path, run_name, approval_id, ttl, ordinal, uuid, trace_sha, manifest_sha, source_sha, binary_sha)=sys.argv[1:]
doc={
 'schema':'fable5-g2-unmodified9-gpu1-guard-run-card-v1',
 'status':'PRE_NVML_G2U9_TRACE_AND_STATIC_AUDITS_PASSED',
 'run_name':run_name,'approval_id':approval_id,'ttl_seconds':int(ttl),
 'gpu_ordinal':int(ordinal),'gpu_uuid':uuid,'trace_sha256':trace_sha,
 'manifest_sha256':manifest_sha,'source_sha256':source_sha,'binary_sha256':binary_sha,
 'scope':'unmodified selected G2 L2 matched correctness only; no workload/timing/performance/recall/C2/C3/deployment claim'
}
with open(path,'w') as f: json.dump(doc,f,indent=2,sort_keys=True); f.write('\n')
PY

now_epoch="$(date +%s)"
age=$(( now_epoch - APPROVAL_EPOCH ))
(( age >= 0 && age <= TTL )) || fail "approval expired before NVML gate (age=${age}s ttl=${TTL}s)"

"$NVML_PYTHON_BIN" - "$GPU_ORDINAL" "$GPU_UUID" "$MAX_IDLE_MEMORY_MIB" >"$RUN_DIR/logs/nvml_preflight.json" <<'PY'
import json,sys
try:
 import pynvml
except Exception as exc:
 raise SystemExit('NVML Python binding unavailable: '+repr(exc))
ordinal,expected_uuid,max_idle=int(sys.argv[1]),sys.argv[2],int(sys.argv[3])
pynvml.nvmlInit()
try:
 handle=pynvml.nvmlDeviceGetHandleByIndex(ordinal)
 uuid=pynvml.nvmlDeviceGetUUID(handle)
 if isinstance(uuid,bytes): uuid=uuid.decode()
 memory=pynvml.nvmlDeviceGetMemoryInfo(handle)
 util=pynvml.nvmlDeviceGetUtilizationRates(handle)
 processes=[]
 for name in ('nvmlDeviceGetComputeRunningProcesses_v3','nvmlDeviceGetComputeRunningProcesses'):
  fn=getattr(pynvml,name,None)
  if fn is not None:
   try: processes=fn(handle)
   except pynvml.NVMLError_NotSupported: processes=[]
   break
 used_mib=memory.used//(1024*1024)
 if uuid!=expected_uuid: raise SystemExit('GPU UUID mismatch: '+str(uuid))
 if processes: raise SystemExit('GPU has active compute processes; no action taken')
 if used_mib>max_idle or util.gpu!=0: raise SystemExit('GPU is not idle; no action taken')
 print(json.dumps({'status':'PASS_NVML_GPU1_IDLE','ordinal':ordinal,'uuid':uuid,'memory_used_mib':used_mib,'utilization_gpu_percent':util.gpu,'compute_process_count':len(processes)},sort_keys=True))
finally:
 pynvml.nvmlShutdown()
PY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ORDINAL"
"$BINARY" --bundle "$BUNDLE" --trace "$TRACE" --policy safe_c1 \
  --out "$RUN_DIR/safe_c1/results.jsonl" --summary "$RUN_DIR/safe_c1/summary.json" \
  --trace-sha256 "$TRACE_SHA256" --leaf-capacity "$LEAF_CAPACITY" --require-coverage 0 \
  >"$RUN_DIR/logs/safe_c1.stdout.log" 2>"$RUN_DIR/logs/safe_c1.stderr.log"
"$BINARY" --bundle "$BUNDLE" --trace "$TRACE" --policy buffer_only \
  --out "$RUN_DIR/buffer_only/results.jsonl" --summary "$RUN_DIR/buffer_only/summary.json" \
  --trace-sha256 "$TRACE_SHA256" --leaf-capacity "$LEAF_CAPACITY" --require-coverage 0 \
  >"$RUN_DIR/logs/buffer_only.stdout.log" 2>"$RUN_DIR/logs/buffer_only.stderr.log"
"$PYTHON_BIN" "$VALIDATOR" --bundle "$BUNDLE" --trace "$TRACE" --trace-sha256 "$TRACE_SHA256" \
  --safe-results "$RUN_DIR/safe_c1/results.jsonl" --buffer-results "$RUN_DIR/buffer_only/results.jsonl" \
  --out "$RUN_DIR/matched_receipt_validation.json" \
  >"$RUN_DIR/logs/receipt_validator.stdout.log" 2>"$RUN_DIR/logs/receipt_validator.stderr.log"

"$PYTHON_BIN" - "$RUN_DIR/run_card.json" <<'PY'
import json,sys
path=sys.argv[1]
doc=json.load(open(path))
doc['status']='PASS_FABLE5_G2_UNMODIFIED9_GUARDED_RUN_PENDING_MANUAL_REVIEW'
with open(path,'w') as f: json.dump(doc,f,indent=2,sort_keys=True); f.write('\n')
PY
printf 'PASS_FABLE5_G2_UNMODIFIED9_GUARDED_RUN_PENDING_MANUAL_REVIEW\n'

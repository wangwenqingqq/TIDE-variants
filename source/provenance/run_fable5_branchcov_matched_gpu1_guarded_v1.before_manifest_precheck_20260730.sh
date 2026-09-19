#!/usr/bin/env bash
# Fable5 constructed-branch-coverage Safe-C1/buffer-only GPU1 guard (v1).
#
# Default behavior is refusal: no NVML query, no CUDA visibility change, no
# binary execution.  `--execute` plus a fresh, timestamped approval is required.
# This guard is only for a constructed branch-coverage micro-witness; it must
# never be used for workload, timing, throughput, latency, recall, C2, C3, or deployment claims.
# This guard never terminates processes and never modifies archived artifacts.
set -euo pipefail

readonly ROOT='/workspace/experiments/tide_safe_c1_20260730/safe_c1_fable5_matched_v1'
readonly REQUIRED_GPU_ORDINAL='1'
readonly REQUIRED_GPU_UUID='GPU-CONFIGURE-ARCHIVE-DEVICE'
readonly MAX_IDLE_MEMORY_MIB='1024'
readonly PYTHON_BIN='python3'
# System python on 8p lacks pynvml. This existing local interpreter is used
# only for the Python NVML binding; all other audits remain on system python.
readonly NVML_PYTHON_BIN='/workspace/project/livsyn/.venv/bin/python'
readonly SOURCE="$ROOT/src/fable5_matched_gpu_runner_v1.cu"
readonly VALIDATOR="$ROOT/tools/validate_fable5_matched_gpu_receipts_v1.py"
readonly STATIC_AUDIT="$ROOT/tools/audit_fable5_branchcov_matched_gpu_static_v1.py"
readonly TRACE_PREFLIGHT="$ROOT/tools/preflight_fable5_branchcov_matched_trace_v1.py"
readonly TRACE_PREFLIGHT_CONTRACT="$ROOT/manifests/fable5_branchcov_matched_trace_preflight_v1.json"

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
CANDIDATE_CONTRACT=''
CANDIDATE_CONTRACT_SHA256=''
MANIFEST=''
MANIFEST_SHA256=''
BINARY=''
BINARY_SHA256=''
SOURCE_SHA256=''
LEAF_CAPACITY=''

usage() {
  cat <<'EOF'
Usage (default is a no-action refusal):
  run_fable5_matched_gpu1_guarded_v1.sh --execute \
    --run-name NAME --approval-id ID --approval-issued-epoch UNIX_SECONDS --ttl 1..120 \
    --gpu-ordinal 1 --gpu-uuid GPU-CONFIGURE-ARCHIVE-DEVICE \
    --bundle BUNDLE --trace BUNDLE/trace.fable5.e1gtrc --trace-sha256 SHA256 \
    --candidate-contract BUNDLE/fable5_matched_trace_candidate_v1.json --candidate-contract-sha256 SHA256 \
    --manifest MANIFEST.json --manifest-sha256 SHA256 \
    --binary GTS_fable5_matched_v1 --binary-sha256 SHA256 \
    --source-sha256 SHA256 --leaf-capacity N
EOF
}

fail() { printf 'FABLE5_BRANCHCOV_GUARD_FAIL: %s\n' "$*" >&2; exit 2; }
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
    --candidate-contract) CANDIDATE_CONTRACT="$(need_value "$1" "${2-}")"; shift 2 ;;
    --candidate-contract-sha256) CANDIDATE_CONTRACT_SHA256="$(need_value "$1" "${2-}")"; shift 2 ;;
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

# This gate intentionally precedes all filesystem mutation, Python imports,
# NVML interactions, environment changes, and executable invocation.
[[ "$EXECUTE" == 1 ]] || { printf 'FABLE5_BRANCHCOV_GUARD_REFUSED: add --execute after fresh approval; no action taken\n' >&2; exit 64; }

[[ -d "$ROOT" && -f "$SOURCE" && -f "$VALIDATOR" && -f "$STATIC_AUDIT" && -f "$TRACE_PREFLIGHT" && -f "$TRACE_PREFLIGHT_CONTRACT" ]] || fail 'isolated Fable5 root is incomplete'
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$ ]] || fail 'invalid run name'
[[ "$APPROVAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{2,120}$ ]] || fail 'invalid approval ID'
[[ "$APPROVAL_EPOCH" =~ ^[0-9]{10,}$ ]] || fail 'approval-issued-epoch must be UNIX seconds'
[[ "$TTL" =~ ^[0-9]+$ ]] && (( TTL >= 1 && TTL <= 120 )) || fail 'TTL must be 1..120 seconds'
now_epoch="$(date +%s)"
age=$(( now_epoch - APPROVAL_EPOCH ))
(( age >= 0 && age <= TTL )) || fail "approval expired or from the future (age=${age}s ttl=${TTL}s)"
[[ "$GPU_ORDINAL" == "$REQUIRED_GPU_ORDINAL" ]] || fail "guard is restricted to GPU ordinal $REQUIRED_GPU_ORDINAL"
[[ "$GPU_UUID" == "$REQUIRED_GPU_UUID" ]] || fail 'GPU UUID does not match the approved GPU1 UUID'
[[ "$LEAF_CAPACITY" =~ ^[1-9][0-9]*$ ]] || fail 'leaf capacity must be a positive integer'
for item in "$TRACE_SHA256" "$CANDIDATE_CONTRACT_SHA256" "$MANIFEST_SHA256" "$BINARY_SHA256" "$SOURCE_SHA256"; do valid_sha256 "$item" || fail 'malformed SHA256'; done
[[ -d "$BUNDLE" && -f "$BUNDLE/pool.i16" && -f "$BUNDLE/queries.i16" && -f "$BUNDLE/stable_id_to_pool_row.i32" && -f "$BUNDLE/initial_base_stable_ids.i32" ]] || fail 'bundle input set is incomplete'
[[ -f "$TRACE" && -f "$CANDIDATE_CONTRACT" && -f "$MANIFEST" && -x "$BINARY" ]] || fail 'trace, candidate contract, manifest, or executable missing'
[[ "$(sha256_of "$TRACE")" == "$TRACE_SHA256" ]] || fail 'trace hash mismatch'
[[ "$(sha256_of "$CANDIDATE_CONTRACT")" == "$CANDIDATE_CONTRACT_SHA256" ]] || fail 'candidate contract hash mismatch'
[[ "$(sha256_of "$MANIFEST")" == "$MANIFEST_SHA256" ]] || fail 'manifest hash mismatch'
[[ "$(sha256_of "$SOURCE")" == "$SOURCE_SHA256" ]] || fail 'runner source hash mismatch'
[[ "$(sha256_of "$BINARY")" == "$BINARY_SHA256" ]] || fail 'runner binary hash mismatch'

# This read-only ABI/layout/identity gate intentionally runs before any run-directory
# creation and before the first NVML import. Its pass does not establish native
# certificate outcomes; the later runner must emit and validate those receipts.
TRACE_PREFLIGHT_JSON=''
if ! TRACE_PREFLIGHT_JSON="$("$PYTHON_BIN" "$TRACE_PREFLIGHT" \
  --root "$ROOT" --bundle "$BUNDLE" --trace "$TRACE" --trace-sha256 "$TRACE_SHA256" \
  --candidate-contract "$CANDIDATE_CONTRACT" --candidate-contract-sha256 "$CANDIDATE_CONTRACT_SHA256" \
  --contract "$TRACE_PREFLIGHT_CONTRACT" --leaf-capacity "$LEAF_CAPACITY")"; then
  printf '%s\n' "$TRACE_PREFLIGHT_JSON" >&2
  fail 'branch-coverage trace preflight failed before run-directory creation/NVML'
fi

[[ -x "$NVML_PYTHON_BIN" ]] || fail 'pinned Python NVML interpreter missing or not executable'
"$NVML_PYTHON_BIN" -c 'import pynvml' >/dev/null 2>&1 || fail 'pinned Python NVML binding unavailable'

RUN_DIR="$ROOT/runs/$RUN_NAME"
[[ ! -e "$RUN_DIR" ]] || fail 'refusing to reuse an existing run directory'
mkdir -p "$RUN_DIR/safe_c1" "$RUN_DIR/buffer_only" "$RUN_DIR/logs"
printf '%s\n' "$TRACE_PREFLIGHT_JSON" > "$RUN_DIR/trace_preflight.json"

# CPU-only static audit is fail-closed before the first NVML import.
"$PYTHON_BIN" "$STATIC_AUDIT" --root "$ROOT" --out "$RUN_DIR/static_audit.json" >"$RUN_DIR/logs/static_audit.stdout.log" 2>"$RUN_DIR/logs/static_audit.stderr.log" || fail 'static audit failed'

"$PYTHON_BIN" - "$RUN_DIR/run_card.json" "$RUN_NAME" "$APPROVAL_ID" "$TTL" "$GPU_ORDINAL" "$GPU_UUID" "$TRACE_SHA256" "$CANDIDATE_CONTRACT_SHA256" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$BINARY_SHA256" <<'PY'
import json, sys
(path, run_name, approval_id, ttl, ordinal, uuid, trace_sha, candidate_sha, manifest_sha, source_sha, binary_sha) = sys.argv[1:]
json.dump({
  'schema': 'fable5-branchcov-matched-gpu1-guard-run-card-v1', 'status': 'PRE_NVML_BRANCHCOV_TRACE_AND_STATIC_AUDITS_PASSED',
  'run_name': run_name, 'approval_id': approval_id, 'ttl_seconds': int(ttl),
  'gpu_ordinal': int(ordinal), 'gpu_uuid': uuid,
  'trace_sha256': trace_sha, 'candidate_contract_sha256': candidate_sha, 'manifest_sha256': manifest_sha,
  'source_sha256': source_sha, 'binary_sha256': binary_sha,
  'scope': 'constructed branch-coverage matched correctness only; no real-workload/timing/performance/recall/C2/C3/deployment claim'
}, open(path, 'w'), indent=2, sort_keys=True)
open(path, 'a').write('\n')
PY

# Re-check expiry after all CPU-only artifact/static gates and immediately
# before the first NVML interaction.
now_epoch="$(date +%s)"
age=$(( now_epoch - APPROVAL_EPOCH ))
(( age >= 0 && age <= TTL )) || fail "approval expired before NVML gate (age=${age}s ttl=${TTL}s)"

# NVML-only idle/identity gate.  Any import/API error is fail-closed.  No
# process is altered; a nonempty process list causes an exit before launch.
"$NVML_PYTHON_BIN" - "$GPU_ORDINAL" "$GPU_UUID" "$MAX_IDLE_MEMORY_MIB" >"$RUN_DIR/logs/nvml_preflight.json" <<'PY'
import json, sys
try:
    import pynvml
except Exception as exc:
    raise SystemExit('NVML Python binding unavailable: ' + repr(exc))
ordinal, expected_uuid, max_idle = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
pynvml.nvmlInit()
try:
    handle = pynvml.nvmlDeviceGetHandleByIndex(ordinal)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes): uuid = uuid.decode()
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    processes = []
    for name in ('nvmlDeviceGetComputeRunningProcesses_v3', 'nvmlDeviceGetComputeRunningProcesses'):
        fn = getattr(pynvml, name, None)
        if fn is not None:
            try:
                processes = fn(handle)
            except pynvml.NVMLError_NotSupported:
                processes = []
            break
    used_mib = memory.used // (1024 * 1024)
    if uuid != expected_uuid:
        raise SystemExit('GPU UUID mismatch: ' + str(uuid))
    if processes:
        raise SystemExit('GPU has active compute processes; no action taken')
    if used_mib > max_idle or util.gpu != 0:
        raise SystemExit('GPU is not idle; no action taken')
    print(json.dumps({'status': 'PASS_NVML_GPU1_IDLE', 'ordinal': ordinal, 'uuid': uuid,
                      'memory_used_mib': used_mib, 'utilization_gpu_percent': util.gpu,
                      'compute_process_count': len(processes)}, sort_keys=True))
finally:
    pynvml.nvmlShutdown()
PY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ORDINAL"
"$BINARY" --bundle "$BUNDLE" --trace "$TRACE" --policy safe_c1 \
  --out "$RUN_DIR/safe_c1/results.jsonl" --summary "$RUN_DIR/safe_c1/summary.json" \
  --trace-sha256 "$TRACE_SHA256" --leaf-capacity "$LEAF_CAPACITY" --require-coverage 1 \
  >"$RUN_DIR/logs/safe_c1.stdout.log" 2>"$RUN_DIR/logs/safe_c1.stderr.log"
"$BINARY" --bundle "$BUNDLE" --trace "$TRACE" --policy buffer_only \
  --out "$RUN_DIR/buffer_only/results.jsonl" --summary "$RUN_DIR/buffer_only/summary.json" \
  --trace-sha256 "$TRACE_SHA256" --leaf-capacity "$LEAF_CAPACITY" --require-coverage 1 \
  >"$RUN_DIR/logs/buffer_only.stdout.log" 2>"$RUN_DIR/logs/buffer_only.stderr.log"
"$PYTHON_BIN" "$VALIDATOR" --safe-results "$RUN_DIR/safe_c1/results.jsonl" \
  --buffer-results "$RUN_DIR/buffer_only/results.jsonl" --trace-sha256 "$TRACE_SHA256" \
  --out "$RUN_DIR/matched_receipt_validation.json" \
  >"$RUN_DIR/logs/receipt_validator.stdout.log" 2>"$RUN_DIR/logs/receipt_validator.stderr.log"

"$PYTHON_BIN" - "$RUN_DIR/run_card.json" <<'PY'
import json, sys
p=sys.argv[1]
r=json.load(open(p))
r['status']='PASS_FABLE5_BRANCHCOV_MATCHED_GUARDED_RUN_PENDING_MANUAL_REVIEW'
with open(p,'w') as f: json.dump(r,f,indent=2,sort_keys=True); f.write('\n')
PY
printf 'PASS_FABLE5_BRANCHCOV_MATCHED_GUARDED_RUN_PENDING_MANUAL_REVIEW\n'

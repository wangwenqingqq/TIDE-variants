#!/usr/bin/env bash
# Manual, fail-closed GPU0 launcher for the C3 same-leaf capacity boundary probe.
# No argument and --verify-static are CPU-only.  --execute is deliberately
# manual and must not be called by unattended automation.
set -Eeuo pipefail

ROOT=/workspace/experiments/tide_safe_c1_20260727/c3_native_direct_insert_probe/same_leaf_capacity_heldout_v1
SRCROOT=/workspace/experiments/tide_safe_c1_20260727/source_gts_incremental
BUNDLE=/workspace/experiments/tide_safe_c1_20260727/runs/e1gi_b_quantized_gts_bundles_v1_20260727/sift128_slice__sift128_slice__seed_20260727
HOST_REQUIRED=CONFIGURE_ARCHIVE_HOST
PYTHON=/workspace/legacy_workspace/GTS/bench_env/bin/python
RUNNER="$ROOT/bin/GTS_c3_same_leaf_capacity_heldout"
VALIDATOR="$ROOT/validate_c3_same_leaf_capacity_heldout.py"
AUDITOR="$ROOT/audit_c3_same_leaf_capacity_static.py"
PROTOCOL="$ROOT/protocol_same_leaf_capacity_heldout_v1.json"
SELECTION="$ROOT/preflight/selection_v2_structure_fingerprint.json"
BUILD_MANIFEST="$ROOT/build_manifest_same_leaf_capacity_v1.json"
LOCK="$ROOT/.locks/c3_same_leaf_capacity_physical_gpu0.lock"
GUARD_SCRIPT="$ROOT/scripts/run_c3_same_leaf_capacity_gpu0_guarded.sh"
GPU0_MAX_MEMORY_USED_MIB=256
# Immutable provenance anchors. verify_frozen_input_provenance runs before the
# first nvidia-smi invocation in --execute.
PROTOCOL_SHA256=c24265f3df29ffaeb1ad5158126635019cf5cbd3af091f8ddb66f9e0790a697e
SELECTION_SHA256=55765f72c9a1a382ffb01a5735412289454d61d697ae592ed64d9a9fea84dfee
BUNDLE_MANIFEST_SHA256=b93e1242c9cd22dc351ac1c408d4211ebc1020074d5af7d50dbe69ca4c554423
LOCK_HELD=0
ACTIVE_RUNNER_PID=""
RUNNER_LAUNCHING=0
SIGNAL_PENDING=""

usage() {
  cat <<'USAGE'
Usage:
  run_c3_same_leaf_capacity_gpu0_guarded.sh --verify-static
  C3_CAPACITY_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE \
    run_c3_same_leaf_capacity_gpu0_guarded.sh --execute

Safety contract:
  * default/no argument does not run CUDA or call nvidia-smi;
  * --verify-static is CPU-only and verifies source/build/input provenance;
  * --execute is refused unless host is 8p, physical GPU0 has no compute
    process, 0% utilization, <=256 MiB used, and all frozen hashes match;
  * physical nvidia-smi GPU0 is mapped by UUID to CUDA_VISIBLE_DEVICES, so
    runner CUDA device 0 cannot silently become a different physical GPU;
  * GPU state is checked before lock and immediately before the isolated
    timeout process; no process is killed/reset by this launcher;
  * a signal reaps the runner process group before the shared-GPU lock is
    released.  The run is correctness only, never performance evidence.
USAGE
}

die() { echo "C3 same-leaf GPU0 guard: $*" >&2; exit 64; }
trim() { local x=$1; x="${x#"${x%%[![:space:]]*}"}"; x="${x%"${x##*[![:space:]]}"}"; printf '%s' "$x"; }

static_audit() {
  local output=$1
  "$PYTHON" "$AUDITOR" \
    --protocol "$PROTOCOL" --selection "$SELECTION" \
    --runner "$ROOT/src/gts_c3_same_leaf_capacity_heldout.cu" \
    --validator "$VALIDATOR" \
    --tree-header "$SRCROOT/include/tree.cuh" --search-header "$SRCROOT/include/search_v2.cuh" \
    --binary "$RUNNER" --build-manifest "$BUILD_MANIFEST" --launcher "$GUARD_SCRIPT" \
    --out "$output"
}

# This CPU-only check validates exact immutable protocol/selection/bundle
# bytes, and emits all observed hashes. It must complete before GPU access.
verify_frozen_input_provenance() {
  local output=$1
  mkdir -p "$(dirname "$output")"
  if ! "$PYTHON" - "$PROTOCOL" "$SELECTION" "$BUNDLE" \
      "$PROTOCOL_SHA256" "$SELECTION_SHA256" "$BUNDLE_MANIFEST_SHA256" >"$output" 2>&1 <<'PY'
import hashlib, json, pathlib, sys
protocol, selection, bundle = map(pathlib.Path, sys.argv[1:4])
expected_protocol, expected_selection, expected_manifest = sys.argv[4:7]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(ok, msg):
    if not ok:
        raise SystemExit('FAIL: ' + msg)

require(digest(protocol) == expected_protocol, 'protocol SHA256 mismatch')
require(digest(selection) == expected_selection, 'selection SHA256 mismatch')
manifest_path = bundle / 'manifest.json'
require(digest(manifest_path) == expected_manifest, 'bundle manifest SHA256 mismatch')
p = json.loads(protocol.read_text())
s = json.loads(selection.read_text())
m = json.loads(manifest_path.read_text())
require(p.get('schema') == 'c3-same-leaf-capacity-heldout-protocol-v1', 'protocol schema')
require(p.get('status') == 'PRE_REGISTERED_AFTER_CPU_SELECTION_AND_COMPILE_ONLY_BEFORE_GPU_EXECUTION', 'protocol status')
require(p['frozen_input']['bundle'] == str(bundle), 'bundle path differs from frozen protocol')
require(p['frozen_input']['bundle_manifest_sha256'] == expected_manifest, 'protocol bundle manifest anchor')
require(p['selection_witness']['path'] == str(selection), 'selection path differs from protocol')
require(p['selection_witness']['sha256'] == expected_selection, 'selection anchor differs from protocol')
require(s.get('schema') == 'c3-same-leaf-capacity-selection-v1', 'selection schema')
require(s.get('status') == 'CPU_ONLY_SELECTION_PASS_NO_CUDA', 'selection status')
require(s['frozen_input']['bundle'] == str(bundle), 'selection bundle path')
require(s['frozen_input']['bundle_manifest_sha256'] == expected_manifest, 'selection bundle manifest anchor')
require(p['candidate_trace']['target_leaf'] == 221 and p['candidate_trace']['initial_target_occupancy'] == 4, 'target contract')
require(p['candidate_trace']['accepted_count'] == 16 and p['candidate_trace']['boundary_stable_id'] == 5430, 'candidate count/boundary')
require(p['heldout_nonself_knn']['query_ids'] == [0,1,2] and p['heldout_nonself_knn']['k_values'] == [1,10,20], 'heldout contract')
require(p['selection_witness']['full_structure_fingerprint'] == 'fnv1a64:a4a1881cd82237fb', 'geometry fingerprint')
require(m.get('schema') == 'e1gi-b-quantized-gts-integration-manifest-v1', 'bundle manifest schema')
for name, expected in sorted(m.get('files_sha256', {}).items()):
    path = bundle / name
    actual = digest(path)
    require(actual == expected, f'input SHA256 mismatch {name}')
    require(p['frozen_input']['bundle_files_sha256'].get(name) == expected, f'protocol file hash differs {name}')
    print(f'{name}_path={path}')
    print(f'{name}_sha256_expected={expected}')
    print(f'{name}_sha256_actual={actual}')
print('protocol_sha256_expected=' + expected_protocol)
print('protocol_sha256_actual=' + digest(protocol))
print('selection_sha256_expected=' + expected_selection)
print('selection_sha256_actual=' + digest(selection))
print('bundle_manifest_sha256_expected=' + expected_manifest)
print('bundle_manifest_sha256_actual=' + digest(manifest_path))
print('frozen_fields=protocol;selection;bundle_manifest;pool;queries;trace;metadata;target;accepted_sequence;boundary;heldout;fingerprint')
print('input_provenance_verdict=PASS')
PY
  then
    die "frozen protocol/input provenance failed before GPU access; see $output"
  fi
}

gpu0_snapshot_and_require_idle() {
  local label=$1 dir=$2
  local status numeric apps apps_clean uuid mem util extra
  status="$(nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader)" || die "cannot query GPU0 status ($label)"
  printf '%s\n' "$status" > "$dir/gpu0_status_${label}.txt"
  numeric="$(nvidia-smi --id=0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)" || die "cannot query GPU0 numeric state ($label)"
  IFS=, read -r uuid mem util extra <<< "$numeric"
  uuid="$(trim "$uuid")"; mem="$(trim "$mem")"; util="$(trim "$util")"
  [[ -z "${extra:-}" ]] || die "unexpected GPU0 CSV fields ($label): $numeric"
  [[ "$uuid" == "$GPU0_UUID" ]] || die "GPU0 UUID mismatch ($label): expected $GPU0_UUID got $uuid"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || die "non-numeric GPU0 state ($label): $numeric"
  [[ "$mem" -le "$GPU0_MAX_MEMORY_USED_MIB" ]] || die "GPU0 memory=${mem}MiB exceeds ${GPU0_MAX_MEMORY_USED_MIB}MiB ($label)"
  [[ "$util" -eq 0 ]] || die "GPU0 utilization=${util}% not 0 ($label)"
  if ! apps="$(nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)"; then
    printf '%s\n' "$apps" > "$dir/gpu0_compute_apps_${label}.txt"
    die "cannot query GPU0 compute apps ($label): $apps"
  fi
  printf '%s\n' "$apps" > "$dir/gpu0_compute_apps_${label}.txt"
  [[ "$apps" != *"No devices were found"* ]] || die "GPU0 compute-app query reported no device ($label)"
  apps_clean="$(printf '%s\n' "$apps" | awk 'NF && $0 !~ /^No running (compute )?processes found$/ { print }')"
  [[ -z "$apps_clean" ]] || die "GPU0 has compute process(es) or unrecognized compute-app output ($label): $apps_clean"
  {
    echo "physical_gpu_uuid=$uuid"
    echo "memory_used_mib=$mem"
    echo "utilization_percent=$util"
    echo "compute_processes=none"
  } > "$dir/gpu0_idle_check_${label}.env"
}

# Lock release is forbidden until the isolated runner is reaped. A signal sent
# during launch is deferred until $! has been recorded, closing the race.
reap_active_runner_before_unlock() {
  local pid="${ACTIVE_RUNNER_PID:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" 2>/dev/null || true
  done
  wait "$pid" 2>/dev/null || true
  ACTIVE_RUNNER_PID=""
}

cleanup_guard() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  reap_active_runner_before_unlock
  if [[ "$LOCK_HELD" == 1 ]]; then
    rmdir "$LOCK"
    LOCK_HELD=0
  fi
  exit "$rc"
}

on_guard_signal() {
  local signal=$1
  if [[ "$RUNNER_LAUNCHING" == 1 && -z "$ACTIVE_RUNNER_PID" ]]; then
    SIGNAL_PENDING="$signal"
    return 0
  fi
  printf 'guard_signal=%s\nutc=%s\n' "$signal" "$(date -u +%FT%TZ)" >> "${OUT:-${PRE:-$ROOT/logs}}/guard_signal.log" 2>/dev/null || true
  case "$signal" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    *) exit 128 ;;
  esac
}

capture_post_status() {
  local destination=$1
  if ! nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$destination" 2>&1; then
    printf 'post_status_query_failed=1\n' >> "$destination"
  fi
}

run_runner() {
  local phase_dir="$OUT"
  gpu0_snapshot_and_require_idle immediately_pre_run "$phase_dir"
  local -a cmd=("$RUNNER" --bundle "$BUNDLE" --out "$OUT/engine.jsonl" --summary "$OUT/runner_summary.json" --initial-geometry "$OUT/geometry_initial.json" --final-geometry "$OUT/geometry_final.json")
  printf '%q ' "${cmd[@]}" > "$OUT/command.shlex"; printf '\n' >> "$OUT/command.shlex"
  RUNNER_LAUNCHING=1
  setsid timeout --foreground -k 30s 35m "${cmd[@]}" > "$OUT/runner_console.log" 2>&1 &
  ACTIVE_RUNNER_PID=$!
  RUNNER_LAUNCHING=0
  if [[ -n "$SIGNAL_PENDING" ]]; then
    local pending="$SIGNAL_PENDING"
    SIGNAL_PENDING=""
    on_guard_signal "$pending"
  fi
  set +e
  wait "$ACTIVE_RUNNER_PID"
  local rc=$?
  set -e
  ACTIVE_RUNNER_PID=""
  cat "$OUT/runner_console.log"
  printf 'runner_exit_code=%s\n' "$rc" > "$OUT/launcher_result.env"
  capture_post_status "$OUT/gpu0_status_after_runner.txt"
  return "$rc"
}

run_validator() {
  set +e
  "$PYTHON" "$VALIDATOR" --protocol "$PROTOCOL" --bundle "$BUNDLE" \
    --initial-geometry "$OUT/geometry_initial.json" --final-geometry "$OUT/geometry_final.json" \
    --engine-jsonl "$OUT/engine.jsonl" --out "$OUT/cpu_validation.json" > "$OUT/validator_console.log" 2>&1
  local rc=$?
  set -e
  cat "$OUT/validator_console.log"
  printf 'validator_exit_code=%s\n' "$rc" >> "$OUT/launcher_result.env"
  return "$rc"
}

[[ $# -eq 1 ]] || { usage; exit 64; }
case "$1" in
  --verify-static)
    [[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
    mkdir -p "$ROOT/logs"
    static_audit "$ROOT/logs/static_audit_launcher_latest.json"
    verify_frozen_input_provenance "$ROOT/logs/input_provenance_launcher_latest.env"
    echo "CPU-only static/provenance audit PASS; no nvidia-smi or CUDA binary was run."
    exit 0 ;;
  --execute) ;;
  *) usage; exit 64 ;;
esac

[[ "${C3_CAPACITY_GPU0_APPROVED:-}" == "I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE" ]] || die "set C3_CAPACITY_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE after checking shared-GPU policy"
[[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || die "unset CUDA_VISIBLE_DEVICES; guard maps physical GPU0 by UUID itself"
[[ -x "$RUNNER" && -r "$VALIDATOR" && -r "$AUDITOR" && -r "$PROTOCOL" && -r "$SELECTION" && -r "$BUILD_MANIFEST" ]] || die "missing experiment artifact"
[[ -d "$BUNDLE" ]] || die "missing frozen bundle $BUNDLE"
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
command -v timeout >/dev/null || die "GNU timeout unavailable"
command -v setsid >/dev/null || die "setsid unavailable for signal-safe runner isolation"

PRE="$ROOT/logs/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$PRE" || die "refusing to overwrite existing preflight directory: $PRE"
static_audit "$PRE/static_audit.json"
verify_frozen_input_provenance "$PRE/input_provenance.env"
GUARD_SHA256="$(sha256sum "$GUARD_SCRIPT" | awk '{print $1}')"
[[ "$GUARD_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "cannot hash guard script"
# All provenance checks above intentionally precede this first GPU command.
GPU0_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ "$GPU0_UUID" == GPU-* ]] || die "cannot resolve physical GPU0 UUID"
gpu0_snapshot_and_require_idle before_lock "$PRE"
mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then die "C3 same-leaf physical-GPU0 lock exists: $LOCK"; fi
LOCK_HELD=1
trap cleanup_guard EXIT
trap 'on_guard_signal INT' INT
trap 'on_guard_signal TERM' TERM

RUN_ID="c3_same_leaf_capacity_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/$RUN_ID"
mkdir "$OUT" || die "refusing to overwrite existing run directory: $OUT"
cp "$PRE/static_audit.json" "$OUT/static_audit.json"
cp "$PRE/input_provenance.env" "$OUT/input_provenance.env"
cp "$PRE"/gpu0_*_before_lock* "$OUT/"
{
  echo "run_id=$RUN_ID"
  echo "utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "guard_script_path=$GUARD_SCRIPT"
  echo "guard_script_sha256=$GUARD_SHA256"
  cat "$PRE/input_provenance.env"
  echo "physical_gpu_index=0"
  echo "physical_gpu_uuid=$GPU0_UUID"
  echo "CUDA_VISIBLE_DEVICES=$GPU0_UUID"
  echo "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  echo "runner_source_sha256=$(sha256sum "$ROOT/src/gts_c3_same_leaf_capacity_heldout.cu" | awk '{print $1}')"
  echo "runner_binary_sha256=$(sha256sum "$RUNNER" | awk '{print $1}')"
  echo "validator_sha256=$(sha256sum "$VALIDATOR" | awk '{print $1}')"
  echo "build_manifest_sha256=$(sha256sum "$BUILD_MANIFEST" | awk '{print $1}')"
  echo "gpu0_idle_policy=compute_processes_none;utilization_percent=0;memory_used_mib<=${GPU0_MAX_MEMORY_USED_MIB};checked_before_lock_and_immediately_pre_run"
  echo "scope=one frozen same-leaf capacity boundary plus external non-self KNN; no buffer merge/performance/complete-C3 claim"
} > "$OUT/run_manifest.env"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
if run_runner; then
  RUNNER_RC=0
  if run_validator; then VALIDATOR_RC=0; else VALIDATOR_RC=$?; fi
else
  RUNNER_RC=$?
  VALIDATOR_RC=not_run_after_runner_failure
  printf 'validator_exit_code=%s\n' "$VALIDATOR_RC" >> "$OUT/launcher_result.env"
fi
capture_post_status "$OUT/gpu0_status_after_all.txt"
if [[ "$RUNNER_RC" == 0 && "$VALIDATOR_RC" == 0 ]]; then
  echo "C3 same-leaf capacity guarded probe PASS within recorded narrow scope: $OUT"
  exit 0
fi
echo "C3 same-leaf capacity guarded probe FAILED (runner=$RUNNER_RC validator=$VALIDATOR_RC): $OUT" >&2
exit 2

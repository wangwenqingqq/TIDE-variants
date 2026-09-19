#!/usr/bin/env bash
# Guarded launcher for the pre-registered full-SIFT1M C2 held-out experiment.
# Default and --verify-static are CPU-only. --execute is intentionally fail-closed.
set -Eeuo pipefail

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_gamma_heldout_scale
HOST_REQUIRED=CONFIGURE_ARCHIVE_HOST
PYTHON=/workspace/legacy_workspace/GTS/bench_env/bin/python
BINARY="$ROOT/bin/GTS_c2_perlevel_sift1m"
SRC="$ROOT/src/gts_c2_perlevel_sift1m.cu"
HEADER="$ROOT/include/residual_pruning.cuh"
PROTOCOL="$ROOT/protocols/submitted_vldb_c2_perlevel_sift1m_v2_paired_baseline.json"
SOURCE_MANIFEST="$ROOT/source_manifest.sha256"
PREFLIGHT="$ROOT/preflight/sift1m_full_raw_v1_20260727T164200CST/preflight.json"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERY=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs
GT=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs
CAL_IDS="$ROOT/preflight/sift1m_full_raw_v1_20260727T164200CST/calibration.ids"
VAL_IDS="$ROOT/preflight/sift1m_full_raw_v1_20260727T164200CST/validation.ids"
TEST_IDS="$ROOT/preflight/sift1m_full_raw_v1_20260727T164200CST/test.ids"
LOCK="$ROOT/.locks/c2_perlevel_physical_gpu0.lock"
GUARD_SCRIPT="$ROOT/tools/run_submitted_vldb_c2_guarded.sh"
GPU0_MAX_MEMORY_USED_MIB=256
# Frozen v2 provenance anchors.  These are checked before *any* GPU query.
PROTOCOL_V2_SHA256=ad93641fa5cd5428b8726aa0b3e2841adf0aec59c6703fac48fa34cba7c798cf
BASE_V2_SHA256=21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816
QUERY_V2_SHA256=f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc
GT_V2_SHA256=2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f
LOCK_HELD=0
ACTIVE_RUNNER_PID=""
PHASE_LAUNCHING=0
SIGNAL_PENDING=""

usage() {
  cat <<'USAGE'
Usage:
  run_submitted_vldb_c2_guarded.sh --verify-static
  C2_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE \
    run_submitted_vldb_c2_guarded.sh --execute

--execute does exactly the frozen sequence:
  calibration (2,000 IDs) -> only if PASS validation (2,000) -> only if PASS sealed test (6,000).
It refuses any non-8p host, GPU0 compute process, GPU0 utilization !=0,
GPU0 memory >256MiB, stale source/binary manifest, pre-existing lock, or absent acknowledgement.
Physical nvidia-smi GPU0 is selected by UUID, then exposed as CUDA device 0.
Each GPU phase has a 60-minute foreground timeout and a fresh idle check immediately before launch.
USAGE
}
die() { echo "C2 GPU0 guard: $*" >&2; exit 64; }
trim() { local x=$1; x="${x#"${x%%[![:space:]]*}"}"; x="${x%"${x##*[![:space:]]}"}"; printf '%s' "$x"; }

static_check() {
  local out=$1
  mkdir -p "$(dirname "$out")"
  {
    echo "static_check_utc=$(date -u -Is)"
    echo "host=$(hostname)"
    echo "guard_script=$GUARD_SCRIPT"
    echo "guard_script_sha256=$(sha256sum "$GUARD_SCRIPT" | awk '{print $1}')"
    sha256sum -c "$SOURCE_MANIFEST"
    "$PYTHON" - "$PROTOCOL" "$PREFLIGHT" "$SRC" "$HEADER" "$BINARY" <<'PY'
import hashlib, json, pathlib, sys
protocol, preflight, src, header, binary = map(pathlib.Path, sys.argv[1:])
p = json.loads(protocol.read_text())
q = json.loads(preflight.read_text())
assert p.get('schema') == 'submitted-vldb-c2-perlevel-heldout-protocol-v2-paired-baseline'
assert p.get('split', {}).get('disjoint_and_exhaustive') is True
assert p.get('heldout_rules', {}).get('test', '').startswith('Only if validation stage passes')
assert q.get('cuda_used') is False
s = src.read_text(); h = header.read_text()
assert 'upload_gamma' in s and 'c_rp_gamma' in s and 'stage_gate' in s
assert '(scale - 1.0f) * dis_lb' in h
assert binary.is_file() and binary.stat().st_size > 0
print('protocol/preflight/C2-Eq2 source contract: PASS')
PY
  } >"$out" 2>&1
}

# This is deliberately CPU-only, and must finish before the first nvidia-smi
# invocation in --execute.  It validates both the immutable v2 protocol fields
# and the bytes actually supplied to the CUDA runner.
verify_frozen_input_provenance() {
  local out=$1
  mkdir -p "$(dirname "$out")"
  if ! "$PYTHON" - "$PROTOCOL" "$BASE" "$QUERY" "$GT" \
      "$PROTOCOL_V2_SHA256" "$BASE_V2_SHA256" "$QUERY_V2_SHA256" "$GT_V2_SHA256" >"$out" 2>&1 <<'PY'
import hashlib, json, pathlib, sys

protocol, base, query, gt = map(pathlib.Path, sys.argv[1:5])
expected_protocol, expected_base, expected_query, expected_gt = sys.argv[5:9]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)

actual_protocol = digest(protocol)
require(actual_protocol == expected_protocol,
        f'protocol SHA256 expected={expected_protocol} actual={actual_protocol}')
p = json.loads(protocol.read_text())
require(p.get('schema') == 'submitted-vldb-c2-perlevel-heldout-protocol-v2-paired-baseline',
        'protocol schema is not frozen v2')
require(p['data']['base']['path'] == str(base), 'base path differs from frozen protocol')
require(p['data']['query']['path'] == str(query), 'query path differs from frozen protocol')
require(p['data']['groundtruth']['path'] == str(gt), 'ground-truth path differs from frozen protocol')
require(p['data']['base']['sha256'] == expected_base, 'base SHA anchor differs in protocol')
require(p['data']['query']['sha256'] == expected_query, 'query SHA anchor differs in protocol')
require(p['data']['groundtruth']['sha256'] == expected_gt, 'ground-truth SHA anchor differs in protocol')
require(p['c2_algorithm1_fidelity']['binary_iterations'] == 16,
        'binary-iteration frozen value differs')
require(p['c2_algorithm1_fidelity']['candidate_interval'] == [1.0, 5.0],
        'candidate-interval frozen value differs')
require(p['c2_algorithm1_fidelity']['freeze_levels'] == 'levels 0,1,2 fixed at gamma=1.0 (L_frz=2)',
        'frozen-level policy differs')
require(p['split']['disjoint_and_exhaustive'] is True, 'split invariant differs')
require(p['resource_policy']['physical_gpu'] == 0, 'physical-GPU policy differs')
require(p['heldout_rules']['test'].startswith('Only if validation stage passes'),
        'sealed-test gate differs')

items = [('base', base, expected_base), ('query', query, expected_query), ('groundtruth', gt, expected_gt)]
print('protocol_path=' + str(protocol))
print('protocol_sha256_expected=' + expected_protocol)
print('protocol_sha256_actual=' + actual_protocol)
for label, path, expected in items:
    actual = digest(path)
    require(actual == expected, f'{label} SHA256 expected={expected} actual={actual}')
    print(f'{label}_path={path}')
    print(f'{label}_sha256_expected={expected}')
    print(f'{label}_sha256_actual={actual}')
print('protocol_frozen_fields=schema;algorithm1_iterations;candidate_interval;freeze_levels;split;physical_gpu;sealed_test_gate')
print('input_provenance_verdict=PASS')
PY
  then
    die "frozen v2 input provenance failed before GPU access; see $out"
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

summary_gate() {
  local summary=$1 expected=$2
  "$PYTHON" - "$summary" "$expected" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
d=json.loads(p.read_text())
assert d.get('status') == expected, d.get('status')
assert d.get('stage_gate_pass') is True, d.get('stage_gate_pass')
assert d.get('archived_incremental_updater_used') is False
assert d.get('dynamic_exact_smoke_used') is False
if expected == 'PASS_C2_CALIBRATION':
    assert pathlib.Path(p.parent/'final_gamma_vector.txt').is_file()
print('summary gate PASS:', expected)
PY
}

# The lock may be removed only after the isolated timeout process has been
# reaped.  On INT/TERM, cleanup first terminates that process group and waits;
# if it cannot exit, this shell remains blocked and the lock remains held.
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
  # A signal can arrive after setsid has started but before Bash assigns $!.
  # Defer only that tiny launch window; run_phase exits immediately after it
  # records the PID, so cleanup can reap the process before unlocking.
  if [[ "$PHASE_LAUNCHING" == 1 && -z "$ACTIVE_RUNNER_PID" ]]; then
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

run_phase() {
  local phase=$1 mode=$2 ids=$3 timed=$4 gamma=${5:-} pending_signal
  local phase_dir="$OUT/$phase"
  mkdir "$phase_dir" || die "refusing to overwrite existing phase directory: $phase_dir"
  # This is the final safety check for this exact phase; it runs after the C2 lock
  # has been acquired and immediately before the CUDA timeout command.
  gpu0_snapshot_and_require_idle "immediately_pre_${phase}" "$phase_dir"
  local -a cmd=("$BINARY" --mode "$mode" --base-fvecs "$BASE" --query-fvecs "$QUERY" --groundtruth-ivecs "$GT" --query-ids "$ids" --out "$phase_dir" --stage "$phase" --warmup-reps 1 --timed-reps "$timed")
  if [[ "$mode" == evaluate ]]; then cmd+=(--gamma-vector-file "$gamma"); fi
  printf '%q ' "${cmd[@]}" > "$phase_dir/command.shlex"; printf '\n' >> "$phase_dir/command.shlex"
  # setsid gives this phase its own process group, so a signal handler can
  # terminate/reap only this launcher and never release the shared-GPU lock early.
  # PHASE_LAUNCHING closes the shell signal window between background spawn and $!.
  PHASE_LAUNCHING=1
  setsid timeout --foreground -k 30s 60m "${cmd[@]}" > "$phase_dir/runner_console.log" 2>&1 &
  ACTIVE_RUNNER_PID=$!
  PHASE_LAUNCHING=0
  if [[ -n "$SIGNAL_PENDING" ]]; then
    pending_signal="$SIGNAL_PENDING"
    SIGNAL_PENDING=""
    on_guard_signal "$pending_signal"
  fi
  set +e
  wait "$ACTIVE_RUNNER_PID"
  local rc=$?
  set -e
  ACTIVE_RUNNER_PID=""
  cat "$phase_dir/runner_console.log"
  printf 'runner_exit_code=%s\n' "$rc" > "$phase_dir/launcher_result.env"
  nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$phase_dir/gpu0_status_after.txt" || true
  return "$rc"
}

[[ $# -eq 1 ]] || { usage; exit 64; }
case "$1" in
  --verify-static)
    [[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
    static_check "$ROOT/logs/static_audit_launcher_latest.log"
    verify_frozen_input_provenance "$ROOT/logs/input_provenance_launcher_latest.env"
    echo "CPU-only static/provenance audit PASS; no GPU query or CUDA binary execution."
    exit 0;;
  --execute) ;;
  *) usage; exit 64;;
esac

[[ "${C2_GPU0_APPROVED:-}" == "I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE" ]] || die "set C2_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE after checking shared GPU policy"
[[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || die "unset CUDA_VISIBLE_DEVICES; this guard maps physical GPU0 by UUID"
[[ -x "$BINARY" && -r "$SRC" && -r "$HEADER" && -r "$PROTOCOL" && -r "$PREFLIGHT" ]] || die "missing C2 artifact"
for p in "$BASE" "$QUERY" "$GT" "$CAL_IDS" "$VAL_IDS" "$TEST_IDS"; do [[ -r "$p" ]] || die "missing input $p"; done
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
command -v timeout >/dev/null || die "GNU timeout unavailable"
command -v setsid >/dev/null || die "setsid unavailable for fail-closed signal handling"

PRE="$ROOT/logs/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$PRE" || die "refusing to overwrite existing preflight directory: $PRE"
static_check "$PRE/static_audit.log"
verify_frozen_input_provenance "$PRE/input_provenance.env"
GUARD_SHA256="$(sha256sum "$GUARD_SCRIPT" | awk '{print $1}')"
[[ "$GUARD_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "cannot hash guard script"
# Input/protocol verification above is intentionally before this first GPU query.
GPU0_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ "$GPU0_UUID" == GPU-* ]] || die "cannot resolve physical GPU0 UUID"
gpu0_snapshot_and_require_idle before_lock "$PRE"
mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then die "C2 physical-GPU0 lock exists: $LOCK"; fi
LOCK_HELD=1
trap cleanup_guard EXIT
trap 'on_guard_signal INT' INT
trap 'on_guard_signal TERM' TERM

RUN_ID="c2_sift1m_heldout_v2_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/$RUN_ID"
mkdir "$OUT" || die "refusing to overwrite existing run directory: $OUT"
cp "$PRE/static_audit.log" "$OUT/static_audit.log"
cp "$PRE/input_provenance.env" "$OUT/input_provenance.env"
cp "$PRE"/gpu0_*_before_lock* "$OUT/" 2>/dev/null || true
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
  echo "protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')"
  echo "source_manifest_sha256=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"
  echo "runner_binary_sha256=$(sha256sum "$BINARY" | awk '{print $1}')"
  echo "runner_source_sha256=$(sha256sum "$SRC" | awk '{print $1}')"
  echo "preflight_sha256=$(sha256sum "$PREFLIGHT" | awk '{print $1}')"
  echo "gpu0_idle_policy=compute_processes_none;utilization_percent=0;memory_used_mib<=${GPU0_MAX_MEMORY_USED_MIB};checked_before_lock_and_before_each_phase"
  echo "sequence=calibration_then_validation_only_if_pass_then_sealed_test_only_if_validation_pass"
  echo "scope=C2 empirical heldout calibration only; no C1/C3 or universal safety claim"
} > "$OUT/run_manifest.env"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"

CAL_RC=99; VAL_RC=not_run; TEST_RC=not_run
if run_phase calibration calibrate "$CAL_IDS" 1; then
  CAL_RC=0
  if summary_gate "$OUT/calibration/summary.json" PASS_C2_CALIBRATION; then
    if run_phase validation evaluate "$VAL_IDS" 3 "$OUT/calibration/final_gamma_vector.txt"; then
      VAL_RC=0
      if summary_gate "$OUT/validation/summary.json" PASS_C2_HELDOUT_STAGE; then
        if run_phase test evaluate "$TEST_IDS" 5 "$OUT/calibration/final_gamma_vector.txt"; then
          TEST_RC=0
          if ! summary_gate "$OUT/test/summary.json" PASS_C2_HELDOUT_STAGE; then TEST_RC=4; fi
        else TEST_RC=$?; fi
      else VAL_RC=4; fi
    else VAL_RC=$?; fi
  else CAL_RC=4; fi
else CAL_RC=$?; fi
{
  echo "calibration_exit_code=$CAL_RC"
  echo "validation_exit_code=$VAL_RC"
  echo "test_exit_code=$TEST_RC"
} > "$OUT/launcher_result.env"
nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu0_status_after_all.txt" || true
if [[ "$CAL_RC" == 0 && "$VAL_RC" == 0 && "$TEST_RC" == 0 ]]; then
  echo "C2 full held-out sequence PASS: $OUT"
  exit 0
fi
if [[ "$CAL_RC" != 0 ]]; then
  echo "C2 stopped after calibration (no held-out evaluation): $OUT" >&2
elif [[ "$VAL_RC" != 0 ]]; then
  echo "C2 validation did not pass; sealed test was not run: $OUT" >&2
else
  echo "C2 sealed test did not pass: $OUT" >&2
fi
exit 3

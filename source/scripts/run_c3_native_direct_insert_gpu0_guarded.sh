#!/usr/bin/env bash
# Manual, guarded launcher for the bounded C3 native direct-insert probe.
# It is intentionally inert unless both --execute and the acknowledgement are
# supplied.  Do not invoke it from unattended automation.
set -euo pipefail

ROOT=/workspace/experiments/tide_safe_c1_20260727/c3_native_direct_insert_probe
SRCROOT=/workspace/experiments/tide_safe_c1_20260727/source_gts_incremental
BUNDLE=/workspace/experiments/tide_safe_c1_20260727/runs/e1gi_b_quantized_gts_bundles_v1_20260727/sift128_slice__sift128_slice__seed_20260727
HOST_REQUIRED=CONFIGURE_ARCHIVE_HOST
PYTHON=/workspace/legacy_workspace/GTS/bench_env/bin/python
RUNNER="$ROOT/bin/GTS_c3_native_direct_insert_probe"
VALIDATOR="$ROOT/validate_c3_native_direct_insert.py"
AUDITOR="$ROOT/audit_c3_native_probe_static.py"
PROTOCOL="$ROOT/protocol_sift128_native_certified_direct_insert_v2_search_upper.json"
BUILD_MANIFEST="$ROOT/build_manifest_native_probe_v2_mode0.json"
LOCK="$ROOT/.locks/c3_native_direct_insert_physical_gpu0.lock"
# Fail closed: an ostensibly idle GPU with substantial allocated memory may
# still be serving an unlisted/MPS/graphics context.  This probe is tiny, so
# it is safe to require a conservative pre-run ceiling.
GPU0_MAX_MEMORY_USED_MIB=256

usage() {
  cat <<'USAGE'
Usage:
  run_c3_native_direct_insert_gpu0_guarded.sh --verify-static
  C3_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE \
    run_c3_native_direct_insert_gpu0_guarded.sh --execute

Safety contract:
  * default/no argument does not run CUDA or call nvidia-smi;
  * --verify-static is CPU-only;
  * --execute refuses a non-8p host, any listed GPU0 compute process,
    GPU0 utilization other than 0%, GPU0 memory use above 256 MiB, a
    pre-existing C3 GPU0 lock, stale source/binary static audit, or a missing
    explicit acknowledgement;
  * it checks those GPU0 conditions twice: before taking the C3 lock and
    again immediately before the CUDA timeout command;
  * physical GPU 0 is selected by its nvidia-smi UUID, not an ambiguous CUDA
    ordinal; the runner then sees it as CUDA device 0;
  * a 20-minute hard wall-time cap is applied.  The first run is a correctness
    probe only; it is not a performance measurement.
USAGE
}

die() { echo "C3 GPU0 guard: $*" >&2; exit 64; }

static_audit() {
  local output=$1
  "$PYTHON" "$AUDITOR" \
    --protocol "$PROTOCOL" \
    --runner "$ROOT/src/gts_c3_native_direct_insert_probe.cu" \
    --validator "$VALIDATOR" \
    --tree-header "$SRCROOT/include/tree.cuh" \
    --search-header "$SRCROOT/include/search_v2.cuh" \
    --binary "$RUNNER" \
    --build-manifest "$BUILD_MANIFEST" \
    --launcher "$0" \
    --out "$output"
}

trim_csv_field() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

# Query both human-readable status and no-units machine fields.  This function
# is deliberately fail-closed: malformed/unsupported output is not idle.
gpu0_snapshot_and_require_idle() {
  local label=$1
  local dir=$2
  local status numeric apps apps_clean uuid memory_used utilization extra
  status="$(nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader)" || die "cannot query GPU0 status ($label)"
  printf '%s\n' "$status" > "$dir/gpu0_status_${label}.txt"
  numeric="$(nvidia-smi --id=0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)" || die "cannot query GPU0 numeric status ($label)"
  IFS=, read -r uuid memory_used utilization extra <<< "$numeric"
  uuid="$(trim_csv_field "$uuid")"
  memory_used="$(trim_csv_field "$memory_used")"
  utilization="$(trim_csv_field "$utilization")"
  [[ -z "${extra:-}" ]] || die "unexpected GPU0 numeric CSV fields ($label): $numeric"
  [[ "$uuid" == "$GPU0_UUID" ]] || die "GPU0 UUID changed or cannot be parsed ($label): expected $GPU0_UUID got $uuid"
  [[ "$memory_used" =~ ^[0-9]+$ ]] || die "GPU0 memory.used is nonnumeric ($label): $memory_used"
  [[ "$utilization" =~ ^[0-9]+$ ]] || die "GPU0 utilization is nonnumeric ($label): $utilization"
  [[ "$memory_used" -le "$GPU0_MAX_MEMORY_USED_MIB" ]] || die "GPU0 memory.used=${memory_used}MiB exceeds ${GPU0_MAX_MEMORY_USED_MIB}MiB ($label)"
  [[ "$utilization" -eq 0 ]] || die "GPU0 utilization=${utilization}% is not 0% ($label)"
  apps="$(nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true)"
  printf '%s\n' "$apps" > "$dir/gpu0_compute_apps_${label}.txt"
  apps_clean="$(printf '%s\n' "$apps" | sed -E '/^[[:space:]]*$/d; /No running (compute )?processes found/d; /No devices were found/d')"
  [[ -z "$apps_clean" ]] || die "physical GPU0 has compute process(es) ($label): $apps_clean"
  {
    echo "physical_gpu_uuid=$uuid"
    echo "memory_used_mib=$memory_used"
    echo "utilization_percent=$utilization"
    echo "compute_processes=none"
  } > "$dir/gpu0_idle_check_${label}.env"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 64
fi
case "$1" in
  --verify-static)
    [[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
    mkdir -p "$ROOT/logs"
    static_audit "$ROOT/logs/static_audit_launcher_latest.json"
    echo "CPU-only static audit completed; no CUDA binary and no GPU command was run."
    exit 0
    ;;
  --execute) ;;
  *) usage; exit 64 ;;
esac

# The acknowledgement is intentionally verbose so this cannot be an accidental
# shell completion or a background call.
[[ "${C3_GPU0_APPROVED:-}" == "I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE" ]] || \
  die "set C3_GPU0_APPROVED=I_CONFIRM_PHYSICAL_GPU0_IS_EXCLUSIVE after checking shared-GPU policy"
[[ "$(hostname)" == "$HOST_REQUIRED" ]] || die "must run on 8p host $HOST_REQUIRED"
[[ -x "$RUNNER" && -r "$VALIDATOR" && -r "$AUDITOR" && -r "$PROTOCOL" && -r "$BUILD_MANIFEST" ]] || die "missing runner/validator/auditor/protocol/build manifest"
[[ -d "$BUNDLE" ]] || die "missing frozen bundle $BUNDLE"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || die "unset CUDA_VISIBLE_DEVICES; guard will map physical GPU0 by UUID itself"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v timeout >/dev/null || die "GNU timeout is required for the 20-minute cap"

# Start with a static contract check before consulting the GPU.  This is CPU only.
PRECHECK_DIR="$ROOT/logs/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$PRECHECK_DIR"
static_audit "$PRECHECK_DIR/static_audit.json"

# Map the physical nvidia-smi index 0 by UUID.  CUDA_VISIBLE_DEVICES=<UUID>
# means the GTS binary's CUDA device 0 is precisely physical GPU0.
GPU0_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ "$GPU0_UUID" == GPU-* ]] || die "could not resolve physical GPU0 UUID"
# First fail-closed check, before taking the local serialization lock.
gpu0_snapshot_and_require_idle "before_lock" "$PRECHECK_DIR"

if ! mkdir "$LOCK" 2>/dev/null; then
  die "C3 physical-GPU0 lock exists: $LOCK (inspect/remove only after confirming no active probe)"
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

RUN_ID="c3_native_sift128_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/$RUN_ID"
mkdir -p "$OUT"
cp "$PRECHECK_DIR/static_audit.json" "$OUT/static_audit.json"
cp "$PRECHECK_DIR/gpu0_status_before_lock.txt" "$OUT/gpu0_status_before_lock.txt"
cp "$PRECHECK_DIR/gpu0_compute_apps_before_lock.txt" "$OUT/gpu0_compute_apps_before_lock.txt"
cp "$PRECHECK_DIR/gpu0_idle_check_before_lock.env" "$OUT/gpu0_idle_check_before_lock.env"
(
  echo "run_id=$RUN_ID"
  echo "utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "physical_gpu_index=0"
  echo "physical_gpu_uuid=$GPU0_UUID"
  echo "CUDA_VISIBLE_DEVICES=$GPU0_UUID"
  echo "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  echo "protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')"
  echo "build_manifest_sha256=$(sha256sum "$BUILD_MANIFEST" | awk '{print $1}')"
  echo "runner_source_sha256=$(sha256sum "$ROOT/src/gts_c3_native_direct_insert_probe.cu" | awk '{print $1}')"
  echo "runner_binary_sha256=$(sha256sum "$RUNNER" | awk '{print $1}')"
  echo "validator_sha256=$(sha256sum "$VALIDATOR" | awk '{print $1}')"
  echo "bundle_manifest_sha256=$(sha256sum "$BUNDLE/manifest.json" | awk '{print $1}')"
  echo "launcher_sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "gpu0_idle_policy=compute_processes_none;utilization_percent=0;memory_used_mib<=${GPU0_MAX_MEMORY_USED_MIB};checked_before_lock_and_immediately_pre_run"
  echo "scope=bounded insert-only native direct tier; no performance or complete-C3 claim"
) > "$OUT/run_manifest.env"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU0_UUID"
# Second fail-closed check after acquiring the lock and immediately before
# launching the CUDA binary.  A newly appearing job prevents this run.
gpu0_snapshot_and_require_idle "immediately_pre_run" "$OUT"
set +e
timeout --foreground 20m "$RUNNER" \
  --bundle "$BUNDLE" \
  --out "$OUT/engine.jsonl" \
  --summary "$OUT/runner_summary.json" \
  --initial-geometry "$OUT/geometry_initial.json" \
  --final-geometry "$OUT/geometry_final.json" \
  --max-accepted 16 2>&1 | tee "$OUT/runner_console.log"
RUNNER_RC=${PIPESTATUS[0]}
set -e
printf 'runner_exit_code=%s\n' "$RUNNER_RC" > "$OUT/launcher_result.env"

# The validator is intentionally invoked for both a successful runner and the
# pre-registered rc=3 "no certified insertion" outcome.  In the latter case
# its FAIL/inconclusive result is evidence against a C3 claim, not a reason to
# retry/reorder candidates.
if [[ "$RUNNER_RC" == 0 || "$RUNNER_RC" == 3 ]]; then
  set +e
  "$PYTHON" "$VALIDATOR" \
    --protocol "$PROTOCOL" \
    --bundle "$BUNDLE" \
    --initial-geometry "$OUT/geometry_initial.json" \
    --final-geometry "$OUT/geometry_final.json" \
    --engine-jsonl "$OUT/engine.jsonl" \
    --out "$OUT/cpu_validation.json" 2>&1 | tee "$OUT/validator_console.log"
  VALIDATOR_RC=${PIPESTATUS[0]}
  set -e
  printf 'validator_exit_code=%s\n' "$VALIDATOR_RC" >> "$OUT/launcher_result.env"
else
  printf 'validator_exit_code=not_run_after_runner_failure\n' >> "$OUT/launcher_result.env"
fi
nvidia-smi --id=0 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu0_status_after.txt" || true

if [[ "$RUNNER_RC" == 0 && "${VALIDATOR_RC:-99}" == 0 ]]; then
  echo "C3 guarded probe PASS subject to the recorded bounded scope: $OUT"
  exit 0
fi
if [[ "$RUNNER_RC" == 3 ]]; then
  echo "C3 guarded probe INCONCLUSIVE: no candidate passed the frozen direct-tier certificate: $OUT" >&2
  exit 3
fi
echo "C3 guarded probe FAILED (runner=$RUNNER_RC validator=${VALIDATOR_RC:-not_run}): $OUT" >&2
exit 2

#!/usr/bin/env bash
# Source-only v4 guard cleanup/failure-integrity contract.
#
# This is deliberately NOT an execution guard: --execute always refuses and
# this file contains no GPU-management or CUDA-binary invocation. It is a
# directly testable replacement pattern for the v3 trap hazard where a cleanup
# error/nounset could prevent an already-claimed stage from being marked failed.
set -Eeuo pipefail

EXECUTE_MODE=0
SUCCESS=0
OUT=""
CURRENT_CLAIMED_STAGE=""
CHILD_SESSION_PID=""
CHILD_PGID=""
CHILD_SID=""
CHILD_WRAPPER_PID=""
CHILD_PHASE=""
FAILURE_RECORD_ATTEMPTED=0
FAILURE_RECORD_RC=0
CLEANUP_RC=0
CLEANUP_ACTION="NOT_ATTEMPTED"
OUTER_EXIT_RECORDED=0
SIMULATE_CLEANUP_FAILURE=0
SIMULATE_FAILURE_RECORD_FAILURE=0

usage() {
  cat <<'USAGE'
usage:
  c2_v4_guard_failure_integrity_source_only.sh --verify-source-only
  c2_v4_guard_failure_integrity_source_only.sh --self-test-failure-integrity
This source-only artifact never permits --execute.
USAGE
}

atomic_json_record() {
  local target=${1:-} body=${2:-}
  [[ -n "$target" ]] || return 64
  local parent
  parent="$(dirname -- "$target")"
  [[ -d "$parent" && ! -L "$parent" ]] || return 65
  local tmp
  tmp="$(mktemp "$parent/.tmp.$(basename -- "$target").XXXXXX")" || return 66
  # No unbound variables or failing cleanup command can preempt the write.
  if ! printf '%s\n' "$body" >"$tmp"; then rm -f -- "$tmp" || true; return 67; fi
  if ! ln -- "$tmp" "$target" 2>/dev/null; then rm -f -- "$tmp" || true; return 68; fi
  rm -f -- "$tmp" || true
  return 0
}

record_current_stage_failed() {
  # This function must be called BEFORE cleanup. All globals use :- defaults
  # so a partially initialized child state cannot trigger nounset.
  local reason=${1:-outer_guard_unknown_failure}
  local stage=${CURRENT_CLAIMED_STAGE:-UNCLAIMED}
  local root=${OUT:-}
  FAILURE_RECORD_ATTEMPTED=1
  [[ -n "$root" && -d "$root" && ! -L "$root" ]] || return 70
  [[ "${SIMULATE_FAILURE_RECORD_FAILURE:-0}" != 1 ]] || return 71
  local target="$root/failure_intent_${stage}.json"
  local body
  body="{\"schema\":\"safe-c2-v4-failure-intent-v1\",\"stage\":\"${stage}\",\"reason\":\"${reason}\",\"state\":\"FAILED_RECORDED_BEFORE_CLEANUP\"}"
  atomic_json_record "$target" "$body"
}

record_emergency_failure_if_needed() {
  local reason=${1:-outer_guard_unknown_failure}
  local root=${OUT:-}
  [[ -n "$root" && -d "$root" && ! -L "$root" ]] || return 0
  local target="$root/EMERGENCY_FAILURE_RECORD.json"
  local body
  body="{\"schema\":\"safe-c2-v4-emergency-failure-record-v1\",\"reason\":\"${reason}\",\"failure_record_rc\":${FAILURE_RECORD_RC:-999},\"cleanup_rc\":${CLEANUP_RC:-999}}"
  atomic_json_record "$target" "$body" || true
}

cleanup_own_child_without_blocking_failure_record() {
  # The executable integration must replace this source-only simulation with
  # verified-PID/PGID cleanup only. This contract's important property is that
  # missing child variables and cleanup errors never escape under set -u/-e.
  local reason=${1:-outer_guard_unknown_failure}
  local pid=${CHILD_SESSION_PID:-}
  local pgid=${CHILD_PGID:-}
  local phase=${CHILD_PHASE:-}
  CLEANUP_ACTION="NO_VERIFIED_CHILD reason=${reason} pid=${pid:-none} pgid=${pgid:-none} phase=${phase:-none}"
  if [[ "${SIMULATE_CLEANUP_FAILURE:-0}" == 1 ]]; then
    CLEANUP_ACTION="SIMULATED_CLEANUP_FAILURE $CLEANUP_ACTION"
    return 99
  fi
  return 0
}

record_cleanup_outcome() {
  local root=${OUT:-}
  [[ -n "$root" && -d "$root" && ! -L "$root" ]] || return 0
  local stage=${CURRENT_CLAIMED_STAGE:-UNCLAIMED}
  local body
  body="{\"schema\":\"safe-c2-v4-cleanup-outcome-v1\",\"stage\":\"${stage}\",\"cleanup_rc\":${CLEANUP_RC:-999},\"action\":\"${CLEANUP_ACTION:-missing}\"}"
  atomic_json_record "$root/cleanup_outcome_${stage}.json" "$body"
}

finalize_failure_integrity() {
  # set +e is intentionally local to the failure-finalization transaction:
  # failure recording is attempted first and cleanup cannot abort it.
  local reason=${1:-outer_guard_unknown_failure}
  local record_rc=0 cleanup_rc=0
  set +e
  record_current_stage_failed "$reason"
  record_rc=$?
  FAILURE_RECORD_RC=$record_rc
  cleanup_own_child_without_blocking_failure_record "$reason"
  cleanup_rc=$?
  CLEANUP_RC=$cleanup_rc
  record_cleanup_outcome
  local outcome_rc=$?
  if [[ "$record_rc" -ne 0 ]]; then
    record_emergency_failure_if_needed "$reason"
  fi
  set -e
  # Preserve diagnostic state. The original outer guard exit status remains
  # authoritative; callers must not treat cleanup success as stage success.
  [[ "$outcome_rc" -eq 0 ]] || true
  return 0
}

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ "${EXECUTE_MODE:-0}" == 1 && "${SUCCESS:-0}" != 1 ]]; then
    finalize_failure_integrity "outer_guard_exit_${rc}" || true
  fi
  set -e
  exit "$rc"
}
trap on_exit EXIT

verify_source_only() {
  [[ "$EXECUTE_MODE" == 0 && "$SUCCESS" == 0 ]] || return 80
  [[ -n "${BASH_VERSION:-}" ]] || return 81
  # Source-only verifier must not create any run/ledger or touch a GPU.
  printf '%s\n' '{"status":"PASS_GUARD_FAILURE_INTEGRITY_SOURCE_ONLY","gpu_binary_executed":false,"nvidia_smi_called":false}'
}

self_test_failure_integrity() {
  local tmp
  tmp="$(mktemp -d /tmp/c2_v4_guard_failure_integrity.XXXXXX)"
  # Deliberately unset child variables under nounset; cleanup must return a
  # controlled code, while the stage failure record is still durable.
  EXECUTE_MODE=1
  SUCCESS=0
  OUT="$tmp"
  CURRENT_CLAIMED_STAGE="calibration"
  SIMULATE_CLEANUP_FAILURE=1
  SIMULATE_FAILURE_RECORD_FAILURE=0
  unset CHILD_SESSION_PID CHILD_PGID CHILD_SID CHILD_WRAPPER_PID CHILD_PHASE
  finalize_failure_integrity "self_test_forced_cleanup_failure"
  [[ -f "$tmp/failure_intent_calibration.json" ]] || { rm -rf -- "$tmp"; return 82; }
  [[ -f "$tmp/cleanup_outcome_calibration.json" ]] || { rm -rf -- "$tmp"; return 83; }
  grep -q 'FAILED_RECORDED_BEFORE_CLEANUP' "$tmp/failure_intent_calibration.json" || { rm -rf -- "$tmp"; return 84; }
  grep -q '"cleanup_rc":99' "$tmp/cleanup_outcome_calibration.json" || { rm -rf -- "$tmp"; return 85; }
  EXECUTE_MODE=0
  SUCCESS=0
  SIMULATE_CLEANUP_FAILURE=0
  printf '%s\n' '{"status":"PASS_GUARD_CLEANUP_NOUNSET_AND_FAILURE_INTEGRITY","failure_record_before_cleanup":true,"forced_cleanup_rc":99,"gpu_binary_executed":false,"nvidia_smi_called":false}'
  rm -rf -- "$tmp"
}

case "${1:-}" in
  --verify-source-only) verify_source_only ;;
  --self-test-failure-integrity) self_test_failure_integrity ;;
  --execute) echo 'REFUSED: v4 guard is source-only; no execution path exists.' >&2; exit 64 ;;
  *) usage >&2; exit 64 ;;
esac

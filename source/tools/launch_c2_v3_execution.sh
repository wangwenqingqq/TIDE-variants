#!/bin/bash
# Human opt-in trust root for a future C2 v3 execution.  It is intentionally
# excluded from PINS to avoid a literal self-hash cycle; it binds guard/PINS
# literals and immediately replaces its environment with env -i.
set -Eeuo pipefail
PATH=/usr/bin:/bin
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH BASH_ENV ENV CDPATH
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3
GUARD="$ROOT/tools/run_c2_v3_execution_guard.sh"
PINS="$ROOT/provenance/c2_v3_execution_pins_v1.json"
PIN_VERIFY="$ROOT/tools/verify_c2_v3_execution_pins.py"
PYTHON=/usr/bin/python3
EXPECTED_GUARD_SHA=82954e824f6cd78d63b16e02e890eb10b0293746482e39a57425f6bc18a5e760
EXPECTED_PINS_SHA=8a601a01442bb1af719b4eb5b7511b891b06b0f35758625f1c73da0feea43eaa
EXPECTED_PINS_VERIFY_SHA=0511fe9613efc16f14ae0f9cc565fb08a59522ed5e0578a92d1baaa0c9dfd051

fail() { echo "SAFE-C2-V3 launcher FAIL: $*" >&2; exit 64; }
sha256_of() { [[ -f "$1" && ! -L "$1" ]] || fail "not a direct regular file: $1"; /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }
validate_chain() {
  [[ "$(/usr/bin/readlink -f "$0")" == "$ROOT/tools/launch_c2_v3_execution.sh" ]] || fail 'launcher invoked through unexpected path'
  [[ "$EXPECTED_GUARD_SHA" =~ ^[0-9a-f]{64}$ && "$EXPECTED_PINS_SHA" =~ ^[0-9a-f]{64}$ && "$EXPECTED_PINS_VERIFY_SHA" =~ ^[0-9a-f]{64}$ ]] || fail 'launcher literals were not finalized'
  [[ "$(sha256_of "$GUARD")" == "$EXPECTED_GUARD_SHA" ]] || fail 'outer guard SHA mismatch'
  [[ "$(sha256_of "$PINS")" == "$EXPECTED_PINS_SHA" ]] || fail 'PINS SHA mismatch'
  [[ "$(sha256_of "$PIN_VERIFY")" == "$EXPECTED_PINS_VERIFY_SHA" ]] || fail 'PINS verifier SHA mismatch'
}
new_run_path() {
  local stamp nonce
  stamp="$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)"
  nonce="$(/usr/bin/od -An -N4 -tu4 /dev/urandom | /usr/bin/tr -d '[:space:]')"
  printf '%s/runs/c2_v3_exec_%s_%09d' "$ROOT" "$stamp" "$((10#$nonce % 1000000000))"
}
case "${1:-}" in
  --verify-source-only)
    validate_chain
    exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C "$PYTHON" "$PIN_VERIFY" --source-only --pins "$PINS" --expected-pins-sha "$EXPECTED_PINS_SHA"
    ;;
  --execute)
    validate_chain
    [[ "${C2_V3_EXECUTE_ACK:-}" == "I_CONFIRM_ONE_IRREVOCABLE_C2_V3_WORKFLOW" ]] || fail 'set C2_V3_EXECUTE_ACK=I_CONFIRM_ONE_IRREVOCABLE_C2_V3_WORKFLOW explicitly'
    out="$(new_run_path)"
    [[ ! -e "$out" && ! -L "$out" ]] || fail 'fresh run-path collision; re-run launcher to get a new name'
    exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C \
      C2_V3_TRUST_LAUNCHER=SAFE_C2_V3_LAUNCHER_V1 \
      C2_V3_TRUST_LAUNCHER_PATH="$ROOT/tools/launch_c2_v3_execution.sh" \
      C2_V3_TRUST_GUARD_SHA="$EXPECTED_GUARD_SHA" \
      C2_V3_TRUST_PINS_SHA="$EXPECTED_PINS_SHA" \
      C2_V3_TRUST_PINS_VERIFY_SHA="$EXPECTED_PINS_VERIFY_SHA" \
      C2_V3_EXECUTE_ACK=I_CONFIRM_ONE_IRREVOCABLE_C2_V3_WORKFLOW \
      C2_V3_RUN_OUT="$out" \
      "$GUARD" --execute
    ;;
  *)
    echo 'usage: launch_c2_v3_execution.sh --verify-source-only|--execute' >&2
    exit 64
    ;;
esac

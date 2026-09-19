#!/usr/bin/env bash
# Reviewed human trust root for the v7 measured guard.  It is intentionally not
# inside hardened_static_pins_v7.json to avoid a self-hash cycle.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_cpu_preflight'
LAUNCHER="$ROOT/launch_c1_v7.sh"
GUARD="$ROOT/run_c1_workspace_microbenchmark_guard_v7.sh"
PINS="$ROOT/hardened_static_pins_v7.json"
PIN_VERIFY="$ROOT/verify_v7_pins.py"
SHA256='/usr/bin/sha256sum'
REALPATH='/usr/bin/realpath'
DATE='/usr/bin/date'
AWK='/usr/bin/awk'
# Source-only CPU-preflight root: inert until a separate post-build review binds exact hashes.
REVIEW_STATE='REVIEW_REQUIRED_V7'
TR='/usr/bin/tr'
GUARD_SHA='2607f3480e67ba2040d0e6bb94715ff0642357673799df91dca410fe69f7733d'
PINS_SHA='a865ff0bdf4eaea7183036b927665bf47b14d0cd654ddd287a6dae02ea0f1997'
PIN_VERIFY_SHA='58b4d8d600325a0e609977dfce349a84fafa283dc97a69702b62fc4bf386fad7'
regular() {
  local p="$1" q
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  q="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$q" == "$p" ]]
}
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
[[ "$REVIEW_STATE" == 'REVIEW_APPROVED_V7' ]] || { echo 'C1-V7-LAUNCHER BLOCKED: review-required v7 launcher; exact v7 guard/PINS/verifier hashes must be independently reviewed and committed first' >&2; exit 69; }
[[ "$0" == "$LAUNCHER" && "$#" -eq 0 ]] || { echo 'Usage: launch_c1_v7.sh' >&2; exit 64; }
regular "$LAUNCHER" && regular "$GUARD" && regular "$PINS" && regular "$PIN_VERIFY" || { echo 'C1-V7-LAUNCHER BLOCKED: trust root path failure' >&2; exit 69; }
[[ "$(sha "$GUARD")" == "$GUARD_SHA" && "$(sha "$PINS")" == "$PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$PIN_VERIFY_SHA" ]] || { echo 'C1-V7-LAUNCHER BLOCKED: reviewed hash mismatch' >&2; exit 69; }
stamp="$("$DATE" -u +%Y%m%dT%H%M%SZ)"
nonce="$("$DATE" -u +%N)"
OUT="${ROOT}/runs/c1_v7_measured_${stamp}_${nonce}"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'C1-V7-LAUNCHER BLOCKED: output collision' >&2; exit 69; }
exec /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' \
  C1_V7_TRUST_ROOT=YES C1_V7_LAUNCHER="$LAUNCHER" \
  C1_V7_TRUST_GUARD_SHA="$GUARD_SHA" C1_V7_TRUST_PINS_SHA="$PINS_SHA" C1_V7_TRUST_PIN_VERIFY_SHA="$PIN_VERIFY_SHA" \
  C1_ALLOW_GPU0=YES C1_RUN_OUT="$OUT" "$GUARD"

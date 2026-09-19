#!/usr/bin/env bash
# Reviewed human trust root for the separate v8 Nsight profile.  It is intentionally
# outside hardened_static_pins_v8.json to avoid a launcher self-hash cycle.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract'
LAUNCHER="$ROOT/launch_c1_profile_v8.sh"
GUARD="$ROOT/run_c1_profile_guard_v8.sh"
PINS="$ROOT/hardened_static_pins_v8.json"
PIN_VERIFY="$ROOT/verify_v8_pins.py"
SHA256='/usr/bin/sha256sum'
REALPATH='/usr/bin/realpath'
DATE='/usr/bin/date'
AWK='/usr/bin/awk'
# v8 source successor is intentionally inert until a human reviews exact post-build hashes.
REVIEW_STATE='REVIEW_APPROVED_V8'
PROFILE_GUARD_SHA='a73f7059285c8473ac7e8747c0cc07e39ce15a17c79d0dedb57ac24bdd8b5fd3'
PINS_SHA='594e2bc84017abb7d64ef539679a50654dfb4d411ac14b2983f11a16897d7f47'
PIN_VERIFY_SHA='ccce421524ca77914b20bab528de2a1af528f5a31d3846c54be34f880c665d6f'
regular() {
  local p="$1" q
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  q="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$q" == "$p" ]]
}
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
[[ "$REVIEW_STATE" == 'REVIEW_APPROVED_V8' ]] || { echo 'C1-V8-LAUNCHER BLOCKED: review-required v8 launcher; exact v8 guard/PINS/verifier hashes must be independently reviewed and committed first' >&2; exit 69; }
[[ "$0" == "$LAUNCHER" && "$#" -eq 0 ]] || { echo 'Usage: launch_c1_profile_v8.sh' >&2; exit 64; }
regular "$LAUNCHER" && regular "$GUARD" && regular "$PINS" && regular "$PIN_VERIFY" || { echo 'C1-V8-PROFILE-LAUNCHER BLOCKED: trust root path failure' >&2; exit 69; }
[[ "$(sha "$GUARD")" == "$PROFILE_GUARD_SHA" && "$(sha "$PINS")" == "$PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$PIN_VERIFY_SHA" ]] || { echo 'C1-V8-PROFILE-LAUNCHER BLOCKED: reviewed hash mismatch' >&2; exit 69; }
stamp="$("$DATE" -u +%Y%m%dT%H%M%SZ)"
nonce="$("$DATE" -u +%N)"
OUT="${ROOT}/profiles/c1_v8_profile_${stamp}_${nonce}"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'C1-V8-PROFILE-LAUNCHER BLOCKED: output collision' >&2; exit 69; }
exec /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' \
  C1_V8_PROFILE_TRUST_ROOT=YES C1_V8_PROFILE_LAUNCHER="$LAUNCHER" \
  C1_V8_PROFILE_TRUST_GUARD_SHA="$PROFILE_GUARD_SHA" C1_V8_PROFILE_TRUST_PINS_SHA="$PINS_SHA" C1_V8_PROFILE_TRUST_PIN_VERIFY_SHA="$PIN_VERIFY_SHA" \
  C1_ALLOW_GPU0=YES C1_PROFILE_OUT="$OUT" "$GUARD"

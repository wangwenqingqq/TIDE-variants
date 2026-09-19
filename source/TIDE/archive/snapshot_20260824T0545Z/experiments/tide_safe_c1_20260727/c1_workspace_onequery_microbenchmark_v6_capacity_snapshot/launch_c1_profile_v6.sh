#!/usr/bin/env bash
# Reviewed human trust root for the separate v6 Nsight profile.  It is intentionally
# outside hardened_static_pins_v6.json to avoid a launcher self-hash cycle.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot'
LAUNCHER="$ROOT/launch_c1_profile_v6.sh"
GUARD="$ROOT/run_c1_profile_guard_v6.sh"
PINS="$ROOT/hardened_static_pins_v6.json"
PIN_VERIFY="$ROOT/verify_v6_pins.py"
SHA256='/usr/bin/sha256sum'
REALPATH='/usr/bin/realpath'
DATE='/usr/bin/date'
AWK='/usr/bin/awk'
PROFILE_GUARD_SHA='86355d90d09d852bf2276d2df54a2e9d38548c19f607a5d2edaa799cad1b0f27'
PINS_SHA='7517e5a38122d2912c9ea228e86764b4551c03ecdacbf1785fa72645aa68a270'
PIN_VERIFY_SHA='c2a88a69018c029e8af1b70562e365814ede23d68934ba2eb3273217c3522abe'
regular() {
  local p="$1" q
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  q="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$q" == "$p" ]]
}
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
[[ "$0" == "$LAUNCHER" && "$#" -eq 0 ]] || { echo 'Usage: launch_c1_profile_v6.sh' >&2; exit 64; }
regular "$LAUNCHER" && regular "$GUARD" && regular "$PINS" && regular "$PIN_VERIFY" || { echo 'C1-V6-PROFILE-LAUNCHER BLOCKED: trust root path failure' >&2; exit 69; }
[[ "$(sha "$GUARD")" == "$PROFILE_GUARD_SHA" && "$(sha "$PINS")" == "$PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$PIN_VERIFY_SHA" ]] || { echo 'C1-V6-PROFILE-LAUNCHER BLOCKED: reviewed hash mismatch' >&2; exit 69; }
stamp="$("$DATE" -u +%Y%m%dT%H%M%SZ)"
nonce="$("$DATE" -u +%N)"
OUT="${ROOT}/profiles/c1_v6_profile_${stamp}_${nonce}"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'C1-V6-PROFILE-LAUNCHER BLOCKED: output collision' >&2; exit 69; }
exec /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' \
  C1_V6_PROFILE_TRUST_ROOT=YES C1_V6_PROFILE_LAUNCHER="$LAUNCHER" \
  C1_V6_PROFILE_TRUST_GUARD_SHA="$PROFILE_GUARD_SHA" C1_V6_PROFILE_TRUST_PINS_SHA="$PINS_SHA" C1_V6_PROFILE_TRUST_PIN_VERIFY_SHA="$PIN_VERIFY_SHA" \
  C1_ALLOW_GPU0=YES C1_PROFILE_OUT="$OUT" "$GUARD"

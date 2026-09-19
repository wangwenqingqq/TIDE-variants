#!/usr/bin/env bash
# Reviewed human trust root for the separate v5 Nsight profile.  It is intentionally
# outside hardened_static_pins_v5.json to avoid a launcher self-hash cycle.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v5_formal'
LAUNCHER="$ROOT/launch_c1_profile_v5.sh"
GUARD="$ROOT/run_c1_profile_guard_v5.sh"
PINS="$ROOT/hardened_static_pins_v5.json"
PIN_VERIFY="$ROOT/verify_v5_pins.py"
SHA256='/usr/bin/sha256sum'
REALPATH='/usr/bin/realpath'
DATE='/usr/bin/date'
AWK='/usr/bin/awk'
PROFILE_GUARD_SHA='78e0ef743ab307aa697f48793e384d06843cba6baa919189eb27d34040036a60'
PINS_SHA='4dbf9ecbb291f191deed073c35cf3f216fe98c582a5cb5d9eb7adcc12f3b2772'
PIN_VERIFY_SHA='266e2ee21cb3f1f8de52b574df700cc4d78ea6f96c8e90cdb2bb90db6605132b'
regular() {
  local p="$1" q
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  q="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$q" == "$p" ]]
}
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
[[ "$0" == "$LAUNCHER" && "$#" -eq 0 ]] || { echo 'Usage: launch_c1_profile_v5.sh' >&2; exit 64; }
regular "$LAUNCHER" && regular "$GUARD" && regular "$PINS" && regular "$PIN_VERIFY" || { echo 'C1-V5-PROFILE-LAUNCHER BLOCKED: trust root path failure' >&2; exit 69; }
[[ "$(sha "$GUARD")" == "$PROFILE_GUARD_SHA" && "$(sha "$PINS")" == "$PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$PIN_VERIFY_SHA" ]] || { echo 'C1-V5-PROFILE-LAUNCHER BLOCKED: reviewed hash mismatch' >&2; exit 69; }
stamp="$("$DATE" -u +%Y%m%dT%H%M%SZ)"
nonce="$("$DATE" -u +%N)"
OUT="${ROOT}/profiles/c1_v5_profile_${stamp}_${nonce}"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'C1-V5-PROFILE-LAUNCHER BLOCKED: output collision' >&2; exit 69; }
exec /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' \
  C1_V5_PROFILE_TRUST_ROOT=YES C1_V5_PROFILE_LAUNCHER="$LAUNCHER" \
  C1_V5_PROFILE_TRUST_GUARD_SHA="$PROFILE_GUARD_SHA" C1_V5_PROFILE_TRUST_PINS_SHA="$PINS_SHA" C1_V5_PROFILE_TRUST_PIN_VERIFY_SHA="$PIN_VERIFY_SHA" \
  C1_ALLOW_GPU0=YES C1_PROFILE_OUT="$OUT" "$GUARD"

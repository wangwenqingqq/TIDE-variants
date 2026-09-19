#!/usr/bin/env bash
# Reviewed human trust root for the v6 measured guard.  It is intentionally not
# inside hardened_static_pins_v6.json to avoid a self-hash cycle.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot'
LAUNCHER="$ROOT/launch_c1_v6.sh"
GUARD="$ROOT/run_c1_workspace_microbenchmark_guard_v6.sh"
PINS="$ROOT/hardened_static_pins_v6.json"
PIN_VERIFY="$ROOT/verify_v6_pins.py"
SHA256='/usr/bin/sha256sum'
REALPATH='/usr/bin/realpath'
DATE='/usr/bin/date'
AWK='/usr/bin/awk'
TR='/usr/bin/tr'
GUARD_SHA='ffa113f907909ab69351d1f9ee11057ec3ab4b3dbcd0176eccd63de45571550d'
PINS_SHA='7517e5a38122d2912c9ea228e86764b4551c03ecdacbf1785fa72645aa68a270'
PIN_VERIFY_SHA='c2a88a69018c029e8af1b70562e365814ede23d68934ba2eb3273217c3522abe'
regular() {
  local p="$1" q
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  q="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$q" == "$p" ]]
}
sha() { "$SHA256" -- "$1" | "$AWK" '{print $1}'; }
[[ "$0" == "$LAUNCHER" && "$#" -eq 0 ]] || { echo 'Usage: launch_c1_v6.sh' >&2; exit 64; }
regular "$LAUNCHER" && regular "$GUARD" && regular "$PINS" && regular "$PIN_VERIFY" || { echo 'C1-V6-LAUNCHER BLOCKED: trust root path failure' >&2; exit 69; }
[[ "$(sha "$GUARD")" == "$GUARD_SHA" && "$(sha "$PINS")" == "$PINS_SHA" && "$(sha "$PIN_VERIFY")" == "$PIN_VERIFY_SHA" ]] || { echo 'C1-V6-LAUNCHER BLOCKED: reviewed hash mismatch' >&2; exit 69; }
stamp="$("$DATE" -u +%Y%m%dT%H%M%SZ)"
nonce="$("$DATE" -u +%N)"
OUT="${ROOT}/runs/c1_v6_measured_${stamp}_${nonce}"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'C1-V6-LAUNCHER BLOCKED: output collision' >&2; exit 69; }
exec /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' \
  C1_V6_TRUST_ROOT=YES C1_V6_LAUNCHER="$LAUNCHER" \
  C1_V6_TRUST_GUARD_SHA="$GUARD_SHA" C1_V6_TRUST_PINS_SHA="$PINS_SHA" C1_V6_TRUST_PIN_VERIFY_SHA="$PIN_VERIFY_SHA" \
  C1_ALLOW_GPU0=YES C1_RUN_OUT="$OUT" "$GUARD"

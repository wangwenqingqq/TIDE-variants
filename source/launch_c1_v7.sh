#!/usr/bin/env bash
# Reviewed human trust root for the v7 measured guard.  It is intentionally not
# inside hardened_static_pins_v7.json to avoid a self-hash cycle.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
unset PYTHONOPTIMIZE PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_manifest_repair'
LAUNCHER="$ROOT/launch_c1_v7.sh"
GUARD="$ROOT/run_c1_workspace_microbenchmark_guard_v7.sh"
PINS="$ROOT/hardened_static_pins_v7.json"
PIN_VERIFY="$ROOT/verify_v7_pins.py"
SHA256='/usr/bin/sha256sum'
REALPATH='/usr/bin/realpath'
DATE='/usr/bin/date'
AWK='/usr/bin/awk'
# v7 source successor is intentionally inert until a human reviews exact post-build hashes.
REVIEW_STATE='REVIEW_APPROVED_V7'
TR='/usr/bin/tr'
GUARD_SHA='f505483705ded9089ab20e77a18b5b3643575a2ab781fe05d2f98cf68af8ac4e'
PINS_SHA='dafba4e8bfadbdcc26381e047c4c2509c66d6ad41019268fb0f8c04f005659b5'
PIN_VERIFY_SHA='935c68c4126286cf67d937ca1ff2f857590b095a6c86590eaf36b95f834c4b73'
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

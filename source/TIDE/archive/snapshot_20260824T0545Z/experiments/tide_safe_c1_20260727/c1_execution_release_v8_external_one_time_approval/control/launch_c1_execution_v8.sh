#!/usr/bin/env bash
# Convenience entrypoint only. The guard independently verifies and consumes
# the external approval; this launcher supplies no trust environment/state.
set -Eeuo pipefail
umask 077
PATH='/usr/bin:/bin'
export PATH
ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_execution_release_v8_external_one_time_approval'
SELF="$ROOT/control/launch_c1_execution_v8.sh"
GUARD="$ROOT/control/run_c1_execution_guard_v8.sh"
REALPATH='/usr/bin/realpath'
regular() {
  local p="$1" resolved
  [[ "$p" == /* && -f "$p" && ! -L "$p" ]] || return 1
  resolved="$("$REALPATH" -e -- "$p")" || return 1
  [[ "$resolved" == "$p" ]]
}
[[ "$0" == "$SELF" && "$#" -eq 2 && "$1" == '--approval-file' && "$2" == /* ]] || { echo 'Usage: launch_c1_execution_v8.sh --approval-file /external/approval.json' >&2; exit 69; }
regular "$SELF" && regular "$GUARD" || { echo 'C1-V8-LAUNCHER BLOCKED: noncanonical release control path' >&2; exit 69; }
exec /usr/bin/env -i PATH='/usr/bin:/bin' HOME='/workspace' LANG='C' "$GUARD" --approval-file "$2"

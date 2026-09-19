#!/bin/sh
# The only supported entry point for the guarded native pilot.
set -eu
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v5_failstop_boundinput_pilot"
BODY="$ROOT/tools/.run_g3_4k_single_tu.body.sh"
[ -f "$BODY" ] && [ ! -L "$BODY" ] || {
  echo "missing or unsafe controlled run body" >&2
  exit 69
}
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent \
  /bin/bash --noprofile --norc "$BODY" "$@"

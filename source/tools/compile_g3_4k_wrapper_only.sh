#!/bin/sh
# The only supported entry point for the controlled compile-only body.
set -eu
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v5_failstop_boundinput_pilot"
BODY="$ROOT/tools/.compile_g3_4k_wrapper_only.body.sh"
[ -f "$BODY" ] && [ ! -L "$BODY" ] || {
  echo "missing or unsafe controlled compile body" >&2
  exit 69
}
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent \
  /bin/bash --noprofile --norc "$BODY" "$@"

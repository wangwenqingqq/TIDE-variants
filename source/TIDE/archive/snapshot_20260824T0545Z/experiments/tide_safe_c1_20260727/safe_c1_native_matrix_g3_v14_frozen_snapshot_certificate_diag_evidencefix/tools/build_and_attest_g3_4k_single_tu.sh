#!/bin/sh
# The only supported entry point for the controlled single-TU link/attestation body.
# It verifies the root-private body bytes before replacing the inherited
# environment; the private body repeats env sanitization as a second boundary.
set -eu
ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v14_frozen_snapshot_certificate_diag_evidencefix"
TOOLS="$ROOT/tools"
BODY="$TOOLS/.build_and_attest_g3_4k_single_tu.body.sh"
EXPECTED_BODY_SHA256="c1d197d141bf089ebbf326dad486484d583052a4b549bf6eb2c0268ae523a7f5"
fail() { /bin/echo "controlled launcher refused: $*" >&2; exit 69; }
private_dir() {
  p="$1"
  [ -d "$p" ] && [ ! -L "$p" ] && [ "$(/usr/bin/readlink -f "$p")" = "$p" ] ||
    fail "unsafe directory $p"
  [ "$(/usr/bin/stat -c '%u:%g:%a' "$p")" = "0:0:700" ] ||
    fail "directory is not root-private 0700 $p"
}
private_body() {
  [ -f "$BODY" ] && [ ! -L "$BODY" ] || fail "missing/unsafe body"
  [ "$(/usr/bin/stat -c '%u:%g:%a:%h' "$BODY")" = "0:0:600:1" ] ||
    fail "body is not root-private single-link 0600"
  actual="$(/usr/bin/sha256sum "$BODY" | /usr/bin/awk '{print $1}')"
  [ "$actual" = "$EXPECTED_BODY_SHA256" ] || fail "body SHA-256 differs from launcher seal"
}
private_dir "$ROOT"
private_dir "$TOOLS"
private_body
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent /bin/bash --noprofile --norc "$BODY" "$@"

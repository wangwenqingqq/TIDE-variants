#!/usr/bin/env bash
# One-shot launcher for the bounded synthetic fixture.  No retry branch exists.
set -euo pipefail
umask 077

ROOT=${1:?missing root}
BUILDER=${2:?missing builder path}
PROTO=${3:?missing protocol path}
MODE=${4:?missing mode}  # --preflight-only or --execute
EXPECTED_TRIAL=8a7b6c5d4e3f2910fedcba9876543210

fail() { printf 'FAIL_CLOSED: %s\n' "$*" >&2; exit 2; }
[ "$(id -u)" = 1001 ] || fail "effective uid must be 1001"
[ "$(id -g)" = 1001 ] || fail "effective gid must be 1001"
[ "$(id -ru)" = 1001 ] || fail "real uid must be 1001"
[ "$(id -rg)" = 1001 ] || fail "real gid must be 1001"
[ -d "$BUILDER" ] && [ -d "$PROTO" ] || fail "source package missing"

verify_package_manifest() {
  /usr/bin/python3 - "$1" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest_path = root / "PACKAGE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = manifest["entries"]
actual = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "PACKAGE_MANIFEST.json":
        actual.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
if actual != expected:
    raise SystemExit("package manifest mismatch: " + str(root))
print("PACKAGE_MANIFEST_PASS", root)
PY
}

verify_package_manifest "$BUILDER"
verify_package_manifest "$PROTO"
/usr/bin/bash -n "$0"

case "$MODE" in
  --preflight-only)
    [ "$ROOT" = "/tmp/tide_e1_v2_syntax_variable_gate_DO_NOT_CREATE" ] || fail "preflight root literal mismatch"
    [ ! -e "$ROOT" ] || fail "preflight path unexpectedly exists"
    printf 'PRELAUNCH_NO_STATE_GATE_PASS root=%s builder=%s protocol=%s trial=%s\n' "$ROOT" "$BUILDER" "$PROTO" "$EXPECTED_TRIAL"
    exit 0
    ;;
  --execute)
    [ "${ROOT##*_}" = "$EXPECTED_TRIAL" ] || fail "root does not bind expected public trial"
    [ ! -e "$ROOT" ] || fail "evidence root already exists"
    [ "$(findmnt -no FSTYPE -T /workspace)" != cifs ] || fail "CIFS evidence root prohibited"
    ;;
  *) fail "unknown mode" ;;
esac

cd /tmp
/usr/bin/python3 "$BUILDER/build_synthetic_e1_fixture.py" --root "$ROOT" --phase prepare
/usr/bin/python3 "$PROTO/tools/validate_trace.py" --catalog "$ROOT/input/raw_catalog.jsonl" --trace "$ROOT/input/sealed_trace.jsonl" > "$ROOT/logs/01_validate_trace.stdout.json" 2> "$ROOT/logs/01_validate_trace.stderr.txt"
/usr/bin/python3 "$PROTO/tools/oracle_replayer.py" --catalog "$ROOT/input/raw_catalog.jsonl" --trace "$ROOT/input/sealed_trace.jsonl" --out "$ROOT/output/oracle_replay.jsonl" > "$ROOT/logs/02_oracle_replayer.stdout.json" 2> "$ROOT/logs/02_oracle_replayer.stderr.txt"
/usr/bin/python3 "$BUILDER/build_synthetic_e1_fixture.py" --root "$ROOT" --phase events > "$ROOT/logs/03_emit_events.stdout.json" 2> "$ROOT/logs/03_emit_events.stderr.txt"
/usr/bin/python3 "$PROTO/tools/validate_e1_bundle.py" --bundle "$ROOT/E1_BUNDLE.json" > "$ROOT/logs/04_validate_e1.stdout.json" 2> "$ROOT/logs/04_validate_e1.stderr.txt"
# Do not redirect this command's stdout into ROOT: that would mutate a file after it is hashed.
/usr/bin/python3 "$PROTO/tools/seal_artifact_manifest.py" create --root "$ROOT" --out "$ROOT/ARTIFACT_MANIFEST.json" --bundle-id "$EXPECTED_TRIAL"
/usr/bin/python3 "$PROTO/tools/seal_artifact_manifest.py" verify --root "$ROOT" --manifest "$ROOT/ARTIFACT_MANIFEST.json"
printf 'FINAL_ROOT=%s\n' "$ROOT"
printf 'FINAL_MODE=%s\n' "$(stat -c %a "$ROOT")"
printf 'FINAL_OWNER=%s:%s\n' "$(stat -c %u "$ROOT")" "$(stat -c %g "$ROOT")"
printf 'ARTIFACT_MANIFEST_SHA256=%s\n' "$(sha256sum "$ROOT/ARTIFACT_MANIFEST.json" | awk '{print $1}')"
printf 'E1_VALIDATOR='; cat "$ROOT/logs/04_validate_e1.stdout.json"
printf 'TRACE_VALIDATOR='; cat "$ROOT/logs/01_validate_trace.stdout.json"
printf 'ORACLE_REPLAYER='; cat "$ROOT/logs/02_oracle_replayer.stdout.json"

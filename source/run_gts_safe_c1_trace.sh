#!/usr/bin/env bash
# Thin adapter for the guarded E1-GI-B launcher.  CUDA policy is enforced by
# the caller; this wrapper only maps generic paths to the compiled runner ABI.
set -Eeuo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727"
BIN="$ROOT/safe_c1_gts_integration/bin/GTS_safe_c1_trace"
BUNDLE=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "usage: $0 --bundle BUNDLE --out OUTPUT_DIR" >&2; exit 64 ;;
  esac
done
[[ -n "$BUNDLE" && -n "$OUT" && -x "$BIN" ]] || exit 64
[[ -f "$BUNDLE/pool.i16" && -f "$BUNDLE/queries.i16" && -f "$BUNDLE/trace.e1gtrc" ]] || exit 66
mkdir -p "$OUT"
# Provenance is written before the GPU runner starts, so every result directory
# names the exact isolated source/binary/wrapper bytes that produced it.
python3 - "$OUT/runner_provenance.json" "$ROOT" "$BIN" "$0" <<'PYJSON'
import hashlib, json, pathlib, sys
out, root, binary, wrapper = map(pathlib.Path, sys.argv[1:])
source = root / "safe_c1_gts_integration/src/gts_safe_c1_trace.cu"
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
json.dump({
    "schema": "e1gi-b-runner-provenance-v1",
    "scope": "isolated 8p Safe-C1/GTS correctness runner; not an archived GTS source",
    "source": {"path": str(source), "sha256": sha(source)},
    "binary": {"path": str(binary), "sha256": sha(binary)},
    "wrapper": {"path": str(wrapper), "sha256": sha(wrapper)},
}, out.open("w"), indent=2, sort_keys=True)
PYJSON
exec "$BIN" --bundle "$BUNDLE" --out "$OUT/engine_results.jsonl" --summary "$OUT/engine_summary.json" --leaf-capacity 64

#!/usr/bin/env bash
# Thin adapter for the guarded C1 held-out gamma gate.  The caller must export
# a fixed, disjoint query block and a candidate gamma before invoking it.
set -Eeuo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727"
BIN="$ROOT/c1_heldout_gamma_probe/bin/GTS_c1_heldout_gamma_probe"
BUNDLE=""; OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    *) echo "usage: $0 --bundle BUNDLE --out OUTPUT_DIR" >&2; exit 64 ;;
  esac
done
[[ -n "$BUNDLE" && -n "$OUT" && -x "$BIN" ]] || exit 64
[[ -f "$BUNDLE/pool.i16" && -f "$BUNDLE/queries.i16" && -f "$BUNDLE/trace.e1gtrc" ]] || exit 66
: "${C1_GAMMA_SCALE:?C1_GAMMA_SCALE is required}"
: "${C1_QUERY_BEGIN:?C1_QUERY_BEGIN is required}"
: "${C1_QUERY_END:?C1_QUERY_END is required}"
[[ "$C1_GAMMA_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid C1_GAMMA_SCALE" >&2; exit 64; }
[[ "$C1_QUERY_BEGIN" =~ ^[0-9]+$ && "$C1_QUERY_END" =~ ^[0-9]+$ ]] || { echo "invalid C1 query block" >&2; exit 64; }
(( C1_QUERY_END > C1_QUERY_BEGIN )) || { echo "empty C1 query block" >&2; exit 64; }
mkdir -p "$OUT"
python3 - "$OUT/runner_provenance.json" "$ROOT" "$BIN" "$0" <<'PY'
import hashlib, json, os, pathlib, sys
out, root, binary, wrapper = map(pathlib.Path, sys.argv[1:])
source = root / "c1_heldout_gamma_probe/src/gts_c1_heldout_gamma_probe.cu"
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
json.dump({
 "schema":"c1-heldout-runner-provenance-v1",
 "scope":"isolated empirical C1 gamma gate; not a general no-false-negative proof",
 "source":{"path":str(source),"sha256":sha(source)},
 "binary":{"path":str(binary),"sha256":sha(binary)},
 "wrapper":{"path":str(wrapper),"sha256":sha(wrapper)},
 "c1_policy":{"gamma_scale":os.environ["C1_GAMMA_SCALE"],"query_block_begin":int(os.environ["C1_QUERY_BEGIN"]),"query_block_end_exclusive":int(os.environ["C1_QUERY_END"])}
},out.open("w"),indent=2,sort_keys=True)
PY
exec "$BIN" --bundle "$BUNDLE" --out "$OUT/engine_results.jsonl" --summary "$OUT/engine_summary.json" \
  --leaf-capacity 64 --c1-gamma-scale "$C1_GAMMA_SCALE" \
  --c1-query-begin "$C1_QUERY_BEGIN" --c1-query-end "$C1_QUERY_END"

#!/usr/bin/env bash
# CPU-only Safe-C1 G1 v2 launch guard. It never starts CUDA/GPU work.
set -Eeuo pipefail

readonly ROOT="/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native"
readonly RUNS_ROOT="/workspace/experiments/tide_safe_c1_20260728/runs"
readonly V1_ROOT="/workspace/experiments/tide_safe_c1_20260727/safe_c1_dynamic_gts_v1"
readonly PYTHON="${PYTHON:-python3}"
readonly BIN="$ROOT/bin/GTS_safe_c1_g1_v2_search_native"
readonly AUDITOR="$ROOT/tools/audit_g1_static_contract.py"
readonly BUNDLE_PREFLIGHT="$ROOT/tools/preflight_g1_v2_bundle.py"

usage() {
  cat <<'EOF'
Usage: prepare_g1_v2_guarded_run.sh --bundle <new-v2-bundle> --out <new-v2-run-root>

CPU-only authorization preparation. It requires a new v2 bundle with a v2
certificate-selection artifact and creates a terminal CPU preflight card. It
never invokes nvidia-smi, CUDA, Nsight, or the GTS binary. GPU execution needs
separate explicit authorization after this card is reviewed.
EOF
}

BUNDLE=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done
[[ -n "$BUNDLE" && -n "$OUT" ]] || { usage >&2; exit 64; }
[[ -d "$ROOT" && "$ROOT" != "$V1_ROOT" ]] || { echo "invalid v2 root boundary" >&2; exit 64; }
case "$ROOT" in "$V1_ROOT"|"$V1_ROOT"/*) echo "v1 root is forbidden" >&2; exit 64;; esac
case "$RUNS_ROOT" in "$V1_ROOT"|"$V1_ROOT"/*) echo "v1 run boundary is forbidden" >&2; exit 64;; esac
[[ -x "$BIN" && -x "$AUDITOR" && -x "$BUNDLE_PREFLIGHT" ]] || {
  echo "required v2 artifact missing" >&2; exit 66;
}

BUNDLE="$($PYTHON - "$BUNDLE" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT="$($PYTHON - "$OUT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
case "$BUNDLE" in "$ROOT"/bundles/*) ;; *) echo "bundle must be under v2 bundles root" >&2; exit 64;; esac
case "$OUT" in "$RUNS_ROOT"/*) ;; *) echo "out must be under new v2 runs root" >&2; exit 64;; esac
[[ ! -e "$OUT" ]] || { echo "out must be a new path (no reuse of v1/v2 evidence)" >&2; exit 73; }

mkdir -p "$OUT/logs" "$OUT/preflight"
STATUS="CPU_PREFLIGHT_RUNNING"
MESSAGE=""
write_card() {
  "$PYTHON" - "$OUT/run_card.json" <<'PY'
import json, os, sys
from datetime import datetime, timezone
out=sys.argv[1]
e=os.environ
obj={
  'schema':'safe-c1-g1-v2-search-native-run-card',
  'status':e['G1V2_STATUS'],
  'updated_at':datetime.now(timezone.utc).isoformat(),
  'scope':'CPU guard only; sibling-boundary Safe-C1 certificate + bounded static-prefix/dynamic-boundary witness domain; no GPU execution or performance claim',
  'gpu_used':False,
  'cuda_binary_executed':False,
  'binary':e['G1V2_BIN'],
  'bundle':e['G1V2_BUNDLE'],
  'output_root':out,
  'message':e.get('G1V2_MESSAGE',''),
  'next_step':'separate explicit GPU authorization required; this guard never launches CUDA',
  'forbidden_reuse':['v1 bundle schema','v1 run output','all-input canonical (distance,stable_id) claim'],
}
json.dump(obj,open(out,'w'),indent=2,sort_keys=True);open(out,'a').write('\n')
PY
}
finish_failure() {
  STATUS="FAILED_CPU_PREFLIGHT"; MESSAGE="$1"
  export G1V2_STATUS="$STATUS" G1V2_MESSAGE="$MESSAGE"
  write_card
  echo "$MESSAGE" >&2
  exit 2
}
export G1V2_STATUS="$STATUS" G1V2_MESSAGE="$MESSAGE" G1V2_BIN="$BIN" G1V2_BUNDLE="$BUNDLE"
write_card
trap 'rc=$?; if [[ $rc -ne 0 && "$STATUS" != "CPU_PREFLIGHT_COMPLETE_AWAITING_EXPLICIT_GPU_AUTHORIZATION" && "$STATUS" != "FAILED_CPU_PREFLIGHT" ]]; then STATUS="FAILED_CPU_PREFLIGHT"; MESSAGE="unexpected CPU guard failure"; export G1V2_STATUS="$STATUS" G1V2_MESSAGE="$MESSAGE"; write_card || true; fi' EXIT

# Both gates are CPU-only. The bundle verifier refuses pending selection or any
# wrong/v1 schema, so no previous v1 result can authorize a v2 run.
"$PYTHON" "$AUDITOR" --root "$ROOT" --out "$OUT/preflight/static_contract_v2.json" \
  >"$OUT/logs/static_audit.stdout.log" 2>"$OUT/logs/static_audit.stderr.log" \
  || finish_failure "v2 static contract audit failed"
"$PYTHON" "$BUNDLE_PREFLIGHT" --root "$ROOT" --bundle "$BUNDLE" \
  --out "$OUT/preflight/bundle_preflight_v2.json" \
  >"$OUT/logs/bundle_preflight.stdout.log" 2>"$OUT/logs/bundle_preflight.stderr.log" \
  || finish_failure "v2 bundle preflight failed or candidate selection remains pending"

STATUS="CPU_PREFLIGHT_COMPLETE_AWAITING_EXPLICIT_GPU_AUTHORIZATION"
MESSAGE="new-v2 static audit + bounded-tie bundle preflight passed; no GPU was queried or used"
export G1V2_STATUS="$STATUS" G1V2_MESSAGE="$MESSAGE"
write_card
printf 'PASS_CPU_ONLY_G1_V2_GUARD out=%s\n' "$OUT"

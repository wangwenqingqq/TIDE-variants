#!/usr/bin/env bash
# One guarded GPU0 selection run: identify strict-certificate IDs sharing a frozen GTS leaf.
set -Eeuo pipefail
ROOT=/workspace/experiments/tide_safe_c1_20260727
D="$ROOT/safe_c1_dynamic_gts_v1"
PY=/workspace/legacy_workspace/GTS/bench_env/bin/python
BIN="$D/bin/GTS_safe_c1_g1"
BUNDLE="$D/bundles/g1b_selector_all_reservoir_v1"
AUDIT="$D/tools/audit_g1_static_contract.py"
RAW="$ROOT/e1g_adapter/verify_quantized_gts_integration_export.py"
EXPECTED_HOST=CONFIGURE_ARCHIVE_HOST
EXPECTED_UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
OUT=""
[[ $# -eq 2 && "$1" == --out ]] || { echo "usage: $0 --out <new isolated runs path>" >&2; exit 64; }
OUT="$2"
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || { echo "not 8p" >&2; exit 77; }
[[ -x "$PY" && -x "$BIN" && -x "$AUDIT" && -f "$RAW" && -d "$BUNDLE" ]] || { echo "artifact missing" >&2; exit 66; }
OUT="$($PY - "$OUT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
case "$OUT" in "$ROOT"/runs/*) ;; *) echo "out must be under isolated runs" >&2; exit 64;; esac
[[ ! -e "$OUT" ]] || { echo "out already exists" >&2; exit 73; }
mkdir -p "$OUT/logs" "$OUT/runner_output" "$OUT/preflight"
"$PY" "$AUDIT" --root "$D" --out "$OUT/preflight/static_contract.json" >"$OUT/logs/static_audit.stdout.log" 2>"$OUT/logs/static_audit.stderr.log"
# Read-only idle check; never controls another process.
ROW="$(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"; printf "%s\n" "$ROW" >"$OUT/logs/preflight_gpu0.csv"
IFS=, read -r UUID MEM UTIL <<<"$ROW"; UUID="$(xargs <<<"$UUID")"; MEM="$(xargs <<<"$MEM")"; UTIL="$(xargs <<<"$UTIL")"
[[ "$UUID" == "$EXPECTED_UUID" && "$MEM" =~ ^[0-9]+$ && "$UTIL" =~ ^[0-9]+$ && "$MEM" -le 256 && "$UTIL" -eq 0 ]] || { echo "GPU0 not idle/expected: $ROW" >&2; exit 2; }
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >"$OUT/logs/preflight_compute_apps.csv" || true
! grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,' "$OUT/logs/preflight_compute_apps.csv" || { echo "GPU0 compute app exists; no action taken" >&2; exit 2; }
cat >"$OUT/run_card.json" <<EOF
{"schema":"safe-c1-g1b-selector-run-card-v1","status":"RUNNING","scope":"candidate selection only; frozen GTS tree; no dynamic-query or performance evidence","host":"$(hostname)","gpu0_uuid":"$UUID","leaf_capacity":4096,"gpu_used":true}
EOF
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_VISIBLE_DEVICES=0
set +e
timeout 900 "$BIN" --bundle "$BUNDLE" --out "$OUT/runner_output/engine_results.jsonl" --summary "$OUT/runner_output/engine_summary.json" --leaf-capacity 4096 >"$OUT/logs/runner.stdout.log" 2>"$OUT/logs/runner.stderr.log"
RC=$?
set -e
[[ $RC -eq 0 ]] || { echo "runner failed=$RC" >&2; exit $RC; }
set +e
CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=none "$PY" "$RAW" --prepared-bundle "$BUNDLE" --engine-jsonl "$OUT/runner_output/engine_results.jsonl" --out "$OUT/validation.raw_quantized_oracle.json" >"$OUT/logs/validator.stdout.log" 2>"$OUT/logs/validator.stderr.log"
VC=$?
set -e
[[ $VC -eq 0 ]] || { echo "update-only oracle failed=$VC" >&2; exit $VC; }
nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >"$OUT/logs/postrun_gpu0.csv" || true
$PY - "$OUT/run_card.json" <<'PY'
import json,sys
p=sys.argv[1]; x=json.load(open(p)); x['status']='COMPLETE';x['message']='PASS_SELECTION_UPDATE_ORACLE';json.dump(x,open(p,'w'),indent=2,sort_keys=True);open(p,'a').write('\n')
PY
echo "PASS_G1B_SELECTOR out=$OUT"

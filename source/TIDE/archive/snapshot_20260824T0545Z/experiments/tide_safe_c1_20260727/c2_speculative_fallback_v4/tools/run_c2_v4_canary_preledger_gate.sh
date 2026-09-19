#!/usr/bin/env bash
# Safe-C2 v4 pre-ledger parser-compatibility gate.
#
# This runs the final binary only with --validate-workload-only. The binary's
# v4 source returns after strict manifest/file-hash binding and receipt O_EXCL,
# before dataset decoding, CUDA allocation/API, output-stage creation, telemetry,
# or any formal ledger claim.
set -Eeuo pipefail
umask 077

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_sift1m"
MANIFEST="$ROOT/inputs/canary_workload_v1/workload_manifest.json"
BINDING="$ROOT/provenance/c2_v4_canary_workload_independent_audit_v1.json"
GATE_PROTOCOL="$ROOT/protocols/c2_v4_preledger_parser_compatibility_gate_v1.json"
LAUNCHER="$ROOT/tools/run_c2_v4_canary_preledger_gate.sh"
RECEIPTS="$ROOT/preledger_receipts"
PARENT="$RECEIPTS/c2_v4_canary_parser_gate_v1"
RECEIPT="$PARENT/receipt.json"
LOG="$PARENT/launcher.log"
CARD="$PARENT/launcher_card.json"
SUCCESS=0

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$SUCCESS" != 1 && -f "$CARD" ]]; then
    set +e
    python3 - "$CARD" "$rc" <<'PYFAIL'
import json,os,sys
card,rc=sys.argv[1:]
x=json.load(open(card))
x["status"]="FAILED_CANARY_VALIDATE_WORKLOAD_ONLY_PRELEDGER"
x["failure_reason"]="outer_launcher_exit_"+rc
tmp=card+".tmp"
with open(tmp,"x",encoding="utf-8") as f:json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,card)
PYFAIL
    set -e
  fi
  exit "$rc"
}
trap on_exit EXIT

for path in "$BIN" "$MANIFEST" "$BINDING" "$GATE_PROTOCOL" "$LAUNCHER"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing/symlink gate input: $path" >&2; exit 64; }
done
for path in "$ROOT/runs" "$ROOT/.locks" "$PARENT" "$RECEIPT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "refusing pre-existing formal/gate state: $path" >&2; exit 65; }
done
[[ -d "$RECEIPTS" && ! -L "$RECEIPTS" ]] || { echo "missing/non-direct receipt root" >&2; exit 65; }
EXPECTED_PRIOR_RECEIPTS=$'c2_v4_parser_gate_v1\nc2_v4_parser_gate_v2'
[[ "$(find "$RECEIPTS" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" == "$EXPECTED_PRIOR_RECEIPTS" ]] || {
  echo "unexpected prior receipt roots" >&2; exit 65;
}
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "e9e6f22a42cc15cad4154b31c134a59c73768674987e096a943ec824901f5693" ]] || {
  echo "manifest hash changed" >&2; exit 66;
}
python3 - "$BINDING" "$GATE_PROTOCOL" <<'PY'
import json,sys
binding=json.load(open(sys.argv[1])); gate=json.load(open(sys.argv[2]))
if binding.get('status')!='PASS_CANARY_WORKLOAD_ISOLATION_CPU_ONLY':
 raise SystemExit('canary workload isolation audit is not PASS')
if gate.get('status')!='DESIGNED_SOURCE_ONLY_NO_FINAL_BINARY_NO_LEDGER':
 raise SystemExit('preledger gate protocol identity mismatch')
print('PASS_CANARY_PRELEDGER_GATE_LAUNCH_PREFLIGHT')
PY

mkdir "$PARENT"
cat >"$CARD" <<EOF
{
  "schema": "safe-c2-v4-canary-preledger-gate-launcher-card-v1",
  "status": "RUNNING_VALIDATE_WORKLOAD_ONLY_NO_LEDGER",
  "scope": "Canary-only strict manifest/file hash validation. No CUDA/dataset decode/canary stage output/formal ledger.",
  "binary_path": "$BIN",
  "binary_sha256": "$(sha256sum "$BIN" | awk '{print $1}')",
  "manifest_path": "$MANIFEST",
  "manifest_sha256": "$(sha256sum "$MANIFEST" | awk '{print $1}')",
  "launcher_path": "$LAUNCHER",
  "launcher_sha256": "$(sha256sum "$LAUNCHER" | awk '{print $1}')",
  "receipt_path": "$RECEIPT",
  "gpu_binary_executed": false,
  "nvidia_smi_called": false
}
EOF

python3 - "$MANIFEST" "$RECEIPT" "$BIN" "$LOG" <<'PY'
import json,subprocess,sys
m,receipt,binary,log=sys.argv[1:]
x=json.load(open(m))
argv=[
 binary,"--validate-workload-only","--mode","calibrate",
 "--base-fvecs",x["base_fvecs_path"],
 "--query-fvecs",x["query_fvecs_path"],
 "--groundtruth-ivecs",x["groundtruth_ivecs_path"],
 "--query-ids",x["calibration_ids_path"],
 "--stage","calibration","--workload-manifest",m,
 "--validation-receipt",receipt,
]
with open(log,"x",encoding="utf-8") as f:
 f.write("scope=validate-workload-only; no CUDA/dataset decode/ledger/stage output\n")
 f.write("argv_json="+json.dumps(argv)+"\n")
 f.flush()
 result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=False)
if result.returncode:
 raise SystemExit(result.returncode)
PY

[[ -f "$RECEIPT" && ! -L "$RECEIPT" && -s "$RECEIPT" ]] || { echo "missing receipt" >&2; exit 67; }
python3 - "$CARD" "$RECEIPT" <<'PY'
import json,os,sys
card,receipt=sys.argv[1:]
x=json.load(open(card))
x["status"]="COMPLETE_CANARY_VALIDATE_WORKLOAD_ONLY_RECEIPT_UNVERIFIED"
x["receipt_sha256"]=__import__('hashlib').sha256(open(receipt,'rb').read()).hexdigest()
tmp=card+".tmp"
with open(tmp,"x",encoding="utf-8") as f:json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,card)
PY
SUCCESS=1
printf '%s\n' "$RECEIPT"

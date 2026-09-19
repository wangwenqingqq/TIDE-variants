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
MANIFEST="$ROOT/inputs/final_workload_v1/workload_manifest.json"
BUILD_AUDIT="$ROOT/provenance/c2_v4_final_binary_build_independent_audit_v1.json"
GATE_PROTOCOL="$ROOT/protocols/c2_v4_preledger_parser_compatibility_gate_v1.json"
LAUNCHER="$ROOT/tools/run_c2_v4_preledger_parser_gate.sh"
RECEIPTS="$ROOT/preledger_receipts"
PARENT="$RECEIPTS/c2_v4_parser_gate_v1"
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
x["status"]="FAILED_VALIDATE_WORKLOAD_ONLY_PRELEDGER"
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

for path in "$BIN" "$MANIFEST" "$BUILD_AUDIT" "$GATE_PROTOCOL" "$LAUNCHER"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing/symlink gate input: $path" >&2; exit 64; }
done
for path in "$ROOT/runs" "$ROOT/.locks" "$RECEIPTS" "$PARENT" "$RECEIPT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "refusing pre-existing formal/gate state: $path" >&2; exit 65; }
done
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "6373431c50806e7c6a993282722e58f5453a01d8a8e9d17cae27018302eaccd7" ]] || {
  echo "manifest hash changed" >&2; exit 66;
}
python3 - "$BUILD_AUDIT" "$GATE_PROTOCOL" <<'PY'
import json,sys
audit=json.load(open(sys.argv[1])); gate=json.load(open(sys.argv[2]))
if audit.get('status')!='PASS_FINAL_BINARY_BUILD_AND_PARSER_VERIFIER_PREFLIGHT':
 raise SystemExit('build audit is not PASS')
if gate.get('status')!='DESIGNED_SOURCE_ONLY_NO_FINAL_BINARY_NO_LEDGER':
 raise SystemExit('preledger gate protocol identity mismatch')
print('PASS_PRELEDGER_GATE_LAUNCH_PREFLIGHT')
PY

mkdir "$RECEIPTS"
mkdir "$PARENT"
cat >"$CARD" <<EOF
{
  "schema": "safe-c2-v4-preledger-gate-launcher-card-v1",
  "status": "RUNNING_VALIDATE_WORKLOAD_ONLY_NO_LEDGER",
  "scope": "Strict manifest/file hash validation only. No CUDA/dataset decode/telemetry/stage output/formal ledger.",
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
x["status"]="COMPLETE_VALIDATE_WORKLOAD_ONLY_RECEIPT_UNVERIFIED"
x["receipt_sha256"]=__import__('hashlib').sha256(open(receipt,'rb').read()).hexdigest()
tmp=card+".tmp"
with open(tmp,"x",encoding="utf-8") as f:json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,card)
PY
SUCCESS=1
printf '%s\n' "$RECEIPT"

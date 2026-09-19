#!/usr/bin/env bash
# CPU-only parser receipt gate for the exact metadata/path-safety corrected binary.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_metadatafix_v1"
MANIFEST="$ROOT/inputs/final_workload_v2/workload_manifest.json"
BUILD="$ROOT/provenance/c2_v4_metadatafix_binary_build_v1.json"
SOURCE_AUDIT="$ROOT/provenance/c2_v4_metadatafix_source_audit_v1.json"
LAUNCHER="$ROOT/tools/run_c2_v4_metadatafix_preledger_gate_v1.sh"
PARENT="$ROOT/preledger_receipts/c2_v4_metadatafix_parser_gate_v1"
RECEIPT="$PARENT/receipt.json"
LOG="$PARENT/launcher.log"
CARD="$PARENT/launcher_card.json"
SUCCESS=0
on_exit() {
 rc=$?; trap - EXIT
 if [[ "$SUCCESS" != 1 && -f "$CARD" ]]; then
  set +e
  python3 - "$CARD" "$rc" <<'PY'
import json,os,sys
p,rc=sys.argv[1:]
x=json.load(open(p));x["status"]="FAILED_METADATAFIX_VALIDATE_WORKLOAD_ONLY";x["failure_reason"]="outer_launcher_exit_"+rc
t=p+".tmp"
with open(t,"w",encoding="utf-8") as f:json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
os.replace(t,p)
PY
  set -e
 fi
 exit "$rc"
}
trap on_exit EXIT
for p in "$BIN" "$MANIFEST" "$BUILD" "$SOURCE_AUDIT" "$LAUNCHER"; do
 [[ -f "$p" && ! -L "$p" ]] || { echo "missing/symlink gate input: $p" >&2; exit 65; }
done
for p in "$PARENT" "$RECEIPT" "$ROOT/runs" "$ROOT/.locks"; do
 [[ ! -e "$p" && ! -L "$p" ]] || { echo "existing state: $p" >&2; exit 66; }
done
[[ -d "$ROOT/preledger_receipts" && ! -L "$ROOT/preledger_receipts" ]] || { echo "receipt root non-direct"; exit 67; }
python3 - "$BUILD" "$SOURCE_AUDIT" "$BIN" "$MANIFEST" <<'PY'
import hashlib,json,sys
build,audit,binary,manifest=sys.argv[1:]
def sha(p):return hashlib.sha256(open(p,"rb").read()).hexdigest()
b=json.load(open(build));a=json.load(open(audit))
if b.get("status")!="COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION":raise SystemExit("build status")
if a.get("status")!="PASS_V4_METADATA_AND_PATH_SAFETY_SOURCE_AUDIT":raise SystemExit("source audit status")
if b.get("binary",{}).get("path")!=binary or b.get("binary",{}).get("sha256")!=sha(binary):raise SystemExit("binary binding")
if sha(manifest)!="7b624940523955d6011fe55da1ca2736f9ac19d79c70e8b013c28f17ad00d7ae":raise SystemExit("manifest hash")
print("PASS_METADATAFIX_PRELEDGER_LAUNCH_PREFLIGHT")
PY
mkdir "$PARENT"
python3 - "$MANIFEST" "$RECEIPT" "$BIN" "$LOG" "$CARD" "$LAUNCHER" <<'PY'
import hashlib,json,os,subprocess,sys
m,receipt,binary,log,card,launcher=sys.argv[1:]
x=json.load(open(m))
def sha(p):return hashlib.sha256(open(p,"rb").read()).hexdigest()
state={"schema":"safe-c2-v4-metadatafix-preledger-launcher-card-v1","status":"RUNNING_VALIDATE_WORKLOAD_ONLY_NO_CUDA_OR_DATASET_LOAD","scope":"strict manifest/file hash parser gate only; no dataset decode, CUDA, telemetry, stage output, or formal ledger","binary":{"path":binary,"sha256":sha(binary)},"manifest":{"path":m,"sha256":sha(m)},"receipt_path":receipt,"launcher":{"path":launcher,"sha256":sha(launcher)},"gpu_binary_executed":False,"nvidia_smi_called":False}
with open(card,"x",encoding="utf-8") as f:json.dump(state,f,sort_keys=True,indent=2);f.write("\n")
argv=[binary,"--validate-workload-only","--mode","calibrate","--base-fvecs",x["base_fvecs_path"],"--query-fvecs",x["query_fvecs_path"],"--groundtruth-ivecs",x["groundtruth_ivecs_path"],"--query-ids",x["calibration_ids_path"],"--stage","calibration","--workload-manifest",m,"--validation-receipt",receipt]
with open(log,"x",encoding="utf-8") as f:
 f.write("scope=validate-workload-only; no CUDA/dataset decode/ledger/stage output\nargv_json="+json.dumps(argv)+"\n");f.flush()
 r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=False)
if r.returncode:raise SystemExit(r.returncode)
state=json.load(open(card));state["status"]="COMPLETE_METADATAFIX_VALIDATE_WORKLOAD_ONLY_RECEIPT_UNVERIFIED";state["receipt_sha256"]=sha(receipt)
tmp=card+".tmp"
with open(tmp,"w",encoding="utf-8") as f:json.dump(state,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,card)
PY
[[ -s "$RECEIPT" && ! -L "$RECEIPT" ]] || { echo "missing receipt"; exit 68; }
SUCCESS=1
printf '%s\n' "$RECEIPT"

#!/usr/bin/env bash
# CPU-only parser receipt for the exact metadatafix binary and isolated qualification manifest.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
BIN="$ROOT/bin/GTS_safe_c2_speculative_fallback_v4_metadatafix_v1"
MANIFEST="$ROOT/inputs/canary_workload_v1/workload_manifest.json"
IDS="$ROOT/inputs/canary_workload_v1/qualification_canary.ids"
LAUNCHER="$ROOT/tools/run_c2_v4_metadatafix_canary_preledger_gate_v2.sh"
PARENT="$ROOT/preledger_receipts/c2_v4_metadatafix_canary_parser_gate_v2"
RECEIPT="$PARENT/receipt.json"
LOG="$PARENT/launcher.log"
CARD="$PARENT/launcher_card.json"
SUCCESS=0
on_exit(){ rc=$?;trap - EXIT;if [[ "$SUCCESS" != 1 && -f "$CARD" ]];then set +e;python3 - "$CARD" "$rc" <<'PY'
import json,os,sys
p,rc=sys.argv[1:];x=json.load(open(p));x["status"]="FAILED_METADATAFIX_CANARY_VALIDATE_ONLY";x["failure_reason"]="outer_launcher_exit_"+rc
t=p+".tmp"
with open(t,"w",encoding="utf-8") as f:json.dump(x,f,sort_keys=True,indent=2);f.write("\n")
os.replace(t,p)
PY
set -e;fi;exit "$rc";}
trap on_exit EXIT
for p in "$BIN" "$MANIFEST" "$IDS" "$LAUNCHER";do [[ -f "$p" && ! -L "$p" ]]||{ echo "missing/symlink $p";exit 65;};done
for p in "$PARENT" "$RECEIPT" "$ROOT/runs" "$ROOT/.locks" "$ROOT/formal_runs_v2";do [[ ! -e "$p" && ! -L "$p" ]]||{ echo "existing prohibited state $p";exit 66;};done
[[ "$(sha256sum "$MANIFEST"|awk '{print $1}')" == "e9e6f22a42cc15cad4154b31c134a59c73768674987e096a943ec824901f5693" ]] || { echo "canary manifest hash";exit 67;}
[[ "$(sha256sum "$IDS"|awk '{print $1}')" == "12975b282b19cc530ba1951b7c58f2efcfe447d2520e707eb0a5d409b88239c0" ]] || { echo "canary ids hash";exit 68;}
mkdir "$PARENT"
python3 - "$BIN" "$MANIFEST" "$IDS" "$RECEIPT" "$LOG" "$CARD" "$LAUNCHER" <<'PY'
import hashlib,json,os,subprocess,sys
b,m,ids,r,l,c,launcher=sys.argv[1:]
def sha(p):return hashlib.sha256(open(p,"rb").read()).hexdigest()
x=json.load(open(m));card={"schema":"safe-c2-v4-metadatafix-canary-preledger-launcher-card-v2","status":"RUNNING_VALIDATE_WORKLOAD_ONLY_NO_CUDA_OR_DATASET_LOAD","scope":"isolated qualification manifest parser binding only; no dataset decode/CUDA/telemetry/stage output/formal ledger","binary":{"path":b,"sha256":sha(b)},"manifest":{"path":m,"sha256":sha(m)},"ids":{"path":ids,"sha256":sha(ids)},"receipt_path":r,"launcher":{"path":launcher,"sha256":sha(launcher)},"gpu_binary_executed":False,"nvidia_smi_called":False}
with open(c,"x",encoding="utf-8") as f:json.dump(card,f,sort_keys=True,indent=2);f.write("\n")
argv=[b,"--validate-workload-only","--mode","calibrate","--base-fvecs",x["base_fvecs_path"],"--query-fvecs",x["query_fvecs_path"],"--groundtruth-ivecs",x["groundtruth_ivecs_path"],"--query-ids",ids,"--stage","calibration","--workload-manifest",m,"--validation-receipt",r]
with open(l,"x",encoding="utf-8") as f:f.write("scope=validate-only canary; no CUDA/dataset/ledger\nargv_json="+json.dumps(argv)+"\n");f.flush();rc=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT).returncode
if rc:raise SystemExit(rc)
card=json.load(open(c));card["status"]="COMPLETE_METADATAFIX_CANARY_VALIDATE_ONLY_RECEIPT_UNVERIFIED";card["receipt_sha256"]=sha(r)
t=c+".tmp"
with open(t,"w",encoding="utf-8") as f:json.dump(card,f,sort_keys=True,indent=2);f.write("\n")
os.replace(t,c)
PY
[[ -s "$RECEIPT" && ! -L "$RECEIPT" ]] || { echo "receipt missing";exit 69;}
SUCCESS=1
printf '%s\n' "$RECEIPT"

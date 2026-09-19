#!/usr/bin/env bash
# Compilation only. It does not call nvidia-smi or execute the CUDA binary.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
NVCC=/usr/local/cuda-13.1/bin/nvcc
SRC="$ROOT/src/aabb_device_canary_v1.cu"
AUDIT="$ROOT/tools/audit_device_canary_source_v1.py"
BIN="$ROOT/bin/aabb_device_canary_v1"
LOG="$ROOT/provenance/device_canary_build_v1.log"
CARD="$ROOT/provenance/device_canary_build_v1.json"
for p in "$NVCC" "$SRC" "$AUDIT"; do [[ -f "$p" && ! -L "$p" ]] || exit 65; done
for p in "$BIN" "$LOG" "$CARD"; do [[ ! -e "$p" && ! -L "$p" ]] || exit 66; done
AUDIT_JSON="$("$AUDIT")"
TMP=/tmp/c2_static_aabb_device_canary.$$
trap 'rm -f "$TMP"' EXIT
{
 echo "started_utc=$(date -u -Is)"
 echo "scope=NVCC compilation only; no CUDA binary execution; no GPU management command"
 echo "compiler=$($NVCC --version | tail -n1)"
 "$NVCC" -std=c++17 -O3 --ftz=false --generate-code=arch=compute_120,code=[compute_120,sm_120] \
   -I"$ROOT/include" "$SRC" -o "$TMP"
 mv "$TMP" "$BIN"; chmod 0750 "$BIN"
 echo "binary_sha256=$(sha256sum "$BIN"|awk '{print $1}')"
 echo "finished_utc=$(date -u -Is)"
} >"$LOG" 2>&1
python3 - "$CARD" "$ROOT" "$SRC" "$AUDIT" "$BIN" "$LOG" "$AUDIT_JSON" <<'PY'
import hashlib,json,os,pathlib,sys
card,root,src,audit,binary,log=map(pathlib.Path,sys.argv[1:7]); audit_result=json.loads(sys.argv[7])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
x={'schema':'safe-c2-static-aabb-device-canary-build-v1','status':'COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION',
   'scope':'NVCC compilation only of static AABB device certificate canary; no workload/GPU execution',
   'source':{'path':str(src),'sha256':sha(src)},'audit':{'path':str(audit),'sha256':sha(audit),'result':audit_result},
   'binary':{'path':str(binary),'sha256':sha(binary),'bytes':binary.stat().st_size},
   'log':{'path':str(log),'sha256':sha(log)},'compile_flags':['-O3','--ftz=false','compute_120/sm_120'],
   'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False}
fd=os.open(card,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f: json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
PY

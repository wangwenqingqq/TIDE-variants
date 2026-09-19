#!/usr/bin/env bash
# Compilation only; no GPU execution.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
NVCC=/usr/local/cuda-13.1/bin/nvcc
SRC="$ROOT/src/static_aabb_perf_probe_v1.cu"
AUDIT="$ROOT/tools/audit_perf_probe_source_v1.py"
BIN="$ROOT/bin/static_aabb_perf_probe_v1"
LOG="$ROOT/provenance/perf_probe_build_v1.log"
CARD="$ROOT/provenance/perf_probe_build_v1.json"
for p in "$NVCC" "$SRC" "$AUDIT";do [[ -f "$p" && ! -L "$p" ]]||exit 65;done
for p in "$BIN" "$LOG" "$CARD";do [[ ! -e "$p" && ! -L "$p" ]]||exit 66;done
A="$("$AUDIT")";T=/tmp/static_aabb_perf_probe.$$
trap 'rm -f "$T"' EXIT
{
 echo "started_utc=$(date -u -Is)";echo "scope=compile only; no CUDA binary execution; no GPU management"
 echo "compiler=$($NVCC --version|tail -n1)"
 "$NVCC" -std=c++17 -O3 --ftz=false --generate-code=arch=compute_120,code=[compute_120,sm_120] -I"$ROOT/include" "$SRC" -o "$T"
 mv "$T" "$BIN";chmod 0750 "$BIN";echo "binary_sha256=$(sha256sum "$BIN"|awk '{print $1}')";echo "finished_utc=$(date -u -Is)"
} >"$LOG" 2>&1
python3 - "$CARD" "$SRC" "$AUDIT" "$BIN" "$LOG" "$A" <<'PY'
import hashlib,json,os,pathlib,sys
card,src,audit,binary,log=map(pathlib.Path,sys.argv[1:6]);a=json.loads(sys.argv[6])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
x={'schema':'safe-c2-static-aabb-perf-probe-build-v1','status':'COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION','scope':'NVCC compilation only; no workload/GPU execution','source':{'path':str(src),'sha256':sha(src)},'audit':{'path':str(audit),'sha256':sha(audit),'result':a},'binary':{'path':str(binary),'sha256':sha(binary),'bytes':binary.stat().st_size},'log':{'path':str(log),'sha256':sha(log)},'compile_flags':['-O3','--ftz=false','compute_120/sm_120'],'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False}
fd=os.open(card,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
PY

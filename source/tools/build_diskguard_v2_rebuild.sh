#!/usr/bin/env bash
# Fresh compilation-only rebuild after discarding incomplete v2 build card.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1
NVCC=/usr/local/cuda-13.1/bin/nvcc
AUDIT="$ROOT/tools/audit_diskguard_v2_sources.py"
TOPSRC="$ROOT/src/static_aabb_topk_canary_v2_diskguard.cu"; PERFSRC="$ROOT/src/static_aabb_perf_probe_v2_diskguard.cu"
TOPBIN="$ROOT/bin/static_aabb_topk_canary_v2_diskguard"; PERFBIN="$ROOT/bin/static_aabb_perf_probe_v2_diskguard"
LOG="$ROOT/provenance/diskguard_v2_rebuild.log";CARD="$ROOT/provenance/diskguard_v2_rebuild.json"
for p in "$NVCC" "$AUDIT" "$TOPSRC" "$PERFSRC";do [[ -f "$p" && ! -L "$p" ]]||exit 65;done
for p in "$TOPBIN" "$PERFBIN" "$LOG" "$CARD";do [[ ! -e "$p" && ! -L "$p" ]]||exit 66;done
A="$("$AUDIT")";T1=/tmp/aabb_topk_v2_rebuild.$$;T2=/tmp/aabb_perf_v2_rebuild.$$
trap 'rm -f "$T1" "$T2"' EXIT
{
 echo "started_utc=$(date -u -Is)"
 echo "scope=NVCC compilation only; no CUDA binary execution; no GPU management"
 echo "compiler=$($NVCC --version|tail -n1)"
 "$NVCC" -std=c++17 -O3 --ftz=false --generate-code=arch=compute_120,code=[compute_120,sm_120] -I"$ROOT/include" "$TOPSRC" -o "$T1"
 "$NVCC" -std=c++17 -O3 --ftz=false --generate-code=arch=compute_120,code=[compute_120,sm_120] -I"$ROOT/include" "$PERFSRC" -o "$T2"
 mv "$T1" "$TOPBIN";mv "$T2" "$PERFBIN";chmod 0750 "$TOPBIN" "$PERFBIN"
 echo "topk_sha256=$(sha256sum "$TOPBIN"|awk '{print $1}')"
 echo "perf_sha256=$(sha256sum "$PERFBIN"|awk '{print $1}')"
 echo "finished_utc=$(date -u -Is)"
} >"$LOG" 2>&1
python3 - "$CARD" "$TOPSRC" "$PERFSRC" "$AUDIT" "$TOPBIN" "$PERFBIN" "$LOG" "$A" <<'PY'
import hashlib,json,pathlib,sys
card,ts,ps,audit,tb,pb,log=(pathlib.Path(x) for x in sys.argv[1:8])
ar=json.loads(sys.argv[8])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
x={'schema':'safe-c2-static-aabb-diskguard-v2-rebuild','status':'COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION','scope':'fresh NVCC rebuild only; no workload/GPU execution','topk_source':{'path':str(ts),'sha256':sha(ts)},'perf_source':{'path':str(ps),'sha256':sha(ps)},'audit':{'path':str(audit),'sha256':sha(audit),'result':ar},'topk_binary':{'path':str(tb),'sha256':sha(tb),'bytes':tb.stat().st_size},'perf_binary':{'path':str(pb),'sha256':sha(pb),'bytes':pb.stat().st_size},'log':{'path':str(log),'sha256':sha(log)},'compile_flags':['-O3','--ftz=false','compute_120/sm_120'],'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False}
card.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
PY

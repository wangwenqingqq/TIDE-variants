#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1;NVCC=/usr/local/cuda-13.1/bin/nvcc
AUDIT="$ROOT/tools/audit_diskguard_v2_1_sources.py";TS="$ROOT/src/static_aabb_topk_canary_v2_1_diskguard.cu";PS="$ROOT/src/static_aabb_perf_probe_v2_1_diskguard.cu";TB="$ROOT/bin/static_aabb_topk_canary_v2_1_diskguard";PB="$ROOT/bin/static_aabb_perf_probe_v2_1_diskguard";L="$ROOT/provenance/diskguard_v2_1_build.log";C="$ROOT/provenance/diskguard_v2_1_build.json"
for p in "$NVCC" "$AUDIT" "$TS" "$PS";do [[ -f "$p" && ! -L "$p" ]]||exit 65;done
for p in "$TB" "$PB" "$L" "$C";do [[ ! -e "$p" && ! -L "$p" ]]||exit 66;done
A="$("$AUDIT")";T=/tmp/aabb_v21_topk.$$;U=/tmp/aabb_v21_perf.$$;trap 'rm -f "$T" "$U"' EXIT
{
 echo "started_utc=$(date -u -Is)";echo "scope=NVCC compile only; no GPU execution"
 "$NVCC" -std=c++17 -O3 --ftz=false --generate-code=arch=compute_120,code=[compute_120,sm_120] -I"$ROOT/include" "$TS" -o "$T"
 "$NVCC" -std=c++17 -O3 --ftz=false --generate-code=arch=compute_120,code=[compute_120,sm_120] -I"$ROOT/include" "$PS" -o "$U"
 mv "$T" "$TB";mv "$U" "$PB";chmod 0750 "$TB" "$PB";echo "finished_utc=$(date -u -Is)"
} >"$L" 2>&1
python3 - "$C" "$TS" "$PS" "$AUDIT" "$TB" "$PB" "$L" "$A" <<'PY'
import hashlib,json,pathlib,sys
c,ts,ps,a,tb,pb,l=(pathlib.Path(x) for x in sys.argv[1:8]);ar=json.loads(sys.argv[8])
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):x.update(z)
 return x.hexdigest()
c.write_text(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2.1-build','status':'COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION','scope':'schema-fixed successor compile only; no GPU execution','topk_source':{'path':str(ts),'sha256':h(ts)},'perf_source':{'path':str(ps),'sha256':h(ps)},'audit':{'path':str(a),'sha256':h(a),'result':ar},'topk_binary':{'path':str(tb),'sha256':h(tb),'bytes':tb.stat().st_size},'perf_binary':{'path':str(pb),'sha256':h(pb),'bytes':pb.stat().st_size},'log':{'path':str(l),'sha256':h(l)},'compile_flags':['-O3','--ftz=false','compute_120/sm_120'],'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False},sort_keys=True,indent=2)+'\n')
PY

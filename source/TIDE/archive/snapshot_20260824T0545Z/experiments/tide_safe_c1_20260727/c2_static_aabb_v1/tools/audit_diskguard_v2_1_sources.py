#!/usr/bin/env python3
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
bound=root/'include/aabb_static_bound_v2_diskguard.cuh';hdr=root/'include/search_static_aabb_v2_diskguard.cuh';top=root/'src/static_aabb_topk_canary_v2_1_diskguard.cu';perf=root/'src/static_aabb_perf_probe_v2_1_diskguard.cu';contract=root/'provenance/sift_integer_disk_contract_v1.json'
for p in (bound,hdr,top,perf,contract):
 if not p.is_file() or p.is_symlink():raise SystemExit('invalid '+str(p))
b=bound.read_text();h=hdr.read_text()
for x in ['static_aabb_next_up_nonnegative','disk_upper','__float_as_uint','__uint_as_float','__fmul_ru']:
 if x not in b:raise SystemExit('missing '+x)
start=h.find('__global__ void nodeProcessKnnStaticAabb(');tail=h[start:]
if start<0 or 'searchIndexKnnStaticAabbV2DiskGuard' not in tail or 'static_aabb_strict_prune' not in tail:raise SystemExit('v2 candidate route')
for bad in ['rp_predict(','recordGammaOnlyPrune','GammaOnlyPruneTrace']:
 if bad in tail:raise SystemExit('legacy in candidate '+bad)
for p,status in ((top,'PASS_STATIC_AABB_TOPK_CANARY_V2_DISKGUARD'),(perf,'PASS_EXPLORATORY_PAIRED_PROBE_V2_DISKGUARD')):
 s=p.read_text()
 if status not in s or status+'_V2_DISKGUARD' in s:raise SystemExit('bad schema status '+str(p))
 if 'searchIndexKnnStaticAabbV2DiskGuard' not in s:raise SystemExit('not wired '+str(p))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
print(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2.1-source-audit','status':'PASS','bound_sha256':sha(bound),'header_sha256':sha(hdr),'topk_source_sha256':sha(top),'perf_source_sha256':sha(perf),'contract_sha256':sha(contract),'scope':'CPU source audit only; no CUDA execution'},sort_keys=True))

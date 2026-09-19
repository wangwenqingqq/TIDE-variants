#!/usr/bin/env python3
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
bound=root/'include/aabb_static_bound_v2_diskguard.cuh';hdr=root/'include/search_static_aabb_v2_diskguard.cuh'
top=root/'src/static_aabb_topk_canary_v2_diskguard.cu';perf=root/'src/static_aabb_perf_probe_v2_diskguard.cu';contract=root/'provenance/sift_integer_disk_contract_v1.json'
for p in (bound,hdr,top,perf,contract):
 if not p.is_file() or p.is_symlink():raise SystemExit('invalid '+str(p))
b=bound.read_text();h=hdr.read_text()
for x in ['static_aabb_next_up_nonnegative','__float_as_uint','__uint_as_float','disk_upper','__fmul_ru']:
 if x not in b:raise SystemExit('missing bound guard '+x)
start=h.find('__global__ void nodeProcessKnnStaticAabb(')
if start<0:raise SystemExit('missing kernel')
tail=h[start:]
for x in ['static_aabb_strict_prune','searchIndexKnnStaticAabbV2DiskGuard']:
 if x not in tail:raise SystemExit('missing candidate '+x)
for bad in ['rp_predict(','recordGammaOnlyPrune','GammaOnlyPruneTrace']:
 if bad in tail:raise SystemExit('legacy leakage '+bad)
for p in (top,perf):
 s=p.read_text().lower()
 if 'search_static_aabb_v2_diskguard.cuh' not in s or 'searchindexknnstaticaabbv2diskguard' not in s:raise SystemExit('v2 runner not wired')
 for bad in ['/c2_speculative_fallback_v4/inputs','searchindexknnv3speculative','upload_gamma','incremental_insert']:
  if bad in s:raise SystemExit('forbidden '+bad)
def sha(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):x.update(z)
 return x.hexdigest()
print(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2-source-audit','status':'PASS','bound_sha256':sha(bound),'header_sha256':sha(hdr),'topk_source_sha256':sha(top),'perf_source_sha256':sha(perf),'sift_integer_contract_sha256':sha(contract),'scope':'CPU source/contract audit only; no CUDA execution'},sort_keys=True))

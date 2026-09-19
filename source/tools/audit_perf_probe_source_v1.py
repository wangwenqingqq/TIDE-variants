#!/usr/bin/env python3
import hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
src=root/'src/static_aabb_perf_probe_v1.cu'; hdr=root/'include/search_static_aabb_v1.cuh'
for p in (src,hdr):
 if not p.is_file() or p.is_symlink():raise SystemExit('invalid '+str(p))
s=src.read_text().lower()
for x in ['read_gt10','evaluate_gt','timed_baseline_probe','timed_aabb_probe','same_snapshot','warmups_per_method','exploratory paired gpu timing']:
 if x.lower() not in s:raise SystemExit('missing '+x)
for bad in ['/c2_speculative_fallback_v4/inputs','searchindexknnv3speculative','upload_gamma','incremental_insert']:
 if bad in s:raise SystemExit('forbidden '+bad)
start=hdr.read_text().find('__global__ void nodeProcessKnnStaticAabb(')
if start<0:raise SystemExit('missing static kernel')
tail=hdr.read_text()[start:]
for x in ['static_aabb_strict_prune','searchIndexKnnStaticAabbV1']:
 if x not in tail:raise SystemExit('missing candidate '+x)
for bad in ['rp_predict(','recordGammaOnlyPrune','GammaOnlyPruneTrace']:
 if bad in tail:raise SystemExit('legacy decision in static kernel')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
print(json.dumps({'schema':'safe-c2-static-aabb-perf-probe-source-audit-v1','status':'PASS','source_sha256':sha(src),'header_sha256':sha(hdr),'scope':'static source audit only; no CUDA execution'},sort_keys=True))

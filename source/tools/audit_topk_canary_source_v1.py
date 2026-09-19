#!/usr/bin/env python3
import hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
src=root/'src/static_aabb_topk_canary_v1.cu'
hdr=root/'include/search_static_aabb_v1.cuh'
for p in (src,hdr):
 if not p.is_file() or p.is_symlink(): raise SystemExit(f'invalid source {p}')
s=src.read_text().lower(); h=hdr.read_text()
for x in ['repair_radial_intervals','build_boxes','exact_oracle','searchindexknnstaticaabbv1','same_snapshot','staticprojectedaabbdeviceview','no timing, update, calibration, heldout, validation, sealed']:
 if x.lower() not in s: raise SystemExit(f'missing source control {x}')
for bad in ['/c2_speculative_fallback_v4/inputs','searchindexknnv3speculative','upload_gamma','incremental_insert']:
 if bad in s: raise SystemExit(f'forbidden runner API/path {bad}')
start=h.find('__global__ void nodeProcessKnnStaticAabb(')
entry_start=h.find('void searchIndexKnnStaticAabbV1(')
if start<0 or entry_start<0: raise SystemExit('missing static AABB kernel/entry')
entry=h[start:]
for x in ['nodeProcessKnnStaticAabb','static_aabb_strict_prune','radial_lb <= disk_at_decision','searchIndexKnnStaticAabbV1']:
 if x not in entry: raise SystemExit(f'missing candidate control {x}')
# The appended section starts after all legacy APIs, so none may appear in it.
for bad in ['rp_predict(','recordGammaOnlyPrune','GammaOnlyPruneTrace']:
 if bad in entry: raise SystemExit(f'legacy decision leaked into static entry {bad}')
def sha(p):
 z=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):z.update(b)
 return z.hexdigest()
print(json.dumps({'schema':'safe-c2-static-aabb-topk-canary-source-audit-v1','status':'PASS','source_sha256':sha(src),'search_header_sha256':sha(hdr),'scope':'static source audit only; no CUDA execution'},sort_keys=True))

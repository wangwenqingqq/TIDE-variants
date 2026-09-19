#!/usr/bin/env python3
import hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
src=root/'src/aabb_device_canary_v1.cu'
hdr=root/'include/aabb_static_bound_v1.cuh'
tree=root/'include/tree.cuh'
for p in (src,hdr,tree):
 if not p.is_file() or p.is_symlink(): raise SystemExit(f'invalid source input: {p}')
s=src.read_text().lower()
for required in ['build_and_verify_boxes','coordinate_cover_violations','static_aabb_lb_sq_rd',
                 'top_variance_dims(base,args.projection_dims)','memcmp','aabb_bound_canary_kernel',
                 'no traversal, top-k, timing, update, calibration, validation, or sealed workload']:
 if required.lower() not in s: raise SystemExit(f'missing required canary control: {required}')
for forbidden in ['searchindexknn','rp_predict','upload_rp_constants','gammaonlyprune','incremental_insert']:
 if forbidden in s: raise SystemExit(f'forbidden legacy API in canary: {forbidden}')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
print(json.dumps({'schema':'safe-c2-static-aabb-device-canary-source-audit-v1','status':'PASS',
                  'source_sha256':sha(src),'bound_header_sha256':sha(hdr),'tree_copy_sha256':sha(tree),
                  'scope':'static source audit only; no CUDA binary execution'},sort_keys=True))

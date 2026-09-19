#!/usr/bin/env python3
import hashlib, json, pathlib, sys
root=pathlib.Path(__file__).resolve().parents[1]
hdr=root/'include/aabb_static_bound_v1.cuh'
src=root/'src/cuda_directed_rounding_compile_v1.cu'
for p in (hdr,src):
    if not p.is_file() or p.is_symlink(): raise SystemExit(f"invalid input: {p}")
text=hdr.read_text()
required=['__fsub_rd','__fmul_rd','__fadd_rd','__fmul_ru','static_aabb_lb_sq_rd','static_aabb_strict_prune','> disk_sq_up']
for x in required:
    if x not in text: raise SystemExit(f'missing {x}')
# Policy words may occur in comments. Reject the legacy executable APIs/types instead.
for forbidden in ['rp_predict(', 'upload_rp_constants', 'gammaonlyprunetrace', 'recordgammaonlyprune']:
    if forbidden in text.lower(): raise SystemExit(f'forbidden legacy executable API in static AABB header: {forbidden}')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
print(json.dumps({"schema":"safe-c2-static-aabb-directed-rounding-source-audit-v1",
                  "status":"PASS",
                  "header_sha256":sha(hdr),"probe_source_sha256":sha(src),
                  "scope":"static source audit only; no CUDA binary execution"},
                 sort_keys=True))

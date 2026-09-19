#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs')
queries=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs')
src=root/'src/aabb_device_canary_v1.cu'
bin=root/'bin/aabb_device_canary_v1'
audit=root/'tools/audit_device_canary_source_v1.py'
card=root/'provenance/device_canary_build_v1.json'
for p in (root,base,queries,src,bin,audit,card):
 if not p.is_file() if p!=root else not p.is_dir(): raise SystemExit(f'missing {p}')
 if p.is_symlink(): raise SystemExit(f'symlink forbidden {p}')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
build=json.loads(card.read_text())
if build.get('status')!='COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION': raise SystemExit('unexpected build status')
if build['source']['sha256']!=sha(src) or build['binary']['sha256']!=sha(bin): raise SystemExit('source/binary build card mismatch')
audit_out=json.loads(subprocess.check_output([str(audit)],text=True))
if audit_out.get('status')!='PASS': raise SystemExit('static source audit did not pass')
source_text=src.read_text().lower()
# Do not reject policy words in explanatory comments. Reject actual legacy paths/APIs.
for bad in ['/c2_speculative_fallback_v4/inputs','searchindexknn','rp_predict','incremental_insert','upload_rp_constants']:
 if bad in source_text: raise SystemExit(f'forbidden source token {bad}')
# CPU-only fvec header checks for exactly the rows this canary will read.
def fvec_header(p,row):
 with p.open('rb') as f:
  f.seek(row*(4+128*4))
  x=f.read(4)
  if len(x)!=4: raise SystemExit(f'short fvec at row {row}')
  return int.from_bytes(x,'little',signed=True)
if fvec_header(base,0)!=128 or fvec_header(base,999999)!=128: raise SystemExit('base fvec header failure')
for row in range(8):
 if fvec_header(queries,row)!=128: raise SystemExit(f'query fvec header failure at {row}')
print(json.dumps({'schema':'safe-c2-static-aabb-device-canary-preflight-v1','status':'PASS_CPU_ONLY',
 'scope':'new static-AABB CUDA canary only; standard SIFT rows 0..7; no old C2 validation/sealed access',
 'base':{'path':str(base),'sha256':sha(base),'bytes':base.stat().st_size},
 'queries':{'path':str(queries),'sha256':sha(queries),'bytes':queries.stat().st_size,'rows':[0,7]},
 'source_sha256':sha(src),'binary_sha256':sha(bin),'source_audit_sha256':sha(audit),
 'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False},sort_keys=True))

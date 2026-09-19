#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs')
queries=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs')
ids=root/'inputs/standard_sift_tiefree_smoke32_v1.ids'
selection=root/'inputs/standard_sift_tiefree_smoke32_v1.json'
src=root/'src/static_aabb_topk_canary_v1.cu'
hdr=root/'include/search_static_aabb_v1.cuh'
binary=root/'bin/static_aabb_topk_canary_v1'
audit=root/'tools/audit_topk_canary_source_v1.py'
card=root/'provenance/topk_canary_build_v1.json'
for p in (base,queries,ids,selection,src,hdr,binary,audit,card):
 if not p.is_file() or p.is_symlink(): raise SystemExit(f'invalid/missing input {p}')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
build=json.loads(card.read_text())
if build.get('status')!='COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION': raise SystemExit('invalid build status')
if build['source']['sha256']!=sha(src) or build['binary']['sha256']!=sha(binary): raise SystemExit('build hash mismatch')
a=json.loads(subprocess.check_output([str(audit)],text=True))
if a.get('status')!='PASS': raise SystemExit('source audit fail')
sel=json.loads(selection.read_text())
ids_list=[int(x) for x in ids.read_text().split()]
if sel.get('output_ids_sha256')!=sha(ids) or ids_list!=sel.get('selected_ids') or ids_list!=list(range(32)): raise SystemExit('selection witness mismatch')
s=src.read_text().lower()
for bad in ['/c2_speculative_fallback_v4/inputs','searchindexknnv3speculative','upload_gamma','incremental_insert']:
 if bad in s: raise SystemExit(f'forbidden runner token {bad}')
print(json.dumps({'schema':'safe-c2-static-aabb-topk-canary-preflight-v1','status':'PASS_CPU_ONLY',
 'scope':'new standard-SIFT tie-free smoke IDs 0..31; static AABB versus CPU exact oracle; no old C2 validation/sealed access',
 'base':{'path':str(base),'sha256':sha(base),'bytes':base.stat().st_size},
 'queries':{'path':str(queries),'sha256':sha(queries),'bytes':queries.stat().st_size},
 'ids':{'path':str(ids),'sha256':sha(ids),'values':ids_list},
 'selection_sha256':sha(selection),'source_sha256':sha(src),'search_header_sha256':sha(hdr),'binary_sha256':sha(binary),
 'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False},sort_keys=True))

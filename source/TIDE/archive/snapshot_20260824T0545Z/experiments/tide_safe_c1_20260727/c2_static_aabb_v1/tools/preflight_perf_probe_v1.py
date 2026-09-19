#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs')
queries=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs')
gt=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs')
ids=root/'inputs/standard_sift_tiefree_perf256_v1.ids'; sel=root/'inputs/standard_sift_tiefree_perf256_v1.json'
src=root/'src/static_aabb_perf_probe_v1.cu'; hdr=root/'include/search_static_aabb_v1.cuh'; binary=root/'bin/static_aabb_perf_probe_v1'; audit=root/'tools/audit_perf_probe_source_v1.py'; card=root/'provenance/perf_probe_build_v1.json'
for p in (base,queries,gt,ids,sel,src,hdr,binary,audit,card):
 if not p.is_file() or p.is_symlink():raise SystemExit('invalid '+str(p))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
b=json.loads(card.read_text())
if b.get('status')!='COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION' or b['source']['sha256']!=sha(src) or b['binary']['sha256']!=sha(binary):raise SystemExit('build contract mismatch')
if json.loads(subprocess.check_output([str(audit)],text=True)).get('status')!='PASS':raise SystemExit('source audit fail')
x=json.loads(sel.read_text()); idsval=[int(i) for i in ids.read_text().split()]
if len(idsval)!=256 or idsval!=x.get('selected_ids') or x.get('output_ids_sha256')!=sha(ids):raise SystemExit('selection contract mismatch')
if idsval[0]!=1000 or idsval[-1]!=1259:raise SystemExit('unexpected fixed probe workload')
s=src.read_text().lower()
for bad in ['/c2_speculative_fallback_v4/inputs','searchindexknnv3speculative','upload_gamma','incremental_insert']:
 if bad in s:raise SystemExit('forbidden '+bad)
print(json.dumps({'schema':'safe-c2-static-aabb-paired-perf-preflight-v1','status':'PASS_CPU_ONLY','scope':'fixed new standard-SIFT nonformal paired probe IDs 1000..1259; no old C2 validation/sealed access','base':{'sha256':sha(base),'bytes':base.stat().st_size},'queries':{'sha256':sha(queries),'bytes':queries.stat().st_size},'groundtruth':{'sha256':sha(gt),'bytes':gt.stat().st_size},'ids':{'sha256':sha(ids),'count':len(idsval),'first':idsval[0],'last':idsval[-1]},'selection_sha256':sha(sel),'source_sha256':sha(src),'header_sha256':sha(hdr),'binary_sha256':sha(binary),'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False},sort_keys=True))

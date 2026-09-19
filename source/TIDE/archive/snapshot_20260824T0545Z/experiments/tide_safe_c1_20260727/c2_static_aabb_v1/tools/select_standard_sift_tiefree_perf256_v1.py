#!/usr/bin/env python3
# CPU-only fixed nonformal query selection for a paired static-AABB timing probe.
import hashlib,json,pathlib,struct
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs')
query=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs')
gt=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs')
idsout=root/'inputs/standard_sift_tiefree_perf256_v1.ids'
jsonout=root/'inputs/standard_sift_tiefree_perf256_v1.json'
for p in (base,query,gt):
 if not p.is_file() or p.is_symlink(): raise SystemExit('invalid input '+str(p))
for p in (idsout,jsonout):
 if p.exists() or p.is_symlink(): raise SystemExit('preexisting output '+str(p))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p,row,width):
 with p.open('rb') as f:
  f.seek(row*(4+width*4)); d=struct.unpack('<i',f.read(4))[0]
  if d!=width: raise RuntimeError('bad dim')
  return struct.unpack('<'+str(width)+('f' if width==128 else 'i'),f.read(width*4))
selected=[]; witnesses=[]
# The selection range begins after the correctness canary IDs and is independent of AABB behavior.
for qid in range(1000,10000):
 q=read(query,qid,128); nn=read(gt,qid,100); ds=[]
 for bid in nn[:11]:
  x=read(base,bid,128); ds.append(sum((a-b)*(a-b) for a,b in zip(q,x)))
 if ds[9]<ds[10]:
  selected.append(qid); witnesses.append({'qid':qid,'d10_sq':ds[9],'d11_sq':ds[10]})
  if len(selected)==256: break
if len(selected)!=256: raise SystemExit('short selection')
idsout.write_text(''.join(str(x)+'\n' for x in selected))
x={'schema':'safe-c2-static-aabb-standard-sift-tiefree-perf-selection-v1','status':'NONFORMAL_EXPLORATORY_ONLY',
 'scope':'fixed standard-SIFT IDs for an exploratory paired timing probe; not calibration, heldout, validation, sealed, or formal paper evidence',
 'selection_policy':'scan query IDs 1000 upward; take first 256 with recomputed GT d10_sq < d11_sq; selection does not inspect AABB, traversal, or timing',
 'selected_ids':selected,'boundary_witness':witnesses,'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (base,query,gt)},'output_ids_sha256':sha(idsout),'gpu_executed':False,'formal_claim_eligible':False}
jsonout.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
print(jsonout)

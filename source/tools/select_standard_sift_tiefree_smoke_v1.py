#!/usr/bin/env python3
# CPU-only selection for a nonformal static-AABB CUDA canary.
import hashlib,json,math,os,pathlib,struct,sys
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs')
query=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs')
gt=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs')
out_ids=root/'inputs/standard_sift_tiefree_smoke32_v1.ids'
out_json=root/'inputs/standard_sift_tiefree_smoke32_v1.json'
for p in (base,query,gt):
 if not p.is_file() or p.is_symlink(): raise SystemExit(f'invalid input {p}')
for p in (out_ids,out_json):
 if p.exists() or p.is_symlink(): raise SystemExit(f'output already exists {p}')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def fvec_row(p,row):
 with p.open('rb') as f:
  f.seek(row*(4+128*4)); d=struct.unpack('<i',f.read(4))[0]
  if d!=128: raise ValueError('bad fvec dim')
  return struct.unpack('<128f',f.read(128*4))
def ivec_row(p,row):
 with p.open('rb') as f:
  f.seek(row*(4+100*4)); d=struct.unpack('<i',f.read(4))[0]
  if d!=100: raise ValueError('bad ivec dim')
  return struct.unpack('<100i',f.read(100*4))
def base_row(p,row):
 with p.open('rb') as f:
  f.seek(row*(4+128*4)); d=struct.unpack('<i',f.read(4))[0]
  if d!=128: raise ValueError('bad base dim')
  return struct.unpack('<128f',f.read(128*4))
# Only the 100 GT-referenced base vectors per candidate query are read.
selected=[]; details=[]
for qid in range(10000):
 q=fvec_row(query,qid); ids=ivec_row(gt,qid)
 ds=[]
 for bid in ids[:11]:
  x=base_row(base,bid)
  ds.append(sum((float(a)-float(b))**2 for a,b in zip(q,x)))
 if ds[9] < ds[10]:
  selected.append(qid); details.append({'qid':qid,'d10_sq':ds[9],'d11_sq':ds[10]})
  if len(selected)==32: break
if len(selected)!=32: raise SystemExit('could not select 32 boundary-tie-free IDs')
out_ids.write_text(''.join(f'{x}\n' for x in selected))
record={'schema':'safe-c2-static-aabb-standard-sift-tiefree-smoke-selection-v1','status':'NONFORMAL_SMOKE_ONLY',
 'scope':'CPU-only fixed first-32 boundary-tie-free selection for a static-AABB CUDA canary; not calibration, heldout, validation, sealed, or paper evidence',
 'selection_policy':'scan standard SIFT query IDs increasing; select first 32 where raw-float64 recomputed GT rank10 squared distance < rank11 squared distance',
 'selected_ids':selected,'boundary_witness':details,
 'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (base,query,gt)},
 'output_ids_sha256':None,'gpu_executed':False,'formal_claim_eligible':False}
record['output_ids_sha256']=sha(out_ids)
out_json.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print(out_json)

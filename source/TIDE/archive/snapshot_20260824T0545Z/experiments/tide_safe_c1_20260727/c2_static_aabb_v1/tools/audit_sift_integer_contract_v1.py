#!/usr/bin/env python3
# CPU-only audit for the SIFT-specific disk-rounding contract.
import hashlib,json,math,pathlib,struct
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
paths=[root/'inputs/../..']
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs')
query=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs')
out=root/'provenance/sift_integer_disk_contract_v1.json'
if out.exists() or out.is_symlink():raise SystemExit('preexisting output')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def scan(p,expected):
 n=0;bad=0;mn=float('inf');mx=-float('inf')
 with p.open('rb') as f:
  for i in range(expected):
   h=f.read(4)
   if len(h)!=4 or struct.unpack('<i',h)[0]!=128:raise SystemExit('bad fvec header '+str(i))
   raw=f.read(512)
   if len(raw)!=512:raise SystemExit('short fvec '+str(i))
   vals=struct.unpack('<128f',raw)
   for v in vals:
    n+=1;mn=min(mn,v);mx=max(mx,v)
    if not math.isfinite(v) or v<0 or v>255 or v!=math.floor(v):bad+=1
  if f.read(1):raise SystemExit('extra fvec payload')
 return {'vectors':expected,'coordinates':n,'min':mn,'max':mx,'integer_range_violations':bad}
b=scan(base,1_000_000);q=scan(query,10_000)
if b['integer_range_violations'] or q['integer_range_violations']:raise SystemExit('SIFT integer contract failed')
# For values 0..255, every float subtraction is an exact integer in [-255,255],
# every float square is an exact integer <=65025, and a 128-term sum is <2^24.
max_sq=128*255*255
x={'schema':'safe-c2-static-aabb-sift-integer-disk-contract-v1','status':'PASS_CPU_ONLY',
'scope':'SIFT1M-specific numerical contract for a one-ULP upward disk guard; not a general-float claim',
'base':dict(b,path=str(base),sha256=sha(base)),'query':dict(q,path=str(query),sha256=sha(query)),
'proof_preconditions':{'coordinate_values':'finite integers in [0,255]','max_abs_difference':255,'max_per_coordinate_squared_difference':255*255,'max_128d_squared_sum':max_sq,'max_128d_squared_sum_lt_2pow24':max_sq<2**24,'consequence':'subtraction, square, and float conversion of the 128d integer sum are exact; sqrtf is the only rounding step before disk storage'},
'gpu_executed':False,'formal_claim_eligible':False}
out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');print(out)

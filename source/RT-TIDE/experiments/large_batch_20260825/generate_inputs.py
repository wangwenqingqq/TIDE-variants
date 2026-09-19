#!/usr/bin/env python3
from pathlib import Path
import json, math, time
import numpy as np

W=Path('/workspace/RT-TIDE')
DS=W/'data/sift1m'
OUT=W/'experiments/large_batch_20260825/inputs'
OUT.mkdir(parents=True, exist_ok=True)
BASE_N=900_000
K=10
NQ=128

def read_vecs(path, dtype):
    raw=np.memmap(path,dtype=dtype,mode='r')
    dim=int(raw[:1].view(np.int32)[0])
    arr=raw.reshape(-1,dim+1)
    dims=arr[:,0].view(np.int32)
    if not np.all(dims==dim): raise RuntimeError(f'bad dims {path}')
    return arr[:,1:]

t0=time.time()
base=read_vecs(DS/'sift_base.fvecs',np.float32)
queries_all=read_vecs(DS/'sift_query.fvecs',np.float32)
gt=read_vecs(DS/'sift_groundtruth.ivecs',np.int32)
queries=np.asarray(queries_all[:NQ],dtype=np.float32)
train=np.asarray(base[:50_000],dtype=np.float64)
mean=train.mean(axis=0)
centered=train-mean
cov=centered.T@centered
evals,evecs=np.linalg.eigh(cov)
basis=evecs[:,np.argsort(evals)[::-1]].astype(np.float32)
mean32=mean.astype(np.float32)

# Chunked projection avoids a large temporary 1M x 128 float buffer.
points=np.memmap(OUT/'points1m_pca2.f32',dtype=np.float32,mode='w+',shape=(1_000_000,2))
for b in range(0,1_000_000,50_000):
    e=min(b+50_000,1_000_000)
    points[b:e]=(np.asarray(base[b:e],dtype=np.float32)-mean32)@basis[:,:2]
points.flush()
qcoords=(queries-mean32)@basis[:,:2]

taus=np.empty(NQ,dtype=np.float32)
seed_ids=[]
for qi in range(NQ):
    valid=np.asarray(gt[qi],dtype=np.int64)
    valid=valid[valid<BASE_N]
    if valid.size<K: raise RuntimeError(f'insufficient GT q={qi}')
    ids=valid[:K]
    seed_ids.append(ids.tolist())
    x=np.asarray(base[ids],dtype=np.float64)-queries[qi].astype(np.float64)
    taus[qi]=np.float32(math.sqrt(float(np.max(np.sum(x*x,axis=1)))))
pad=np.maximum(np.float32(1e-3),np.abs(taus)*np.float32(1e-5))
tr=np.nextafter(taus+pad,np.float32(np.inf))
boxes=np.stack((qcoords[:,0]-tr,qcoords[:,1]-tr,qcoords[:,0]+tr,qcoords[:,1]+tr),axis=1).astype(np.float32)
boxes.tofile(OUT/'query_boxes_q128.f32')
np.stack((taus,tr),axis=1).astype(np.float32).tofile(OUT/'taus_q128.f32')
qcoords.astype(np.float32).tofile(OUT/'query_coords_q128.f32')
mean32.tofile(OUT/'pca_mean.f32')
basis[:,:2].tofile(OUT/'pca_basis2.f32')

# Candidate aggregates for each prefix size and q prefix. Exact 128D
# threshold counts are checked independently by the GPU benchmark.
Ns=[100_000,250_000,500_000,750_000,1_000_000]
Qs=[1,4,8,16,32,64,128]
per_query=[]
points_arr=np.asarray(points)
for qi in range(NQ):
    box=boxes[qi]
    pmask=(points_arr[:,0]>=box[0])&(points_arr[:,0]<=box[2])&(points_arr[:,1]>=box[1])&(points_arr[:,1]<=box[3])
    cand_counts={str(n):int(np.count_nonzero(pmask[:n])) for n in Ns}
    per_query.append({"query":qi,"tau":float(taus[qi]),"candidate_counts":cand_counts})
summary={}
for n in Ns:
    for q in Qs:
        c=sum(x["candidate_counts"][str(n)] for x in per_query[:q])
        summary[f"n{n}_q{q}"]={"candidate_pairs":c,"candidate_ratio":c/(n*q)}

# Reproduce/compare old q32 artifact as a provenance check.
old=W/'results/sift1m_pool1m_q32_k10/pca2_rt_case'
oldp=np.fromfile(old/'points.f32',dtype=np.float32).reshape(-1,2)
oldb=np.fromfile(old/'query_boxes.f32',dtype=np.float32).reshape(-1,4)
oldt=np.fromfile(old/'taus.f32',dtype=np.float32).reshape(-1,2)
comparison={
  'points_max_abs_diff':float(np.max(np.abs(oldp-points_arr))),
  'boxes_first32_max_abs_diff':float(np.max(np.abs(oldb-boxes[:32]))),
  'taus_first32_max_abs_diff':float(np.max(np.abs(oldt-np.stack((taus,tr),axis=1)[:32]))),
  'points_byte_equal':bool(np.array_equal(oldp,points_arr)),
  'boxes_first32_byte_equal':bool(np.array_equal(oldb,boxes[:32])),
  'taus_first32_byte_equal':bool(np.array_equal(oldt,np.stack((taus,tr),axis=1)[:32])),
}
meta={'dataset':'SIFT1M','base_threshold_count':BASE_N,'k':K,'queries':NQ,'sizes':Ns,'query_batches':Qs,
      'projection':'PCA2 trained on first 50k vectors; same deterministic procedure as prior validation',
      'threshold':'safe kth distance from GT entries inside frozen Base900k',
      'summary':summary,'comparison_to_prior_q32':comparison,'elapsed_s':time.time()-t0,'per_query':per_query}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'elapsed_s':meta['elapsed_s'],'comparison':comparison,'summary':summary},indent=2))

#!/usr/bin/env python3
"""Run the stable-ID Safe-C1 oracle harness on an explicit SIFT base/reservoir split."""
from __future__ import annotations
import argparse, hashlib, json, socket, sys, time
from pathlib import Path
import numpy as np

from safe_c1_oracle import SafeC1Index, execute_trace, make_normal_trace, sha256_file, json_default

def load_fbin(path: Path, rows: int | None = None) -> np.ndarray:
    with path.open('rb') as f:
        header=np.fromfile(f,dtype=np.int32,count=2)
        if len(header)!=2: raise ValueError('bad fbin header')
        n,d=map(int,header)
        take=n if rows is None else min(n,int(rows))
        data=np.fromfile(f,dtype=np.float32,count=take*d)
        if len(data)!=take*d: raise ValueError('truncated fbin')
    return data.reshape(take,d).astype(np.float64)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--query',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20260727,20260728,20260729])
    p.add_argument('--base-n',type=int,default=4096); p.add_argument('--reservoir-n',type=int,default=2048);p.add_argument('--query-n',type=int,default=128)
    p.add_argument('--events',type=int,default=512);p.add_argument('--fanout',type=int,default=8);p.add_argument('--base-leaf-size',type=int,default=32);p.add_argument('--leaf-extension-cap',type=int,default=8);p.add_argument('--delta-capacity',type=int,default=8);p.add_argument('--certificate-margin',type=float,default=1e-7);p.add_argument('--k',type=int,default=10);p.add_argument('--audit-queries',type=int,default=4)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('refusing to overwrite output')
    pool=load_fbin(a.base,a.base_n+a.reservoir_n)
    queries=load_fbin(a.query,a.query_n)
    if pool.shape[1]!=queries.shape[1]: raise ValueError('dimension mismatch')
    # A fixed, data-derived radius: median nearest-base distance, expanded by 5%.
    nearest=[]
    base=pool[:a.base_n]
    for q in queries:
        nearest.append(float(np.min(np.linalg.norm(base-q,axis=1))))
    radius=float(np.median(nearest)*1.05)
    policy={'fanout':a.fanout,'base_leaf_size':a.base_leaf_size,'leaf_extension_cap':a.leaf_extension_cap,'delta_capacity':a.delta_capacity,'certificate_margin':a.certificate_margin}
    a.out.mkdir(parents=True)
    results=[]
    for seed in a.seeds:
        trace=make_normal_trace(seed,a.base_n,a.reservoir_n,a.query_n,a.events)
        rd=a.out/'normal'/f'seed_{seed}';rd.mkdir(parents=True)
        np.save(rd/'pool.npy',pool);np.save(rd/'queries.npy',queries)
        (rd/'trace.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True) for x in trace)+'\n')
        safe=SafeC1Index(pool,range(a.base_n),allow_direct=True,**policy)
        buffer=SafeC1Index(pool,range(a.base_n),allow_direct=False,**policy)
        audit=list(range(min(a.audit_queries,a.query_n)))
        sr=execute_trace(safe,trace,queries,radius=radius,k=a.k,audit_qids=audit,output=rd/'safe_answers.jsonl')
        br=execute_trace(buffer,trace,queries,radius=radius,k=a.k,audit_qids=audit,output=rd/'buffer_answers.jsonl')
        same=safe.active_set_digest()==buffer.active_set_digest()
        if not same: raise AssertionError('active sets diverged')
        summary={'scenario':'sift128_slice','seed':seed,'policy':policy,'data':{'dataset':'SIFT1M fbin explicit slice','metric':'L2','dimension':int(pool.shape[1]),'base_n':a.base_n,'reservoir_n':a.reservoir_n,'query_n':a.query_n,'base_source':str(a.base),'query_source':str(a.query),'base_sha256':sha256_file(a.base),'query_sha256':sha256_file(a.query)},'query_contract':{'k':a.k,'range_radius':radius},'trace':{'events':len(trace),'sha256':sha256_file(rd/'trace.jsonl')},'safe_c1':sr,'buffer_only':br,'oracle_pass':sr['checks']['knn_mismatch']==0 and sr['checks']['range_mismatch']==0 and br['checks']['knn_mismatch']==0 and br['checks']['range_mismatch']==0,'same_final_active_set':same,'final_active_set_hash':safe.active_set_digest(),'files':{x:sha256_file(rd/x) for x in ['pool.npy','queries.npy','trace.jsonl','safe_answers.jsonl','buffer_answers.jsonl']}}
        (rd/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,default=json_default)+'\n')
        results.append(summary)
    suite={'schema':'safe-c1-oracle-sift-slice-v1','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'command':sys.argv,'host':socket.gethostname(),'python':sys.version,'numpy':np.__version__,'gpu_used':False,'harness_sha256':sha256_file(Path(__file__).with_name('safe_c1_oracle.py')),'source_manifest_sha256':sha256_file(a.source_manifest),'results':results}
    suite['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (a.out/'suite_summary.json').write_text(json.dumps(suite,indent=2,sort_keys=True,default=json_default)+'\n')
    print(json.dumps({'runs':len(results),'all_oracle_pass':all(r['oracle_pass'] for r in results),'radius':radius},sort_keys=True))
if __name__=='__main__':main()

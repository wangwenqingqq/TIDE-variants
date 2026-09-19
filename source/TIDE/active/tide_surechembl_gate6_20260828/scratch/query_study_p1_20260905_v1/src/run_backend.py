#!/usr/bin/env python3
"""Common complete-ID batch driver. Run only through guarded_run.py."""
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from make_oracle import load_oracle
from prepare import digest


class Native:
    def __init__(self, name, root, data):
        libname='libp1_gpusim.so' if name=='gpusim' else 'libp1_tide.so'
        self.path=root/'build/bridges_v2'/libname
        self.lib=C.CDLL(str(self.path))
        ptr=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
        self.lib.p1_create.argtypes=[C.c_char_p,C.c_int]; self.lib.p1_create.restype=C.c_void_p
        self.lib.p1_destroy.argtypes=[C.c_void_p]; self.lib.p1_destroy.restype=None
        self.lib.p1_error.restype=C.c_char_p
        self.lib.p1_search.argtypes=[C.c_void_p,ptr,C.c_uint,C.c_uint,ptr,C.c_uint64]
        self.lib.p1_search.restype=C.c_int64
        self.handle=self.lib.p1_create(str(data).encode(),int(name=='tide_bounded'))
        if not self.handle: raise RuntimeError(self.lib.p1_error().decode())
        self.output=np.empty(65536,dtype=np.uint64)
        self.info={'adapter':'serial batch, original native query core',
                   'library':str(self.path),'library_sha256':digest(self.path)}

    def run(self, queries, indices, p, d):
        results=[]; completed=[]; start=time.perf_counter_ns()
        for qi in indices:
            n=self.lib.p1_search(self.handle,queries[qi],p,d,self.output,len(self.output))
            if n<0: raise RuntimeError(self.lib.p1_error().decode())
            results.append(self.output[:n].copy())
            completed.append((time.perf_counter_ns()-start)/1e6)
        elapsed=(time.perf_counter_ns()-start)/1e6
        return results,elapsed,completed

    def close(self):
        if self.handle: self.lib.p1_destroy(self.handle); self.handle=None


class NvMolKit:
    def __init__(self, name, root, data, queries):
        import torch
        from nvmolkit.similarity import crossTanimotoSimilarity
        if importlib.metadata.version('nvmolkit')!='0.6.0': raise RuntimeError('unfrozen version')
        if torch.cuda.device_count()!=1: raise RuntimeError('require exactly one visible GPU')
        torch.set_num_threads(1)
        self.torch=torch; self.cross=crossTanimotoSimilarity; self.name=name
        self.stream=torch.cuda.Stream()
        ids=np.fromfile(data/'id_i64.bin',dtype='<u8')
        if np.any(ids>np.iinfo(np.int64).max) or np.any(queries[:,0]>np.iinfo(np.int64).max):
            raise ValueError('signed torch ID encoding would overflow')
        fp=np.fromfile(data/'fp_u64x4.bin',dtype='<u4').reshape(-1,8)
        if name=='nv_column' and (len(ids)+63)//64>65535:
            raise ValueError('target-major grid.y exceeds hardware launch limit')
        with torch.cuda.stream(self.stream):
            self.database=torch.from_numpy(fp).to('cuda')
            self.ids=torch.from_numpy(ids.view(np.int64)).to('cuda')
        self.stream.synchronize()
        self.query_host=torch.from_numpy(np.ascontiguousarray(queries[:,1:5]).view('<u4').reshape(-1,8)).pin_memory()
        self.qid_host=torch.from_numpy(np.ascontiguousarray(queries[:,0]).view(np.int64)).pin_memory()
        self.info={'adapter':'native matrix batch with vectorized threshold/ID gather',
                   'orientation':name,'torch':torch.__version__,'nvmolkit':'0.6.0',
                   'cutoff':'one float32 nextafter below 7/10 or 4/5, complete oracle required'}

    def run(self, queries, indices, p, d):
        torch=self.torch
        begin=int(indices[0]); end=begin+len(indices)
        if list(indices)!=list(range(begin,end)): raise ValueError('batch must be contiguous')
        cutoff=float(np.nextafter(np.float32(p/d),np.float32(-np.inf)))
        start=time.perf_counter_ns()
        with torch.cuda.stream(self.stream):
            q=self.query_host[begin:end].to('cuda',non_blocking=True)
            qid=self.qid_host[begin:end].to('cuda',non_blocking=True)
            if self.name=='nv_row':
                scores=self.cross(q,self.database,stream=self.stream).torch()
                if tuple(scores.shape)!=(len(indices),len(self.ids)): raise RuntimeError('shape')
                eligible=(scores>=cutoff)&(self.ids[None,:]!=qid[:,None])
                coords=torch.nonzero(eligible,as_tuple=False)
                pairs=torch.stack((coords[:,0],self.ids[coords[:,1]]),dim=1)
            else:
                scores=self.cross(self.database,q,stream=self.stream).torch()
                if tuple(scores.shape)!=(len(self.ids),len(indices)): raise RuntimeError('shape')
                eligible=(scores>=cutoff)&(self.ids[:,None]!=qid[None,:])
                coords=torch.nonzero(eligible,as_tuple=False)
                pairs=torch.stack((coords[:,1],self.ids[coords[:,0]]),dim=1)
            if scores.dtype!=torch.float64: raise RuntimeError('score dtype changed')
            host=pairs.to('cpu',non_blocking=False).numpy()
        results=[np.array(host[host[:,0]==i,1],dtype=np.uint64) for i in range(len(indices))]
        elapsed=(time.perf_counter_ns()-start)/1e6
        return results,elapsed,None

    def close(self):
        self.stream.synchronize()
        del self.database,self.ids,self.query_host,self.qid_host
        self.torch.cuda.empty_cache()


def validate_inputs(data):
    manifest=json.loads((data/'MANIFEST.json').read_text())
    for name,info in manifest['files'].items():
        if digest(data/name)!=info['sha256']: raise ValueError('input hash: '+name)
    om=json.loads((data/'ORACLE_MANIFEST.json').read_text())
    if digest(data/'oracle.bin')!=om['oracle_sha256']: raise ValueError('oracle hash mismatch')
    queries=np.fromfile(data/'queries_u64x6.bin',dtype='<u8').reshape(-1,6)
    if np.any(queries[:,5]==0): raise ValueError('zero query outside contract')
    return manifest,queries,load_oracle(data/'oracle.bin')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['nv_row','nv_column','gpusim','tide_bounded','tide_unbounded'],required=True)
    ap.add_argument('--data',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=['check','timing','churn'],default='check')
    ap.add_argument('--batch-sizes',default='1,8,64'); ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--warmup',type=int,default=16); ap.add_argument('--churn-repeats',type=int,default=10)
    args=ap.parse_args()
    if os.environ.get('P1_GUARDED_UUID')!='GPU-CONFIGURE-ARCHIVE-DEVICE':
        raise RuntimeError('run through the GPU-3 admission/occupation guard')
    if args.output.exists() or args.output.with_suffix('.metadata.json').exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]; data=args.data.resolve()
    manifest,queries,oracle=validate_inputs(data)
    sizes=[int(x) for x in args.batch_sizes.split(',')]
    if set(sizes)-{1,8,64} or len(set(sizes))!=len(sizes): raise ValueError('unfrozen batch set')
    thresholds=[(7,10),(4,5)]
    cohorts=list(dict.fromkeys(manifest['cohorts']))
    if args.reverse: sizes.reverse(); thresholds.reverse(); cohorts.reverse()
    def create():
        return NvMolKit(args.backend,root,data,queries) if args.backend.startswith('nv_') else Native(args.backend,root,data)
    verified=0; observed=0; context_info=None; setup_times=[]
    def check(indices,p,d,results):
        nonlocal verified
        for qi,ids in zip(indices,results):
            expected=oracle[int(qi),p,d]
            got=np.sort(ids)
            if not np.array_equal(got,expected):
                failure={'query':int(qi),'p':p,'d':d,'got':got.tolist(),'expected':expected.tolist()}
                args.output.with_suffix('.failure.json').write_text(json.dumps(failure)+'\n')
                raise AssertionError(f'complete oracle mismatch: {qi}, {p}/{d}')
            verified+=1
    with args.output.open('x') as output:
        repeats=args.churn_repeats if args.mode=='churn' else 1
        for repetition in range(repeats):
            setup=time.perf_counter(); backend=create(); setup_times.append(time.perf_counter()-setup)
            context_info=backend.info
            try:
                for cohort in cohorts:
                    group=np.flatnonzero(np.array(manifest['cohorts'])==cohort)
                    if args.mode=='churn': group=group[:64]
                    for size in sizes:
                        if len(group)%size: raise ValueError('ragged batch not admitted')
                        for p,d in thresholds:
                            if args.mode=='timing':
                                for _ in range(args.warmup):
                                    r,_,_=backend.run(queries,group[:size],p,d); check(group[:size],p,d,r)
                            for start in range(0,len(group),size):
                                indices=group[start:start+size]
                                results,ms,completions=backend.run(queries,indices,p,d)
                                check(indices,p,d,results)
                                row={'backend':args.backend,'mode':args.mode,'cohort':cohort,
                                     'batch':size,'threshold_num':p,'threshold_den':d,
                                     'query_start':int(indices[0]),'query_ids_sha256':hashlib.sha256(queries[indices,0].tobytes()).hexdigest(),
                                     'service_ms':ms,'batch_throughput_queries_s':1000*size/ms,
                                     'individual_completion_ms':completions,'counts':[len(x) for x in results],
                                     'complete_oracle_equal':True,'repetition':repetition}
                                output.write(json.dumps(row,separators=(',',':'))+'\n'); observed+=1
                output.flush()
            finally: backend.close()
    meta={'backend':args.backend,'mode':args.mode,'request_scope':context_info,
          'data_manifest_sha256':digest(data/'MANIFEST.json'),'oracle_sha256':digest(data/'oracle.bin'),
          'query_sha256':digest(data/'queries_u64x6.bin'),'rows':manifest['rows'],
          'gpu_uuid':os.environ['P1_GUARDED_UUID'],'pid':os.getpid(),
          'verified_complete_vectors_including_warmups':verified,'batch_records':observed,
          'setup_seconds_excluded':setup_times,'peak_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'warmup_batches_per_cell':args.warmup if args.mode=='timing' else 0,
          'reverse':args.reverse,'timing_admissible':args.mode=='timing',
          'script_sha256':digest(Path(__file__)),'python':sys.version}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__': main()

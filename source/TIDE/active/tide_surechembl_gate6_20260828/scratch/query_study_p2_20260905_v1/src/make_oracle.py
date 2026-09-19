#!/usr/bin/env python3
"""Materialize complete CPU result vectors and independent Python spot checks."""
import argparse
import ctypes as C
import json
import struct
import time
from pathlib import Path
import numpy as np
from prepare import digest, pc


def load_oracle(path):
    raw=path.read_bytes()
    if raw[:8] != b'P1ORCL01':
        raise ValueError('bad oracle magic')
    nq=struct.unpack_from('<Q',raw,8)[0]; offset=16; result={}
    for q in range(nq):
        for p,d in ((7,10),(4,5)):
            size=struct.unpack_from('<Q',raw,offset)[0]; offset+=8
            result[q,p,d]=np.frombuffer(raw,dtype='<u8',count=size,offset=offset)
            offset+=8*size
    if offset != len(raw): raise ValueError('oracle trailing bytes')
    return result


def python_oracle(ids, fp, q, p, d):
    qbits=sum(int(q[w+1]) << (64*w) for w in range(4))
    out=[]
    for identifier,words in zip(ids,fp):
        if int(identifier)==int(q[0]): continue
        bits=sum(int(words[w]) << (64*w) for w in range(4))
        i=(bits&qbits).bit_count(); u=(bits|qbits).bit_count()
        if u and d*i >= p*u: out.append(int(identifier))
    return np.array(sorted(out),dtype='<u8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--library',type=Path,required=True); args=ap.parse_args()
    root=args.data; outpath=root/'oracle.bin'
    if outpath.exists(): raise FileExistsError(outpath)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name, info in manifest['files'].items():
        if digest(root/name)!=info['sha256']: raise ValueError('input hash mismatch')
    ids=np.fromfile(root/'id_i64.bin',dtype='<u8')
    fp=np.fromfile(root/'fp_u64x4.bin',dtype='<u8').reshape(-1,4)
    queries=np.fromfile(root/'queries_u64x6.bin',dtype='<u8').reshape(-1,6)
    assert np.array_equal(pc(fp),np.fromfile(root/'popcnt_u16.bin',dtype='<u2'))
    assert np.array_equal(pc(queries[:,1:5]),queries[:,5])
    lib=C.CDLL(str(args.library.resolve())); ptr=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    lib.p1_oracle.argtypes=[ptr,ptr,C.c_uint64,ptr,C.c_uint,C.c_uint,ptr]
    lib.p1_oracle.restype=C.c_uint64
    out=np.empty(len(ids),dtype='<u8'); t=time.monotonic(); counts=[]
    with outpath.open('xb') as f:
        f.write(b'P1ORCL01'+struct.pack('<Q',len(queries)))
        for qi,q in enumerate(queries):
            for p,d in ((7,10),(4,5)):
                n=lib.p1_oracle(fp,ids,len(ids),q,p,d,out)
                f.write(struct.pack('<Q',n)); f.write(out[:n].tobytes()); counts.append(int(n))
            if qi % 128==0: print('oracle',qi,'elapsed',round(time.monotonic()-t,2),flush=True)
    oracle=load_oracle(outpath)
    # All prefix queries plus high-entropy/self-exclusion endpoints for fixture;
    # four independent complete scans on real data, selected before results.
    checks=list(range(256))+[256,495,496,511] if len(ids)<10000 else [0,511,512,1023]
    for qi in checks:
        for p,d in ((7,10),(4,5)):
            expected=python_oracle(ids,fp,queries[qi],p,d)
            if not np.array_equal(expected,oracle[qi,p,d]): raise AssertionError((qi,p,d))
    metadata={'queries':len(queries),'rows':len(ids),'exact_vectors':2*len(queries),
              'python_complete_scans':2*len(checks),'max_hits':max(counts),'sum_hits':sum(counts),
              'seconds_cpu_preparation_not_query_benchmark':time.monotonic()-t,
              'oracle_sha256':digest(outpath),'library_sha256':digest(args.library),
              'self_excluded':True,'unpruned':True,'zero_query_admitted':False}
    (root/'ORACLE_MANIFEST.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()

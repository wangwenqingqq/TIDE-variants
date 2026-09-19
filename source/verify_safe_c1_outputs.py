#!/usr/bin/env python3
"""Independent stable-ID oracle verifier for Safe-C1 JSONL outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

EPS=1e-8

def expected(pool, active, q, kind, radius, k):
    ids=np.flatnonzero(active)
    dist=np.sqrt(np.sum((pool[ids]-q)**2,axis=1))
    if kind=='range':
        keep=dist <= radius + EPS
        out=[(int(i),float(d)) for i,d,yes in zip(ids,dist,keep) if yes]
    else:
        order=np.lexsort((ids,dist))[:k]
        out=[(int(ids[i]),float(dist[i])) for i in order]
    return sorted(out,key=lambda x:(x[1],x[0]))

def verify(run_dir, name, base_n, radius, k):
    pool=np.load(run_dir/'pool.npy')
    queries=np.load(run_dir/'queries.npy')
    active=np.zeros(len(pool),dtype=bool); active[:base_n]=True
    nquery=nupdate=0
    for line_no,line in enumerate((run_dir/name).read_text().splitlines(),1):
        r=json.loads(line)
        if r['record']=='update':
            e=r['event']; nupdate+=1
            if e['op']=='insert':
                if active[e['id']]: raise AssertionError(f'{name}:{line_no}: duplicate insert')
                active[e['id']]=True
            elif e['op']=='delete':
                if not active[e['id']]: raise AssertionError(f'{name}:{line_no}: invalid delete')
                active[e['id']]=False
            else: raise AssertionError(f'{name}:{line_no}: unknown update')
            continue
        if r['record']!='query': raise AssertionError(f'{name}:{line_no}: unknown record')
        nquery+=1
        got=[(int(i),float(d)) for i,d in r['actual']]
        want=expected(pool,active,queries[r['query_id']],r['kind'],radius,k)
        if [i for i,_ in got] != [i for i,_ in want]:
            raise AssertionError(f'{name}:{line_no}: ID mismatch')
        if any(abs(a[1]-b[1])>EPS for a,b in zip(got,want)):
            raise AssertionError(f'{name}:{line_no}: distance mismatch')
    return {'file':name,'updates':nupdate,'queries':nquery,'final_active':int(active.sum())}

def main():
    a=argparse.ArgumentParser();a.add_argument('run_dir',type=Path);a.add_argument('--base-n',type=int,required=True);a.add_argument('--radius',type=float,required=True);a.add_argument('--k',type=int,required=True);args=a.parse_args()
    out=[verify(args.run_dir,n,args.base_n,args.radius,args.k) for n in ('safe_answers.jsonl','buffer_answers.jsonl')]
    print(json.dumps({'pass':True,'files':out},sort_keys=True))
if __name__=='__main__':main()

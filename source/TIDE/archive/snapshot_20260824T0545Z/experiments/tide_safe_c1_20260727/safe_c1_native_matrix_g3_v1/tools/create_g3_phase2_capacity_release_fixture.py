#!/usr/bin/env python3
"""Create a CPU-only G3 capacity-release fixture from the 4K capacity witness."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from g3_common import exact_results, stable_set_sha256, write_json


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load_lines(path: Path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def dump_lines(path: Path, rows): path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--dest',type=Path,required=True); args=ap.parse_args()
    src=args.source.resolve(); dst=args.dest.resolve()
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)
    meta=json.loads((dst/'metadata.json').read_text()); header=meta['header']; pool_n=int(header['pool_n']); dim=int(header['dim']); k=int(header['k'])
    physical_pool=np.fromfile(dst/'pool.i16',dtype='<i2').reshape(pool_n,dim)
    mapping=np.fromfile(dst/'stable_id_to_pool_row.i32',dtype='<i4')
    if mapping.size!=pool_n or len(np.unique(mapping))!=pool_n: raise ValueError('bad source stable mapping')
    pool=physical_pool[mapping.astype(np.int64)]
    queries=np.fromfile(dst/'queries.i16',dtype='<i2').reshape(int(header['query_n']),dim)
    old=load_lines(dst/'trace.jsonl')
    if not (old[13]['op']=='delete' and old[13]['stable_id']==4258 and old[16]['op']=='delete' and old[16]['stable_id']==5711):
        raise ValueError('source capacity trace shape changed')
    extension=[
        {'op':'insert','stable_id':5711,'expected_role':'direct_same_leaf','expected_leaf':190,'witness_label':'capacity_slot_released_direct'},
        {'op':'knn','query_id':129,'witness_label':'capacity_slot_released_direct_visible'},
        {'op':'range','query_id':129,'radius_sq':1,'witness_label':'capacity_slot_released_direct_visible'},
        {'op':'delete','stable_id':5711,'expected_prior':'direct','witness_label':'capacity_slot_released_direct_deleted'},
        {'op':'knn','query_id':129,'witness_label':'capacity_slot_released_direct_absent'},
        {'op':'range','query_id':129,'radius_sq':1,'witness_label':'capacity_slot_released_direct_absent'},
    ]
    trace=old[:17]+extension+old[17:]
    for index,row in enumerate(trace): row['op_index']=index
    # Recompute all trace-level rebuild hashes / expected delta cardinalities.
    base=np.fromfile(dst/'initial_base_stable_ids.i32',dtype='<i4').astype(int).tolist()
    active=set(base); placement={sid:'base' for sid in active}; oracle=[]
    for row in trace:
        op=row['op']
        if op=='insert':
            sid=int(row['stable_id']);
            if sid in active: raise ValueError(f'duplicate insert {sid}')
            active.add(sid); placement[sid]='direct' if row.get('expected_role') in {'direct','direct_same_leaf'} else 'delta'
        elif op=='delete':
            sid=int(row['stable_id'])
            if placement.get(sid)!=row.get('expected_prior'): raise ValueError(f'bad expected prior {sid}')
            active.remove(sid); placement[sid]='deleted'
        elif op=='rebuild':
            row['expected_live_ids_sha256']=stable_set_sha256(active)
            row['expected_delta_live']=sum(1 for sid in active if placement.get(sid)=='delta')
            for sid in active: placement[sid]='base'
        elif op in {'knn','range'}:
            qid=int(row['query_id'])
            values=exact_results(pool,queries[qid],active,op,k,row.get('radius_sq'))
            oracle.append({'record':'oracle_query','op_index':row['op_index'],'kind':op,'query_id':qid,
                           'radius_sq':row.get('radius_sq'),'active_ids_sha256':stable_set_sha256(active),'results':values})
        else: raise ValueError(op)
    dump_lines(dst/'trace.jsonl',trace); dump_lines(dst/'oracle_expected.jsonl',oracle)
    source_selection=json.loads((dst/'selection_receipt.json').read_text())
    cap=next(x for x in source_selection['events'] if x.get('stable_id')==5711 and x.get('role')=='capacity_delta')
    release=dict(cap); release['placement']='direct'; release['role']='direct_same_leaf'; release['sidecar_leaf_id']=190; release.pop('certified_leaf_id',None); release.pop('fallback',None)
    source_selection['events'].append(release)
    source_selection['phase2_capacity_release']={
        'selector':'cpu_source_mirror_not_native_receipt','stable_id':5711,'leaf':190,
        'precondition':'delete direct stable_id 4258 frees one L=8 sidecar slot',
        'required_native_behavior':'reinsert 5711 as direct only after actual frozen-tree certificate and sidecar size=7 check',
        'not_native_executed':True,
    }
    write_json(dst/'selection_receipt.json',source_selection)
    meta['header']['event_count']=len(trace)
    meta['selection']['events'].append(release)
    meta['selection']['phase2_capacity_release']=source_selection['phase2_capacity_release']
    meta['status']='CPU_PREPARED_NOT_NATIVE_EXECUTED'
    meta['scope']='G3 Safe-C1 CPU fixture including capacity-slot-release; no CUDA/native execution or performance claim'
    meta['trace_contract']['roles']['direct_insert']=int(meta['trace_contract']['roles'].get('direct_insert',0))+1
    meta['trace_contract']['roles']['delete_direct']=int(meta['trace_contract']['roles'].get('delete_direct',0))+1
    write_json(dst/'metadata.json',meta)
    manifest=json.loads((dst/'manifest.json').read_text())
    manifest['files_sha256']={name:sha(dst/name) for name in sorted(manifest['files_sha256'])}
    manifest['phase2_extension']='capacity_slot_release_reinsert_direct'
    write_json(dst/'manifest.json',manifest)
    print(json.dumps({'status':'CPU_PREPARED_NOT_NATIVE_EXECUTED','gpu_used':False,'source':str(src),'dest':str(dst),'events':len(trace),'oracle_queries':len(oracle),'manifest_sha256':sha(dst/'manifest.json')},sort_keys=True))
if __name__=='__main__': main()

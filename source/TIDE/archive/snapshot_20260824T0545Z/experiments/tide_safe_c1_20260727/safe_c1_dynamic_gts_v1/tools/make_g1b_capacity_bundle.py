#!/usr/bin/env python3
"""Freeze a G1B capacity-fallback witness trace from the guarded selector receipt."""
from __future__ import annotations
import argparse, hashlib, json, shutil, struct
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/experiments/tide_safe_c1_20260727')
D=ROOT/'safe_c1_dynamic_gts_v1'
H=struct.Struct('<8sI6IfQ'); E=struct.Struct('<IB3xi')

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def write(p:Path,o:object)->None:p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
def exact(pool,q,active,k):
 ids=np.asarray(sorted(active),dtype=np.int64); d=pool[ids].astype(np.int64)-q.astype(np.int64); ds=np.sqrt(np.sum(d*d,axis=1,dtype=np.int64).astype(np.float64)); ix=np.lexsort((ids,ds))[:k];return [(int(ids[i]),float(ds[i])) for i in ix]
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--selection-run',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 run=a.selection_run.resolve();out=a.out.resolve()
 if out.exists():raise SystemExit(f'refuse overwrite {out}')
 source=D/'bundles/g1b_selector_all_reservoir_v1'
 selection=[json.loads(x) for x in (run/'runner_output/engine_results.jsonl').read_text().splitlines() if x.strip()]
 ins=[x for x in selection if x.get('record')=='update' and x.get('op')=='insert']
 groups=defaultdict(list); cert_delta=[]
 for x in ins:
  if x.get('placement')=='direct':groups[int(x['sidecar_leaf_id'])].append(int(x['stable_id']))
  elif x.get('placement')=='delta':cert_delta.append(int(x['stable_id']))
 choices=[(len(ids),leaf,sorted(ids)) for leaf,ids in groups.items() if len(ids)>=3]
 if not choices or not cert_delta:raise SystemExit('selector did not yield same-leaf triple and certificate delta')
 # Deterministic: densest group then smallest leaf, first three IDs; independent
 # G1B witnesses will check that c falls to delta only after a/b fill L=2.
 _,leaf,ids=sorted(choices,key=lambda x:(-x[0],x[1]))[0]
 a_id,b_id,c_id=ids[:3]; d_id=min(cert_delta)
 raw=(source/'trace.e1gtrc').read_bytes(); magic,ver,dim,base,res,pooln,_,k,radius,_=H.unpack_from(raw)
 pool=np.fromfile(source/'pool.i16',dtype='<i2').reshape(pooln,dim)
 qids=[a_id,b_id,c_id,a_id,c_id,d_id];queries=np.asarray(pool[qids],dtype='<i2',order='C')
 events=[(1,a_id),(3,0),(1,b_id),(3,1),(1,c_id),(3,2),(2,a_id),(3,3),(2,c_id),(3,4),(1,d_id),(3,5)]
 active=set(range(base));cpu=[]
 for oi,(code,arg) in enumerate(events):
  if code==1:active.add(arg)
  elif code==2:active.remove(arg)
  else:
   ans=exact(pool,queries[arg],active,k);target=qids[arg]
   cpu.append({'op_index':oi,'query_id':arg,'target_stable_id':target,'top1':ans[0][0],'target_in_topk':any(i==target for i,_ in ans)})
 # Positive insertion witnesses and deleted-object absence have deterministic CPU guarantees.
 for row in cpu:
  if row['op_index'] in {1,3,5,11} and row['top1']!=row['target_stable_id']:raise SystemExit(f'nonunique insert witness {row}')
  if row['op_index'] in {7,9} and row['target_in_topk']:raise SystemExit(f'deleted witness remains visible {row}')
 out.mkdir(parents=True);(out/'pool.i16').write_bytes(pool.tobytes(order='C'));(out/'queries.i16').write_bytes(queries.tobytes(order='C'))
 (out/'trace.e1gtrc').write_bytes(H.pack(b'E1GTRC01',1,dim,base,res,pooln,len(queries),k,np.float32(radius),len(events))+b''.join(E.pack(i,code,arg) for i,(code,arg) in enumerate(events)))
 for n in ('initial_base_stable_ids.i32','stable_id_to_pool_row.i32'):shutil.copyfile(source/n,out/n)
 selection_receipt={'schema':'safe-c1-g1b-selection-receipt-v1','scope':'guarded high-capacity selection prerequisite, not dynamic-query evidence','selection_run':str(run),'selection_engine_sha256':sha(run/'runner_output/engine_results.jsonl'),'selection_summary_sha256':sha(run/'runner_output/engine_summary.json'),'selection_validator_sha256':sha(run/'validation.raw_quantized_oracle.json'),'frozen_tree_hash':json.loads((run/'runner_output/engine_summary.json').read_text())['base_tree']['frozen_hash'],'same_leaf_id':leaf,'same_leaf_strict_certificate_ids':[a_id,b_id,c_id],'certificate_delta_id':d_id,'selector_leaf_capacity':4096,'g1b_leaf_capacity':2}
 write(out/'selection_receipt.json',selection_receipt)
 contract={'schema':'safe-c1-g1b-capacity-witness-contract-v1','expected_insert_receipts':[
   {'op_index':0,'stable_id':a_id,'placement':'direct','sidecar_leaf_id':leaf},
   {'op_index':2,'stable_id':b_id,'placement':'direct','sidecar_leaf_id':leaf},
   {'op_index':4,'stable_id':c_id,'placement':'delta','fallback':'capacity','certified_leaf_id':leaf},
   {'op_index':10,'stable_id':d_id,'placement':'delta','fallback':'certificate'}],
   'events':[{'op_index':1,'mode':'inserted_top1','stable_id':a_id},{'op_index':3,'mode':'inserted_top1','stable_id':b_id},{'op_index':5,'mode':'inserted_top1','stable_id':c_id},{'op_index':7,'mode':'deleted_absent','stable_id':a_id},{'op_index':9,'mode':'deleted_absent','stable_id':c_id},{'op_index':11,'mode':'inserted_top1','stable_id':d_id}],
   'cpu_preflight':cpu}
 write(out/'witness_contract.json',contract)
 meta={'schema':'e1gi-b-quantized-gts-integration-bundle-v1','scope':'CPU-prepared G1B Safe-C1 capacity witness; top-k correctness only; no range/rebuild/base-delete/latency claim','gpu_used':False,
       'header':{'magic':'E1GTRC01','version':1,'dim':dim,'base_n':base,'reservoir_n':res,'pool_n':pooln,'query_n':len(queries),'k':k,'radius':float(radius),'event_count':len(events)},
       'quantization':{'scale':100.0,'coordinate_type':'little-endian signed int16'},'metric_contract':{'metric':'L2','topk_tie_break':'(distance, stable_id)'},
       'g1b_scope':{'leaf_capacity':2,'coverage':['two certified direct inserts into one leaf','third certified same-leaf insert falls to exact delta by capacity','certificate fallback','direct/delta delete visibility'],'not_covered':['rebuild','base delete','range','latency']},
       'selection_receipt_file':'selection_receipt.json'}
 write(out/'metadata.json',meta);write(out/'manifest.json',{'schema':'safe-c1-g1b-bundle-manifest-v1','status':'prepared_cpu_only','files_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
 print(json.dumps({'status':'PASS_CPU_ONLY_G1B_BUNDLE','out':str(out),'same_leaf':leaf,'direct_ids':[a_id,b_id,c_id],'certificate_delta':d_id,'gpu_used':False},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

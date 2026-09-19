#!/usr/bin/env python3
"""CPU-only input for selecting same-leaf Safe-C1 candidates from a real frozen GTS tree."""
from __future__ import annotations
import hashlib, json, shutil, struct
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/experiments/tide_safe_c1_20260727')
SRC=ROOT/'safe_c1_dynamic_gts_v1/bundles/g1a_witness_pre_rebuild_v1'
OUT=ROOT/'safe_c1_dynamic_gts_v1/bundles/g1b_selector_all_reservoir_v1'
H=struct.Struct('<8sI6IfQ'); E=struct.Struct('<IB3xi')

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def write(p:Path,o:object)->None:p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
def main()->int:
 if OUT.exists():raise SystemExit(f'refuse overwrite {OUT}')
 raw=(SRC/'trace.e1gtrc').read_bytes(); magic,ver,dim,base,res,pool,qnum,k,radius,_=H.unpack_from(raw)
 if magic!=b'E1GTRC01' or ver!=1 or res!=pool-base:raise SystemExit('unexpected G1A input header')
 pool_data=np.fromfile(SRC/'pool.i16',dtype='<i2').reshape(pool,dim)
 # One frozen static probe query only; the events themselves are update-only.
 queries=np.asarray(pool_data[[base]],dtype='<i2',order='C')
 events=[(1,sid) for sid in range(base,pool)]
 OUT.mkdir(parents=True)
 (OUT/'pool.i16').write_bytes(pool_data.tobytes(order='C'))
 (OUT/'queries.i16').write_bytes(queries.tobytes(order='C'))
 (OUT/'trace.e1gtrc').write_bytes(H.pack(b'E1GTRC01',1,dim,base,res,pool,1,k,np.float32(radius),len(events))+b''.join(E.pack(i,c,a) for i,(c,a) in enumerate(events)))
 for n in ('initial_base_stable_ids.i32','stable_id_to_pool_row.i32'):shutil.copyfile(SRC/n,OUT/n)
 meta={'schema':'e1gi-b-quantized-gts-integration-bundle-v1','scope':'CPU-only G1B candidate-selection input; no query evidence or performance result','gpu_used':False,
       'header':{'magic':'E1GTRC01','version':1,'dim':dim,'base_n':base,'reservoir_n':res,'pool_n':pool,'query_n':1,'k':k,'radius':float(radius),'event_count':len(events)},
       'quantization':{'scale':100.0,'coordinate_type':'little-endian signed int16'},
       'stable_id_contract':{'initial_base_ids':'[0, base_n)','reservoir_ids':'[base_n,base_n+reservoir_n)','delete_semantics':'stable ID'},
       'selection_scope':{'operation':'insert every reservoir ID exactly once with sidecar capacity=4096; group strict-certificate direct IDs by frozen leaf; selection result is preparation, not evidence'},
       'source':{'path':str(SRC),'pool_i16_sha256':sha(SRC/'pool.i16')}}
 write(OUT/'metadata.json',meta)
 write(OUT/'manifest.json',{'schema':'safe-c1-g1b-selector-bundle-v1','status':'prepared_cpu_only','files_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
 print(json.dumps({'status':'PASS_CPU_ONLY_G1B_SELECTOR_BUNDLE','out':str(OUT),'events':len(events),'gpu_used':False},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())

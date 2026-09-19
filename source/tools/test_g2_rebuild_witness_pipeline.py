#!/usr/bin/env python3
"""CPU-only positive/negative pipeline test for G2 v4 tools.
No GPU query, telemetry, CUDA binary, or NVCC invocation occurs here.
"""
from __future__ import annotations
import json, math, subprocess, sys, tempfile
from pathlib import Path
from safe_c1_v4_rebuild_bundle_contract import cint, fnv1a_i32, load_i16, parse_trace, read_contract, replay_g2_trace, topk_exact
from preflight_g2_rebuild_witness import ROOT, validate_bundle
from finalize_g2_rebuild_witness import active_hash, float_pool_hash, validate_engine

BUNDLE=ROOT/'bundles/g2_rebuild_witness_20260728T120500Z_cpu'

def write_jsonl(path:Path, rows:list[dict]) -> None:
    path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf-8')

def main()->int:
    report,errors=validate_bundle(ROOT,BUNDLE)
    if errors: raise SystemExit('preflight fixture unexpectedly fails:'+repr(errors))
    header,events=parse_trace(BUNDLE/'trace.g2trc'); contract=read_contract(BUNDLE/'g2_rebuild_contract.txt'); replay=replay_g2_trace(header,events,contract)
    a,b,c=(cint(contract,k) for k in ('stable_id_a','stable_id_b','stable_id_c')); leaf=cint(contract,'sidecar_leaf_id'); base=cint(contract,'base_delete_id')
    initial=replay['initial_base_ids']; live=replay['active_after_rebuild']; ah=active_hash(header.pool_n,live)
    e0={'record':'epoch','epoch':0,'phase':'E0','tree_hash':101,'leaf_stable_id_hash':fnv1a_i32(initial),'pivot_stable_id_hash':fnv1a_i32([]),'live_stable_id_hash':fnv1a_i32(initial),'live_count':len(initial),'leaf_set_exact':True,'pivot_set_subset_of_live':True,'pivot_stable_ids':[]}
    e1={'record':'epoch','epoch':1,'phase':'E1','tree_hash':202,'leaf_stable_id_hash':fnv1a_i32(live),'pivot_stable_id_hash':fnv1a_i32([]),'live_stable_id_hash':fnv1a_i32(live),'live_count':len(live),'leaf_set_exact':True,'pivot_set_subset_of_live':True,'pivot_stable_ids':[]}
    pool_i16=load_i16(BUNDLE/'pool.i16',header.pool_n*header.dimension)
    queries_i16=load_i16(BUNDLE/'queries.i16',header.query_n*header.dimension)
    pool_hash=float_pool_hash(pool_i16)
    # Regression for the G2 physical-pool encoding: archive config maps its
    # historical `short` spelling to float, so the engine hash must be float32
    # bytes, not the source pool.i16 bytes.
    import struct
    raw_i16_hash=__import__('safe_c1_v4_rebuild_bundle_contract').fnv1a_u64_bytes((BUNDLE/'pool.i16').read_bytes())
    assert pool_hash != raw_i16_hash, 'float32 GtsScalar hash regressed to raw int16 bytes'
    source=(ROOT/'src/safe_c1_dynamic_gts.cu').read_text(encoding='utf-8')
    assert 'using GtsScalar = short;' in source and 'sizeof(GtsScalar)' in source
    assert 'std::vector<float> copy(pool_values)' not in source
    query_ids={2:cint(contract,'query_id_before'),5:cint(contract,'query_id_middle'),8:cint(contract,'query_id_after')}
    def res(op:int):
      return [[ident,math.sqrt(square)] for square,ident in topk_exact(pool_i16,queries_i16,replay['query_active_ids'][str(op)],header,query_ids[op])]
    rows=[
      {'record':'meta','schema':'safe-c1-g2-rebuild-witness-v4','scope':'no timing and no performance claim','legacy_incremental_updater_used':False,'stable_id_layout':'identity_full_immutable_pool_seeded_stable_ids_v4','host_full_pool_hash':pool_hash},e0,
      {'record':'update','op_index':0,'op':'insert','stable_id':a,'placement':'direct','sidecar_leaf_id':leaf,'certified_leaf_id':leaf,'fallback_reason':'none'},
      {'record':'update','op_index':1,'op':'insert','stable_id':b,'placement':'delta','sidecar_leaf_id':-1,'certified_leaf_id':leaf,'fallback_reason':'capacity_full'},
      {'record':'query','op_index':2,'query_id':cint(contract,'query_id_before'),'gts_visited_leaf_ids':[leaf],'sidecar_ids':[a],'delta_ids':[b],'oracle_full_active_set_checked':True,'results':res(2)},
      {'record':'update','op_index':3,'op':'delete','stable_id':a,'placement':'deleted','sidecar_leaf_id':leaf,'certified_leaf_id':leaf,'fallback_reason':'none'},
      {'record':'update','op_index':4,'op':'insert','stable_id':c,'placement':'direct','sidecar_leaf_id':leaf,'certified_leaf_id':leaf,'fallback_reason':'none'},
      {'record':'query','op_index':5,'query_id':cint(contract,'query_id_middle'),'gts_visited_leaf_ids':[leaf],'sidecar_ids':[c],'delta_ids':[b],'oracle_full_active_set_checked':True,'results':res(5)},
      {'record':'update','op_index':6,'op':'delete','stable_id':base,'placement':'deleted','sidecar_leaf_id':-1,'certified_leaf_id':-1,'fallback_reason':'none'},
      {'record':'rebuild','op_index':7,'trigger':'base_delete_immediate','metadata_only_release':True,'legacy_incremental_updater_used':False,'seeded_base_stable_ids':live,'pre_rebuild_active_hash':ah,'post_rebuild_active_hash':ah,'immutable_pool_hash_before':pool_hash,'immutable_pool_hash_after':pool_hash,'transient_direct_before':[c],'transient_delta_before':[b],'transient_direct_after':[],'transient_delta_after':[],'E1_tree_hash':202,'E1_leaf_stable_id_hash':fnv1a_i32(live),'E1_pivot_stable_id_hash':fnv1a_i32([]),'E1_leaf_set_exact':True,'E1_pivot_set_subset_of_live':True}, e1,
      {'record':'query','op_index':8,'query_id':cint(contract,'query_id_after'),'gts_visited_leaf_ids':[],'sidecar_ids':[],'delta_ids':[],'oracle_full_active_set_checked':True,'results':res(8)}]
    summary={'schema':'safe-c1-g2-rebuild-witness-v4','status':'PASS_G2_REBUILD_WITNESS_PENDING_INDEPENDENT_VALIDATOR','scope':'strict witness no performance claim','archived_incremental_updater_used':False,'rebuild':{'metadata_only_release':True,'full_pool_preserved':True,'active_set_preserved':True,'transient_tiers_cleared':True},'E0':{'leaf_set_exact':True,'pivot_set_subset_of_live':True},'E1':{'leaf_set_exact':True,'pivot_set_subset_of_live':True},'active_hash':{'before_rebuild':ah,'after_rebuild':ah},'immutable_full_pool_hash':{'host':pool_hash,'device_after_rebuild':pool_hash}}
    good=validate_engine(BUNDLE,report,rows,summary)
    if good: raise SystemExit('synthetic positive engine rejected:'+repr(good))
    bad=[dict(x) for x in rows]
    for x in bad:
      if x.get('record')=='rebuild': x['transient_delta_after']=[b]
    bad_errors=validate_engine(BUNDLE,report,bad,summary)
    if not any('rebuild_delta_after' in x for x in bad_errors): raise SystemExit('negative transient-clear fixture accepted:'+repr(bad_errors))
    with tempfile.TemporaryDirectory(prefix='g2_v4_cpu_test_') as d:
      d=Path(d); pre=d/'preflight.json'; r=d/'engine.jsonl'; s=d/'summary.json'; out=d/'final.json'; pre.write_text(json.dumps(report),encoding='utf-8'); write_jsonl(r,rows); s.write_text(json.dumps(summary),encoding='utf-8')
      cmd=[sys.executable,str(ROOT/'tools/finalize_g2_rebuild_witness.py'),'--root',str(ROOT),'--bundle',str(BUNDLE),'--preflight',str(pre),'--engine-results',str(r),'--engine-summary',str(s),'--out',str(out)]
      run=subprocess.run(cmd,text=True,capture_output=True,check=False)
      if run.returncode!=0: raise SystemExit('CLI positive finalizer failed:'+run.stdout+run.stderr)
      write_jsonl(r,bad); run=subprocess.run(cmd,text=True,capture_output=True,check=False)
      if run.returncode==0: raise SystemExit('CLI negative finalizer unexpectedly passed')
    print('PASS_CPU_ONLY_G2_REBUILD_PIPELINE_TEST')
    return 0
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
"""Independent CPU finalizer for a future G2 v4 guarded engine run.

It never invokes CUDA, GPU telemetry, or an engine binary.  It validates that a
submitted engine JSONL/summary still realizes the exact nine-event witness
which the CPU bundle preflight prepared.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct
from pathlib import Path
from typing import Any
from safe_c1_v4_rebuild_bundle_contract import (cint, fnv1a_i32, fnv1a_u64_bytes, load_i16, parse_trace, read_contract, replay_g2_trace, topk_exact)
from preflight_g2_rebuild_witness import ROOT, validate_bundle


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink(): raise ValueError('symlink_forbidden:'+str(path))
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError('not_object:'+str(path))
    return value

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink(): raise ValueError('symlink_forbidden:'+str(path))
    rows=[]
    for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): raise ValueError('blank_jsonl_line:'+str(n))
        v=json.loads(line)
        if not isinstance(v,dict): raise ValueError('jsonl_not_object:'+str(n))
        rows.append(v)
    if not rows: raise ValueError('empty_jsonl')
    return rows

def active_hash(pool_n:int, ids:list[int]) -> int:
    bit=bytearray(pool_n)
    for ident in ids: bit[ident]=1
    return fnv1a_u64_bytes(bytes(bit))

def float_pool_hash(values: Any) -> int:
    # C++ stores each quantized coordinate in the macro-expanded `short` type,
    # i.e. IEEE-754 float32, then FNV-1a hashes the entire immutable pool.
    return fnv1a_u64_bytes(b''.join(struct.pack('<f', float(value)) for value in values))

def require(v: bool, errors:list[str], text:str) -> None:
    if not v: errors.append(text)

def equal_list(value:Any, expected:list[int], errors:list[str], text:str) -> None:
    require(value==expected,errors,text)

def validate_engine(bundle:Path, preflight:dict[str,Any], rows:list[dict[str,Any],], summary:dict[str,Any]) -> list[str]:
    errors:list[str]=[]
    try:
        header,events=parse_trace(bundle/'trace.g2trc'); contract=read_contract(bundle/'g2_rebuild_contract.txt')
        replay=replay_g2_trace(header,events,contract)
    except Exception as exc:
        return ['contract_replay_exception:'+str(exc)]
    require(preflight.get('status')=='PASS_CPU_ONLY_G2_REBUILD_WITNESS_PREFLIGHT',errors,'preflight_not_pass')
    a,b,c=(cint(contract,k) for k in ('stable_id_a','stable_id_b','stable_id_c'))
    base_delete=cint(contract,'base_delete_id'); leaf=cint(contract,'sidecar_leaf_id')
    pool_i16=load_i16(bundle/'pool.i16', header.pool_n * header.dimension)
    queries_i16=load_i16(bundle/'queries.i16', header.query_n * header.dimension)
    expected_pool_hash=float_pool_hash(pool_i16)
    by_record:dict[str,list[dict[str,Any]]]={}
    for row in rows: by_record.setdefault(str(row.get('record')),[]).append(row)
    require(len(by_record.get('meta',[]))==1,errors,'need_one_meta')
    require(len(by_record.get('epoch',[]))==2,errors,'need_two_epochs')
    require(len(by_record.get('update',[]))==5,errors,'need_five_updates')
    require(len(by_record.get('query',[]))==3,errors,'need_three_queries')
    require(len(by_record.get('rebuild',[]))==1,errors,'need_one_rebuild')
    allowed={'meta','epoch','update','query','rebuild'}
    require(set(by_record)<=allowed,errors,'unexpected_record_type')
    if by_record.get('meta'):
        meta=by_record['meta'][0]
        require(meta.get('schema')=='safe-c1-g2-rebuild-witness-v4',errors,'meta_schema')
        require(meta.get('legacy_incremental_updater_used') is False,errors,'meta_legacy_updater')
        require(meta.get('stable_id_layout')=='identity_full_immutable_pool_seeded_stable_ids_v4',errors,'meta_layout')
        require(meta.get('host_full_pool_hash')==expected_pool_hash,errors,'meta_full_pool_hash')
        scope=str(meta.get('scope','')).lower()
        require('performance' in scope and 'no timing' in scope,errors,'meta_scope_missing_limits')
    initial=list(range(header.base_n)); live=replay['active_after_rebuild']
    expected_e={0:(initial,'E0'),1:(live,'E1')}
    got_epochs={int(r.get('epoch',-1)):r for r in by_record.get('epoch',[]) if isinstance(r.get('epoch'),int)}
    require(set(got_epochs)=={0,1},errors,'epoch_ids')
    for epoch,(ids,label) in expected_e.items():
        row=got_epochs.get(epoch,{})
        require(row.get('phase')==label,errors,'epoch_phase_'+label)
        require(row.get('leaf_set_exact') is True,errors,'epoch_leaf_exact_'+label)
        require(row.get('pivot_set_subset_of_live') is True,errors,'epoch_pivot_subset_'+label)
        require(row.get('live_count')==len(ids),errors,'epoch_live_count_'+label)
        require(row.get('live_stable_id_hash')==fnv1a_i32(ids),errors,'epoch_live_hash_'+label)
        require(row.get('leaf_stable_id_hash')==fnv1a_i32(ids),errors,'epoch_leaf_hash_'+label)
        require(isinstance(row.get('tree_hash'),int) and isinstance(row.get('pivot_stable_id_hash'),int),errors,'epoch_hash_types_'+label)
        piv=row.get('pivot_stable_ids'); require(isinstance(piv,list) and all(isinstance(x,int) and x in ids for x in piv),errors,'epoch_pivot_ids_'+label)
    updates={int(r.get('op_index',-1)):r for r in by_record.get('update',[]) if isinstance(r.get('op_index'),int)}
    require(set(updates)=={0,1,3,4,6},errors,'update_ops')
    def upd(op:int,sid:int,placement:str,side:int,cert:int,fallback:str)->None:
        x=updates.get(op,{})
        require(x.get('stable_id')==sid and x.get('placement')==placement and x.get('sidecar_leaf_id')==side and x.get('certified_leaf_id')==cert and x.get('fallback_reason')==fallback,errors,'update_contract_op'+str(op))
    upd(0,a,'direct',leaf,leaf,'none'); upd(1,b,'delta',-1,leaf,'capacity_full')
    upd(3,a,'deleted',leaf,leaf,'none'); upd(4,c,'direct',leaf,leaf,'none')
    x6=updates.get(6,{})
    require(x6.get('stable_id')==base_delete and x6.get('placement')=='deleted',errors,'update_contract_op6')
    queries={int(r.get('op_index',-1)):r for r in by_record.get('query',[]) if isinstance(r.get('op_index'),int)}
    require(set(queries)=={2,5,8},errors,'query_ops')
    expected_query_ids={2:cint(contract,'query_id_before'),5:cint(contract,'query_id_middle'),8:cint(contract,'query_id_after')}
    def q(op:int,side:list[int],delta:list[int],top:int)->None:
        x=queries.get(op,{})
        equal_list(x.get('sidecar_ids'),side,errors,'query_sidecar_'+str(op)); equal_list(x.get('delta_ids'),delta,errors,'query_delta_'+str(op))
        require(x.get('query_id')==expected_query_ids[op],errors,'query_id_'+str(op))
        require(x.get('oracle_full_active_set_checked') is True,errors,'query_oracle_'+str(op))
        visited=x.get('gts_visited_leaf_ids')
        require(isinstance(visited,list) and (op not in {2,5} or leaf in visited),errors,'query_visibility_receipt_'+str(op))
        res=x.get('results')
        expected=topk_exact(pool_i16,queries_i16,replay['query_active_ids'][str(op)],header,expected_query_ids[op])
        require(isinstance(res,list) and len(res)==header.k and bool(res) and isinstance(res[0],list) and res[0][0]==top,errors,'query_top_'+str(op))
        got_ids=[]
        if isinstance(res,list):
            for rank,row in enumerate(res):
                if not (isinstance(row,list) and len(row)==2 and isinstance(row[0],int) and isinstance(row[1],(int,float))):
                    errors.append('query_result_shape_'+str(op)); continue
                got_ids.append(row[0])
                if rank < len(expected):
                    expected_distance=math.sqrt(expected[rank][0])
                    require(math.isfinite(float(row[1])) and abs(float(row[1])-expected_distance) <= 1e-8*max(1.0,expected_distance),errors,'query_distance_'+str(op)+'_'+str(rank))
        require(got_ids==[ident for _,ident in expected],errors,'query_exact_full_oracle_ids_'+str(op))
    q(2,[a],[b],a); q(5,[c],[b],c); q(8,[],[],c)
    if by_record.get('rebuild'):
        rb=by_record['rebuild'][0]
        require(rb.get('op_index')==7 and rb.get('trigger')=='base_delete_immediate',errors,'rebuild_trigger')
        require(rb.get('metadata_only_release') is True and rb.get('legacy_incremental_updater_used') is False,errors,'rebuild_safety_flags')
        equal_list(rb.get('seeded_base_stable_ids'),live,errors,'rebuild_seed_ids')
        require(rb.get('pre_rebuild_active_hash')==active_hash(header.pool_n,live),errors,'rebuild_pre_active_hash')
        require(rb.get('post_rebuild_active_hash')==active_hash(header.pool_n,live),errors,'rebuild_post_active_hash')
        require(rb.get('immutable_pool_hash_before')==expected_pool_hash and rb.get('immutable_pool_hash_after')==expected_pool_hash,errors,'rebuild_pool_hash')
        equal_list(rb.get('transient_direct_before'),[c],errors,'rebuild_direct_before')
        equal_list(rb.get('transient_delta_before'),[b],errors,'rebuild_delta_before')
        equal_list(rb.get('transient_direct_after'),[],errors,'rebuild_direct_after')
        equal_list(rb.get('transient_delta_after'),[],errors,'rebuild_delta_after')
        require(rb.get('E1_leaf_set_exact') is True and rb.get('E1_pivot_set_subset_of_live') is True,errors,'rebuild_e1_set_flags')
        require(rb.get('E1_tree_hash')==got_epochs.get(1,{}).get('tree_hash'),errors,'rebuild_e1_tree_bind')
        require(rb.get('E1_leaf_stable_id_hash')==fnv1a_i32(live),errors,'rebuild_e1_leaf_hash')
    require(summary.get('schema')=='safe-c1-g2-rebuild-witness-v4',errors,'summary_schema')
    require(summary.get('status')=='PASS_G2_REBUILD_WITNESS_PENDING_INDEPENDENT_VALIDATOR',errors,'summary_status')
    require(summary.get('archived_incremental_updater_used') is False,errors,'summary_legacy')
    rbsummary=summary.get('rebuild',{})
    require(isinstance(rbsummary,dict) and all(rbsummary.get(k) is True for k in ('metadata_only_release','full_pool_preserved','active_set_preserved','transient_tiers_cleared')),errors,'summary_rebuild')
    require(summary.get('active_hash',{}).get('before_rebuild')==active_hash(header.pool_n,live),errors,'summary_active_pre')
    require(summary.get('active_hash',{}).get('after_rebuild')==active_hash(header.pool_n,live),errors,'summary_active_post')
    pool_summary=summary.get('immutable_full_pool_hash',{})
    require(isinstance(pool_summary,dict) and pool_summary.get('host')==expected_pool_hash and pool_summary.get('device_after_rebuild')==expected_pool_hash,errors,'summary_full_pool_hash')
    for label in ('E0','E1'):
        require(isinstance(summary.get(label),dict) and summary[label].get('leaf_set_exact') is True and summary[label].get('pivot_set_subset_of_live') is True,errors,'summary_'+label)
    forbidden=('throughput','latency result','speedup','qps')
    scope=str(summary.get('scope','')).lower()
    require(not any(word in scope for word in forbidden),errors,'summary_scope_overclaim')
    return errors

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--preflight',type=Path,required=True); p.add_argument('--engine-results',type=Path,required=True); p.add_argument('--engine-summary',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    errors=[]; report:dict[str,Any]={}
    try:
        if a.root.resolve()!=ROOT.resolve(): errors.append('root_not_canonical')
        cpu_report,cpu_errors=validate_bundle(a.root.resolve(),a.bundle.resolve())
        if cpu_errors: errors.append('bundle_revalidation_failed')
        pre=load_json(a.preflight); rows=load_jsonl(a.engine_results); summ=load_json(a.engine_summary)
        errors.extend(validate_engine(a.bundle.resolve(),pre,rows,summ))
        report={'schema':'safe-c1-g2-v4-independent-finalizer','status':'PASS_G2_REBUILD_WITNESS_INDEPENDENTLY_VALIDATED' if not errors else 'FAIL_G2_REBUILD_WITNESS_INDEPENDENT_VALIDATION','gpu_used_by_finalizer':False,'cuda_binary_executed_by_finalizer':False,'scope':'validates one strict witness artifact; does not establish performance/general rebuild/range/C2/all-input tie claims','root':str(a.root.resolve()),'bundle':str(a.bundle.resolve()),'engine_results_sha256':hashlib.sha256(a.engine_results.read_bytes()).hexdigest(),'engine_summary_sha256':hashlib.sha256(a.engine_summary.read_bytes()).hexdigest(),'errors':errors}
    except Exception as exc:
        errors.append('finalizer_exception:'+str(exc)); report={'schema':'safe-c1-g2-v4-independent-finalizer','status':'FAIL_G2_REBUILD_WITNESS_INDEPENDENT_VALIDATION','gpu_used_by_finalizer':False,'cuda_binary_executed_by_finalizer':False,'errors':errors}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(report['status']); return 0 if not errors else 2
if __name__=='__main__': raise SystemExit(main())

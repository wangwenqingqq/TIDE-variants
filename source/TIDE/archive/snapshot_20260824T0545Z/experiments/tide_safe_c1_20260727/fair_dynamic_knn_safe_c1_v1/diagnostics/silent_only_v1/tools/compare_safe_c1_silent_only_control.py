#!/usr/bin/python3.12
"""Standalone exact-oracle comparator for the silent-only diagnostic.

It fixes the v1b control path and hashes; it does not import any runner,
validator, CUDA, NVML, or subprocess facility. It independently validates both
raw JSONLs before comparing semantic fields. It intentionally does not
interpret latency and cannot witness concrete leaf-list identity.
"""
from __future__ import annotations
import argparse
import array
import hashlib
import json
import os
import re
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT=Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
DIAG=ROOT/"diagnostics/silent_only_v1"
BUNDLE_PATH=ROOT/"inputs/e1_frozen_base_knn_projection_v1"
CONTROL=ROOT/"runs/safe-c1-tradeoff-pilot-v1b-20260730"
SCHEMA_CONTROL="fair-safe-c1-latency-tradeoff-e1-v1"
SCHEMA_CANDIDATE="fair-safe-c1-latency-silent-only-e1-v1"
VARIANT="optimization_diagnostic_silent_only"
MANIFEST_SCHEMA="fair-safe-c1-silent-only-diagnostic-source-manifest-v1"
HASH_DOMAIN="fair-safe-c1-latency-tradeoff-e1-merged-result-v1"
TRACE_MAGIC=b"E1GTRC01"; TRACE_HEADER=struct.Struct("<8sI6IfQ"); TRACE_EVENT=struct.Struct("<IB3xi")
INSERT,KNN=1,3
RUN_RE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
INPUT_HASHES={
 "manifest.json":"68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
 "metadata.json":"2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
 "trace.e1gtrc":"9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
 "pool.i16":"899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
 "queries.i16":"18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
 "stable_id_to_pool_row.i32":"93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
 "initial_base_stable_ids.i32":"6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}
CONTROL_HASHES={
 "engine.jsonl":"4a32720f49ac49bed6bd6f2755532231ea8a04e5d3c797e342e56ef22cb35f11",
 "summary.json":"5d9f197cf7297cb970a69f8a5fdb7255efad9d06c37adac343377af5a185c103",
 "guard_receipt.json":"1400941c7d898c2f8daa4884c85c8fa8e1a027f1ffdcc2f88af4a574070b63f8",
 "independent_validation.json":"2719fcffa018e352d4c7b8da480c889545a57d5a39605bff424511fb8acb042e",
 "admission.env":"a979ba830a2f39bf12ddace378b1be7106835233cd2e0f8e1bd8917175c75721",
}
NATIVE_PATH="real_gts_vector_topk_receipt_all_native_leaf_rows"
FALLBACK_PATH="exact_full_immutable_base_range_fallback_no_gts_receipt"
CONTROL_SCOPE=("native query_knn candidate API versus current exact query_range(UINT64_MAX) "
               "full-result immutable-base fallback; not a fair same-API speedup")
CANDIDATE_SCOPE=("silent-only diagnostic: only two vector-topk native stdout statements are suppressed; "
                 "native query_knn candidate API versus current exact query_range(UINT64_MAX) "
                 "full-result immutable-base fallback; not a fair same-API speedup")
CONTROL_LIMITS=["not_same_api","no CUDA event/kernel timing","external delta is CPU wrapper work","no delete/range-workload/rebuild/direct-sidecar claim"]
CANDIDATE_LIMITS=CONTROL_LIMITS+["optimization_diagnostic_silent_only","non_publication_variant"]
COMMON_KEYS={"record","not_same_api","timing_claim","condition","pass","case_ordinal","op_index","query_id","external_global_delta_live","api_host_ns","post_api_external_delta_merge_excluded_from_api_host_ns","api_result_count_before_adapter_truncation","base_candidate_count","visited_leaf_count","full_immutable_base_candidate_count","exact_full_immutable_base_fallback","base_path","merged_overlap_at_k","merged_exact_match","merged_result_sha256","merged"}

class Fail(RuntimeError): pass
def req(x:bool,m:str)->None:
    if not x: raise Fail(m)
def integer(x:Any,lo:int=0)->bool: return isinstance(x,int) and not isinstance(x,bool) and x>=lo
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def regular(p:Path,label:str,private:bool=True)->None:
    s=p.lstat(); req(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode),"unsafe "+label)
    req(s.st_uid==0 and s.st_gid==0,label+" ownership")
    req((stat.S_IMODE(s.st_mode)&(0o077 if private else 0o022))==0,label+" permissions")
def directory(p:Path,label:str,private:bool=True)->None:
    s=p.lstat();req(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode),"unsafe "+label)
    req(s.st_uid==0 and s.st_gid==0,label+" ownership")
    if private:req(stat.S_IMODE(s.st_mode)==0o700,label+" not 0700")
def read_json(p:Path,label:str,private:bool=True)->dict[str,Any]:
    regular(p,label,private)
    try:x=json.loads(p.read_text("utf-8"))
    except Exception as e:raise Fail("invalid "+label+" JSON") from e
    req(isinstance(x,dict),label+" object");return x
def read_jsonl(p:Path)->list[dict[str,Any]]:
    regular(p,"engine JSONL");o=[]
    with p.open("r",encoding="utf-8",newline="") as f:
        for i,line in enumerate(f,1):
            req(line.endswith("\n") and line!="\n","JSONL LF "+str(i))
            try:x=json.loads(line)
            except Exception as e:raise Fail("JSONL parse "+str(i)) from e
            req(isinstance(x,dict),"JSONL object "+str(i));o.append(x)
    return o
def arr(p:Path,t:str,w:int,label:str)->array.array:
    regular(p,label,False);raw=p.read_bytes();req(len(raw)%w==0,label+" bytes")
    a=array.array(t);req(a.itemsize==w,label+" width");a.frombytes(raw)
    if sys.byteorder!="little":a.byteswap()
    return a
@dataclass(frozen=True)
class Bundle:
    dim:int;base_n:int;pool_n:int;query_n:int;k:int;events:tuple[tuple[int,int,int],...];pool:array.array;queries:array.array;mapping:array.array;base:array.array
@dataclass(frozen=True)
class Case:
    ordinal:int;op:int;qid:int;delta:int;active:bytes;oracle:tuple[tuple[int,int],...]
def bundle(p:Path)->Bundle:
    req(p==BUNDLE_PATH and not p.is_symlink(),"wrong sealed bundle");directory(p,"sealed bundle",False)
    for n,h in INPUT_HASHES.items():regular(p/n,"sealed "+n,False);req(sha(p/n)==h,"sealed hash "+n)
    raw=(p/"trace.e1gtrc").read_bytes();req(len(raw)>=TRACE_HEADER.size,"trace short")
    magic,v,d,b,r,pn,qn,k,_rad,cnt=TRACE_HEADER.unpack_from(raw)
    req(magic==TRACE_MAGIC and v==1 and d>0 and b>=k>0 and pn==b+r and qn>0,"trace header")
    req(len(raw)==TRACE_HEADER.size+cnt*TRACE_EVENT.size,"trace bytes");events=[]
    for i in range(cnt):
        op,code,arg=TRACE_EVENT.unpack_from(raw,TRACE_HEADER.size+i*TRACE_EVENT.size)
        req(op==i and code in (INSERT,KNN),"trace op")
        req((code==INSERT and b<=arg<pn) or (code==KNN and 0<=arg<qn),"trace argument");events.append((op,code,arg))
    pool=arr(p/"pool.i16","h",2,"pool");queries=arr(p/"queries.i16","h",2,"queries");mapping=arr(p/"stable_id_to_pool_row.i32","i",4,"mapping");base=arr(p/"initial_base_stable_ids.i32","i",4,"base")
    req(len(pool)==pn*d and len(queries)==qn*d and len(mapping)==pn and len(base)==b,"bundle shapes")
    req(len(set(mapping))==pn and len(set(base))==b and min(base)>=0 and max(base)<pn,"bundle ids")
    return Bundle(d,b,pn,qn,k,tuple(events),pool,queries,mapping,base)
def l2(b:Bundle,s:int,q:int)->int:
    r=int(b.mapping[s]);x=r*b.dim;y=q*b.dim;t=0
    for i in range(b.dim):
        z=int(b.pool[x+i])-int(b.queries[y+i]);t+=z*z
    return t
def make_cases(b:Bundle)->tuple[Case,...]:
    active=bytearray(b.pool_n)
    for s in b.base:active[int(s)]=1
    delta=0;o=[]
    for op,code,arg in b.events:
        if code==INSERT:req(not active[arg],"duplicate insert");active[arg]=1;delta+=1
        else:
            xs=sorted((l2(b,s,arg),s) for s,on in enumerate(active) if on)[:b.k]
            o.append(Case(len(o),op,arg,delta,bytes(active),tuple((s,d) for d,s in xs)))
    req(len(o)==136 and delta==169,"trace cardinality");return tuple(o)
def getrows(v:Any,b:Bundle,c:Case,label:str)->tuple[tuple[int,int],...]:
    req(isinstance(v,list) and len(v)==b.k,label+" size");out=[];seen=set();prior=None
    for i,x in enumerate(v):
        req(isinstance(x,list) and len(x)==2,label+" pair")
        s,d=x;req(integer(s) and s<b.pool_n and integer(d),label+" types")
        req(s not in seen and c.active[s],label+" active")
        req(l2(b,s,c.qid)==d,label+" l2")
        key=(d,s);req(prior is None or prior<key,label+" order");prior=key;seen.add(s);out.append((s,d))
    return tuple(out)
def reshash(xs:tuple[tuple[int,int],...])->str:
    return hashlib.sha256((HASH_DOMAIN+"\n"+"".join(str(s)+":"+str(d)+"\n" for s,d in xs)).encode("ascii")).hexdigest()
def distro(v:list[int])->dict[str,int]:
    req(v,"distribution");a=sorted(v);return {"n":len(a),"min_ns":a[0],"p50_ns":a[(50*(len(a)-1))//100],"p95_ns":a[(95*(len(a)-1))//100],"max_ns":a[-1],"mean_ns":sum(a)//len(a)}
def admission(p:Path)->None:
    regular(p,"candidate admission");d={}
    for l in p.read_text("ascii").splitlines():
        if not l or l.startswith("#"):continue
        req(l.count("=")==1,"admission syntax");k,v=l.split("=",1);req(k not in d and v,"admission duplicate");d[k]=v
    req(d.get("schema")=="e1-frozen-base-knn-projection-admission-v2" and d.get("status")=="PASS","admission label")
    req(d.get("ops")=="insert,knn" and d.get("excluded_ops")=="delete,range" and d.get("base_immutable")=="true","admission semantics")

def manifest(p:Path)->dict[str,Any]:
    m=read_json(p,"variant manifest");req(m.get("schema")==MANIFEST_SCHEMA and m.get("diagnostic_variant")==VARIANT and m.get("publication_eligible") is False,"manifest label")
    req(m.get("algorithm_data_receipt_contract",{}).get("merged_result_hash_domain")==HASH_DOMAIN,"manifest domain")
    rd=m.get("reviewed_diff",{});rp=Path(rd.get("path",""));req(rp==DIAG/"provenance/silent_only_reviewed_diff.patch","manifest diff path")
    regular(rp,"reviewed diff");req(rd.get("sha256")==sha(rp) and rd.get("audit_result")=="PASS_EXACT_ALLOWED_SOURCE_DIFFERENCES_ONLY","manifest diff")
    closure=m.get("source_closure");req(isinstance(closure,list) and len(closure)>=9,"manifest closure")
    lines=[]
    for x in closure:
        req(isinstance(x,dict),"closure entry");q=Path(x.get("path",""));req(q.is_relative_to(ROOT),"closure escape")
        regular(q,"closure",q.is_relative_to(DIAG));req(x.get("sha256")==sha(q),"closure hash")
        lines.append(str(q).replace(str(ROOT)+"/","")+"\t"+x["sha256"])
    req(m.get("full_source_closure_sha256")==hashlib.sha256(("\n".join(sorted(lines))+"\n").encode("ascii")).hexdigest(),"closure digest")
    return m

def check_records(rs:list[dict[str,Any],],b:Bundle,cs:tuple[Case,...],candidate:bool)->dict[str,Any]:
    req(len(rs)==272,"record count");nv=[];fv=[];ne=no=fe=0;sem=[];at=0
    for c in cs:
        order=("native_query_knn_candidate","exact_query_range_full_base_fallback") if c.ordinal%2==0 else ("exact_query_range_full_base_fallback","native_query_knn_candidate")
        for cond in order:
            r=rs[at];at+=1;expected={"schema"}|COMMON_KEYS
            if candidate:expected|={"diagnostic_variant","publication_eligible"}
            req(set(r)==expected,"record field set "+str(at))
            req(r.get("schema")== (SCHEMA_CANDIDATE if candidate else SCHEMA_CONTROL) and r.get("record")=="measurement","record schema "+str(at))
            if candidate:req(r.get("diagnostic_variant")==VARIANT and r.get("publication_eligible") is False,"candidate tag "+str(at))
            req(r.get("not_same_api") is True and r.get("timing_claim")=="latency_quality_tradeoff_only","record claim "+str(at))
            req(r.get("condition")==cond and r.get("pass")==0 and r.get("case_ordinal")==c.ordinal and r.get("op_index")==c.op and r.get("query_id")==c.qid and r.get("external_global_delta_live")==c.delta,"record trace "+str(at))
            req(integer(r.get("api_host_ns"),1) and r.get("post_api_external_delta_merge_excluded_from_api_host_ns") is True,"record timing "+str(at))
            x=getrows(r["merged"],b,c,"merged");req(r.get("merged_result_sha256")==reshash(x),"record hash "+str(at))
            ov=len({s for s,_ in x}&{s for s,_ in c.oracle});ex=x==c.oracle
            req(r.get("merged_overlap_at_k")==ov and r.get("merged_exact_match") is ex,"record quality "+str(at))
            if cond=="native_query_knn_candidate":
                req(r.get("api_result_count_before_adapter_truncation")==b.k and integer(r.get("base_candidate_count"),b.k) and r["base_candidate_count"]<=b.base_n and integer(r.get("visited_leaf_count")) and r.get("full_immutable_base_candidate_count")==0 and r.get("exact_full_immutable_base_fallback") is False and r.get("base_path")==NATIVE_PATH,"native contract "+str(at));nv.append(r["api_host_ns"]);ne+=int(ex);no+=ov
            else:
                req(r.get("api_result_count_before_adapter_truncation")==b.base_n and r.get("base_candidate_count")==b.base_n and integer(r.get("visited_leaf_count")) and r.get("full_immutable_base_candidate_count")==b.base_n and r.get("exact_full_immutable_base_fallback") is True and r.get("base_path")==FALLBACK_PATH and ex,"fallback contract "+str(at));fv.append(r["api_host_ns"]);fe+=1
            semantic={k:r[k] for k in sorted(COMMON_KEYS-{"record","api_host_ns"})}
            sem.append(semantic)
    return {"native":nv,"fallback":fv,"quality":{"native_exact_set_count":ne,"native_overlap_sum":no,"fallback_exact_set_count":fe,"fallback_expected_exact_set_count":136},"semantic":sem,"semantic_fingerprint_sha256":hashlib.sha256((json.dumps(sem,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()}

def check_summary(s:dict[str,Any],m:dict[str,Any],candidate:bool)->None:
    expected={"schema","mode","status","not_same_api","timing_claim","scope","warmup_passes_discarded","measured_passes","measurement_records","post_api_external_delta_merge_excluded_from_api_host_ns","json_and_oracle_excluded_from_api_host_ns","native_api_host_ns","fallback_api_host_ns","quality","limitations"}
    if candidate:expected|={"diagnostic_variant","publication_eligible"}
    req(set(s)==expected,"summary fields")
    req(s.get("schema")== (SCHEMA_CANDIDATE if candidate else SCHEMA_CONTROL) and s.get("mode")=="timing" and s.get("status")==("PASS_SILENT_ONLY_DIAGNOSTIC_TIMING" if candidate else "PASS_TRADEOFF_TIMING"),"summary status")
    if candidate:req(s.get("diagnostic_variant")==VARIANT and s.get("publication_eligible") is False,"summary candidate tag")
    req(s.get("not_same_api") is True and s.get("timing_claim")=="latency_quality_tradeoff_only" and s.get("scope")== (CANDIDATE_SCOPE if candidate else CONTROL_SCOPE),"summary scope")
    req(s.get("warmup_passes_discarded")==1 and s.get("measured_passes")==1 and s.get("measurement_records")==272,"summary count")
    req(s.get("post_api_external_delta_merge_excluded_from_api_host_ns") is True and s.get("json_and_oracle_excluded_from_api_host_ns") is True,"summary envelope")
    req(s.get("native_api_host_ns")==distro(m["native"]) and s.get("fallback_api_host_ns")==distro(m["fallback"]) and s.get("quality")==m["quality"],"summary metrics")
    req(s.get("limitations")== (CANDIDATE_LIMITS if candidate else CONTROL_LIMITS),"summary limits")

def atomic(p:Path,x:dict[str,Any])->None:
    req(not p.exists() and not p.is_symlink(),"refuse overwrite")
    fd,t=tempfile.mkstemp(prefix=".silent-compare.",dir=p.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(json.dumps(x,sort_keys=True,indent=2)+"\n");f.flush();os.fsync(f.fileno())
        os.chmod(t,0o600);os.replace(t,p)
    except BaseException:
        try:os.unlink(t)
        except FileNotFoundError:pass
        raise

def run(a:argparse.Namespace)->dict[str,Any]:
    bpath=Path(a.bundle).absolute();mpath=Path(a.variant_manifest).absolute();out=Path(a.out).absolute()
    req(bpath==BUNDLE_PATH and mpath==DIAG/"provenance/silent_only_source_manifest.json","fixed bundle/manifest")
    req(RUN_RE.fullmatch(a.candidate_run_name) is not None and a.candidate_run_name!="safe-c1-tradeoff-pilot-v1b-20260730","candidate name")
    cand=ROOT/"runs"/a.candidate_run_name;req(out==cand/"silent_vs_control_validation.json","canonical comparison output");directory(cand,"candidate run")
    admission(cand/"admission.env");m=manifest(mpath)
    iv=read_json(cand/"independent_validation.json","candidate independent validation")
    req(iv.get("schema")=="fair-safe-c1-silent-only-output-validator-v1" and iv.get("status")=="PASS_SEMANTIC_OUTPUT_ONLY" and iv.get("events_sha256")==sha(cand/"engine.jsonl") and iv.get("summary_sha256")==sha(cand/"summary.json") and iv.get("variant_manifest_sha256")==sha(mpath),"candidate validation binding")
    directory(CONTROL,"fixed control")
    for n,h in CONTROL_HASHES.items():regular(CONTROL/n,"control "+n);req(sha(CONTROL/n)==h,"control hash "+n)
    civ=read_json(CONTROL/"independent_validation.json","control validation");req(civ.get("status")=="PASS","control validation status")
    b=bundle(bpath);cs=make_cases(b)
    cr=read_jsonl(CONTROL/"engine.jsonl");csum=read_json(CONTROL/"summary.json","control summary")
    sr=read_jsonl(cand/"engine.jsonl");ssum=read_json(cand/"summary.json","candidate summary")
    cm=check_records(cr,b,cs,False);sm=check_records(sr,b,cs,True)
    check_summary(csum,cm,False);check_summary(ssum,sm,True)
    req(cm["semantic"]==sm["semantic"],"per-query non-timing semantic fields differ from v1b")
    return {"schema":"fair-safe-c1-silent-only-control-comparator-v1","status":"PASS_SEMANTIC_EQUIVALENCE_COUNTS_ONLY","diagnostic_variant":VARIANT,"publication_eligible":False,"gpu_workload_launched_by_comparator":False,"scope":"fixed-v1b per-query output/candidate-count/visited-leaf-count/quality equivalence only; timing is not interpreted, this is not a same-API speedup and not publication evidence","control":{"run_name":"safe-c1-tradeoff-pilot-v1b-20260730","engine_jsonl_sha256":CONTROL_HASHES["engine.jsonl"],"summary_json_sha256":CONTROL_HASHES["summary.json"],"semantic_fingerprint_sha256":cm["semantic_fingerprint_sha256"]},"candidate":{"run_name":a.candidate_run_name,"engine_jsonl_sha256":sha(cand/"engine.jsonl"),"summary_json_sha256":sha(cand/"summary.json"),"semantic_fingerprint_sha256":sm["semantic_fingerprint_sha256"],"variant_manifest_sha256":sha(mpath)},"comparison":{"measurement_records_per_run":272,"native_records_per_run":136,"fallback_records_per_run":136,"matched_fields":["merged","merged_result_sha256","base_candidate_count","visited_leaf_count","merged_overlap_at_k","merged_exact_match","base_path","API result counts","trace/ABBA fields"],"timing":"NOT_INTERPRETED"},"leaf_identity_equivalence":{"status":"NOT_WITNESSED","control_has_leaf_list_witness":False,"reason":"v1b exposes only base_candidate_count and visited_leaf_count; it contains no ordered leaf-ID list, leaf-list hash, leaf-set hash, or candidate-ID-set witness."},"missing_equivalence_witnesses":["visited leaf membership/order is not witnessed by count equality","control has no per-stage allocator, synchronization, kernel, or device timing counters"]}

def cpu_self_check()->dict[str,Any]:
    # Deliberately CPU/filesystem only: no CUDA/NVML helper/subprocess import or call.
    manifest(DIAG/"provenance/silent_only_source_manifest.json")
    b=bundle(BUNDLE_PATH); cs=make_cases(b)
    directory(CONTROL,"fixed control")
    for n,h in CONTROL_HASHES.items():
        regular(CONTROL/n,"control "+n);req(sha(CONTROL/n)==h,"control hash "+n)
    civ=read_json(CONTROL/"independent_validation.json","control validation")
    req(civ.get("status")=="PASS","control validation status")
    return {"schema":"fair-safe-c1-silent-only-control-comparator-v1","status":"PASS_CPU_ONLY_STATIC_SELFCHECK","gpu_workload_launched":False,"nvml_used":False,"knn_cases":len(cs),"control_engine_jsonl_sha256":CONTROL_HASHES["engine.jsonl"],"variant_manifest_sha256":sha(DIAG/"provenance/silent_only_source_manifest.json")}

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--self-check",action="store_true");p.add_argument("--bundle");p.add_argument("--candidate-run-name");p.add_argument("--variant-manifest");p.add_argument("--out");a=p.parse_args()
    try:
        if a.self_check:
            req(all(x is None for x in (a.bundle,a.candidate_run_name,a.variant_manifest,a.out)),"self-check cannot accept comparison arguments")
            x=cpu_self_check();print(json.dumps(x,sort_keys=True));return 0
        req(all(x is not None for x in (a.bundle,a.candidate_run_name,a.variant_manifest,a.out)),"all arguments required")
        x=run(a);atomic(Path(a.out).absolute(),x);print(json.dumps(x,sort_keys=True));return 0
    except Fail as e:print("FAIL: "+str(e),file=sys.stderr);return 2
if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/python3.12
"""Pure-stdlib semantic validator for the non-publication silent-only variant.

This module never imports the CUDA runner/GTS code and never launches a GPU.
It independently replays the sealed integer-L2 trace and validates only
candidate output semantics and source provenance. A future separate guard must
validate authorization/device state before any candidate run.
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

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
DIAG = ROOT / "diagnostics/silent_only_v1"
BUNDLE_PATH = ROOT / "inputs/e1_frozen_base_knn_projection_v1"
SCHEMA = "fair-safe-c1-latency-silent-only-e1-v1"
MANIFEST_SCHEMA = "fair-safe-c1-silent-only-diagnostic-source-manifest-v1"
VARIANT = "optimization_diagnostic_silent_only"
HASH_DOMAIN = "fair-safe-c1-latency-tradeoff-e1-merged-result-v1"
TRACE_MAGIC = b"E1GTRC01"
TRACE_HEADER = struct.Struct("<8sI6IfQ")
TRACE_EVENT = struct.Struct("<IB3xi")
INSERT, KNN = 1, 3
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
INPUT_HASHES = {
    "manifest.json": "68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
    "metadata.json": "2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
    "trace.e1gtrc": "9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
    "pool.i16": "899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
    "queries.i16": "18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
    "stable_id_to_pool_row.i32": "93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
    "initial_base_stable_ids.i32": "6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}
NATIVE_PATH = "real_gts_vector_topk_receipt_all_native_leaf_rows"
FALLBACK_PATH = "exact_full_immutable_base_range_fallback_no_gts_receipt"
SCOPE = ("silent-only diagnostic: only two vector-topk native stdout statements are suppressed; "
         "native query_knn candidate API versus current exact query_range(UINT64_MAX) "
         "full-result immutable-base fallback; not a fair same-API speedup")
LIMITS = [
    "not_same_api", "no CUDA event/kernel timing", "external delta is CPU wrapper work",
    "no delete/range-workload/rebuild/direct-sidecar claim",
    "optimization_diagnostic_silent_only", "non_publication_variant",
]
MEAS_KEYS = {
    "schema", "diagnostic_variant", "publication_eligible", "record", "not_same_api",
    "timing_claim", "condition", "pass", "case_ordinal", "op_index", "query_id",
    "external_global_delta_live", "api_host_ns",
    "post_api_external_delta_merge_excluded_from_api_host_ns",
    "api_result_count_before_adapter_truncation", "base_candidate_count",
    "visited_leaf_count", "full_immutable_base_candidate_count",
    "exact_full_immutable_base_fallback", "base_path", "merged_overlap_at_k",
    "merged_exact_match", "merged_result_sha256", "merged",
}
SUMMARY_KEYS = {
    "schema", "diagnostic_variant", "publication_eligible", "mode", "status",
    "not_same_api", "timing_claim", "scope", "warmup_passes_discarded",
    "measured_passes", "measurement_records",
    "post_api_external_delta_merge_excluded_from_api_host_ns",
    "json_and_oracle_excluded_from_api_host_ns", "native_api_host_ns",
    "fallback_api_host_ns", "quality", "limitations",
}

class Fail(RuntimeError): pass
def require(ok: bool, msg: str) -> None:
    if not ok: raise Fail(msg)
def plain_int(v: Any, lo: int = 0) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo
def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def regular(p: Path, label: str, private: bool) -> None:
    st=p.lstat()
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode), "unsafe "+label)
    require(st.st_uid==0 and st.st_gid==0, label+" is not root owned")
    if private: require(stat.S_IMODE(st.st_mode)&0o077==0, label+" is not root private")
    else: require(stat.S_IMODE(st.st_mode)&0o022==0, label+" is non-root writable")
def directory(p: Path, label: str, private: bool=True) -> None:
    st=p.lstat()
    require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode), "unsafe "+label)
    require(st.st_uid==0 and st.st_gid==0, label+" is not root owned")
    if private: require(stat.S_IMODE(st.st_mode)==0o700, label+" is not root-private 0700")
def json_file(p: Path, label: str, private: bool=True) -> dict[str,Any]:
    regular(p,label,private)
    try: x=json.loads(p.read_text("utf-8"))
    except Exception as e: raise Fail("invalid "+label+" JSON: "+str(e)) from e
    require(isinstance(x,dict), label+" root is not object")
    return x
def jsonl(p: Path) -> list[dict[str,Any]]:
    regular(p,"engine JSONL",True); out=[]
    with p.open("r",encoding="utf-8",newline="") as f:
        for n,line in enumerate(f,1):
            require(line.endswith("\n") and line!="\n", "invalid JSONL line "+str(n))
            try: x=json.loads(line)
            except json.JSONDecodeError as e: raise Fail("bad JSONL "+str(n)) from e
            require(isinstance(x,dict), "JSONL object "+str(n)); out.append(x)
    return out
def read_env(p: Path) -> dict[str,str]:
    regular(p,"admission",True); d={}
    for line in p.read_text("ascii").splitlines():
        if not line or line.startswith("#"): continue
        require(line.count("=")==1 and line.index("=")>0, "malformed admission")
        k,v=line.split("=",1)
        require(k not in d and v and not any(c.isspace() for c in v), "bad admission field")
        d[k]=v
    return d
def need(d:dict[str,str], k:str) -> str:
    require(k in d,"admission omits "+k); return d[k]
def env_int(d:dict[str,str], k:str) -> int:
    x=need(d,k); require(x.isdecimal(),"admission decimal "+k); return int(x)
def le_array(p:Path, typecode:str, width:int, label:str)->array.array:
    regular(p,label,False); raw=p.read_bytes(); require(len(raw)%width==0,label+" bytes")
    a=array.array(typecode); require(a.itemsize==width,label+" host width"); a.frombytes(raw)
    if sys.byteorder!="little": a.byteswap()
    return a

@dataclass(frozen=True)
class Bundle:
    dim:int; base_n:int; pool_n:int; query_n:int; k:int
    events:tuple[tuple[int,int,int],...]; pool:array.array; queries:array.array
    mapping:array.array; base:array.array
@dataclass(frozen=True)
class Case:
    ordinal:int; op:int; qid:int; delta:int; active:bytes; oracle:tuple[tuple[int,int],...]

def load_bundle(p:Path)->Bundle:
    require(p==BUNDLE_PATH and not p.is_symlink(),"wrong sealed bundle")
    directory(p,"sealed bundle",False)
    for name,h in INPUT_HASHES.items():
        f=p/name; regular(f,"sealed input "+name,False); require(sha_file(f)==h,"sealed drift "+name)
    raw=(p/"trace.e1gtrc").read_bytes(); require(len(raw)>=TRACE_HEADER.size,"short trace")
    magic,ver,dim,base_n,reservoir,pool_n,query_n,k,_radius,count=TRACE_HEADER.unpack_from(raw)
    require(magic==TRACE_MAGIC and ver==1 and dim>0 and base_n>=k>0,"trace header")
    require(pool_n==base_n+reservoir and query_n>0,"trace populations")
    require(len(raw)==TRACE_HEADER.size+count*TRACE_EVENT.size,"trace bytes")
    events=[]
    for i in range(count):
        op,code,arg=TRACE_EVENT.unpack_from(raw,TRACE_HEADER.size+i*TRACE_EVENT.size)
        require(op==i and code in (INSERT,KNN),"bad trace op")
        require((code==INSERT and base_n<=arg<pool_n) or (code==KNN and 0<=arg<query_n),"bad trace arg")
        events.append((op,code,arg))
    pool=le_array(p/"pool.i16","h",2,"pool"); queries=le_array(p/"queries.i16","h",2,"queries")
    mapping=le_array(p/"stable_id_to_pool_row.i32","i",4,"mapping"); base=le_array(p/"initial_base_stable_ids.i32","i",4,"base")
    require(len(pool)==pool_n*dim and len(queries)==query_n*dim,"matrix shape")
    require(len(mapping)==pool_n and len(base)==base_n and len(set(mapping))==pool_n,"mapping shape")
    require(len(set(base))==base_n and min(base)>=0 and max(base)<pool_n,"base ids")
    return Bundle(dim,base_n,pool_n,query_n,k,tuple(events),pool,queries,mapping,base)
def distance(b:Bundle,stable:int,qid:int)->int:
    r=int(b.mapping[stable]); x=r*b.dim; y=qid*b.dim; total=0
    for i in range(b.dim):
        d=int(b.pool[x+i])-int(b.queries[y+i]); total+=d*d
    return total
def exact_topk(b:Bundle,active:bytearray,qid:int)->tuple[tuple[int,int],...]:
    vals=[(distance(b,s,qid),s) for s,on in enumerate(active) if on]
    vals.sort(); require(len(vals)>=b.k,"too few active")
    return tuple((s,d) for d,s in vals[:b.k])
def cases(b:Bundle)->tuple[tuple[Case,...],str]:
    active=bytearray(b.pool_n)
    for s in b.base: active[int(s)]=1
    delta=0; out=[]
    for op,code,arg in b.events:
        if code==INSERT:
            require(not active[arg],"duplicate insert"); active[arg]=1; delta+=1
        else:
            out.append(Case(len(out),op,arg,delta,bytes(active),exact_topk(b,active,arg)))
    final=hashlib.sha256("".join(str(i)+"\n" for i,x in enumerate(active) if x).encode("ascii")).hexdigest()
    require(len(out)==136 and delta==169, "trace cardinality")
    return tuple(out),final
def rows(v:Any,b:Bundle,c:Case,label:str)->tuple[tuple[int,int],...]:
    require(isinstance(v,list) and len(v)==b.k,label+" rows")
    result=[]; seen=set(); prior=None
    for i,row in enumerate(v):
        require(isinstance(row,list) and len(row)==2,label+" pair "+str(i))
        s,d=row
        require(plain_int(s) and s<b.pool_n and plain_int(d),label+" value "+str(i))
        require(s not in seen and c.active[s],label+" active/unique "+str(i))
        require(d==distance(b,s,c.qid),label+" integer L2 "+str(i))
        key=(d,s); require(prior is None or prior<key,label+" canonical order")
        prior=key; seen.add(s); result.append((s,d))
    return tuple(result)
def result_hash(x:tuple[tuple[int,int],...])->str:
    data=HASH_DOMAIN+"\n"+"".join(str(s)+":"+str(d)+"\n" for s,d in x)
    return hashlib.sha256(data.encode("ascii")).hexdigest()
def dist(vals:list[int])->dict[str,int]:
    require(vals,"empty distribution"); a=sorted(vals)
    return {"n":len(a),"min_ns":a[0],"p50_ns":a[(50*(len(a)-1))//100],
            "p95_ns":a[(95*(len(a)-1))//100],"max_ns":a[-1],"mean_ns":sum(a)//len(a)}

def validate_admission(b:Bundle,p:Path)->dict[str,str]:
    e=read_env(p); require(need(e,"schema")=="e1-frozen-base-knn-projection-admission-v2" and need(e,"status")=="PASS","admission schema")
    require(Path(need(e,"bundle_realpath"))==BUNDLE_PATH,"admission bundle")
    require(need(e,"ops")=="insert,knn" and need(e,"excluded_ops")=="delete,range","admission ops")
    require(need(e,"base_immutable")=="true" and need(e,"direct_sidecar_allowed")=="false","admission policy")
    for k,v in {"dimension":b.dim,"base_n":b.base_n,"pool_n":b.pool_n,"query_n":b.query_n,"k":b.k,"event_count":305,"insert_count":169,"knn_count":136,"final_active_count":4265}.items():
        require(env_int(e,k)==v,"admission "+k)
    return e

def validate_manifest(p:Path)->dict[str,Any]:
    m=json_file(p,"variant manifest")
    require(m.get("schema")==MANIFEST_SCHEMA and m.get("diagnostic_variant")==VARIANT and m.get("publication_eligible") is False,"variant manifest label")
    require(m.get("execution_state")=="static_copy_only_no_gpu_launch","variant manifest phase")
    require(m.get("algorithm_data_receipt_contract",{}).get("merged_result_hash_domain")==HASH_DOMAIN,"hash domain")
    rd=m.get("reviewed_diff")
    require(isinstance(rd,dict) and rd.get("audit_result")=="PASS_EXACT_ALLOWED_SOURCE_DIFFERENCES_ONLY","reviewed diff audit")
    rdp=Path(rd.get("path","")); require(rdp==DIAG/"provenance/silent_only_reviewed_diff.patch","reviewed diff path")
    regular(rdp,"reviewed diff",True); require(rd.get("sha256")==sha_file(rdp),"reviewed diff hash")
    copied=m.get("copied_sources"); require(isinstance(copied,list) and len(copied)==3,"copied source list")
    want={DIAG/"src/g3_safe_search_v2_silent_only.cuh",DIAG/"src/g3_safe_c1_native_matrix_silent_only.cu",DIAG/"runner/fair_safe_c1_latency_silent_only_e1_runner.cu"}
    got=set()
    for x in copied:
        require(isinstance(x,dict),"copied source entry"); q=Path(x.get("path","")); got.add(q)
        regular(q,"copied source",True); require(isinstance(x.get("sha256"),str) and x["sha256"]==sha_file(q),"copied source hash")
    require(got==want,"copied source paths")
    closure=m.get("source_closure"); require(isinstance(closure,list) and len(closure)>=9,"source closure")
    lines=[]
    for x in closure:
        require(isinstance(x,dict),"closure entry"); q=Path(x.get("path",""))
        require(q.is_relative_to(ROOT),"closure escapes experiment root")
        regular(q,"closure member",q.is_relative_to(DIAG)); require(x.get("sha256")==sha_file(q),"closure hash")
        lines.append(str(q).replace(str(ROOT)+"/","")+"\t"+x["sha256"])
    full=hashlib.sha256(("\n".join(sorted(lines))+"\n").encode("ascii")).hexdigest()
    require(m.get("full_source_closure_sha256")==full,"full source closure hash")
    missing=m.get("known_missing_equivalence_witnesses")
    require(isinstance(missing,list) and any("visited_leaf_count only" in str(x) for x in missing),"missing leaf witness disclosure")
    return m

def validate_records(b:Bundle,cs:tuple[Case,...],rs:list[dict[str,Any]])->dict[str,Any]:
    require(len(rs)==len(cs)*2,"candidate measurement count")
    nv=[]; fv=[]; ne=no=fe=0; seq=hashlib.sha256(); at=0
    for c in cs:
        order=("native_query_knn_candidate","exact_query_range_full_base_fallback") if c.ordinal%2==0 else ("exact_query_range_full_base_fallback","native_query_knn_candidate")
        for condition in order:
            r=rs[at]; at+=1
            require(set(r)==MEAS_KEYS,"candidate measurement fields "+str(at))
            require(r.get("schema")==SCHEMA and r.get("diagnostic_variant")==VARIANT and r.get("publication_eligible") is False and r.get("record")=="measurement","candidate record label "+str(at))
            require(r.get("not_same_api") is True and r.get("timing_claim")=="latency_quality_tradeoff_only","candidate claim "+str(at))
            require(r.get("condition")==condition and r.get("pass")==0 and r.get("case_ordinal")==c.ordinal and r.get("op_index")==c.op and r.get("query_id")==c.qid and r.get("external_global_delta_live")==c.delta,"candidate ABBA trace "+str(at))
            require(plain_int(r.get("api_host_ns"),1) and r.get("post_api_external_delta_merge_excluded_from_api_host_ns") is True,"candidate timing field "+str(at))
            z=rows(r.get("merged"),b,c,"candidate merged")
            require(r.get("merged_result_sha256")==result_hash(z),"candidate hash "+str(at))
            ov=len({s for s,_ in z}&{s for s,_ in c.oracle}); ex=z==c.oracle
            require(r.get("merged_overlap_at_k")==ov and r.get("merged_exact_match") is ex,"candidate quality "+str(at))
            if condition=="native_query_knn_candidate":
                require(r.get("api_result_count_before_adapter_truncation")==b.k and plain_int(r.get("base_candidate_count"),b.k) and r["base_candidate_count"]<=b.base_n and plain_int(r.get("visited_leaf_count")) and r.get("full_immutable_base_candidate_count")==0 and r.get("exact_full_immutable_base_fallback") is False and r.get("base_path")==NATIVE_PATH,"candidate native API "+str(at))
                nv.append(r["api_host_ns"]); ne+=int(ex); no+=ov
            else:
                require(r.get("api_result_count_before_adapter_truncation")==b.base_n and r.get("base_candidate_count")==b.base_n and plain_int(r.get("visited_leaf_count")) and r.get("full_immutable_base_candidate_count")==b.base_n and r.get("exact_full_immutable_base_fallback") is True and r.get("base_path")==FALLBACK_PATH and ex,"candidate fallback API "+str(at))
                fv.append(r["api_host_ns"]); fe+=1
            seq.update((str(c.ordinal)+":"+condition+":"+str(c.op)+":"+str(c.qid)+":").encode("ascii"))
            for s,d in z: seq.update((str(s)+":"+str(d)+",").encode("ascii"))
            seq.update(b"\n")
    return {"native":nv,"fallback":fv,"native_exact_set_count":ne,"native_overlap_sum":no,"fallback_exact_set_count":fe,"sequence_sha256":seq.hexdigest()}

def validate_summary(s:dict[str,Any],metrics:dict[str,Any],case_count:int)->None:
    require(set(s)==SUMMARY_KEYS,"candidate summary fields")
    require(s.get("schema")==SCHEMA and s.get("diagnostic_variant")==VARIANT and s.get("publication_eligible") is False and s.get("mode")=="timing" and s.get("status")=="PASS_SILENT_ONLY_DIAGNOSTIC_TIMING","candidate summary label")
    require(s.get("not_same_api") is True and s.get("timing_claim")=="latency_quality_tradeoff_only" and s.get("scope")==SCOPE,"candidate summary scope")
    require(s.get("warmup_passes_discarded")==1 and s.get("measured_passes")==1 and s.get("measurement_records")==case_count*2,"candidate summary counts")
    require(s.get("post_api_external_delta_merge_excluded_from_api_host_ns") is True and s.get("json_and_oracle_excluded_from_api_host_ns") is True,"candidate summary timing")
    require(s.get("native_api_host_ns")==dist(metrics["native"]) and s.get("fallback_api_host_ns")==dist(metrics["fallback"]),"candidate summary distributions")
    q={"native_exact_set_count":metrics["native_exact_set_count"],"native_overlap_sum":metrics["native_overlap_sum"],"fallback_exact_set_count":metrics["fallback_exact_set_count"],"fallback_expected_exact_set_count":case_count}
    require(s.get("quality")==q and s.get("limitations")==LIMITS,"candidate summary quality/limits")

def write_once(p:Path,x:dict[str,Any])->None:
    require(not p.exists() and not p.is_symlink(),"refuse output overwrite")
    fd,tmp=tempfile.mkstemp(prefix=".silent-validator.",dir=p.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            f.write(json.dumps(x,sort_keys=True,indent=2)+"\n");f.flush();os.fsync(f.fileno())
        os.chmod(tmp,0o600);os.replace(tmp,p)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def run(a:argparse.Namespace)->dict[str,Any]:
    for k in ("bundle","admission","events","summary","manifest","out"):
        setattr(a,k,Path(getattr(a,k)).absolute())
    require(a.bundle==BUNDLE_PATH and a.manifest==DIAG/"provenance/silent_only_source_manifest.json","fixed bundle/manifest paths")
    run=a.events.parent
    require(a.events.name=="engine.jsonl" and a.summary.name=="summary.json" and a.admission.name=="admission.env" and a.out.name=="independent_validation.json","canonical candidate names")
    require(a.summary.parent==a.admission.parent==a.out.parent==run and run.parent==ROOT/"runs" and RUN_RE.fullmatch(run.name) is not None,"candidate run path")
    directory(run,"candidate run")
    b=load_bundle(a.bundle); env=validate_admission(b,a.admission); cs,final=cases(b)
    require(need(env,"final_active_set_sha256")==final,"admission final active hash")
    m=validate_manifest(a.manifest); rs=jsonl(a.events); summary=json_file(a.summary,"candidate summary")
    metrics=validate_records(b,cs,rs); validate_summary(summary,metrics,len(cs))
    return {"schema":"fair-safe-c1-silent-only-output-validator-v1","status":"PASS_SEMANTIC_OUTPUT_ONLY","diagnostic_variant":VARIANT,"publication_eligible":False,"gpu_workload_launched_by_validator":False,"scope":"independent exact-oracle validation of silent-only candidate output semantics; not authorization evidence, not a same-API speedup, and not publication evidence","variant_manifest_sha256":sha_file(a.manifest),"events_sha256":sha_file(a.events),"summary_sha256":sha_file(a.summary),"validated_measurements":len(rs),"knn_cases_per_pass":len(cs),"native_exact_set_count":metrics["native_exact_set_count"],"native_overlap_sum":metrics["native_overlap_sum"],"fallback_exact_set_count":metrics["fallback_exact_set_count"],"measurement_sequence_sha256":metrics["sequence_sha256"],"known_missing_equivalence_witness":"This validator observes visited_leaf_count only; it cannot prove leaf-ID/list/set or traversal-order equivalence to v1b."}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--self-check",action="store_true")
    for n in ("bundle","admission","events","summary","manifest","out"): p.add_argument("--"+n.replace("_","-"))
    a=p.parse_args()
    try:
        if a.self_check:
            require(all(getattr(a,n) is None for n in ("bundle","admission","events","summary","manifest","out")),"self-check args")
            b=load_bundle(BUNDLE_PATH); cs,final=cases(b)
            print(json.dumps({"schema":"fair-safe-c1-silent-only-output-validator-v1","status":"PASS_CPU_ONLY_ORACLE_SELFCHECK","gpu_workload_launched":False,"knn_cases":len(cs),"final_active_set_sha256":final},sort_keys=True)); return 0
        require(all(getattr(a,n) is not None for n in ("bundle","admission","events","summary","manifest","out")),"all output args required")
        r=run(a); write_once(Path(a.out).absolute(),r); print(json.dumps(r,sort_keys=True)); return 0
    except Fail as e:
        print("FAIL: "+str(e),file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())

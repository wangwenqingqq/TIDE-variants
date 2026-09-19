#!/usr/bin/env python3
"""Fail-closed CPU-only integrity and source-policy verifier for C1 v6."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, pathlib, re

ROOT_DEFAULT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot"
VARIANTS={
    "E_G_c1_off_reference":("OFF","OFF"),
    "P_G_workspace_only":("ON","OFF"),
    "E_F_fastpath_only":("OFF","ON"),
    "P_F_full_C1":("ON","ON"),
}
HELPERS={"guard","measured_engine","profile_engine","profile_guard","profile_manifest_utility","strict_semantic_verifier","measured_analyzer","profile_verifier","measured_finalizer","formal_aggregator","manifest_utility","pin_verifier","pin_generator","build_script","build_manifest_creator","capacity_snapshot_contract","capacity_snapshot_fixture","source_pin_verifier","source_pin_generator"}
PROTOCOLS={"formal_supersession","frozen_protocol","gpu_uuid_safety","idle_poll_safety","idle_poll_logdir","implementation_correction","execution_plan","profile_execution_plan"}
INPUTS={"base","derived_query_only_trace","derived_trace_provenance","source_update_trace"}

def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def safe_file(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_file():
        raise SystemExit(f"{label}: absolute regular non-symlink file required: {p}")
    q=p.resolve(strict=True)
    if p!=q: raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p

def safe_dir(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_dir():
        raise SystemExit(f"{label}: absolute non-symlink directory required: {p}")
    q=p.resolve(strict=True)
    if p!=q: raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p

def under(p:pathlib.Path,root:pathlib.Path)->bool:
    return str(p).startswith(str(root)+"/")

def entry_file(e:dict,label:str,root:pathlib.Path,allowed:set[str])->pathlib.Path:
    if not isinstance(e,dict) or set(("path","sha256"))-set(e): raise SystemExit(f"{label}: missing path/sha256")
    if not isinstance(e["sha256"],str) or re.fullmatch(r"[0-9a-f]{64}",e["sha256"]) is None:
        raise SystemExit(f"{label}: bad SHA format")
    p=safe_file(str(e["path"]),label)
    if not (under(p,root) or str(p) in allowed): raise SystemExit(f"{label}: path outside root/allowed externals: {p}")
    got=sha(p)
    if got!=e["sha256"]: raise SystemExit(f"{label}: SHA mismatch {p}: expected {e['sha256']} got {got}")
    return p

def load(p:pathlib.Path,label:str)->dict:
    try: x=json.loads(p.read_text())
    except Exception as e: raise SystemExit(f"{label}: invalid JSON: {e}")
    if not isinstance(x,dict): raise SystemExit(f"{label}: JSON object required")
    return x

def cmake_values(p:pathlib.Path)->dict[str,str]:
    out={}
    for line in p.read_text(errors="strict").splitlines():
        if not line or line.startswith("//") or line.startswith("#"): continue
        m=re.match(r"([^:=]+):[^=]*=(.*)$",line)
        if m: out[m.group(1)]=m.group(2)
    return out

def check_cache(p:pathlib.Path,name:str,persist:str,fast:str)->None:
    c=cmake_values(p)
    if c.get("C1_PERSISTENT_WORKSPACE")!=persist or c.get("C1_ONE_QUERY_FASTPATH")!=fast:
        raise SystemExit(f"{name}: CMake C1 gate mismatch")
    if c.get("CMAKE_BUILD_TYPE")!="Release": raise SystemExit(f"{name}: CMake build type is not Release")
    if c.get("CMAKE_CUDA_ARCHITECTURES")!="120": raise SystemExit(f"{name}: CUDA architecture cache is not 120")
    if c.get("CMAKE_CUDA_COMPILER")!="/usr/local/cuda-13.1/bin/nvcc":
        raise SystemExit(f"{name}: unexpected CUDA compiler cache value {c.get('CMAKE_CUDA_COMPILER')!r}")

def verify_profile_execution_plan(pins:dict)->None:
    entry=pins["protocols"]["profile_execution_plan"]
    path=safe_file(str(entry["path"]),"profile execution plan")
    plan=load(path,"profile execution plan")
    reports=["cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_mem_size_sum","um_sum"]
    if plan.get("schema")!="gtspp-c1-v6-profile-execution-plan-v1": raise SystemExit("profile execution plan schema")
    if plan.get("schedule")!=[{"replicate":1,"variant":"E_G_c1_off_reference"},{"replicate":1,"variant":"P_F_full_C1"}]: raise SystemExit("profile execution plan schedule")
    nsys=plan.get("nsys",{})
    if nsys.get("executable")!="/usr/local/bin/nsys" or nsys.get("profile_arguments")!=["profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true"]: raise SystemExit("profile execution nsys command")
    if nsys.get("stats_arguments")!=["stats","--force-export=true","--force-overwrite=true","--format","csv"] or nsys.get("reports")!=reports: raise SystemExit("profile execution report set")
    if nsys.get("artifact_layout",{}).get("nsys_rep")!="nsys/profile.nsys-rep" or nsys.get("artifact_layout",{}).get("sqlite")!="nsys/profile.sqlite" or nsys.get("artifact_layout",{}).get("csv_reports")!=["nsys/reports/profile_"+x+".csv" for x in reports]: raise SystemExit("profile execution artifact layout")
    constraint=pins.get("semantic_constraints",{})
    if constraint.get("profile_execution_plan_sha256")!=entry.get("sha256"): raise SystemExit("profile execution plan constraint hash")

def verify_trace_provenance(pins:dict,root:pathlib.Path)->None:
    """Fail before GPU launch if the v6-local trace provenance is inconsistent."""
    inputs=pins["inputs"]; protocols=pins["protocols"]
    base=safe_file(str(inputs["base"]["path"]),"provenance base")
    source=safe_file(str(inputs["source_update_trace"]["path"]),"provenance source trace")
    derived=safe_file(str(inputs["derived_query_only_trace"]["path"]),"provenance derived trace")
    prov_path=safe_file(str(inputs["derived_trace_provenance"]["path"]),"provenance manifest")
    protocol=safe_file(str(protocols["frozen_protocol"]["path"]),"provenance frozen protocol")
    formal=safe_file(str(protocols["formal_supersession"]["path"]),"provenance formal supersession")
    prov=load(prov_path,"trace provenance")
    if prov.get("schema")!="gtspp-c1-query-only-trace-provenance-v2": raise SystemExit("trace provenance schema")
    def bound(key:str,path:pathlib.Path,pin:dict)->None:
        x=prov.get(key)
        if not isinstance(x,dict) or x.get("path")!=str(path) or x.get("sha256")!=pin.get("sha256"):
            raise SystemExit(f"trace provenance pin binding failed: {key}")
    bound("base",base,inputs["base"])
    bound("source_update_trace",source,inputs["source_update_trace"])
    bound("derived_trace",derived,inputs["derived_query_only_trace"])
    bound("protocol",protocol,protocols["frozen_protocol"])
    bound("formal_supersession",formal,protocols["formal_supersession"])
    header=base.open().readline().strip()
    if header!="128 1000000 2": raise SystemExit("trace provenance base header")
    source_lines=[x.strip() for x in source.read_text().splitlines() if x.strip()]
    derived_lines=[x.strip() for x in derived.read_text().splitlines() if x.strip()]
    if len(source_lines)!=10001 or source_lines[0]!="10000": raise SystemExit("trace provenance source header/length")
    if len(derived_lines)!=1025 or derived_lines[0]!="1024": raise SystemExit("trace provenance derived header/length")
    events=[]
    for ordinal,line in enumerate(source_lines[1:]):
        fields=line.split()
        if len(fields)!=2 or fields[0] not in {"0","1","2"}: raise SystemExit(f"trace provenance malformed source row {ordinal}")
        events.append((int(fields[0]),int(fields[1])))
    selected=[(i,op,qid) for i,(op,qid) in enumerate(events) if op==2][:1024]
    if len(selected)!=1024: raise SystemExit("trace provenance lacks 1024 type-2 rows")
    ords=prov.get("selected_source_event_ordinals_zero_based")
    lnums=prov.get("selected_source_line_numbers_one_based")
    rows=prov.get("selected_type2_rows")
    if ords!=[x[0] for x in selected] or lnums!=[x[0]+2 for x in selected] or not isinstance(rows,list) or len(rows)!=1024:
        raise SystemExit("trace provenance selection witness mismatch")
    qids=[]
    for i,(ordinal,op,qid) in enumerate(selected):
        row=f"2 {qid}"
        if derived_lines[i+1]!=row: raise SystemExit(f"trace provenance derived reconstruction mismatch {i}")
        witness=rows[i]
        if not isinstance(witness,dict) or witness.get("source_event_ordinal_zero_based")!=ordinal or witness.get("source_line_number_one_based")!=ordinal+2 or witness.get("row")!=row:
            raise SystemExit(f"trace provenance row witness mismatch {i}")
        if not 0<=qid<1000000: raise SystemExit(f"trace provenance qid out of range {i}")
        qids.append(qid)
    if prov.get("selection_rule")!="first 1024 source rows with flag==2 in original source-file order; serialize each as `2 original_id` after a 1024 header":
        raise SystemExit("trace provenance selection rule")
    if prov.get("source_type2_rows_exactly_reconstruct_derived_trace") is not True:
        raise SystemExit("trace provenance reconstruction assertion")
    if prov.get("selected_id_min")!=min(qids) or prov.get("selected_id_max")!=max(qids):
        raise SystemExit("trace provenance ID span")

def cpp_function(text:str,signature:str)->str:
    start=text.index(signature)
    opening=text.index("{",start)
    depth=0
    for i in range(opening,len(text)):
        ch=text[i]
        if ch=="{": depth+=1
        elif ch=="}":
            depth-=1
            if depth==0: return text[start:i+1]
    raise SystemExit(f"unterminated C++ function {signature}")

def assert_capacity_snapshot_source_policy(c1:str)->None:
    """Reject the stale post-release telemetry failure mode without compiling/running CUDA."""
    body=cpp_function(c1,"QueryResult run_query")
    snapshot="const int c1_capacity_snapshot_pre_release = update_result_ws_cap;"
    release="releaseC1QueryWorkspace(qresult_count, qresult_count_prefix, result_id, result_dis);"
    if body.count(snapshot)!=1: raise SystemExit("capacity snapshot declaration must occur exactly once")
    if body.count(release)!=1: raise SystemExit("run_query must contain exactly one ephemeral release site")
    snapshot_pos=body.index(snapshot); release_pos=body.index(release)
    if snapshot_pos>=release_pos: raise SystemExit("capacity snapshot is not captured before ephemeral release")
    suffix=body[release_pos:]
    if re.search(r"\bupdate_result_ws_cap\b",suffix):
        raise SystemExit("global update_result_ws_cap read after release is forbidden")
    for field in ("update_result_capacity_slots","total_result_capacity_slots"):
        expected=f"result.{field} = c1_capacity_snapshot_pre_release;"
        if body.count(expected)!=1: raise SystemExit(f"{field} is not exported solely from the local pre-release snapshot")
    if body.count('result.capacity_snapshot_stage = "pre_ephemeral_release";')!=1:
        raise SystemExit("capacity snapshot stage assignment missing")
    if 'std::string capacity_snapshot_stage;' not in c1:
        raise SystemExit("QueryResult capacity snapshot stage field missing")
    if r'\"capacity_snapshot_stage\":\"' not in c1:
        raise SystemExit("capacity snapshot stage is not serialized to JSON")

def static_source_policy(root:pathlib.Path)->None:
    c1=(root/"worktree/src/c1_microbench.cu").read_text()
    up=(root/"worktree/include/update.cuh").read_text()
    tree=(root/"worktree/include/tree.cuh").read_text()
    main=(root/"worktree/src/main.cu").read_text()
    for token in (
        "--profile-only","UNMEASURED_NSYS_PROFILE_DO_NOT_USE","\\\"cuda_status\\\":\\\"success\\\"",
        "update-result capacity/overflow violation","query-only result count outside C1 workspace capacity",
        "cudaDeviceGetPCIBusId","visible_cuda_device_pci_bus_id","visible_cuda_device_uuid","cuda_uuid_string","CUDA UUID unavailable","CHECK(cudaRuntimeGetVersion",
        "setup_query_only_state","c1_capacity_snapshot_pre_release","capacity_snapshot_stage","pre_ephemeral_release",
    ):
        if token not in c1: raise SystemExit(f"missing C1 v6 policy token {token!r}")
    for token in ("validateC1SearchNumForWorkspace","c1FailClosedWorkspace","CHECK(cudaDeviceSynchronize());","CHECK(cudaGetLastError());"):
        if token not in up: raise SystemExit(f"missing update safety token {token!r}")
    for text,token in ((up,"collectLeafNodesSingleQuery error:"),(up,"compactResultSingleQuery error:"),(tree,"initIndexData error:"),(tree,"getPivotDis error:"),(main,"Error uploading c_rp_")):
        if token in text: raise SystemExit(f"nonfatal C1-path CUDA handler remains: {token}")
    for api in ("updateIndexRnn","initIncrementalInsert","incrementalInsert","findTargetLeaf","ensureRebuildInsertWorkspace","ensureDeletePrefixValid","mergeTotalResult"):
        if re.search(r"\b"+re.escape(api)+r"\s*\(",c1):
            raise SystemExit(f"direct C3 mutation/session API is forbidden in c1_microbench.cu: {api}")
    assert_capacity_snapshot_source_policy(c1)
    measured_guard=(root/"run_c1_workspace_microbenchmark_guard_v6.sh").read_text()
    measured_verifier=(root/"verify_c1_semantics_v6.py").read_text()
    profile_guard=(root/"run_c1_profile_guard_v6.sh").read_text()
    profile_manifest=(root/"c1_v6_profile_manifest.py").read_text()
    profile_verifier=(root/"verify_c1_profile_artifacts_v6.py").read_text()
    finalizer=(root/"finalize_c1_evidence_v6.py").read_text()
    aggregator=(root/"aggregate_c1_formal_evidence_v6.py").read_text()
    for token in ("visible_cuda_device_uuid","device UUID/session provenance"):
        if token not in measured_guard: raise SystemExit(f"missing measured guard UUID policy token {token!r}")
    for token in ("C1_V6_TRUST_ROOT=''","C1_V6_LAUNCHER=''","C1_V6_TRUST_GUARD_SHA=''","C1_V6_TRUST_PINS_SHA=''","C1_V6_TRUST_PIN_VERIFY_SHA=''","C1_ALLOW_GPU0=''","C1_RUN_OUT=''"):
        if token in measured_guard: raise SystemExit(f"measured guard erases inherited trust environment {token!r}")
    for token in ("C1_V6_PROFILE_TRUST_ROOT=''","C1_V6_PROFILE_LAUNCHER=''","C1_V6_PROFILE_TRUST_GUARD_SHA=''","C1_V6_PROFILE_TRUST_PINS_SHA=''","C1_V6_PROFILE_TRUST_PIN_VERIFY_SHA=''","C1_ALLOW_GPU0=''","C1_PROFILE_OUT=''"):
        if token in profile_guard: raise SystemExit(f"profile guard erases inherited trust environment {token!r}")
    for token in ("visible_cuda_device_uuid","CUDA UUID must exactly match manifest/session UUID","telemetry PCI discontinuity","capacity_snapshot_violations","capacity_snapshot_contract_v6.py"):
        if token not in measured_verifier: raise SystemExit(f"missing measured verifier binding token {token!r}")
    for token in ("--trace=cuda","--cuda-memory-usage=true","--profile-only","profile_nsys_stats_complete","um_sum","C1_V6_PROFILE_TRUST_GUARD_SHA","profile_execution_plan_v6.json","CUDA UUID/session"):
        if token not in profile_guard: raise SystemExit(f"missing profile guard policy token {token!r}")
    for token in ("profile_run_manifest_v6.json","nsys_argv","profile_artifacts","profile.sqlite","um_sum"):
        if token not in profile_manifest: raise SystemExit(f"missing profile manifest policy token {token!r}")
    for token in ("profile_run_manifest_v6.json","outer guarded nsys invocation manifest","profile_nsys_stats_complete","SQLite format 3","um_sum","CUDA UUID differs from manifest/session UUID","profile telemetry PCI differs from outer prelaunch","capacity_snapshot_violations","capacity snapshot"):
        if token not in profile_verifier: raise SystemExit(f"missing profile verifier policy token {token!r}")
    for token in ("gtspp-c1-v6-measured-evidence-v2","gpu_pci_bus_id_normalized"):
        if token not in finalizer: raise SystemExit(f"missing measured finalizer binding token {token!r}")
    for token in ("gtspp-c1-v6-profile-verification-v3","cross-root GPU PCI disagreement","gtspp-c1-v6-formal-evidence-aggregate-v3","aggregate self pin/provenance mismatch","aggregate_script_sha256","SELF_PATH"):
        if token not in aggregator: raise SystemExit(f"missing formal aggregate binding token {token!r}")
    body=cpp_function(tree,"void indexConstru")
    for m in re.finditer(r"\b(cudaMemcpy|cudaFree|cudaDeviceSynchronize)\s*\(",body):
        stmt=body[body.rfind(";",0,m.start())+1:m.start()]
        if "CHECK(" not in stmt:
            raise SystemExit(f"unwrapped indexConstru CUDA call: {body[m.start():m.start()+100].splitlines()[0]}")

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=pathlib.Path,required=True)
    ap.add_argument("--pins",type=pathlib.Path,required=True)
    ap.add_argument("--variant")
    ap.add_argument("--static-source",action="store_true")
    a=ap.parse_args()
    root=safe_dir(str(a.root),"root")
    pins_path=safe_file(str(a.pins),"pins")
    p=load(pins_path,"pins")
    if p.get("schema")!="gtspp-c1-v6-pins-v1" or p.get("root")!=str(root): raise SystemExit("pins schema/root mismatch")
    allowed=p.get("allowed_external_paths",[])
    if not isinstance(allowed,list) or len(allowed)!=2 or not all(isinstance(x,str) for x in allowed): raise SystemExit("pins external allowlist invalid")
    allowed_set=set(allowed)
    if set(p.get("inputs",{}))!=INPUTS: raise SystemExit("pins input set incomplete")
    if set(p.get("protocols",{}))!=PROTOCOLS: raise SystemExit("pins protocol set incomplete")
    if set(p.get("runtime_helpers",{}))!=HELPERS: raise SystemExit("pins helper set incomplete")
    required=["source_build_manifest","worktree_source_files","variant_binaries","canonical_inspector","semantic_constraints"]
    if any(k not in p for k in required): raise SystemExit("pins required top-level key missing")
    all_entries=[]
    for k,e in p["inputs"].items(): all_entries.append(("input:"+k,e))
    for k,e in p["protocols"].items(): all_entries.append(("protocol:"+k,e))
    all_entries.append(("source_build_manifest",p["source_build_manifest"]))
    for i,e in enumerate(p["worktree_source_files"]): all_entries.append(("source:"+str(i),e))
    for i,e in enumerate(p["variant_binaries"]):
        all_entries.append(("variant:"+str(i),e))
        all_entries.append(("variant_cmake_cache:"+str(i),e.get("cmake_cache")))
    for k,e in p["runtime_helpers"].items(): all_entries.append(("helper:"+k,e))
    ci=p["canonical_inspector"]
    all_entries.extend([
        ("canonical_inspector.source",{"path":ci.get("source"),"sha256":ci.get("source_sha256")}),
        ("canonical_inspector.binary",{"path":ci.get("binary"),"sha256":ci.get("binary_sha256")}),
    ])
    seen={}
    for label,e in all_entries:
        f=entry_file(e,label,root,allowed_set)
        if str(f) in seen and seen[str(f)]!=e["sha256"]: raise SystemExit(f"duplicate inconsistent pin {f}")
        seen[str(f)]=e["sha256"]
    bm_path=entry_file(p["source_build_manifest"],"build manifest pin",root,allowed_set)
    bm=load(bm_path,"build manifest")
    if bm.get("schema")!="gtspp-c1-v6-build-manifest-v1": raise SystemExit("build manifest schema")
    source_map={x.get("path"):x.get("sha256") for x in bm.get("source_files",[])}
    if len(source_map)!=len(p["worktree_source_files"]): raise SystemExit("source list/build manifest mismatch")
    for e in p["worktree_source_files"]:
        if source_map.get(e.get("path"))!=e.get("sha256") or e.get("expected_from_build_manifest")!=e.get("sha256"):
            raise SystemExit("source build hash disagreement")
    bm_variants={x.get("name"):x for x in bm.get("variants",[])}
    got_variants={x.get("name"):x for x in p["variant_binaries"]}
    if set(got_variants)!=set(VARIANTS) or set(bm_variants)!=set(VARIANTS): raise SystemExit("variant set incomplete")
    for n,(persist,fast) in VARIANTS.items():
        e=got_variants[n]; b=bm_variants[n]
        if (e.get("C1_PERSISTENT_WORKSPACE"),e.get("C1_ONE_QUERY_FASTPATH"))!=(persist,fast): raise SystemExit(f"variant pin gates {n}")
        if (b.get("C1_PERSISTENT_WORKSPACE"),b.get("C1_ONE_QUERY_FASTPATH"))!=(persist,fast): raise SystemExit(f"variant manifest gates {n}")
        binary=root/"builds"/n/"bin"/"C1Microbench"; cache=root/"builds"/n/"CMakeCache.txt"
        if pathlib.Path(e.get("path",""))!=binary or pathlib.Path(str(e.get("cmake_cache",{}).get("path","")))!=cache:
            raise SystemExit(f"variant build location {n}")
        cache_e=e.get("cmake_cache",{})
        if e.get("sha256")!=b.get("binary_sha256") or e.get("expected_from_build_manifest")!=b.get("binary_sha256"):
            raise SystemExit(f"binary provenance disagreement {n}")
        if cache_e.get("sha256")!=b.get("cmake_cache_sha256") or e.get("expected_cmake_cache_from_build_manifest")!=b.get("cmake_cache_sha256"):
            raise SystemExit(f"CMakeCache provenance disagreement {n}")
        check_cache(cache,n,persist,fast)
    verify_trace_provenance(p,root)
    verify_profile_execution_plan(p)
    constraints=p.get("semantic_constraints",{})
    if constraints.get("correctness_scope")!="cross_variant_differential_equivalence_only; no independent absolute range-search oracle is claimed by this pin set":
        raise SystemExit("semantic correctness scope is not explicit")
    if constraints.get("capacity_snapshot_stage")!="pre_ephemeral_release" or constraints.get("capacity_snapshot_fields")!=["update_result_capacity_slots","total_result_capacity_slots"]:
        raise SystemExit("capacity snapshot semantic constraints are not explicit")
    if a.variant:
        if a.variant not in VARIANTS: raise SystemExit("unknown variant")
        wanted=[got_variants[a.variant],got_variants[a.variant]["cmake_cache"],*p["inputs"].values(),p["source_build_manifest"],*p["worktree_source_files"],*p["protocols"].values(),*p["runtime_helpers"].values(),{"path":ci["source"],"sha256":ci["source_sha256"]},{"path":ci["binary"],"sha256":ci["binary_sha256"]}]
        for i,e in enumerate(wanted): entry_file(e,"runtime:"+str(i),root,allowed_set)
    if a.static_source: static_source_policy(root)
    print(json.dumps({"pass":True,"pins_sha256":sha(pins_path),"checked_unique":len(seen),"variant":a.variant,"static_source":a.static_source},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())

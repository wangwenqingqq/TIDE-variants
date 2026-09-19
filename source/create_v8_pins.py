#!/usr/bin/env python3
"""Create the complete CPU-only C1 v8 integrity pin set after a successful build."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, pathlib, re
from datetime import datetime, timezone

ROOT_DEFAULT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract"
VARIANTS={
    "E_G_c1_off_reference":("OFF","OFF"),
    "P_G_workspace_only":("ON","OFF"),
    "E_F_fastpath_only":("OFF","ON"),
    "P_F_full_C1":("ON","ON"),
}
HELPER_PATHS={
    "guard":"run_c1_workspace_microbenchmark_guard_v8.sh",
    "measured_engine":"run_c1_four_variants_v8.sh",
    "profile_engine":"run_c1_profile_primary_v8.sh",
    "profile_guard":"run_c1_profile_guard_v8.sh",
    "profile_manifest_utility":"c1_v8_profile_manifest.py",
    "strict_semantic_verifier":"verify_c1_semantics_v8.py",
    "measured_analyzer":"analyze_c1_microbenchmark_v8.py",
    "profile_verifier":"verify_c1_profile_artifacts_v8.py",
    "measured_finalizer":"finalize_c1_evidence_v8.py",
    "formal_aggregator":"aggregate_c1_formal_evidence_v8.py",
    "manifest_utility":"c1_v8_manifest.py",
    "pin_verifier":"verify_v8_pins.py",
    "pin_generator":"create_v8_pins.py",
    "build_script":"build_v8_variants.sh",
    "build_manifest_creator":"create_v8_build_manifest.py",
    "capacity_snapshot_contract":"capacity_snapshot_contract_v8.py",
    "capacity_snapshot_fixture":"tools/test_c1_v8_capacity_snapshot_fixtures.py",
    "profile_manifest_fixture":"tools/test_c1_v8_profile_manifest_fixtures.py",
    "profile_terminal_fixture":"tools/test_c1_v8_profile_terminal_fixture.py",
    "guard_session_fixture":"tools/test_c1_v8_guard_session_fixtures.py",
    "profile_child_sessions_fixture":"tools/test_c1_v8_profile_child_sessions_fixture.py",
    "source_pin_verifier":"verify_v8_source_pins.py",
    "source_pin_generator":"create_v8_source_pins.py",
}
PROTOCOL_PATHS={
    "formal_supersession":"protocol_c1_v8_formal_supersession_v1.json",
    "frozen_protocol":"protocol_c1_workspace_onequery_sift1m_v1_v8_local.json",
    "gpu_uuid_safety":"protocol_c1_gpu_uuid_safety_v8_local.json",
    "idle_poll_safety":"protocol_c1_gpu_idle_poll_safety_v8_local.json",
    "idle_poll_logdir":"protocol_c1_gpu_idle_poll_logdir_v8_local.json",
    "implementation_correction":"implementation_correction_capacity_snapshot_v1.json",
    "execution_plan":"measured_execution_plan_v8.json",
    "profile_execution_plan":"profile_execution_plan_v8.json",
}

def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def raw_file(p:pathlib.Path,label:str)->pathlib.Path:
    if not p.is_absolute() or p.is_symlink() or not p.is_file():
        raise SystemExit(f"{label}: absolute regular non-symlink required: {p}")
    q=p.resolve(strict=True)
    if q!=p: raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p

def raw_dir(p:pathlib.Path,label:str)->pathlib.Path:
    if not p.is_absolute() or p.is_symlink() or not p.is_dir():
        raise SystemExit(f"{label}: absolute directory without symlink required: {p}")
    q=p.resolve(strict=True)
    if q!=p: raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p

def entry(p:pathlib.Path,label:str)->dict:
    p=raw_file(p,label)
    return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}

def load_json(p:pathlib.Path,label:str)->dict:
    p=raw_file(p,label)
    try: x=json.loads(p.read_text())
    except Exception as e: raise SystemExit(f"{label}: invalid JSON: {e}")
    if not isinstance(x,dict): raise SystemExit(f"{label}: JSON object required")
    return x

def cmake_values(p:pathlib.Path)->dict[str,str]:
    values={}
    for line in p.read_text(errors="strict").splitlines():
        if not line or line.startswith("//") or line.startswith("#"): continue
        m=re.match(r"([^:=]+):[^=]*=(.*)$",line)
        if m: values[m.group(1)]=m.group(2)
    return values

def check_cache(p:pathlib.Path,name:str,persist:str,fast:str)->None:
    c=cmake_values(p)
    if c.get("C1_PERSISTENT_WORKSPACE")!=persist or c.get("C1_ONE_QUERY_FASTPATH")!=fast:
        raise SystemExit(f"{name}: C1 gate mismatch in CMakeCache")
    if c.get("CMAKE_BUILD_TYPE")!="Release": raise SystemExit(f"{name}: build type is not Release")
    if c.get("CMAKE_CUDA_ARCHITECTURES")!="120": raise SystemExit(f"{name}: CUDA architecture cache must be 120")
    compiler=c.get("CMAKE_CUDA_COMPILER","")
    if compiler!="/usr/local/cuda-13.1/bin/nvcc": raise SystemExit(f"{name}: unexpected CUDA compiler {compiler!r}")

def validate_v8_trace_provenance(inputs:dict,protocols:dict)->None:
    base=pathlib.Path(inputs["base"]["path"]); source=pathlib.Path(inputs["source_update_trace"]["path"])
    derived=pathlib.Path(inputs["derived_query_only_trace"]["path"]); prov_path=pathlib.Path(inputs["derived_trace_provenance"]["path"])
    prov=load_json(prov_path,"trace provenance")
    if prov.get("schema")!="gtspp-c1-query-only-trace-provenance-v2": raise SystemExit("trace provenance schema")
    for key,path,pin in (("base",base,inputs["base"]),("source_update_trace",source,inputs["source_update_trace"]),("derived_trace",derived,inputs["derived_query_only_trace"]),("protocol",pathlib.Path(protocols["frozen_protocol"]["path"]),protocols["frozen_protocol"]),("formal_supersession",pathlib.Path(protocols["formal_supersession"]["path"]),protocols["formal_supersession"])):
        x=prov.get(key,{})
        if not isinstance(x,dict) or x.get("path")!=str(path) or x.get("sha256")!=pin.get("sha256"): raise SystemExit(f"trace provenance binding {key}")
    src=[x.strip() for x in source.read_text().splitlines() if x.strip()]
    der=[x.strip() for x in derived.read_text().splitlines() if x.strip()]
    if len(src)!=10001 or src[0]!="10000" or len(der)!=1025 or der[0]!="1024": raise SystemExit("trace provenance headers")
    events=[]
    for i,line in enumerate(src[1:]):
        a=line.split()
        if len(a)!=2 or a[0] not in {"0","1","2"}: raise SystemExit("trace provenance source row")
        events.append((int(a[0]),int(a[1])))
    selected=[(i,op,qid) for i,(op,qid) in enumerate(events) if op==2][:1024]
    rows=prov.get("selected_type2_rows")
    if len(selected)!=1024 or prov.get("selected_source_event_ordinals_zero_based")!=[i for i,_,_ in selected] or prov.get("selected_source_line_numbers_one_based")!=[i+2 for i,_,_ in selected] or not isinstance(rows,list) or len(rows)!=1024: raise SystemExit("trace provenance selection")
    for j,(ordinal,_,qid) in enumerate(selected):
        row=f"2 {qid}"; witness=rows[j]
        if der[j+1]!=row or not isinstance(witness,dict) or witness.get("source_event_ordinal_zero_based")!=ordinal or witness.get("source_line_number_one_based")!=ordinal+2 or witness.get("row")!=row: raise SystemExit("trace provenance reconstruction")
    if prov.get("source_type2_rows_exactly_reconstruct_derived_trace") is not True: raise SystemExit("trace provenance assertion")

def expected_child_session_contract() -> dict:
    return {
        "top_level_directories": ["E_G_c1_off_reference", "P_F_full_C1", "gpu_snapshots", "child_sessions"],
        "directory": "child_sessions",
        "per_variant": {
            "ready_filename_template": "profile_{variant}.ready",
            "go_filename_template": "profile_{variant}.go",
            "ready_exact_format": "pid=<positive decimal>\npgid=<same positive decimal>\nsid=<same positive decimal>\n",
            "go_bytes": 0,
            "cleanup_record_template": "{variant}/child_session_provenance/owned_session_cleanup_v8.json",
            "cleanup_schema": "gtspp-c1-v8-profile-owned-child-cleanup-v1",
            "required_terminal": {
                "phase": "direct",
                "reason": "direct child exited",
                "action": "WAITED_FOR_VERIFIED_OWNED_SESSION",
                "child_returncode": 0,
                "verified_owned_session": True,
                "still_alive_after_cleanup": False,
            },
        },
        "strictness": "The profile verifier rejects any missing, extra, symlinked, non-regular, malformed, or mismatched ready/go/cleanup proof artifact; child_sessions is verified evidence, never a permissive extra directory.",
    }


def validate_profile_execution_plan(protocols:dict)->None:
    path=pathlib.Path(protocols["profile_execution_plan"]["path"])
    plan=load_json(path,"profile execution plan")
    reports=["cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_mem_size_sum","um_sum"]
    if plan.get("schema")!="gtspp-c1-v8-profile-execution-plan-v1": raise SystemExit("profile plan schema")
    if plan.get("schedule")!=[{"replicate":1,"variant":"E_G_c1_off_reference"},{"replicate":1,"variant":"P_F_full_C1"}]: raise SystemExit("profile plan schedule")
    nsys=plan.get("nsys",{})
    if nsys.get("executable")!="/usr/local/bin/nsys" or nsys.get("profile_arguments")!=["profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true"]: raise SystemExit("profile plan nsys command")
    if nsys.get("stats_arguments")!=["stats","--force-export=true","--force-overwrite=true","--format","csv"] or nsys.get("reports")!=reports: raise SystemExit("profile plan stats reports")
    expected_layout={"nsys_rep":"nsys/profile.nsys-rep","sqlite":"nsys/profile.sqlite","stats_stdout":"nsys/stats_stdout.log","csv_reports":["nsys/reports/profile_"+x+".csv" for x in reports]}
    expected_um={"allowed_report":"nsys/reports/profile_um_sum.csv","nonempty_reports":["nsys/reports/profile_"+x+".csv" for x in reports[:-1]],"zero_byte_requires":{"stats_stdout":"nsys/stats_stdout.log","processed_marker":"/um_sum.py] to [{absolute profile_um_sum.csv path}]... PROCESSED (EMPTY RESULTS)","no_page_fault_marker":"does not contain CUDA Unified Memory CPU page faults data."}}
    if nsys.get("artifact_layout")!=expected_layout: raise SystemExit("profile plan artifact layout")
    if plan.get("um_zero_byte_contract")!=expected_um: raise SystemExit("profile plan UM zero-byte contract")
    if plan.get("child_session_contract")!=expected_child_session_contract(): raise SystemExit("profile plan child-session contract")

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=pathlib.Path,default=ROOT_DEFAULT)
    ap.add_argument("--out",type=pathlib.Path)
    a=ap.parse_args()
    root=raw_dir(a.root,"root")
    build=root/"worktree_build_manifest_v8.json"
    bm=load_json(build,"build manifest")
    if bm.get("schema")!="gtspp-c1-v8-build-manifest-v1": raise SystemExit("build manifest schema")
    if set(x.get("name") for x in bm.get("variants",[]))!=set(VARIANTS): raise SystemExit("build manifest variants")
    base=pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt")
    source_trace=pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/update_workloads/sift_1m_10k_442.txt")
    inputs={
        "base":entry(base,"base"),
        "derived_query_only_trace":entry(root/"inputs/sift1m_10k_442_first1024_type2_query_only.txt","derived trace"),
        "derived_trace_provenance":entry(root/"inputs/sift1m_10k_442_first1024_type2_provenance.json","derived provenance"),
        "source_update_trace":entry(source_trace,"source trace"),
    }
    variants=[]
    for b in bm["variants"]:
        name=b.get("name")
        if name not in VARIANTS: raise SystemExit(f"unexpected build variant {name}")
        persist,fast=VARIANTS[name]
        if (b.get("C1_PERSISTENT_WORKSPACE"),b.get("C1_ONE_QUERY_FASTPATH"))!=(persist,fast):
            raise SystemExit(f"{name}: build-manifest gates")
        binary=entry(pathlib.Path(str(b.get("binary",""))),f"{name} binary")
        cache=entry(pathlib.Path(str(b.get("cmake_cache",""))),f"{name} CMakeCache")
        expected_binary=root/"builds"/name/"bin"/"C1Microbench"
        expected_cache=root/"builds"/name/"CMakeCache.txt"
        if pathlib.Path(binary["path"])!=expected_binary or pathlib.Path(cache["path"])!=expected_cache:
            raise SystemExit(f"{name}: noncanonical build artifact location")
        if binary["sha256"]!=b.get("binary_sha256") or cache["sha256"]!=b.get("cmake_cache_sha256"):
            raise SystemExit(f"{name}: build artifact drift from manifest")
        check_cache(expected_cache,name,persist,fast)
        binary.update({
            "name":name,
            "C1_PERSISTENT_WORKSPACE":persist,
            "C1_ONE_QUERY_FASTPATH":fast,
            "expected_from_build_manifest":b["binary_sha256"],
            "cmake_cache":cache,
            "expected_cmake_cache_from_build_manifest":b["cmake_cache_sha256"],
        })
        variants.append(binary)
    helpers={k:entry(root/v,"helper "+k) for k,v in HELPER_PATHS.items()}
    protocols={k:entry(root/v,"protocol "+k) for k,v in PROTOCOL_PATHS.items()}
    validate_v8_trace_provenance(inputs,protocols)
    validate_profile_execution_plan(protocols)
    sources=[]
    for x in bm.get("source_files",[]):
        q=entry(pathlib.Path(str(x.get("path",""))),"worktree source")
        if q["sha256"]!=x.get("sha256"): raise SystemExit(f"source drift {q['path']}")
        q["expected_from_build_manifest"]=x["sha256"]
        sources.append(q)
    if not sources: raise SystemExit("empty source list")
    ci=bm.get("canonical_inspector",{})
    cis=entry(pathlib.Path(str(ci.get("source",""))),"canonical inspector source")
    cib=entry(pathlib.Path(str(ci.get("binary",""))),"canonical inspector binary")
    if cis["sha256"]!=ci.get("source_sha256") or cib["sha256"]!=ci.get("binary_sha256"):
        raise SystemExit("canonical inspector drift")
    out={
        "schema":"gtspp-c1-v8-pins-v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "root":str(root),
        "scope":"All v8 C1 artifacts pinned after CPU/NVCC compilation only; no CUDA binary, nsys, nvidia-smi, or GPU workload executed.",
        "allowed_external_paths":[str(base),str(source_trace)],
        "inputs":inputs,
        "protocols":protocols,
        "source_build_manifest":entry(build,"build manifest"),
        "worktree_source_files":sources,
        "canonical_inspector":{"source":cis["path"],"source_sha256":cis["sha256"],"binary":cib["path"],"binary_sha256":cib["sha256"]},
        "variant_binaries":sorted(variants,key=lambda x:x["name"]),
        "runtime_helpers":helpers,
        "semantic_constraints":{
            "radius":500.0,"replicates":5,"C2_residual_pruning_mode":0,
            "C3_measured_mutations":"forbidden","qnum_per_engine_call":1,
            "variants":sorted(VARIANTS),"measured_schedule_sha256":protocols["execution_plan"]["sha256"],
            "profile_primary_variants":["E_G_c1_off_reference","P_F_full_C1"],
            "profile_execution_plan_sha256":protocols["profile_execution_plan"]["sha256"],
            "correctness_scope":"cross_variant_differential_equivalence_only; no independent absolute range-search oracle is claimed by this pin set",
            "capacity_snapshot_stage":"pre_ephemeral_release",
            "capacity_snapshot_fields":["update_result_capacity_slots","total_result_capacity_slots"],
            "capacity_snapshot_contract":"result_count must be an integer >= 0 and <= both positive equal capacity fields from one local pre-release snapshot",
        },
    }
    target=a.out or root/"hardened_static_pins_v8.json"
    if not target.is_absolute() or target.is_symlink() or target.resolve(strict=False)!=target: raise SystemExit("unsafe pin output path")
    tmp=target.with_name(target.name+".tmp")
    tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    tmp.replace(target)
    print(json.dumps({"pins":str(target),"sha256":sha(target),"sources":len(sources),"variants":len(variants),"helpers":len(helpers)},sort_keys=True))
if __name__=="__main__": main()

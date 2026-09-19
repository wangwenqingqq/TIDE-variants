#!/usr/bin/env python3
"""Strict CPU-only verifier for the separately outer-guarded v6 Nsight profile."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, importlib.util, json, pathlib, re, subprocess
from datetime import datetime, timezone
from typing import Any

ROOT_DEFAULT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot"
PROFILES_ROOT=pathlib.Path(ROOT_DEFAULT)/"profiles"
PINS_PATH=pathlib.Path(ROOT_DEFAULT)/"hardened_static_pins_v6.json"
PRIMARY=("E_G_c1_off_reference","P_F_full_C1")
GATES={"E_G_c1_off_reference":(0,0),"P_F_full_C1":(1,1)}
REPORTS=("cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_mem_size_sum","um_sum")
PHASES=(("cold",128),("provision",1024),("warmup",128),("steady",1024))
CONTRACT_PATH=pathlib.Path(ROOT_DEFAULT)/"capacity_snapshot_contract_v6.py"
_contract_spec=importlib.util.spec_from_file_location("gtspp_c1_v6_capacity_snapshot_contract",CONTRACT_PATH)
if _contract_spec is None or _contract_spec.loader is None: raise RuntimeError("cannot load v6 capacity snapshot contract")
_contract=importlib.util.module_from_spec(_contract_spec); _contract_spec.loader.exec_module(_contract)
capacity_snapshot_violations=_contract.capacity_snapshot_violations

def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def raw_file(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_file():raise SystemExit(f"{label}: absolute regular non-symlink file required: {p}")
    q=p.resolve(strict=True)
    if q!=p:raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p
def raw_dir(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_dir():raise SystemExit(f"{label}: absolute non-symlink directory required: {p}")
    q=p.resolve(strict=True)
    if q!=p:raise SystemExit(f"{label}: noncanonical/symlink traversal: {p} -> {q}")
    return p
def load(p:pathlib.Path,label:str)->dict[str,Any]:
    raw_file(str(p),label)
    try:x=json.loads(p.read_text())
    except Exception as e:raise SystemExit(f"{label}: invalid JSON: {e}")
    if not isinstance(x,dict):raise SystemExit(f"{label}: JSON object required")
    return x
def need(x:bool,m:str)->None:
    if not x:raise SystemExit(m)
def norm_pci(value:Any)->str|None:
    s=str(value).strip().lower()
    m=re.fullmatch(r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])",s)
    return m.group(1) if m else None
def flatten(x:Any,out:list[dict[str,Any]])->None:
    if isinstance(x,dict):
        if set(("path","sha256")).issubset(x):out.append(x)
        for v in x.values():flatten(v,out)
    elif isinstance(x,list):
        for v in x:flatten(v,out)
def parse_telemetry(p:pathlib.Path,uuid:str,label:str)->str:
    raw_file(str(p),label)
    lines=[x.strip() for x in p.read_text(errors="replace").splitlines() if x.strip()]
    need(len(lines)==1,label+": one telemetry row required")
    fields=[x.strip() for x in lines[0].split(",")]
    need(len(fields)>=5 and fields[1]=="0" and fields[2]==uuid,label+": GPU0 UUID telemetry mismatch")
    pci=norm_pci(fields[4]);need(pci is not None,label+": malformed PCI bus id")
    return pci
def check_telemetry(root:pathlib.Path,manifest:dict[str,Any],uuid:str)->str:
    d=raw_dir(str(root/"gpu_snapshots"),"profile snapshot directory")
    snapshot=pathlib.Path(str(manifest.get("gpu",{}).get("telemetry_snapshot","")))
    need(str(snapshot).startswith(str(d)+"/"),"manifest initial profile telemetry boundary")
    outer_pci=parse_telemetry(snapshot,uuid,"outer prelaunch telemetry")
    required=[snapshot,d/"outer_postrun_telemetry_gpu0.csv",d/"outer_exit_telemetry_gpu0.csv"]
    for v in PRIMARY:
        tag="profile_"+v
        pre=sorted(d.glob("pre_"+tag+"_attempt_*_telemetry_gpu0.csv"))
        need(bool(pre),"missing profile prelaunch telemetry "+v)
        required.extend(pre+[d/("post_"+tag+"_telemetry_gpu0.csv")])
        idle=d/("pre_"+tag+"_strict_idle_poll.log")
        raw_file(str(idle),"profile strict idle log")
        need("strict_idle=PASS" in idle.read_text(errors="replace"),"no strict idle PASS "+v)
    for telemetry in required:
        observed_pci=parse_telemetry(telemetry,uuid,"profile telemetry "+telemetry.name)
        need(observed_pci==outer_pci,"profile telemetry PCI differs from outer prelaunch "+telemetry.name)
        prefix=str(telemetry).removesuffix("_telemetry_gpu0.csv")
        for suffix in ("_safety_gpu0.csv","_compute_gpu0.csv","_compute_all_visible.csv"):
            component=pathlib.Path(prefix+suffix);raw_file(str(component),"profile telemetry component")
            if suffix=="_safety_gpu0.csv":need(uuid in component.read_text(errors="replace"),"profile safety UUID mismatch "+component.name)
    return outer_pci
def pin_entry(pins:dict[str,Any],section:str,name:str)->dict[str,Any]:
    e=pins.get(section,{}).get(name)
    need(isinstance(e,dict) and isinstance(e.get("path"),str) and isinstance(e.get("sha256"),str),f"missing pin {section}/{name}")
    p=raw_file(e["path"],"pin "+section+"/"+name);need(sha(p)==e["sha256"],"pin hash mismatch "+section+"/"+name)
    return e
def run_pin_verifier(root:pathlib.Path,pins_path:pathlib.Path,pins:dict[str,Any])->None:
    verifier=pin_entry(pins,"runtime_helpers","pin_verifier")
    r=subprocess.run([sys.executable,"-B","-I",verifier["path"],"--root",str(root),"--pins",str(pins_path),"--static-source"],text=True,capture_output=True)
    need(r.returncode==0,"pinned static verifier failed: "+r.stderr[-500:])
    for v in PRIMARY:
        r=subprocess.run([sys.executable,"-B","-I",verifier["path"],"--root",str(root),"--pins",str(pins_path),"--variant",v],text=True,capture_output=True)
        need(r.returncode==0,"pinned variant verifier failed "+v+": "+r.stderr[-500:])
def expected_plan(pins:dict[str,Any])->dict[str,Any]:
    e=pin_entry(pins,"protocols","profile_execution_plan")
    plan=load(pathlib.Path(e["path"]),"profile execution plan")
    need(plan.get("schema")=="gtspp-c1-v6-profile-execution-plan-v1","profile plan schema")
    need(plan.get("schedule")==[{"replicate":1,"variant":"E_G_c1_off_reference"},{"replicate":1,"variant":"P_F_full_C1"}],"profile plan schedule")
    nsys=plan.get("nsys",{})
    need(nsys.get("executable")=="/usr/local/bin/nsys" and nsys.get("profile_arguments")==["profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true"],"profile nsys command plan")
    need(nsys.get("stats_arguments")==["stats","--force-export=true","--force-overwrite=true","--format","csv"] and nsys.get("reports")==list(REPORTS),"profile stats plan")
    return plan
def verify_child(root:pathlib.Path,pins:dict[str,Any],manifest:dict[str,Any],uuid:str,outer_pci:str,plan:dict[str,Any],child:dict[str,Any],artifact_record:dict[str,Any],expected_variant:str)->dict[str,Any]:
    v=expected_variant;out=raw_dir(str(root/v),"profile variant directory "+v)
    need(child.get("replicate")==1 and child.get("variant")==v,"profile child schedule identity "+v)
    binary_pin={x.get("name"):x for x in pins.get("variant_binaries",[])}.get(v,{})
    binary=child.get("binary",{});cache=child.get("cmake_cache",{})
    need(binary.get("path")==binary_pin.get("path") and binary.get("sha256")==binary_pin.get("sha256"),"profile child binary provenance "+v)
    need(cache.get("path")==binary_pin.get("cmake_cache",{}).get("path") and cache.get("sha256")==binary_pin.get("cmake_cache",{}).get("sha256"),"profile child CMakeCache provenance "+v)
    need(child.get("variant_output")==str(out) and child.get("env_i") is True,"profile child output/env-i "+v)
    env=child.get("environment",{})
    need(env=={"CUDA_VISIBLE_DEVICES":uuid,"NVIDIA_VISIBLE_DEVICES":uuid,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","C1_EXECUTION_MODE":"PROFILE"},"profile child UUID environment "+v)
    base="/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt";trace=str(pathlib.Path(ROOT_DEFAULT)/"inputs/sift1m_10k_442_first1024_type2_query_only.txt")
    binary_argv=[binary_pin.get("path"),"--base",base,"--trace",trace,"--radius","500","--out",str(out),"--replicate","1","--variant",v,"--profile-only"]
    need(child.get("binary_argv")==binary_argv,"profile binary argv "+v)
    prefix=out/"nsys"/"profile"
    nsys_argv=["/usr/local/bin/nsys","profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true","--output",str(prefix),"--",*binary_argv]
    need(child.get("nsys_argv")==nsys_argv,"outer guarded nsys invocation manifest "+v)
    card=load(out/"run_card.json",v+" run card");done=load(out/"completion.json",v+" completion")
    need(card.get("schema")=="gtspp-c1-microbench-run-card-v6" and done.get("schema")=="gtspp-c1-microbench-completion-v6",v+": schemas")
    need(card.get("run_mode")=="UNMEASURED_NSYS_PROFILE_DO_NOT_USE" and card.get("profile_only") is True and done.get("run_mode")=="UNMEASURED_NSYS_PROFILE_DO_NOT_USE",v+": profile mode")
    need(card.get("requested_variant")==v==card.get("compiled_variant")==done.get("compiled_variant") and card.get("replicate")==1,v+": variant contract")
    need(card.get("base")==base and card.get("trace")==trace and float(card.get("radius"))==500.0 and card.get("trace_limit")==1024 and card.get("C2_residual_mode")==0,v+": profile input contract")
    need((card.get("C1_PERSISTENT_WORKSPACE"),card.get("C1_ONE_QUERY_FASTPATH"))==GATES[v],v+": compile gates")
    need(done.get("tree_invariance_pass") is True and int(card.get("cuda_runtime_version",0))>0 and int(card.get("cuda_driver_version",0))>0,v+": profile completion/runtime")
    need(str(card.get("visible_cuda_device_name","")).startswith("NVIDIA RTX PRO 6000"),v+": profile CUDA device")
    need(isinstance(card.get("visible_cuda_device_uuid"),str) and card.get("visible_cuda_device_uuid")==uuid,v+": CUDA UUID differs from manifest/session UUID")
    need(norm_pci(card.get("visible_cuda_device_pci_bus_id",""))==outer_pci,v+": CUDA PCI differs from outer telemetry")
    for phase,expected_count in PHASES:
        op_path=raw_file(str(out/("phase_"+phase+"_ops.jsonl")),v+" "+phase+" capacity JSONL")
        records=[json.loads(line) for line in op_path.read_text().splitlines() if line.strip()]
        need(len(records)==expected_count,v+": "+phase+" capacity record count")
        for ordinal,record in enumerate(records):
            violations=capacity_snapshot_violations(record)
            need(not violations,v+": invalid capacity snapshot "+phase+"/"+str(ordinal)+": "+"; ".join(violations))
    expected=[out/"nsys"/"profile.nsys-rep",out/"nsys"/"profile.sqlite",*[out/"nsys"/"reports"/("profile_"+r+".csv") for r in REPORTS]]
    need(artifact_record.get("replicate")==1 and artifact_record.get("variant")==v,"profile artifact record identity "+v)
    artifacts=artifact_record.get("artifacts",[])
    need(isinstance(artifacts,list) and [x.get("path") for x in artifacts]==[str(x) for x in expected],"profile artifact manifest layout "+v)
    outputs=[]
    for i,path in enumerate(expected):
        path=raw_file(str(path),v+" artifact")
        need(path.stat().st_size>0,"empty profile artifact "+str(path))
        entry=artifacts[i];need(entry.get("sha256")==sha(path) and entry.get("bytes")==path.stat().st_size,"profile artifact hash mismatch "+path.name)
        if path.suffix==".sqlite":need(path.open("rb").read(16)==b"SQLite format 3\x00",v+": invalid Nsight SQLite")
        if path.suffix==".csv":
            lines=[x for x in path.read_text(errors="replace").splitlines() if x.strip()]
            need(len(lines)>=2 and not any("no data" in x.lower() for x in lines),v+": empty/no-data Nsight report "+path.name)
        outputs.append({"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size})
    return {"variant":v,"run_card_sha256":sha(out/"run_card.json"),"completion_sha256":sha(out/"completion.json"),"cuda_pci_bus_id_normalized":outer_pci,"artifacts":outputs}
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--profile-root",type=pathlib.Path,required=True);ap.add_argument("--pins",type=pathlib.Path,required=True);a=ap.parse_args()
    root=raw_dir(str(a.profile_root),"profile root")
    need(root.parent==PROFILES_ROOT and root.name.startswith("c1_v6_profile_"),"profile root must be direct c1_v6_profile_* child of v6 profiles")
    pins_path=raw_file(str(a.pins),"pins");need(pins_path==PINS_PATH,"pins path must be canonical v6 hardened pin file")
    pins=load(pins_path,"pins");need(pins.get("schema")=="gtspp-c1-v6-pins-v1","pins schema")
    run_pin_verifier(pathlib.Path(ROOT_DEFAULT),pins_path,pins)
    plan=expected_plan(pins)
    manifest=load(root/"profile_run_manifest_v6.json","profile run manifest");guard=load(root/"profile_guard_card_v6.json","profile guard card")
    need(manifest.get("schema")=="gtspp-c1-v6-profile-run-manifest-v1" and manifest.get("execution_mode")=="PROFILE" and manifest.get("status")=="COMPLETE","profile manifest mode/status")
    need(manifest.get("root")==ROOT_DEFAULT and manifest.get("profile_root")==str(root) and manifest.get("pins",{}).get("path")==str(pins_path) and manifest.get("pins",{}).get("sha256")==sha(pins_path),"profile manifest root/pins")
    launcher=raw_file(str(manifest.get("trust_root_launcher",{}).get("path","")),"profile launcher manifest path")
    need(launcher==pathlib.Path(ROOT_DEFAULT)/"launch_c1_profile_v6.sh" and sha(launcher)==manifest.get("trust_root_launcher",{}).get("sha256"),"profile launcher provenance")
    profile_guard_pin=pin_entry(pins,"runtime_helpers","profile_guard")
    outer_guard=raw_file(str(manifest.get("outer_profile_guard",{}).get("path","")),"profile guard manifest path")
    need(outer_guard==pathlib.Path(profile_guard_pin["path"]) and sha(outer_guard)==profile_guard_pin["sha256"] and manifest.get("outer_profile_guard",{}).get("sha256")==profile_guard_pin["sha256"],"outer profile guard provenance")
    uuid=str(manifest.get("gpu",{}).get("uuid",""));need(manifest.get("gpu",{}).get("physical_index")==0 and uuid.startswith("GPU-"),"profile GPU provenance")
    need(guard.get("schema")=="gtspp-c1-v6-profile-guard-card-v1" and guard.get("status")=="COMPLETE" and guard.get("execution_mode")=="PROFILE" and guard.get("physical_gpu_index")==0 and guard.get("physical_gpu_uuid")==uuid,"profile guard card provenance")
    expected_entries=[];flatten(pins,expected_entries);expected_map={x["path"]:x["sha256"] for x in expected_entries}
    observed={x.get("path"):x for x in manifest.get("pinned_artifacts",[])}
    need(set(observed)==set(expected_map),"profile manifest pinned artifact set")
    for path,h in expected_map.items():need(observed[path].get("sha256")==h and observed[path].get("actual_sha256")==h,"profile manifest pin drift "+path)
    root_dirs={x.name for x in root.iterdir() if x.is_dir()};need(root_dirs==set(PRIMARY)|{"gpu_snapshots"},"profile root directory set")
    outer_pci=check_telemetry(root,manifest,uuid)
    children=manifest.get("children",[]);records=manifest.get("profile_artifacts",[])
    need(isinstance(children,list) and isinstance(records,list) and len(children)==2 and len(records)==2,"exactly two profile children/artifact records")
    schedule=plan["schedule"]
    need([(x.get("replicate"),x.get("variant")) for x in children]==[(x["replicate"],x["variant"]) for x in schedule],"profile child schedule")
    need([(x.get("replicate"),x.get("variant")) for x in records]==[(x["replicate"],x["variant"]) for x in schedule],"profile artifact schedule")
    events=manifest.get("events",[])
    need(any(x.get("kind")=="outer_prelaunch_pass" for x in events) and any(x.get("kind")=="profile_schedule_verified" for x in events),"profile outer events")
    for v in PRIMARY:
        need(sum(1 for x in events if x.get("kind")=="profile_variant_prelaunch_pass" and x.get("variant")==v)==1,"profile prelaunch event "+v)
        need(sum(1 for x in events if x.get("kind")=="profile_variant_postlaunch_telemetry" and x.get("variant")==v)==1,"profile post telemetry event "+v)
        need(sum(1 for x in events if x.get("kind")=="profile_nsys_stats_complete" and x.get("variant")==v)==1,"profile nsys stats event "+v)
        need(sum(1 for x in events if x.get("kind")=="profile_variant_completion_contract_pass" and x.get("variant")==v)==1,"profile completion event "+v)
    rows=[verify_child(root,pins,manifest,uuid,outer_pci,plan,children[i],records[i],v) for i,v in enumerate(PRIMARY)]
    out={"schema":"gtspp-c1-v6-profile-verification-v3","verified_utc":datetime.now(timezone.utc).isoformat(),"profile_root":str(root),"pins_sha256":sha(pins_path),"profile_run_manifest_sha256":sha(root/"profile_run_manifest_v6.json"),"profile_guard_card_sha256":sha(root/"profile_guard_card_v6.json"),"gpu_uuid":uuid,"gpu_pci_bus_id_normalized":outer_pci,"pass":True,"formal_profile_eligible":True,"profiles":rows,"warning":"Explicitly unmeasured Nsight allocation evidence only; no profile value is timing data and this verifier never launches CUDA/nsys."}
    target=root/"profile_verification_v6.json"
    if target.is_symlink():raise SystemExit("refuse symlink profile verification output")
    tmp=target.with_name(target.name+".tmp");tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");tmp.replace(target)
    print(json.dumps({"pass":True,"output":str(target)}));return 0
if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""Atomic profile provenance manifest utility; it never launches a GPU or nsys itself."""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, platform, subprocess, sys
from datetime import datetime, timezone
from typing import Any

MANIFEST_NAME="profile_run_manifest_v5.json"
ROOT_DEFAULT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v5_formal"
PRIMARY=("E_G_c1_off_reference","P_F_full_C1")
REPORTS=("cuda_api_sum","cuda_gpu_mem_time_sum","cuda_gpu_mem_size_sum","um_sum")
ENV_ALLOW=("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","CUDA_HOME","PATH","LD_LIBRARY_PATH","C1_EXECUTION_MODE")

def utc()->str:return datetime.now(timezone.utc).isoformat()
def sha(path:pathlib.Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def secure_file(value:str,label:str)->pathlib.Path:
    raw=pathlib.Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():raise SystemExit(f"{label} must be absolute regular non-symlink: {raw}")
    resolved=raw.resolve(strict=True)
    if raw!=resolved:raise SystemExit(f"{label} noncanonical/symlink traversal: {raw} -> {resolved}")
    return raw
def secure_dir(value:str,label:str)->pathlib.Path:
    raw=pathlib.Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_dir():raise SystemExit(f"{label} must be absolute non-symlink directory: {raw}")
    resolved=raw.resolve(strict=True)
    if raw!=resolved:raise SystemExit(f"{label} noncanonical/symlink traversal: {raw} -> {resolved}")
    return raw
def future_under(path:pathlib.Path,parent:pathlib.Path,label:str)->pathlib.Path:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=False)!=path or not str(path).startswith(str(parent)+"/"):
        raise SystemExit(f"{label} unsafe future path: {path}")
    return path
def load(path:pathlib.Path,label:str)->dict[str,Any]:
    secure_file(str(path),label)
    try:x=json.loads(path.read_text())
    except Exception as exc:raise SystemExit(f"invalid JSON {label}: {exc}")
    if not isinstance(x,dict):raise SystemExit(f"JSON object required: {label}")
    return x
def flatten(x:Any,out:list[dict[str,Any]])->None:
    if isinstance(x,dict):
        if set(("path","sha256")).issubset(x):out.append(x)
        for value in x.values():flatten(value,out)
    elif isinstance(x,list):
        for value in x:flatten(value,out)
def atomic(path:pathlib.Path,obj:dict[str,Any])->None:
    if path.is_symlink():raise SystemExit("refuse symlink profile manifest output")
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)
def manifest_path(out:str)->pathlib.Path:
    d=secure_dir(out,"profile output")
    p=d/MANIFEST_NAME
    if p.is_symlink():raise SystemExit("profile manifest symlink")
    return p
def command_text(argv:list[str])->str:
    try:return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as exc:return f"UNAVAILABLE: {type(exc).__name__}: {exc}"
def root_and_out(a:argparse.Namespace)->tuple[pathlib.Path,pathlib.Path]:
    root=secure_dir(a.root,"v5 root");out=secure_dir(a.out,"profile output")
    if root!=pathlib.Path(ROOT_DEFAULT) or out.parent!=root/"profiles" or not out.name.startswith("c1_v5_profile_"):
        raise SystemExit("profile root/output boundary")
    return root,out
def init(a:argparse.Namespace)->None:
    root,out=root_and_out(a)
    pins=secure_file(a.pins,"pins");launcher=secure_file(a.launcher,"profile launcher");guard=secure_file(a.guard,"profile guard")
    if launcher!=root/"launch_c1_profile_v5.sh" or guard!=root/"run_c1_profile_guard_v5.sh":raise SystemExit("canonical profile trust path required")
    pin=load(pins,"pins")
    entries=[];flatten(pin,entries)
    if pin.get("schema")!="gtspp-c1-v5-pins-v1" or not entries:raise SystemExit("invalid pins")
    artifacts={}
    for entry in entries:
        item=secure_file(str(entry.get("path","")),"pinned artifact")
        got=sha(item)
        if got!=entry.get("sha256"):raise SystemExit(f"pinned artifact drift {item}")
        artifacts[str(item)]={"path":str(item),"sha256":entry["sha256"],"actual_sha256":got}
    obj={
        "schema":"gtspp-c1-v5-profile-run-manifest-v1","created_utc":utc(),"updated_utc":utc(),"status":"PREPARED","execution_mode":"PROFILE",
        "root":str(root),"profile_root":str(out),"invocation_argv":a.argv,
        "host":{"nodename":os.uname().nodename,"sysname":os.uname().sysname,"release":os.uname().release,"machine":platform.machine()},
        "environment":{k:os.environ[k] for k in ENV_ALLOW if k in os.environ},
        "toolchain":{"python":sys.version,"cmake":command_text(["/usr/bin/cmake","--version"]),"nvcc":command_text(["/usr/local/cuda-13.1/bin/nvcc","--version"]),"nsys":command_text(["/usr/local/bin/nsys","--version"])},
        "pins":{"path":str(pins),"sha256":sha(pins),"schema":pin.get("schema")},
        "trust_root_launcher":{"path":str(launcher),"sha256":sha(launcher)},"outer_profile_guard":{"path":str(guard),"sha256":sha(guard)},
        "pinned_artifacts":sorted(artifacts.values(),key=lambda x:x["path"]),"gpu":None,"events":[],"children":[],"profile_artifacts":[],
    }
    atomic(out/MANIFEST_NAME,obj)
def event(a:argparse.Namespace)->None:
    p=manifest_path(a.out);x=load(p,"profile manifest")
    x.setdefault("events",[]).append({"utc":utc(),"kind":a.kind,"label":a.label,"variant":a.variant,"detail":a.detail})
    x["updated_utc"]=utc();atomic(p,x)
def gpu(a:argparse.Namespace)->None:
    if not a.uuid.startswith("GPU-"):raise SystemExit("invalid GPU UUID")
    p=manifest_path(a.out);x=load(p,"profile manifest");snap=secure_file(a.snapshot,"profile GPU telemetry snapshot")
    out=secure_dir(a.out,"profile output")
    if not str(snap).startswith(str(out/"gpu_snapshots")+"/"):raise SystemExit("profile telemetry outside output")
    x["gpu"]={"physical_index":0,"uuid":a.uuid,"telemetry_snapshot":str(snap)};x["updated_utc"]=utc();atomic(p,x)
def child(a:argparse.Namespace)->None:
    p=manifest_path(a.out);x=load(p,"profile manifest");root=secure_dir(x["root"],"manifest root");out=secure_dir(a.out,"profile output")
    if a.variant not in PRIMARY or a.rep!=1 or not a.uuid.startswith("GPU-"):raise SystemExit("invalid profile child identity")
    binary=secure_file(a.binary,"profile binary");cache=secure_file(a.cmake_cache,"profile CMakeCache");variant_out=secure_dir(a.variant_out,"profile variant output")
    if binary!=root/"builds"/a.variant/"bin"/"C1Microbench" or cache!=root/"builds"/a.variant/"CMakeCache.txt" or variant_out!=out/a.variant:raise SystemExit("profile child path boundary")
    if a.base!="/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt" or a.trace!=str(root/"inputs/sift1m_10k_442_first1024_type2_query_only.txt"):raise SystemExit("profile child input boundary")
    prefix=future_under(variant_out/"nsys"/"profile",variant_out,"nsys output prefix")
    binary_argv=[str(binary),"--base",a.base,"--trace",a.trace,"--radius","500","--out",str(variant_out),"--replicate","1","--variant",a.variant,"--profile-only"]
    nsys_argv=["/usr/local/bin/nsys","profile","--trace=cuda","--cuda-memory-usage=true","--force-overwrite=true","--output",str(prefix),"--",*binary_argv]
    x.setdefault("children",[]).append({"replicate":1,"variant":a.variant,"binary":{"path":str(binary),"sha256":sha(binary)},"cmake_cache":{"path":str(cache),"sha256":sha(cache)},"variant_output":str(variant_out),"binary_argv":binary_argv,"nsys_argv":nsys_argv,"env_i":True,"environment":{"CUDA_VISIBLE_DEVICES":a.uuid,"NVIDIA_VISIBLE_DEVICES":a.uuid,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","C1_EXECUTION_MODE":"PROFILE"}})
    x["updated_utc"]=utc();atomic(p,x)
def artifacts(a:argparse.Namespace)->None:
    p=manifest_path(a.out);x=load(p,"profile manifest");out=secure_dir(a.out,"profile output")
    if a.variant not in PRIMARY or a.rep!=1:raise SystemExit("invalid artifact identity")
    d=out/a.variant
    expected=[d/"nsys"/"profile.nsys-rep",d/"nsys"/"profile.sqlite",*[d/"nsys"/"reports"/("profile_"+r+".csv") for r in REPORTS]]
    supplied=[pathlib.Path(a.rep),pathlib.Path(a.sqlite),*[pathlib.Path(z) for z in a.report]]
    if supplied!=expected:raise SystemExit("profile artifact layout differs from fixed plan")
    rows=[]
    for item in expected:
        item=secure_file(str(item),"profile artifact")
        if item.stat().st_size<=0:raise SystemExit(f"empty profile artifact {item}")
        rows.append({"path":str(item),"sha256":sha(item),"bytes":item.stat().st_size})
    x.setdefault("profile_artifacts",[]).append({"replicate":1,"variant":a.variant,"artifacts":rows})
    x["updated_utc"]=utc();atomic(p,x)
def final(a:argparse.Namespace)->None:
    p=manifest_path(a.out);x=load(p,"profile manifest")
    if not a.status or not a.reason:raise SystemExit("profile status/reason required")
    x["status"]=a.status;x["reason"]=a.reason;x["updated_utc"]=utc();atomic(p,x)
def main()->None:
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("init");q.add_argument("--out",required=True);q.add_argument("--root",required=True);q.add_argument("--pins",required=True);q.add_argument("--launcher",required=True);q.add_argument("--guard",required=True);q.add_argument("argv",nargs=argparse.REMAINDER)
    q=sub.add_parser("event");q.add_argument("--out",required=True);q.add_argument("--kind",required=True);q.add_argument("--label",default="");q.add_argument("--variant",default="");q.add_argument("--detail",default="")
    q=sub.add_parser("gpu");q.add_argument("--out",required=True);q.add_argument("--uuid",required=True);q.add_argument("--snapshot",required=True)
    q=sub.add_parser("child");q.add_argument("--out",required=True);q.add_argument("--rep",type=int,required=True);q.add_argument("--variant",required=True);q.add_argument("--binary",required=True);q.add_argument("--cmake-cache",required=True);q.add_argument("--variant-out",required=True);q.add_argument("--base",required=True);q.add_argument("--trace",required=True);q.add_argument("--uuid",required=True)
    q=sub.add_parser("artifacts");q.add_argument("--out",required=True);q.add_argument("--variant",required=True);q.add_argument("--rep",type=int,required=True);q.add_argument("--sqlite",required=True);q.add_argument("--report",action="append",required=True)
    q=sub.add_parser("final");q.add_argument("--out",required=True);q.add_argument("--status",required=True);q.add_argument("--reason",required=True)
    a=ap.parse_args();{"init":init,"event":event,"gpu":gpu,"child":child,"artifacts":artifacts,"final":final}[a.cmd](a)
if __name__=="__main__":main()

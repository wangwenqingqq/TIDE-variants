#!/usr/bin/env python3
"""Atomic, CPU-only provenance manifest utility for C1 v6."""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, platform, subprocess, sys
from datetime import datetime, timezone
from typing import Any

MANIFEST_NAME="run_manifest_v6.json"
ENV_ALLOW=("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","CUDA_HOME","PATH","LD_LIBRARY_PATH","C1_EXECUTION_MODE")

def utc() -> str: return datetime.now(timezone.utc).isoformat()
def sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
def secure_file(value: str, label: str) -> pathlib.Path:
    raw=pathlib.Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise SystemExit(f"{label} must be an absolute regular non-symlink file: {raw}")
    resolved=raw.resolve(strict=True)
    if raw != resolved:
        raise SystemExit(f"{label} may not traverse a symlink or noncanonical path: {raw} -> {resolved}")
    return raw
def secure_dir(value: str, label: str) -> pathlib.Path:
    raw=pathlib.Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_dir():
        raise SystemExit(f"{label} must be an absolute directory without symlink: {raw}")
    resolved=raw.resolve(strict=True)
    if raw != resolved: raise SystemExit(f"{label} noncanonical/symlink path: {raw}")
    return raw
def load(path: pathlib.Path) -> dict[str,Any]:
    try:
        x=json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"invalid JSON {path}: {exc}")
    if not isinstance(x,dict): raise SystemExit(f"JSON object required: {path}")
    return x
def flatten(x: Any, out: list[dict[str,Any]]) -> None:
    if isinstance(x,dict):
        if set(("path","sha256")).issubset(x): out.append(x)
        for v in x.values(): flatten(v,out)
    elif isinstance(x,list):
        for v in x: flatten(v,out)
def atomic_write(path:pathlib.Path,obj:dict[str,Any])->None:
    if path.is_symlink(): raise SystemExit(f"refuse symlink manifest output {path}")
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)
def command_text(argv:list[str])->str:
    try: return subprocess.check_output(argv,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as exc: return f"UNAVAILABLE: {type(exc).__name__}: {exc}"
def manifest_path(out:str)->pathlib.Path:
    d=secure_dir(out,"run output")
    p=d/MANIFEST_NAME
    if p.is_symlink(): raise SystemExit("manifest path is symlink")
    return p

def init(a:argparse.Namespace)->None:
    root=secure_dir(a.root,"root")
    out=secure_dir(a.out,"run output")
    pins=secure_file(a.pins,"pins")
    launcher=secure_file(a.launcher,"launcher")
    if launcher != root/"launch_c1_v6.sh": raise SystemExit("launcher must be canonical v6 trust-root path")
    if a.mode not in ("MEASURED","PROFILE"): raise SystemExit("invalid mode")
    p=load(pins)
    entries:list[dict[str,Any]]=[]; flatten(p,entries)
    if not entries: raise SystemExit("pins contain no hash entries")
    unique={}
    for e in entries:
        if not isinstance(e.get("sha256"),str) or len(e["sha256"])!=64:
            raise SystemExit("invalid pin hash")
        artifact=secure_file(str(e["path"]),"pinned artifact")
        unique[str(artifact)]={"path":str(artifact),"sha256":e["sha256"],"actual_sha256":sha(artifact)}
    mismatch=[x for x in unique.values() if x["actual_sha256"]!=x["sha256"]]
    if mismatch: raise SystemExit(f"cannot initialize manifest with pin mismatch: {mismatch[:1]}")
    obj={"schema":"gtspp-c1-v6-run-manifest-v2","created_utc":utc(),"updated_utc":utc(),"status":"PREPARED",
         "root":str(root),"run_root":str(out),"execution_mode":a.mode,"invocation_argv":a.argv,
         "host":{"nodename":os.uname().nodename,"sysname":os.uname().sysname,"release":os.uname().release,"machine":platform.machine()},
         "environment":{k:os.environ[k] for k in ENV_ALLOW if k in os.environ},
         "toolchain":{"python":sys.version,"cmake":command_text(["/usr/bin/cmake","--version"]),
                      "nvcc":command_text(["/usr/local/cuda-13.1/bin/nvcc","--version"]),
                      "nsys":command_text(["/usr/local/bin/nsys","--version"])},
         "pins":{"path":str(pins),"sha256":sha(pins),"schema":p.get("schema")},
         "trust_root_launcher":{"path":str(launcher),"sha256":sha(launcher)},
         "pinned_artifacts":sorted(unique.values(),key=lambda x:x["path"]),"gpu":None,"events":[]}
    atomic_write(out/MANIFEST_NAME,obj)
def event(a:argparse.Namespace)->None:
    p=manifest_path(a.out); x=load(p)
    if x.get("status") not in ("PREPARED","COMPLETE","ENGINE_FAILED","CONTRACT_FAILED","SEMANTIC_VERIFICATION_FAILED","ANALYSIS_FAILED","TELEMETRY_INCOMPLETE","INTERRUPTED","FINAL_EVIDENCE_FAILED","POSTRUN_PIN_VERIFICATION_FAILED","MANIFEST_INCOMPLETE","GUARD_CARD_INCOMPLETE","SESSION_CLEANUP_INCOMPLETE"):
        raise SystemExit("unexpected manifest status")
    x.setdefault("events",[]).append({"utc":utc(),"kind":a.kind,"label":a.label,"variant":a.variant,"detail":a.detail})
    x["updated_utc"]=utc(); atomic_write(p,x)
def gpu(a:argparse.Namespace)->None:
    if not a.uuid.startswith("GPU-"): raise SystemExit("invalid GPU UUID")
    p=manifest_path(a.out); x=load(p)
    snapshot=secure_file(a.snapshot,"GPU telemetry snapshot")
    x["gpu"]={"physical_index":0,"uuid":a.uuid,"telemetry_snapshot":str(snapshot)}
    x["updated_utc"]=utc(); atomic_write(p,x)
def final(a:argparse.Namespace)->None:
    p=manifest_path(a.out); x=load(p)
    if not a.status or not a.reason: raise SystemExit("status and reason are required")
    x["status"]=a.status; x["reason"]=a.reason; x["updated_utc"]=utc(); atomic_write(p,x)
def bind_plan(a:argparse.Namespace)->None:
    p=manifest_path(a.out); x=load(p); plan=secure_file(a.execution_plan,"execution plan")
    q=load(plan); tuples=q.get("schedule")
    if not isinstance(tuples,list) or len(tuples)!=20: raise SystemExit("execution plan must contain 20 tuples")
    x["execution_plan"]={"path":str(plan),"sha256":sha(plan),"schedule":tuples,"child_launch_template":q.get("child_launch_template")}
    x["updated_utc"]=utc(); atomic_write(p,x)
def child(a:argparse.Namespace)->None:
    p=manifest_path(a.out); x=load(p)
    binary=secure_file(a.binary,"child binary"); out=secure_dir(a.variant_out,"child output")
    if not a.uuid.startswith("GPU-") or a.rep not in range(1,6): raise SystemExit("invalid child identity")
    argv=[str(binary),"--base",a.base,"--trace",a.trace,"--radius","500","--out",str(out),"--replicate",str(a.rep),"--variant",a.variant]
    x.setdefault("children",[]).append({"replicate":a.rep,"variant":a.variant,"binary":{"path":str(binary),"sha256":sha(binary)},"argv":argv,
      "env_i":True,"environment":{"CUDA_VISIBLE_DEVICES":a.uuid,"NVIDIA_VISIBLE_DEVICES":a.uuid,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","C1_EXECUTION_MODE":"MEASURED"}})
    x["updated_utc"]=utc(); atomic_write(p,x)

def main()->None:
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("init"); q.add_argument("--out",required=True); q.add_argument("--root",required=True); q.add_argument("--pins",required=True); q.add_argument("--launcher",required=True); q.add_argument("--mode",required=True); q.add_argument("argv",nargs=argparse.REMAINDER)
    q=sub.add_parser("event"); q.add_argument("--out",required=True); q.add_argument("--kind",required=True); q.add_argument("--label",default=""); q.add_argument("--variant",default=""); q.add_argument("--detail",default="")
    q=sub.add_parser("gpu"); q.add_argument("--out",required=True); q.add_argument("--uuid",required=True); q.add_argument("--snapshot",required=True)
    q=sub.add_parser("final"); q.add_argument("--out",required=True); q.add_argument("--status",required=True); q.add_argument("--reason",required=True)
    q=sub.add_parser("bind-plan"); q.add_argument("--out",required=True); q.add_argument("--execution-plan",required=True)
    q=sub.add_parser("child"); q.add_argument("--out",required=True); q.add_argument("--rep",type=int,required=True); q.add_argument("--variant",required=True); q.add_argument("--binary",required=True); q.add_argument("--variant-out",required=True); q.add_argument("--base",required=True); q.add_argument("--trace",required=True); q.add_argument("--uuid",required=True)
    a=ap.parse_args()
    {"init":init,"event":event,"gpu":gpu,"final":final,"bind-plan":bind_plan,"child":child}[a.cmd](a)
if __name__=="__main__": main()


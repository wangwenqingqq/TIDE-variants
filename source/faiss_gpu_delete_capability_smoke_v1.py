#!/usr/bin/env python3
"""Capability smoke test for FAISS GpuIndexIVFFlat dynamic deletion.

This is deliberately a small fixed-data API test, not a benchmark.  It only
answers whether the installed FAISS GPU IVF-Flat index supports immediate
remove_ids under the exact API required by a dynamic insert/delete workload.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, socket, sys, time
from pathlib import Path
import numpy as np
import faiss


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name("."+path.name+".tmp."+str(os.getpid()))
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,separators=(",",":")); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--device",type=int,default=0,help="logical device after CUDA_VISIBLE_DEVICES filtering")
    p.add_argument("--dim",type=int,default=128)
    p.add_argument("--nlist",type=int,default=32)
    p.add_argument("--train-n",type=int,default=4096)
    p.add_argument("--add-n",type=int,default=4096)
    p.add_argument("--remove-n",type=int,default=512)
    p.add_argument("--seed",type=int,default=2026080203)
    a=p.parse_args()
    if not (a.dim>0 and a.nlist>0 and a.train_n>=a.nlist and a.add_n>0 and 0<a.remove_n<=a.add_n):
        raise ValueError("invalid smoke dimensions/counts")
    rng=np.random.default_rng(a.seed)
    train=np.ascontiguousarray(rng.standard_normal((a.train_n,a.dim),dtype=np.float32))
    values=np.ascontiguousarray(rng.standard_normal((a.add_n,a.dim),dtype=np.float32))
    ids=np.arange(a.add_n,dtype=np.int64)
    res=None; index=None
    result={
      "schema":"tide-faiss-gpu-delete-capability-smoke-v1",
      "evidence_scope":"api_capability_smoke_only",
      "claim_boundary":["small synthetic API smoke only","not a dynamic performance benchmark","not a FAISS-vs-GTS comparison","not a claim about all FAISS index types"],
      "input":{"seed":a.seed,"dimension":a.dim,"nlist":a.nlist,"train_n":a.train_n,"add_n":a.add_n,"remove_n":a.remove_n},
      "environment":{"hostname":socket.gethostname(),"platform":platform.platform(),"python":sys.version,"faiss_version":getattr(faiss,"__version__","unknown"),"faiss_module":str(Path(faiss.__file__).resolve()),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"logical_device":a.device,"gpu_used":True},
      "remove_ids":{"supported":None,"removed":None,"error_type":None,"error":None,"elapsed_ns":None},
      "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "runner":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())},
    }
    try:
        res=faiss.StandardGpuResources()
        cfg=faiss.GpuIndexIVFFlatConfig(); cfg.device=a.device
        index=faiss.GpuIndexIVFFlat(res,a.dim,a.nlist,faiss.METRIC_L2,cfg)
        index.train(train); res.syncDefaultStreamCurrentDevice()
        index.add_with_ids(values,ids); res.syncDefaultStreamCurrentDevice()
        selector=faiss.IDSelectorBatch(np.arange(a.remove_n,dtype=np.int64))
        t=time.perf_counter_ns()
        removed=index.remove_ids(selector)
        res.syncDefaultStreamCurrentDevice()
        result["remove_ids"].update({"supported":True,"removed":int(removed),"elapsed_ns":time.perf_counter_ns()-t})
    except Exception as exc:
        result["remove_ids"].update({"supported":False,"error_type":type(exc).__name__,"error":str(exc)})
    finally:
        del index; del res
    result["status"]="REMOVE_IDS_SUPPORTED" if result["remove_ids"]["supported"] else "REMOVE_IDS_UNSUPPORTED_OR_FAILED"
    out=Path(a.out).resolve(); write_json_new(out,result)
    print(json.dumps({"status":result["status"],"out":str(out),"remove_ids":result["remove_ids"]},sort_keys=True))

if __name__=="__main__": main()

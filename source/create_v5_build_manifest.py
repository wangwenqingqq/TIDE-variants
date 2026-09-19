#!/usr/bin/env python3
"""CPU-only manifest for the isolated v5 source and compiled binaries."""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, subprocess
from datetime import datetime, timezone

def sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    a=ap.parse_args()
    root=a.root.resolve(); wt=root/"worktree"
    files=[]
    for p in sorted([wt/"CMakeLists.txt", *sorted((wt/"include").rglob("*")), *sorted((wt/"src").rglob("*"))]):
        if p.is_file() and not p.is_symlink():
            files.append({"path":str(p), "relative_path":str(p.relative_to(wt)), "bytes":p.stat().st_size, "sha256":sha(p)})
    variants=[]
    for name,persist,fast in [
        ("E_G_c1_off_reference","OFF","OFF"),("P_G_workspace_only","ON","OFF"),
        ("E_F_fastpath_only","OFF","ON"),("P_F_full_C1","ON","ON")]:
        b=root/"builds"/name/"bin"/"C1Microbench"
        if not b.is_file() or b.is_symlink(): raise SystemExit(f"missing binary {b}")
        variants.append({"name":name,"C1_PERSISTENT_WORKSPACE":persist,"C1_ONE_QUERY_FASTPATH":fast,
                         "binary":str(b),"binary_bytes":b.stat().st_size,"binary_sha256":sha(b),
                         "cmake_cache":str(root/"builds"/name/"CMakeCache.txt"),
                         "cmake_cache_sha256":sha(root/"builds"/name/"CMakeCache.txt")})
    inspector_source=root/"tools"/"canonical_inspect_v5.cpp"
    inspector_binary=root/"builds"/"canonical_inspect_v5"
    if not inspector_source.is_file() or inspector_source.is_symlink() or not inspector_binary.is_file() or inspector_binary.is_symlink():
        raise SystemExit("missing CPU canonical inspector source/binary")
    inspector={"source":str(inspector_source),"source_sha256":sha(inspector_source),
               "binary":str(inspector_binary),"binary_bytes":inspector_binary.stat().st_size,
               "binary_sha256":sha(inspector_binary)}
    out={"schema":"gtspp-c1-v5-build-manifest-v1","generated_utc":datetime.now(timezone.utc).isoformat(),
         "host":os.uname().nodename,"scope":"CPU/NVCC compilation only; no CUDA binary, nsys, or GPU workload executed",
         "worktree":str(wt),"source_files":files,"variants":variants,"canonical_inspector":inspector,
         "toolchain":{"python":text(["python3","--version"]),"cmake":text(["cmake","--version"]),
                      "cxx":text([os.environ.get("CXX","c++"),"--version"]),
                      "nvcc":text(["/usr/local/cuda-13.1/bin/nvcc","--version"])}}
    (root/"worktree_build_manifest_v5.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"source_files":len(files),"variants":len(variants),"manifest":str(root/"worktree_build_manifest_v5.json")}))

if __name__=="__main__":
    main()

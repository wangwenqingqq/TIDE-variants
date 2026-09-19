#!/usr/bin/env python3
"""Create or verify a cooperative content-addressed artifact manifest.

No network, signing, host execution, or retry semantics are provided. Creation
uses exclusive output creation; callers must use a new artifact directory.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from tide_protocol_lib import (ProtocolError, canonical_json_bytes, load_json, print_result,
    require, require_mapping, require_relative_path, require_sha256, sha256_file)

def entries(root: Path, exclude: Path | None = None):
    out=[]
    for path in sorted(root.rglob("*")):
        if path == exclude: continue
        if path.is_symlink(): raise ProtocolError(f"unsafe symlink: {path}")
        if path.is_file():
            rel=path.relative_to(root).as_posix(); out.append({"path":rel,"bytes":path.stat().st_size,"sha256":sha256_file(path)})
        elif not path.is_dir(): raise ProtocolError(f"unsafe nonregular entry: {path}")
    require(out,"artifact root has no files")
    return out

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="cmd",required=True)
    create=sub.add_parser("create"); create.add_argument("--root",required=True,type=Path); create.add_argument("--out",required=True,type=Path); create.add_argument("--bundle-id",required=True)
    verify=sub.add_parser("verify"); verify.add_argument("--root",required=True,type=Path); verify.add_argument("--manifest",required=True,type=Path)
    args=ap.parse_args()
    try:
        if args.cmd=="create":
            require(args.root.is_dir(),"root is not directory"); require(args.out.parent==args.root,"manifest must be directly under root")
            manifest={"schema":"tide.artifact-manifest.v1","bundle_id":args.bundle_id,"storage_scope":"cooperative_no_concurrent_mutation_only","entries":entries(args.root,args.out)}
            with args.out.open("xb") as h: h.write(canonical_json_bytes(manifest))
            print_result({"status":"MANIFEST_CREATED","entries":len(manifest["entries"]),"out":str(args.out)}); return 0
        manifest=load_json(args.manifest); require(manifest.get("schema")=="tide.artifact-manifest.v1","manifest schema")
        expected=require_mapping({row["path"]:row for row in manifest.get("entries",[])},"entries")
        actual={row["path"]:row for row in entries(args.root,args.manifest)}
        require(expected==actual,"manifest entries/hash/bytes mismatch")
        print_result({"status":"MANIFEST_VERIFIED_COOPERATIVE_ONLY","entries":len(actual)}); return 0
    except (ProtocolError,OSError,KeyError) as error: print_result({"status":"FAIL_CLOSED","error":str(error)}); return 2
if __name__=="__main__": sys.exit(main())

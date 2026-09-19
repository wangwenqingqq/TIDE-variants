#!/usr/bin/env python3
"""Source-only static audit; deliberately does not execute protocol inputs."""
from __future__ import annotations
import argparse, ast, json, py_compile, sys
from pathlib import Path
from tide_protocol_lib import ProtocolError, canonical_json_bytes, print_result, require
REQUIRED={"README.md","NONCLAIMS.md","PROTOCOL.md","INTEGRATION_HANDOFF.md","STATIC_AUDIT_PROTOCOL.md","tools/tide_protocol_lib.py","tools/validate_trace.py","tools/oracle_replayer.py","tools/validate_e1_bundle.py","tools/validate_e2e3_plan.py","tools/seal_artifact_manifest.py"}
BANNED={"socket","subprocess","requests","urllib","http","torch","cupy","cuda","ctypes"}
def main()->int:
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--root",required=True,type=Path); ap.add_argument("--write-result",required=True,type=Path); args=ap.parse_args()
 try:
  root=args.root.resolve(); require(root.is_dir(),"root missing"); require(args.write_result.parent.resolve()==root,"result must be direct child of root")
  files=[]
  for path in root.rglob("*"):
   if path.is_symlink(): raise ProtocolError(f"symlink prohibited: {path}")
   if path.is_file(): files.append(path)
  rels={p.relative_to(root).as_posix() for p in files}; require(REQUIRED<=rels,f"missing required files: {sorted(REQUIRED-rels)}")
  for path in files:
   if path.suffix==".json": json.loads(path.read_text(encoding="utf-8"))
   if path.suffix==".py":
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path)); py_compile.compile(str(path),doraise=True)
    for node in ast.walk(tree):
     if isinstance(node,(ast.Import,ast.ImportFrom)):
      names=[n.name.split(".")[0] for n in node.names] if isinstance(node,ast.Import) else [node.module.split(".")[0]] if node.module else []
      require(not (set(names)&BANNED),f"banned runtime import in {path}: {set(names)&BANNED}")
  readme=(root/"README.md").read_text(encoding="utf-8"); require("not evidence" in readme,"README must state non-evidence")
  result={"schema":"tide.static-audit.v1","status":"PASS_SOURCE_ONLY_STATIC_AUDIT","files":len(files),"nonclaim":"No GTS/GPU/vector/trace/host runner was executed."}
  with args.write_result.open("xb") as h:h.write(canonical_json_bytes(result))
  print_result(result); return 0
 except (ProtocolError,OSError,ValueError,json.JSONDecodeError,py_compile.PyCompileError) as error: print_result({"status":"FAIL_CLOSED","error":str(error)}); return 2
if __name__=="__main__":sys.exit(main())

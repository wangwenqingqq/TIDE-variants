#!/usr/bin/env python3
"""Fail closed unless writerflush v1 is exactly the provenance-only successor."""
from __future__ import annotations
import hashlib,json,pathlib
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
OLD=ROOT/"src/gts_speculative_fallback_v4_metadatafix_v1.cu"
NEW=ROOT/"src/gts_speculative_fallback_v4_writerflush_v1.cu"
def sha(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def transform(s:str)->str:
 header="""// Safe-C2 v4 speculative gamma traversal plus gamma-only fallback skeleton.
// This isolated implementation never mutates v2."""
 successor="""// Safe-C2 v4 speculative gamma traversal plus gamma-only fallback skeleton.
// Writer-integrity successor: closes each per-query JSONL before readback FNV.
// This isolated implementation never mutates v2."""
 if s.count(header)!=1: raise RuntimeError("header anchor")
 s=s.replace(header,successor,1)
 before='''  if (!out) fail("failed writing v4 per-query JSONL");
  std::ifstream input(path, std::ios::binary);
'''
 after='''  // The summary binds the byte-level FNV of the completed artifact, not an
  // implementation-defined view while the output stream is still buffered.
  out.flush();
  if (!out) fail("failed flushing v4 per-query JSONL");
  out.close();
  if (!out) fail("failed closing v4 per-query JSONL");
  std::ifstream input(path, std::ios::binary);
'''
 if s.count(before)!=1: raise RuntimeError("writer anchor")
 return s.replace(before,after,1)
def main()->int:
 if not OLD.is_file() or OLD.is_symlink() or not NEW.is_file() or NEW.is_symlink():
  raise SystemExit("FAIL_WRITERFLUSH_SOURCE_AUDIT: source path")
 old=OLD.read_text(); new=NEW.read_text()
 if sha(OLD)!="810245916ff301a923c14847bb3e89f9d9d0e85523421c7856bc0bf34b83bf89":
  raise SystemExit("FAIL_WRITERFLUSH_SOURCE_AUDIT: predecessor hash")
 if new!=transform(old):
  raise SystemExit("FAIL_WRITERFLUSH_SOURCE_AUDIT: non-permitted source difference")
 lo=new.index("std::uint64_t write_per_query_jsonl")
 hi=new.index("\n}",lo)
 block=new[lo:hi]
 if not (block.index("out.flush();") < block.index("out.close();") < block.index("std::ifstream input")):
  raise SystemExit("FAIL_WRITERFLUSH_SOURCE_AUDIT: write/read ordering")
 print(json.dumps({
  "schema":"safe-c2-v4-writerflush-source-audit-v1",
  "status":"PASS_WRITERFLUSH_PROVENANCE_ONLY_SOURCE_AUDIT",
  "predecessor_source":{"path":str(OLD),"sha256":sha(OLD)},
  "successor_source":{"path":str(NEW),"sha256":sha(NEW)},
  "permitted_change_scope":["close completed per-query JSONL before byte-level FNV readback"],
  "explicit_non_changes":["no CUDA kernel/header modification","no gamma-selection constant or control-flow modification","no query/update/fallback algorithm modification","no workload split or result modification"],
  "gpu_binary_executed":False,
  "formal_claim_eligible":False
 },sort_keys=True))
 return 0
if __name__=="__main__": raise SystemExit(main())

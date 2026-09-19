#!/usr/bin/env python3
"""Fail closed unless the Safe-C2 v4 metadata/path-safety source is an exact allowed transform."""
from __future__ import annotations
import hashlib,json,pathlib,sys
ROOT=pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
OLD=ROOT/"src/gts_speculative_fallback_v4_sift1m.cu"
NEW=ROOT/"src/gts_speculative_fallback_v4_metadatafix_v1.cu"
def sha(p:pathlib.Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def transform(s:str)->str:
 def once(a:str,b:str)->None:
  nonlocal s
  if s.count(a)!=1:raise RuntimeError("expected one source fragment: "+a[:80])
  s=s.replace(a,b,1)
 def exact(a:str,b:str,n:int)->None:
  nonlocal s
  if s.count(a)!=n:raise RuntimeError("expected source fragment count "+str(n)+": "+a)
  s=s.replace(a,b)
 once("// historical provenance only and is explicitly forbidden for v3 tuning/runs.","// historical provenance only and is explicitly forbidden for v4 tuning/runs.")
 once("// V3 binds a separate compact workload manifest before any executable run.","// V4 binds a strict fresh-workload manifest before any executable run.")
 once("// independently frozen compact workload manifest.  Do not substitute a shell","// strict manifest-bound workload.  Do not substitute a shell")
 once('''bool strict_child_path(const std::string& path, const std::string& root) {
  return path.size() > root.size() && path.compare(0, root.size(), root) == 0 &&
         path[root.size()] == '/';
}
''','''bool strict_child_path(const std::string& path, const std::string& root) {
  return path.size() > root.size() && path.compare(0, root.size(), root) == 0 &&
         path[root.size()] == '/';
}

// For normal executable inputs/outputs, lexical containment alone is not
// sufficient: reject any path that traverses a symlink or whose canonical root
// differs from the pinned v4 root. Receipt paths are handled separately because
// they are created only after parser validation succeeds.
bool strict_existing_child_path(const std::string& path, const std::string& root) {
  if (!strict_child_path(path, root)) return false;
  char resolved_root[PATH_MAX] = {};
  char resolved_path[PATH_MAX] = {};
  if (::realpath(root.c_str(), resolved_root) == nullptr ||
      ::realpath(path.c_str(), resolved_path) == nullptr) return false;
  const std::string canonical_root(resolved_root);
  const std::string canonical_path(resolved_path);
  return root == canonical_root && path == canonical_path &&
         strict_child_path(canonical_path, canonical_root);
}
''')
 once('if (strict_child_path(path, kV2Root)) fail("v2 root is forbidden for " + label);','if (strict_existing_child_path(path, kV2Root)) fail("v2 root is forbidden for " + label);')
 once('if (must_live_under_v4 && !strict_child_path(path, kV4Root)) {','if (must_live_under_v4 && !strict_existing_child_path(path, kV4Root)) {')
 once('if (!strict_child_path(args.workload_manifest, kV4Root)) {','if (!strict_existing_child_path(args.workload_manifest, kV4Root)) {')
 once('if (!args.gamma_vector_file.empty() && !strict_child_path(args.gamma_vector_file, kV4Root)) {','if (!args.gamma_vector_file.empty() && !strict_existing_child_path(args.gamma_vector_file, kV4Root)) {')
 once('''if (!args.validate_workload_only && !strict_child_path(args.output_dir, kV4Root)) {
    fail("v4 output directory must reside under isolated v4 root");
  }''','''if (!args.validate_workload_only && !strict_existing_child_path(args.output_dir, kV4Root)) {
    fail("v4 output directory must reside under isolated v4 root without symlink traversal");
  }
  if (!args.validate_workload_only) {
    require_direct_directory(args.output_dir, "v4 output directory");
  }''')
 once("cannot write v3 per-query JSONL","cannot write v4 per-query JSONL")
 once("safe-c2-v3-per-query-v1","safe-c2-v4-per-query-v1")
 once("failed writing v3 per-query JSONL","failed writing v4 per-query JSONL")
 once("cannot reopen v3 per-query JSONL","cannot reopen v4 per-query JSONL")
 once("safe-c2-v3-baseline-invalid-diagnostic-v1","safe-c2-v4-baseline-invalid-diagnostic-v1")
 exact("v3_speculative_fallback","v4_speculative_fallback",2)
 once("forbidden for v3 input/tuning","forbidden for v4 input/tuning")
 once("safe-c2-speculative-fallback-v3-run-v1","safe-c2-speculative-fallback-v4-run-v1")
 once("v3 isolated Safe-C2 speculative gamma traversal with gamma-only fallback on an independently frozen compact workload; empirical guarded-output evaluation only","Safe-C2 v4 speculative gamma traversal with gamma-only fallback on a strict fresh manifest-bound workload; calibration selects a fixed gamma vector only and held-out evaluation is separately gated")
 once("v3 speculative-fallback skeleton; not v2 and not the submitted legacy C2 kernel","v4 speculative-fallback skeleton; not v2 and not the submitted legacy C2 kernel")
 once("forbidden for v3 gamma tuning, calibration, evaluation, retry, or input binding","forbidden for v4 gamma tuning, calibration, evaluation, retry, or input binding")
 once("gts-v4-compact-learn-workload-v1",'" << kV4WorkloadSchema << "')
 return s
def main()->int:
 old=OLD.read_text()
 new=NEW.read_text()
 expected=transform(old)
 if new!=expected:raise SystemExit("FAIL_METADATAFIX_SOURCE_AUDIT: new source differs from exact permitted transform")
 for stale in ("safe-c2-v3-per-query-v1","safe-c2-v3-baseline-invalid-diagnostic-v1","safe-c2-speculative-fallback-v3-run-v1","v3_speculative_fallback","gts-v4-compact-learn-workload-v1","cannot reopen v3 per-query JSONL"):
  if stale in new:raise SystemExit("FAIL_METADATAFIX_SOURCE_AUDIT: stale output identity "+stale)
 if "strict_existing_child_path(args.output_dir, kV4Root)" not in new or 'require_direct_directory(args.output_dir, "v4 output directory")' not in new:
  raise SystemExit("FAIL_METADATAFIX_SOURCE_AUDIT: output path defense absent")
 print(json.dumps({
  "schema":"safe-c2-v4-metadatafix-source-audit-v1",
  "status":"PASS_V4_METADATA_AND_PATH_SAFETY_SOURCE_AUDIT",
  "old_source":{"path":str(OLD),"sha256":sha(OLD)},
  "new_source":{"path":str(NEW),"sha256":sha(NEW)},
  "permitted_change_scope":[
   "output metadata corrected from stale v3/compact labels to v4/fresh labels",
   "workload schema emitted from kV4WorkloadSchema",
   "direct canonical-path enforcement for existing normal inputs and output directory"
  ],
  "explicit_non_changes":[
   "no CUDA kernel/header modification",
   "no gamma-selection constant or control-flow modification",
   "no query/update algorithm modification"
  ]
 },sort_keys=True))
 return 0
if __name__=="__main__":raise SystemExit(main())

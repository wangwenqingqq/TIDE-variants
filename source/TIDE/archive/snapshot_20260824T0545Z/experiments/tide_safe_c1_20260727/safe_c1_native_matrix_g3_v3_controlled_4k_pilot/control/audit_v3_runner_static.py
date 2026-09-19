#!/usr/bin/env python3
"""CPU-only static admission audit; it never invokes CUDA, a compiler, or a binary."""
from __future__ import annotations
import argparse, hashlib, json, stat
from pathlib import Path

ROOT=Path("/workspace/experiments/tide_safe_c1_20260727/safe_c1_native_matrix_g3_v3_controlled_4k_pilot")
SOURCE=ROOT/"src/g3_safe_c1_native_matrix.cu"
HEADER=ROOT/"src/g3_safe_search_v2.cuh"
WRAPPER=ROOT/"runner/g3_4k_pilot_wrapper.cu"
FIXTURE=ROOT/"inputs/g3_sift4096_branchstress_l2_d3"
BOOT=ROOT/"preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3"
EXPECTED={
 "src/g3_safe_c1_native_matrix.cu":"c22a7682f5aa56e4d96ae2a912348eaf4bcb379971950a21bd15fd89f5c5ca81",
 "src/g3_safe_search_v2.cuh":"e52ae9a14db32967ef0816d880a6fb46ed7793e86ebc85175b9f96986f46b4f0",
 "runner/g3_4k_pilot_wrapper.cu":"6b46dab9906db657996c7ab6ca4b1f39c0175ef78faede4c78625948307b8242",
 "reference/include/tree.cuh":"f812c385d254d77e84bbd515e4055e94cc8163ee7a7d57216f3659c09f569c2b",
 "reference/include/file.cuh":"b8be03246ec82cadd3228b75a6dfd91c552ccfa3124d2ca9470e28e0031ffd8a",
 "reference/include/config.cuh":"622d0977e49d80d5791364bc12463de9bce5aaca324d6a681512004c20d100cb",
 "reference/include/mlp_constant.cuh":"cf6545624c0cbc51744b978298e6d138591521c7b600ece8812b6195f735e559",
 "reference/include/residual_pruning.cuh":"745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2",
}
RAW=("upload_rp_constants","searchIndexKnnV2","indexConstru","max_dis_d","c_rp_mode","cudaMemcpyToSymbol")
ART=(".o",".so",".a",".ptx",".cubin")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def reg(p,err,label):
 try:s=p.lstat()
 except FileNotFoundError:err.append("missing:"+label);return False
 if stat.S_ISLNK(s.st_mode) or not stat.S_ISREG(s.st_mode) or p.resolve()!=p:
  err.append("unsafe:"+label);return False
 return True
def main():
 a=argparse.ArgumentParser();a.add_argument("--out",required=True,type=Path);x=a.parse_args()
 e=[]; hashes={}
 if not ROOT.is_dir() or ROOT.is_symlink() or ROOT.resolve()!=ROOT:e.append("unsafe_root")
 for rel,want in EXPECTED.items():
  p=ROOT/rel
  if reg(p,e,rel):
   got=sha(p);hashes[rel]=got
   if got!=want:e.append("hash:"+rel)
 src=SOURCE.read_text() if SOURCE.exists() else ""; wr=WRAPPER.read_text() if WRAPPER.exists() else ""
 for t in ("int tree_height = 0;","int node_count = 0;","int fanout = 0;","prepared.tree_height = candidate_snapshot.tree_height;","prepared.node_count = candidate_snapshot.node_count;","prepared.fanout = candidate_snapshot.fanout;","\\\"tree_height\\\":","\\\"node_count\\\":","\\\"fanout\\\":"):
  if t not in src:e.append("source_token:"+t)
 if wr.count('#include "../src/g3_safe_c1_native_matrix.cu"')!=1:e.append("single_tu_include")
 if wr.count("int main(")!=1:e.append("single_main")
 for t in RAW:
  if t in wr:e.append("raw_wrapper_token:"+t)
 for t in ("exact_oracle(","std::int64_t sum = 0;","matrix.build_initial_base(","matrix.rebuild_from_current_live()","verify_first_post_rebuild_query","bootstrap_attestation.json","ticket_attestation.jsonl","engine.jsonl","G3_DIRECT_SIDECAR_RUNTIME_GUARD_OPEN","require_native_certificate_contract","\"record\\\":\\\"update\\\"","\"direct_runtime_guard_open\\\":false","\"placement\\\":\\\"","\"prior_placement\\\":\\\""):
  if t not in wr:e.append("wrapper_token:"+t)
 mp=FIXTURE/"manifest.json"
 if reg(mp,e,"manifest"):
  try:
   m=json.loads(mp.read_text())
   if m.get("schema")!="safe-c1-g3-bundle-manifest-v1" or m.get("status")!="CPU_PREPARED_NOT_NATIVE_EXECUTED":e.append("manifest_contract")
   fs=m.get("files_sha256",{})
   if not isinstance(fs,dict):e.append("manifest_map")
   else:
    for n,want in fs.items():
     p=FIXTURE/n
     if reg(p,e,"fixture:"+n):
      got=sha(p);hashes["inputs/g3_sift4096_branchstress_l2_d3/"+n]=got
      if got!=want:e.append("fixture_hash:"+n)
  except Exception as q:e.append("manifest_parse:"+str(q))
 for n,want in {"bootstrap_oracle.json":"4c74995c19f6a9a1309d75bb1611850ceec877b5c66f813f7f0e120e42e0ad1f","bootstrap_oracle_validation.json":"8f5ea4416ca20ece3c960b683420844e2aa84931c6990ac958120ab2e1e4e851"}.items():
  p=BOOT/n
  if reg(p,e,"bootstrap:"+n):
   got=sha(p);hashes["preflight/bootstrap_oracles/g3_sift4096_branchstress_l2_d3/"+n]=got
   if got!=want:e.append("bootstrap_hash:"+n)
 arts=[]
 for p in ROOT.rglob("*"):
  if p.is_symlink():e.append("symlink:"+str(p.relative_to(ROOT)))
  if p.is_file() and p.suffix in ART:arts.append(str(p.relative_to(ROOT)))
 if arts:e.append("precompile_artifacts:"+",".join(arts))
 runs=ROOT/"runs"
 if not runs.is_dir() or runs.is_symlink():e.append("unsafe_runs")
 elif any(runs.iterdir()):e.append("runs_not_empty")
 out={"schema":"safe-c1-g3-v3-static-runner-admission-v1","status":"PASS_STATIC_DESIGN_NO_COMPILE" if not e else "FAIL","scope":"CPU-only source/input audit; no compiler, CUDA, GPU telemetry, native execution, correctness, or performance result.","release_root":str(ROOT),"source_hashes":hashes,"single_wrapper_translation_unit":wr.count('#include "../src/g3_safe_c1_native_matrix.cu"')==1,"direct_runtime_guard_expected_open":False,"range_scope":"branch-aligned predicate mirror plus exact filter; not archive-native/direct-range","linked_executable":False,"native_engine_executed":False,"gpu_used":False,"error_count":len(e),"errors":e}
 x.out.parent.mkdir(parents=True,exist_ok=True);tmp=x.out.with_name(x.out.name+".tmp");tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");tmp.replace(x.out);return 0 if not e else 1
if __name__=="__main__":raise SystemExit(main())

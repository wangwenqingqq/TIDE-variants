#!/usr/bin/env python3
"""Strict CPU-only differential-semantic gate for isolated C1 v7 measured runs."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, importlib.util, json, math, pathlib, re, subprocess
from datetime import datetime, timezone
from typing import Any

ROOT_DEFAULT="/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v7_profile_manifest_repair"
RUNS_ROOT=pathlib.Path(ROOT_DEFAULT)/"runs"
PINS_PATH=pathlib.Path(ROOT_DEFAULT)/"hardened_static_pins_v7.json"
VARIANTS=("E_G_c1_off_reference","P_G_workspace_only","E_F_fastpath_only","P_F_full_C1")
PHASES=(("cold",128),("provision",1024),("warmup",128),("steady",1024))
TREE_LABELS=("after_build_and_session_setup","after_cold","after_provision","after_warmup","after_steady")
GATES={"E_G_c1_off_reference":(0,0),"P_G_workspace_only":(1,0),"E_F_fastpath_only":(0,1),"P_F_full_C1":(1,1)}
CONTRACT_PATH=pathlib.Path(ROOT_DEFAULT)/"capacity_snapshot_contract_v7.py"
_contract_spec=importlib.util.spec_from_file_location("gtspp_c1_v7_capacity_snapshot_contract",CONTRACT_PATH)
if _contract_spec is None or _contract_spec.loader is None: raise RuntimeError("cannot load v7 capacity snapshot contract")
_contract=importlib.util.module_from_spec(_contract_spec); _contract_spec.loader.exec_module(_contract)
CAPACITY_SNAPSHOT_STAGE=_contract.CAPACITY_SNAPSHOT_STAGE
capacity_snapshot_violations=_contract.capacity_snapshot_violations

def utc()->str: return datetime.now(timezone.utc).isoformat()
def sha(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def raw_regular(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_file():
        raise ValueError(f"{label}: not absolute regular non-symlink {p}")
    r=p.resolve(strict=True)
    if p!=r: raise ValueError(f"{label}: noncanonical or symlink traversal {p} -> {r}")
    return p
def raw_dir(value:str,label:str)->pathlib.Path:
    p=pathlib.Path(value)
    if not p.is_absolute() or p.is_symlink() or not p.is_dir():
        raise ValueError(f"{label}: not absolute non-symlink directory {p}")
    r=p.resolve(strict=True)
    if p!=r: raise ValueError(f"{label}: noncanonical or symlink traversal {p} -> {r}")
    return p
def under(p:pathlib.Path,parent:pathlib.Path)->bool:
    return str(p).startswith(str(parent)+"/")
def read_json(p:pathlib.Path,label:str)->dict[str,Any]:
    raw_regular(str(p),label)
    x=json.loads(p.read_text())
    if not isinstance(x,dict): raise ValueError(f"{label}: JSON object required")
    return x
class Gate:
    def __init__(self): self.failures:list[str]=[]
    def need(self,x:bool,msg:str)->bool:
        if not x: self.failures.append(msg)
        return x
    def attempt(self,label:str,fn):
        try: return fn()
        except Exception as e:
            self.failures.append(f"{label}: {type(e).__name__}: {e}")
            return None

def pinned_path(pins:dict[str,Any],section:str,name:str,g:Gate)->pathlib.Path|None:
    e=pins.get(section,{}).get(name,{})
    try:
        p=raw_regular(str(e["path"]),f"pin {section}/{name}")
        g.need(sha(p)==e.get("sha256"),f"pin SHA mismatch {section}/{name}")
        return p
    except Exception as exc:
        g.need(False,f"pin {section}/{name}: {exc}")
        return None

def provenance_qids(pins:dict[str,Any],g:Gate)->list[int]:
    base=pinned_path(pins,"inputs","base",g)
    source=pinned_path(pins,"inputs","source_update_trace",g)
    derived=pinned_path(pins,"inputs","derived_query_only_trace",g)
    provenance=pinned_path(pins,"inputs","derived_trace_provenance",g)
    protocol=pinned_path(pins,"protocols","frozen_protocol",g)
    formal=pinned_path(pins,"protocols","formal_supersession",g)
    if any(x is None for x in (base,source,derived,provenance,protocol,formal)): return []
    try:
        prov=read_json(provenance,"trace provenance")
    except Exception as exc:
        g.need(False,f"trace provenance JSON: {exc}")
        return []
    g.need(prov.get("schema")=="gtspp-c1-query-only-trace-provenance-v2","trace provenance schema")
    def bound(key:str,p:pathlib.Path,e:dict[str,Any])->None:
        x=prov.get(key,{})
        g.need(isinstance(x,dict) and x.get("path")==str(p) and x.get("sha256")==e.get("sha256"),f"provenance {key} pin binding")
    bound("base",base,pins["inputs"]["base"])
    bound("source_update_trace",source,pins["inputs"]["source_update_trace"])
    bound("derived_trace",derived,pins["inputs"]["derived_query_only_trace"])
    bound("protocol",protocol,pins["protocols"]["frozen_protocol"])
    bound("formal_supersession",formal,pins["protocols"]["formal_supersession"])
    try:
        base_header=base.open().readline().strip().split()
        g.need(len(base_header)==3 and base_header==["128","1000000","2"],"base header")
        src_lines=[x.strip() for x in source.read_text().splitlines() if x.strip()]
        der_lines=[x.strip() for x in derived.read_text().splitlines() if x.strip()]
        g.need(len(src_lines)==10001 and src_lines[0]=="10000","source trace header/length")
        g.need(len(der_lines)==1025 and der_lines[0]=="1024","derived trace header/length")
        events=[]
        for ordinal,line in enumerate(src_lines[1:]):
            fields=line.split()
            g.need(len(fields)==2 and fields[0] in {"0","1","2"},f"source trace row {ordinal}")
            if len(fields)==2 and fields[0] in {"0","1","2"}: events.append((int(fields[0]),int(fields[1])))
        selected=[(i,op,qid) for i,(op,qid) in enumerate(events) if op==2][:1024]
        ords=prov.get("selected_source_event_ordinals_zero_based")
        lnums=prov.get("selected_source_line_numbers_one_based")
        rows=prov.get("selected_type2_rows")
        g.need(isinstance(ords,list) and len(ords)==1024 and ords==[x[0] for x in selected],"provenance selected ordinals")
        g.need(isinstance(lnums,list) and len(lnums)==1024 and lnums==[x[0]+2 for x in selected],"provenance selected line numbers")
        g.need(isinstance(rows,list) and len(rows)==1024,"provenance selected rows")
        qids=[]
        for j,(ordinal,op,qid) in enumerate(selected):
            expected=f"2 {qid}"
            g.need(der_lines[j+1]==expected,f"derived trace reconstruction row {j}")
            if isinstance(rows,list) and j<len(rows):
                row=rows[j]
                g.need(isinstance(row,dict) and row.get("source_event_ordinal_zero_based")==ordinal and row.get("source_line_number_one_based")==ordinal+2 and row.get("row")==expected,f"provenance row witness {j}")
            g.need(0<=qid<1000000,f"derived qid range {j}")
            qids.append(qid)
        g.need(prov.get("selection_rule")=="first 1024 source rows with flag==2 in original source-file order; serialize each as `2 original_id` after a 1024 header","provenance selection rule")
        g.need(prov.get("source_type2_rows_exactly_reconstruct_derived_trace") is True,"provenance reconstruction assertion")
        g.need(prov.get("selected_id_min")==min(qids) and prov.get("selected_id_max")==max(qids),"provenance ID span")
        return qids
    except Exception as exc:
        g.need(False,f"trace/provenance reconstruction: {type(exc).__name__}: {exc}")
        return []

def parse_jsonl(p:pathlib.Path,g:Gate,label:str)->list[dict[str,Any]]:
    out=[]
    try:
        raw_regular(str(p),label)
        for n,line in enumerate(p.read_text().splitlines(),1):
            if line.strip(): out.append(json.loads(line))
    except Exception as e: g.need(False,f"{label}: invalid JSONL line: {e}")
    return out

def inspect_canonical(inspector:pathlib.Path,artifact:pathlib.Path,g:Gate,label:str)->list[tuple[int,int,int,str]]:
    r=subprocess.run([str(inspector),str(artifact)],text=True,capture_output=True)
    if r.returncode!=0:
        g.need(False,f"{label}: canonical inspector rc={r.returncode}: {r.stderr[-500:]}")
        return []
    rows=[]; summary=None
    for line in r.stdout.splitlines():
        f=line.split("\t")
        if f and f[0]=="REC" and len(f)==6:
            try: rows.append((int(f[1]),int(f[2]),int(f[3]),f[4]))
            except Exception: g.need(False,f"{label}: malformed inspector record")
        elif f and f[0]=="SUMMARY" and len(f)==3:
            try: summary=(int(f[1]),int(f[2]))
            except Exception: g.need(False,f"{label}: malformed inspector summary")
        else: g.need(False,f"{label}: unexpected inspector output {line[:160]}")
    g.need(summary is not None and summary[0]==len(rows),f"{label}: inspector summary mismatch")
    return rows

def normalize_pci(value:Any)->str|None:
    s=str(value).strip().lower()
    m=re.fullmatch(r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])",s)
    return m.group(1) if m else None

def telemetry_pci(p:pathlib.Path,expected_uuid:str,g:Gate,label:str)->str|None:
    try:
        raw_regular(str(p),label)
        lines=[x.strip() for x in p.read_text(errors="replace").splitlines() if x.strip()]
        g.need(len(lines)==1,f"{label}: expected exactly one telemetry row")
        fields=[x.strip() for x in lines[0].split(",")] if lines else []
        g.need(len(fields)>=5 and fields[1]=="0" and fields[2]==expected_uuid,f"{label}: telemetry identity fields")
        norm=normalize_pci(fields[4] if len(fields)>=5 else "")
        g.need(norm is not None,f"{label}: malformed PCI bus id")
        return norm
    except Exception as exc:
        g.need(False,f"{label}: {exc}")
        return None

def check_logs(out:pathlib.Path,g:Gate,label:str)->None:
    bad=re.compile(r"(C1Microbench error|cuda[^\n]{0,80}(?:error|invalid)|workspace safety violation|\bFATAL\b|\bASSERT\b)",re.I)
    stdout=out/"stdout.log"; stderr=out/"stderr.log"
    for p in (stdout,stderr):
        try: raw_regular(str(p),f"{label} {p.name}")
        except Exception as e: g.need(False,str(e))
    if stderr.is_file() and not stderr.is_symlink(): g.need(stderr.stat().st_size==0,f"{label}: stderr.log must be empty")
    if stdout.is_file() and not stdout.is_symlink() and bad.search(stdout.read_text(errors="replace")):
        g.need(False,f"{label}: error signature in stdout.log")

def check_telemetry(root:pathlib.Path,manifest:dict[str,Any],g:Gate)->str|None:
    d=root/"gpu_snapshots"; gpu=manifest.get("gpu") or {}; expected=str(gpu.get("uuid",""))
    try:
        raw_dir(str(d),"GPU snapshot directory")
    except Exception as exc: g.need(False,str(exc))
    snap=pathlib.Path(str(gpu.get("telemetry_snapshot","")))
    g.need(gpu.get("physical_index")==0 and expected.startswith("GPU-") and under(snap,d),"manifest GPU provenance/snapshot boundary")
    outer_pci=telemetry_pci(snap,expected,g,"outer prelaunch telemetry")
    telemetry=[snap,d/"outer_postrun_telemetry_gpu0.csv"]
    idle_logs=[]
    for rep in range(1,6):
        for v in VARIANTS:
            safe=f"rep{rep}_{v}"
            pre=list(d.glob(f"pre_{safe}_attempt_*_telemetry_gpu0.csv")); post=d/f"post_{safe}_telemetry_gpu0.csv"
            g.need(bool(pre),f"missing prelaunch telemetry rep{rep}/{v}")
            telemetry.extend(pre+[post]); idle_logs.append(d/f"pre_{safe}_strict_idle_poll.log")
    for p in telemetry:
        try:
            raw_regular(str(p),"telemetry")
            prefix=str(p).removesuffix("_telemetry_gpu0.csv")
            safety=pathlib.Path(prefix+"_safety_gpu0.csv"); apps0=pathlib.Path(prefix+"_compute_gpu0.csv"); appsall=pathlib.Path(prefix+"_compute_all_visible.csv")
            for q in (safety,apps0,appsall): raw_regular(str(q),"snapshot component")
            g.need(expected in p.read_text(errors="replace") and expected in safety.read_text(errors="replace"),f"telemetry UUID mismatch {p.name}")
            observed_pci=telemetry_pci(p,expected,g,"telemetry PCI "+p.name)
            g.need(observed_pci is not None and outer_pci is not None and observed_pci==outer_pci,f"telemetry PCI discontinuity {p.name}")
        except Exception as exc: g.need(False,f"telemetry missing/unreadable {p}: {exc}")
    for p in idle_logs:
        try:
            raw_regular(str(p),"strict idle log")
            g.need("strict_idle=PASS" in p.read_text(errors="replace"),f"no strict-idle PASS in {p.name}")
        except Exception as exc: g.need(False,f"idle log missing/unreadable {p}: {exc}")
    return outer_pci

def check_one(root:pathlib.Path,rep:int,v:str,qids:list[int],inspector:pathlib.Path,expected_uuid:str,expected_pci:str|None,g:Gate)->dict[str,Any]:
    out=root/f"rep{rep}"/v; label=f"rep{rep}/{v}"
    try: raw_dir(str(out),label+" output")
    except Exception as exc: g.need(False,str(exc))
    for name in ("run_card.json","completion.json","allocation_by_phase.json","tree_fingerprints.jsonl"):
        try: raw_regular(str(out/name),f"{label} {name}")
        except Exception as e: g.need(False,str(e))
    for phase,_ in PHASES:
        for suffix in ("ops.jsonl","canonical.bin"):
            try: raw_regular(str(out/f"phase_{phase}_{suffix}"),f"{label} phase {phase} {suffix}")
            except Exception as e: g.need(False,str(e))
    card=g.attempt(label+" run card",lambda:read_json(out/"run_card.json",label+" run card")) or {}
    done=g.attempt(label+" completion",lambda:read_json(out/"completion.json",label+" completion")) or {}
    expected=GATES[v]
    g.need(card.get("schema")=="gtspp-c1-microbench-run-card-v7",f"{label}: run-card schema")
    g.need(done.get("schema")=="gtspp-c1-microbench-completion-v7",f"{label}: completion schema")
    g.need(card.get("run_mode")=="MEASURED_PROTOCOL" and done.get("run_mode")=="MEASURED_PROTOCOL" and card.get("profile_only") is False,f"{label}: profile/smoke entered measured data")
    g.need(card.get("requested_variant")==v==card.get("compiled_variant")==done.get("compiled_variant"),f"{label}: variant contract")
    g.need(card.get("replicate")==rep and card.get("C2_residual_mode")==0,f"{label}: replicate/C2 contract")
    g.need((card.get("C1_PERSISTENT_WORKSPACE"),card.get("C1_ONE_QUERY_FASTPATH"))==expected,f"{label}: compile gates")
    try: g.need(math.isclose(float(card.get("radius")),500.0,rel_tol=0,abs_tol=0) and card.get("trace_limit")==1024,f"{label}: radius/trace")
    except Exception: g.need(False,f"{label}: radius invalid")
    g.need(int(card.get("cuda_runtime_version",0))>0 and int(card.get("cuda_driver_version",0))>0,f"{label}: CUDA provenance")
    g.need(str(card.get("visible_cuda_device_name","")).startswith("NVIDIA RTX PRO 6000"),f"{label}: CUDA device")
    card_uuid=card.get("visible_cuda_device_uuid")
    g.need(isinstance(card_uuid,str) and card_uuid==expected_uuid,f"{label}: CUDA UUID must exactly match manifest/session UUID")
    card_pci=normalize_pci(card.get("visible_cuda_device_pci_bus_id",""))
    g.need(card_pci is not None and expected_pci is not None and card_pci==expected_pci,f"{label}: CUDA PCI must match outer prelaunch telemetry")
    g.need(done.get("tree_invariance_pass") is True,f"{label}: tree-invariance flag")
    fps=[]
    for key in ("initial_tree_fingerprint","cold_tree_fingerprint","provision_tree_fingerprint","warmup_tree_fingerprint","steady_tree_fingerprint"):
        x=done.get(key); fps.append(x); g.need(isinstance(x,str) and re.fullmatch(r"[0-9a-f]{16}",x or "") is not None,f"{label}: invalid {key}")
    g.need(len(set(fps))==1,f"{label}: completion tree fingerprints differ")
    tree=parse_jsonl(out/"tree_fingerprints.jsonl",g,label+" tree JSONL")
    g.need([x.get("label") for x in tree]==list(TREE_LABELS),f"{label}: tree labels")
    tree_values=[x.get("fnv1a64") for x in tree]
    g.need(len(tree_values)==5 and len(set(tree_values))==1 and tree_values[0]==done.get("initial_tree_fingerprint"),f"{label}: tree JSONL fingerprint gate")
    alloc=g.attempt(label+" allocation",lambda:read_json(out/"allocation_by_phase.json",label+" allocation")) or {}
    phases=alloc.get("phases",{})
    for phase_name in ("setup","cold","provision","warmup","steady_state","teardown"):
        x=phases.get(phase_name,{})
        g.need(isinstance(x,dict),f"{label}: allocation phase {phase_name} missing")
        for k in ("cudaMalloc_calls","cudaMallocManaged_calls","cudaFree_calls","cudaMalloc_requested_bytes","cudaMallocManaged_requested_bytes","failed_calls","api_host_ms"):
            try: g.need(float(x.get(k,-1))>=0 and math.isfinite(float(x.get(k,-1))),f"{label}: bad allocation {phase_name}.{k}")
            except Exception: g.need(False,f"{label}: nonnumeric allocation {phase_name}.{k}")
        g.need(x.get("failed_calls")==0,f"{label}: allocator failure {phase_name}")
    steady=phases.get("steady_state",{})
    if expected[0]==1:
        g.need(sum(int(steady.get(k,-1)) for k in ("cudaMalloc_calls","cudaMallocManaged_calls","cudaFree_calls"))==0,f"{label}: persistent steady allocation/free nonzero")
    else:
        g.need(int(steady.get("cudaMalloc_calls",0))+int(steady.get("cudaMallocManaged_calls",0))>0 and int(steady.get("cudaFree_calls",0))>0,f"{label}: ephemeral steady allocation/free absent")
    artifact_sha={}; initial=done.get("initial_tree_fingerprint")
    for phase,count in PHASES:
        records=parse_jsonl(out/f"phase_{phase}_ops.jsonl",g,f"{label} {phase} JSONL")
        g.need(len(records)==count,f"{label}: {phase} record count {len(records)} != {count}")
        for i,x in enumerate(records):
            g.need(x.get("phase")==phase and x.get("ordinal")==i and x.get("qid")==(qids[i] if i<len(qids) else None),f"{label}: {phase} record identity {i}")
            try:
                g.need(int(x.get("buffered_result_count",-1))==0,f"{label}: buffered result count {phase}/{i}")
                for violation in capacity_snapshot_violations(x):
                    g.need(False,f"{label}: {violation} {phase}/{i}")
                g.need(math.isfinite(float(x.get("end_to_end_wall_ms"))) and float(x.get("end_to_end_wall_ms"))>=0 and math.isfinite(float(x.get("gpu_event_ms"))) and float(x.get("gpu_event_ms"))>=0,f"{label}: timing value {phase}/{i}")
            except Exception: g.need(False,f"{label}: malformed numeric record {phase}/{i}")
            g.need(x.get("cuda_status")=="success" and x.get("canonical_hash_algorithm")=="fnv1a64_sorted_id_distance_bits" and re.fullmatch(r"[0-9a-f]{16}",str(x.get("canonical_hash",""))) is not None,f"{label}: CUDA/hash record {phase}/{i}")
            alloc_delta=x.get("allocation_delta",{})
            g.need(isinstance(alloc_delta,dict) and alloc_delta.get("failed_calls")==0,f"{label}: per-operation allocator failure {phase}/{i}")
        artifact=out/f"phase_{phase}_canonical.bin"
        try:
            artifact=raw_regular(str(artifact),f"{label} {phase} canonical")
            artifact_sha[phase]=sha(artifact)
            seen=inspect_canonical(inspector,artifact,g,f"{label} {phase}")
            g.need(len(seen)==len(records),f"{label}: canonical record count {phase}")
            for i,row in enumerate(seen):
                if i>=len(records): break
                ordinal,qid,count0,h=row; x=records[i]
                g.need((ordinal,qid,count0,h)==(x.get("ordinal"),x.get("qid"),x.get("result_count"),x.get("canonical_hash")),f"{label}: canonical content mismatch {phase}/{i}")
        except Exception as exc: g.need(False,f"{label}: canonical {phase}: {exc}")
    check_logs(out,g,label)
    return {"initial_tree_fingerprint":initial,"canonical_sha256":artifact_sha}

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=pathlib.Path,required=True)
    ap.add_argument("--pins",type=pathlib.Path,required=True)
    a=ap.parse_args()
    g=Gate()
    try:
        root=raw_dir(str(a.run_root),"run root")
        if root.parent!=RUNS_ROOT or not root.name.startswith("c1_v7_measured_"):
            raise ValueError("run root must be a direct c1_v7_measured_* child of the v7 runs root")
        raw_regular(str(root/"run_manifest_v7.json"),"run manifest")
        pins_path=raw_regular(str(a.pins),"pins")
        if pins_path!=PINS_PATH: raise ValueError("pins path must be canonical v7 hardened pin file")
        pins=read_json(pins_path,"pins")
    except Exception as exc: raise SystemExit(f"setup failed: {exc}")
    g.need(pins.get("schema")=="gtspp-c1-v7-pins-v1","pins schema")
    pin_sha=sha(pins_path)
    inspector_entry=pins.get("canonical_inspector",{})
    inspector=g.attempt("canonical inspector",lambda:raw_regular(str(inspector_entry["binary"]),"canonical inspector"))
    if inspector is not None:
        g.need(sha(inspector)==inspector_entry.get("binary_sha256"),"canonical inspector binary hash mismatch")
        source=g.attempt("canonical inspector source",lambda:raw_regular(str(inspector_entry["source"]),"canonical inspector source"))
        if source is not None: g.need(sha(source)==inspector_entry.get("source_sha256"),"canonical inspector source hash mismatch")
    qids=provenance_qids(pins,g)
    manifest=g.attempt("run manifest",lambda:read_json(root/"run_manifest_v7.json","run manifest")) or {}
    guard=g.attempt("guard card",lambda:read_json(root/"guard_run_card_v7.json","guard card")) or {}
    g.need(manifest.get("schema")=="gtspp-c1-v7-run-manifest-v2" and manifest.get("execution_mode")=="MEASURED","manifest measured schema/mode")
    m_uuid=str(manifest.get("gpu",{}).get("uuid",""))
    g.need(manifest.get("pins",{}).get("sha256")==pin_sha and manifest.get("gpu",{}).get("physical_index")==0 and m_uuid.startswith("GPU-"),"manifest pin/GPU provenance")
    g.need(guard.get("schema")=="gtspp-c1-v7-guard-card-v2" and guard.get("physical_gpu_index")==0 and guard.get("physical_gpu_uuid")==m_uuid,"guard/manifest GPU provenance")
    g.need(guard.get("status") not in {"ENGINE_FAILED","CONTRACT_FAILED","SEMANTIC_VERIFICATION_FAILED","ANALYSIS_FAILED","TELEMETRY_INCOMPLETE","INTERRUPTED","FINAL_EVIDENCE_FAILED","POSTRUN_PIN_VERIFICATION_FAILED","MANIFEST_INCOMPLETE","GUARD_CARD_INCOMPLETE","SESSION_CLEANUP_INCOMPLETE"},"guard failure status")
    outer_pci=check_telemetry(root,manifest,g)
    reps=[root/f"rep{i}" for i in range(1,6)]
    rep_names={x.name for x in root.iterdir() if x.is_dir() and x.name.startswith("rep")}
    g.need(rep_names=={f"rep{i}" for i in range(1,6)} and all(x.is_dir() and not x.is_symlink() for x in reps),"replicate directories are not exactly rep1..rep5")
    g.need(not any("profile" in x.name.lower() for x in root.iterdir()),"profile artifact contaminates measured run root")
    report=[]
    for rep in range(1,6):
        actual={x.name for x in (root/f"rep{rep}").iterdir() if x.is_dir()} if (root/f"rep{rep}").is_dir() else set()
        g.need(actual==set(VARIANTS),f"rep{rep}: variants are not exact 4-way set")
        one={v:check_one(root,rep,v,qids,inspector,m_uuid,outer_pci,g) if inspector else {} for v in VARIANTS}
        vals=[one[v].get("initial_tree_fingerprint") for v in VARIANTS]
        g.need(len(set(vals))==1 and vals[0] is not None,f"rep{rep}: cross-variant initial tree mismatch")
        for phase,_ in PHASES:
            hs=[one[v].get("canonical_sha256",{}).get(phase) for v in VARIANTS]
            g.need(len(set(hs))==1 and hs[0] is not None,f"rep{rep}: cross-variant exact canonical mismatch {phase}")
        report.append({"replicate":rep,"initial_tree_fingerprints":{v:one[v].get("initial_tree_fingerprint") for v in VARIANTS},"canonical_sha256_by_phase":{phase:{v:one[v].get("canonical_sha256",{}).get(phase) for v in VARIANTS} for phase,_ in PHASES}})
    result={
        "schema":"gtspp-c1-v7-semantic-verification-v1","verified_utc":utc(),"run_root":str(root),"pins_sha256":pin_sha,
        "pass":not g.failures,"failure_count":len(g.failures),"failures":g.failures,"replicates":report,
        "correctness_scope":"Differential equivalence only: every canonical artifact is checked, and each non-reference variant must exactly equal E_G under the same initial tree. No independent absolute range-search oracle is implied.",
        "outer_prelaunch_pci_bus_id_normalized":outer_pci,
        "capacity_snapshot_contract":{"stage":CAPACITY_SNAPSHOT_STAGE,"per_operation_gate":"both capacity fields are integer > 0, equal, and captured before ephemeral release"},
    }
    out=root/"semantic_verification_v7.json"
    if out.is_symlink(): raise SystemExit("refuse symlink semantic output")
    tmp=out.with_name(out.name+".tmp"); tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); tmp.replace(out)
    print(json.dumps({"pass":result["pass"],"failures":len(g.failures),"output":str(out)}))
    return 0 if result["pass"] else 1
if __name__=="__main__": raise SystemExit(main())

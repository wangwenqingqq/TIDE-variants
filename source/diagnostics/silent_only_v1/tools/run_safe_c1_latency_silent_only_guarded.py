#!/usr/bin/python3.12
"""Root-private GPU1 guard for the non-publication silent-only diagnostic.

Normal mode is deliberately narrow: it admits one direct CUDA child only after
three UUID-bound idle NVML snapshots, runs 1 warmup plus 1 measured pass, takes
a postrun snapshot, then performs CPU-only exact-oracle validation and a fixed
v1b counts-only comparison. --self-check never invokes NVML or a CUDA binary.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT=Path("/workspace/experiments/tide_safe_c1_20260727/fair_dynamic_knn_safe_c1_v1")
RUNS=ROOT/"runs"
BUNDLE=ROOT/"inputs/e1_frozen_base_knn_projection_v1"
DIAG=ROOT/"diagnostics/silent_only_v1"
RUNNER=DIAG/"runner/fair_safe_c1_latency_silent_only_e1_runner.cu"
HEADER=DIAG/"src/g3_safe_search_v2_silent_only.cuh"
MATRIX=DIAG/"src/g3_safe_c1_native_matrix_silent_only.cu"
COMPILE_HELPER=DIAG/"tools/compile_safe_c1_latency_silent_only.sh"
VALIDATOR=DIAG/"tools/validate_safe_c1_silent_only_output.py"
COMPARATOR=DIAG/"tools/compare_safe_c1_silent_only_control.py"
BINARY=DIAG/"bin/fair_safe_c1_latency_silent_only_e1_runner.sm_80"
OBJECT=DIAG/"build/fair_safe_c1_latency_silent_only_e1_runner.sm_80.o"
BUILD_RECEIPT=DIAG/"build/silent_only_static_build_receipt.env"
MANIFEST=DIAG/"provenance/silent_only_source_manifest.json"
REVIEWED_DIFF=DIAG/"provenance/silent_only_reviewed_diff.patch"
STATIC_AUDIT=DIAG/"provenance/silent_only_static_audit.json"
PREFLIGHT=ROOT/"tools/preflight_frozen_base_knn_bundle.py"
NVML=ROOT/"tools/.nvml_idle_snapshot.py"
SYSTEM_PYTHON=Path("/usr/bin/python3.12")
CONTROL=RUNS/"safe-c1-tradeoff-pilot-v1b-20260730"

GPU_ORDINAL=1
GPU_UUID="GPU-CONFIGURE-ARCHIVE-DEVICE"
MAX_TTL=120
SAMPLES=3
SAMPLE_INTERVAL=1
WARMUP=1
MEASURED=1
SCHEMA="fair-safe-c1-latency-silent-only-guard-v1"
VARIANT="optimization_diagnostic_silent_only"
SCOPE=("silent-only source-level optimization diagnostic: only the two informational stdout statements "
       "in the invoked vector-topk traversal are suppressed; native query_knn candidate API versus "
       "current exact query_range(UINT64_MAX) full-result immutable-base fallback; not a same-API "
       "speedup, not a current-implementation result, and not publication evidence")
SNAPSHOT_SCHEMA="safe-c1-g3-nvml-snapshot-v2"
SNAPSHOT_KEYS=("schema","gpu_ordinal","gpu_uuid","gpu_pci_bus_id","compute_process_count",
               "graphics_process_count","python_realpath","python_sha256","nvml_library_realpath",
               "nvml_library_sha256","nvml_driver_version")
EXPECTED_STDOUT="Index construction...\nTree height: 4\n"
NAME_RE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
APPROVAL_RE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
NONCE_RE=re.compile(r"^[0-9a-f]{64}$")
SHA_RE=re.compile(r"^[0-9a-f]{64}$")
UUID_RE=re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
PCI_RE=re.compile(r"^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
INPUT_HASHES={
 "manifest.json":"68dbf15788793a828c8a9503d11304da576acbdca2f609dd0f8799ae39cd9a4c",
 "metadata.json":"2f515a5cf3bef61d6084f4c0d90075ee16cb4e365971b75102a24bdbbc588798",
 "trace.e1gtrc":"9402c609710fc9076f46653dd5bf30527013e158c63b4576dd665c27f9973ae5",
 "pool.i16":"899adaa59b265ee788841f1a48b667b7da39df166972ba9568c4e94727b31170",
 "queries.i16":"18e0ebbe8ddcdcf6e2312e1e48310111a4d96c1fcb752622d1e9ec9508c21c1e",
 "stable_id_to_pool_row.i32":"93710cce11c994b6b1934713842c93cfcec76a3563fc47574abf419137f4c5c8",
 "initial_base_stable_ids.i32":"6b0751ba5e64fc9c13ddfb44778fa7d6a1f7d7aa9d6a5e38a1f0a1502c3fb9e3",
}
CONTROL_HASHES={
 "engine.jsonl":"4a32720f49ac49bed6bd6f2755532231ea8a04e5d3c797e342e56ef22cb35f11",
 "summary.json":"5d9f197cf7297cb970a69f8a5fdb7255efad9d06c37adac343377af5a185c103",
 "guard_receipt.json":"1400941c7d898c2f8daa4884c85c8fa8e1a027f1ffdcc2f88af4a574070b63f8",
 "independent_validation.json":"2719fcffa018e352d4c7b8da480c889545a57d5a39605bff424511fb8acb042e",
 "admission.env":"a979ba830a2f39bf12ddace378b1be7106835233cd2e0f8e1bd8917175c75721",
}
EXPECTED={
 "runner_source":"43f4509eb943444b1bb760d027325657c6efe659ae640658772734f3dfc0b592",
 "header_source":"9a97a2e76a21b38755775816679c8a47cd9c56406090dcfaa54f5b5fe3fe3f5f",
 "matrix_source":"b0b4d5a9a38681170130028aa67bd581b6f7ad0e6b19ebbd4f43a23389c0f6dc",
 "compile_helper":"47c47d9a6407ce4bb878779b5cdd5e0252c57fa40955dbd335a8935ce48bfc9b",
 "validator":"87ea08316d3824019082712c8862fc26bbbc2217dc23fcabe6e9f1e47a28f68c",
 "comparator":"d4c811cba3b8801fcdcaa0fc52125b894ef291fb0c19d68f358d11485451750d",
 "object":"fa5cc8585dd2b5bba295d53b01145c872e4ab81d40ba563d610ed9cb756e1404",
 "binary":"61462eed3edecf5b46bde550a7fb8887cdbb7b9bc159332919df7aff8af3bb57",
 "manifest":"075f8214b3f1a59442b91692185fda5da538283f6661273e96b60d5832374ac3",
 "reviewed_diff":"0400611429fd90b7a1f6afc62eaeddf54f524eba4e6ab18a6337f8b8f836b423",
 "static_audit":"415b41b58477f44a61f0e3f4cc6ab50083cebf7ffa15bdc990243c85fe0f813c",
 "build_receipt":"dcc2a55de1e893d2552f39e4607d25cdc707a946874be8338a1e10704e0ce913",
 "preflight":"62258f1461055744bfc7f74bb8825acb135291873f0289845038eb3ff0037b7d",
 "nvml":"7d039a4317bf0f0ebed6d7fd1cb363d536b541ea1444caf3d73ee8098c755f34",
}
EXPECTED_CLOSURE="45d4429d066999ea1a6556319d6db90cac8362e76934281605552f5188e2ab26"
EXPECTED_ADMISSION_SHA="a979ba830a2f39bf12ddace378b1be7106835233cd2e0f8e1bd8917175c75721"

class GateError(RuntimeError): pass
def require(x:bool,msg:str)->None:
    if not x:raise GateError(msg)
def boot_ns()->int:return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def private_dir(p:Path,label:str)->None:
    s=p.lstat();require(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode),"unsafe "+label+" directory")
    require(s.st_uid==0 and s.st_gid==0 and stat.S_IMODE(s.st_mode)==0o700,label+" not root-private 0700")
def private_file(p:Path,label:str,exe:bool=False)->None:
    s=p.lstat();require(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode),"unsafe "+label)
    require(s.st_uid==0 and s.st_gid==0 and stat.S_IMODE(s.st_mode)&0o077==0,label+" not root-private")
    if exe:require(stat.S_IMODE(s.st_mode)&0o100,label+" not owner executable")
def trusted_file(p:Path,label:str,exe:bool=False)->None:
    s=p.lstat();require(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode),"unsafe trusted "+label)
    require(s.st_uid==0 and s.st_gid==0 and stat.S_IMODE(s.st_mode)&0o022==0,label+" writable by non-root")
    if exe:require(stat.S_IMODE(s.st_mode)&0o100,label+" not executable")
def tree_private(p:Path)->None:
    private_dir(p,"diagnostic root")
    for q in p.rglob("*"):
        s=q.lstat();require(not stat.S_ISLNK(s.st_mode),"diagnostic tree symlink: "+str(q))
        require(s.st_uid==0 and s.st_gid==0 and stat.S_IMODE(s.st_mode)&0o077==0,"nonprivate diagnostic entry: "+str(q))
        require(stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode),"unexpected diagnostic entry: "+str(q))
def atomic_json(p:Path,x:dict[str,Any])->None:
    private_dir(p.parent,"receipt parent");tmp=p.with_name("."+p.name+".tmp")
    require(not tmp.exists() and not tmp.is_symlink(),"stale receipt temporary")
    with tmp.open("x",encoding="utf-8") as f:f.write(json.dumps(x,sort_keys=True,indent=2)+"\n");f.flush();os.fsync(f.fileno())
    os.chmod(tmp,0o600);os.replace(tmp,p)
def write_new(p:Path,text:str,encoding:str="utf-8")->None:
    private_dir(p.parent,"log parent");require(not p.exists() and not p.is_symlink(),"refuse overwrite "+str(p))
    with p.open("x",encoding=encoding) as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.chmod(p,0o600)
def json_private(p:Path,label:str)->dict[str,Any]:
    private_file(p,label)
    try:x=json.loads(p.read_text("utf-8"))
    except Exception as e:raise GateError("bad "+label+" JSON") from e
    require(isinstance(x,dict),label+" object");return x
def parse_env(p:Path)->dict[str,str]:
    private_file(p,"admission");d={}
    for line in p.read_text("ascii").splitlines():
        if not line or line.startswith("#"):continue
        require(line.count("=")==1 and line.index("=")>0,"admission syntax")
        k,v=line.split("=",1);require(k not in d and v and not any(c.isspace() for c in v),"admission field");d[k]=v
    return d

def diff_ops(a:list[str],b:list[str])->list[tuple[Any,...]]:
    return [(t,i1,i2,j1,j2,a[i1:i2],b[j1:j2]) for t,i1,i2,j1,j2 in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if t!="equal"]
def between(s:str,start:str,end:str)->str:
    i=s.index(start);return s[i:s.index(end,i)]
def verify_exact_source_policy()->None:
    h0=(ROOT/"src/g3_safe_search_v2.cuh").read_text().splitlines()
    h1=HEADER.read_text().splitlines()
    expected=[("replace",2036,2037,2036,2037,['\tcout << "Searching..." << endl;'],['\tstatic_cast<void>(0);']),("replace",2262,2263,2262,2263,['\t\t\t\tprintf("qnum_l_low: %lu\\n", static_cast<unsigned long>(qnum_l_low));'],['\t\t\t\tstatic_cast<void>(0);'])]
    require(diff_ops(h0,h1)==expected,"silent header differs beyond exactly two invoked vector-topk stdout statements")
    m0=(ROOT/"src/g3_safe_c1_native_matrix.cu").read_text().splitlines();m1=MATRIX.read_text().splitlines()
    expected_m=[("replace",40,41,40,41,['#include "g3_safe_search_v2.cuh"'],['#include "g3_safe_search_v2_silent_only.cuh"'])]
    require(diff_ops(m0,m1)==expected_m,"silent matrix differs beyond include redirection")
    r0=(ROOT/"runner/fair_safe_c1_latency_tradeoff_e1_runner.cu").read_text();r1=RUNNER.read_text()
    require(between(r0,"Observation measure_one(","void run_phase(")==between(r1,"Observation measure_one(","void run_phase("),"timed measure_one/ABBA body drift")
    require(r1.count('#include "../src/g3_safe_c1_native_matrix_silent_only.cu"')==1,"silent runner include redirect")
    require(r1.count("fair-safe-c1-latency-tradeoff-e1-merged-result-v1")==1,"merged hash domain drift")
    require(r1.count("SAFE_C1_LATENCY_TRADEOFF_SILENT_ONLY_GUARD_NONCE")==1 and "SAFE_C1_LATENCY_TRADEOFF_GUARD_NONCE" not in r1,"runner nonce boundary drift")

def verify_manifest()->dict[str,Any]:
    private_file(MANIFEST,"variant manifest");require(sha(MANIFEST)==EXPECTED["manifest"],"variant manifest hash drift")
    m=json_private(MANIFEST,"variant manifest")
    require(m.get("schema")=="fair-safe-c1-silent-only-diagnostic-source-manifest-v1" and m.get("diagnostic_variant")==VARIANT and m.get("publication_eligible") is False,"variant manifest label")
    rd=m.get("reviewed_diff");require(isinstance(rd,dict) and rd.get("path")==str(REVIEWED_DIFF) and rd.get("sha256")==EXPECTED["reviewed_diff"] and rd.get("audit_result")=="PASS_EXACT_ALLOWED_SOURCE_DIFFERENCES_ONLY","reviewed diff binding")
    closure=m.get("source_closure");require(isinstance(closure,list) and len(closure)==11,"source closure cardinality")
    lines=[]
    for x in closure:
        require(isinstance(x,dict) and isinstance(x.get("path"),str) and SHA_RE.fullmatch(str(x.get("sha256",""))) is not None,"source closure entry")
        q=Path(x["path"]);require(q.is_relative_to(ROOT),"source closure escapes root")
        private_file(q,"source closure member",q.suffix in (".py",".sh"))
        require(sha(q)==x["sha256"],"source closure member hash drift")
        lines.append(str(q).replace(str(ROOT)+"/","")+"\t"+x["sha256"])
    require(hashlib.sha256(("\n".join(sorted(lines))+"\n").encode("ascii")).hexdigest()==EXPECTED_CLOSURE and m.get("full_source_closure_sha256")==EXPECTED_CLOSURE,"source closure digest drift")
    control=m.get("control_artifacts");require(isinstance(control,dict) and control.get("engine_jsonl_sha256")==CONTROL_HASHES["engine.jsonl"] and control.get("summary_json_sha256")==CONTROL_HASHES["summary.json"],"manifest control binding")
    return m

def verify_build_receipt()->None:
    private_file(BUILD_RECEIPT,"static build receipt");require(sha(BUILD_RECEIPT)==EXPECTED["build_receipt"],"build receipt hash drift")
    d={}
    for line in BUILD_RECEIPT.read_text("ascii").splitlines():
        require(line.count("=")==1,"build receipt syntax");k,v=line.split("=",1);require(k not in d and v,"build receipt field");d[k]=v
    require(d.get("schema")=="fair-safe-c1-silent-only-static-build-receipt-v1" and d.get("diagnostic_variant")==VARIANT and d.get("publication_eligible")=="false" and d.get("gpu_workload_launched")=="false" and d.get("arch")=="sm_80" and d.get("compile_mode")=="all","build receipt label")
    pairs={"runner_source_sha256":"runner_source","header_source_sha256":"header_source","matrix_source_sha256":"matrix_source","compile_helper_sha256":"compile_helper","object_sha256":"object","binary_sha256":"binary","source_manifest_sha256":"manifest"}
    for k,n in pairs.items():require(d.get(k)==EXPECTED[n],"build receipt binding "+k)

def verify_control()->None:
    private_dir(CONTROL,"fixed v1b control")
    for n,h in CONTROL_HASHES.items():private_file(CONTROL/n,"fixed control "+n);require(sha(CONTROL/n)==h,"fixed control hash "+n)
    cv=json_private(CONTROL/"independent_validation.json","fixed control validation")
    require(cv.get("status")=="PASS","fixed control validation status")

def verify_static()->dict[str,str]:
    for q,label in ((ROOT,"experiment root"),(RUNS,"runs"),(DIAG,"diagnostic root"),(DIAG/"src","diagnostic source"),(DIAG/"runner","diagnostic runner"),(DIAG/"tools","diagnostic tools"),(DIAG/"provenance","diagnostic provenance"),(DIAG/"build","diagnostic build"),(DIAG/"bin","diagnostic bin"),(BUNDLE,"sealed bundle")):private_dir(q,label)
    tree_private(DIAG)
    paths={"runner_source":(RUNNER,False),"header_source":(HEADER,False),"matrix_source":(MATRIX,False),"compile_helper":(COMPILE_HELPER,True),"validator":(VALIDATOR,True),"comparator":(COMPARATOR,True),"object":(OBJECT,False),"binary":(BINARY,True),"manifest":(MANIFEST,False),"reviewed_diff":(REVIEWED_DIFF,False),"static_audit":(STATIC_AUDIT,False),"build_receipt":(BUILD_RECEIPT,False),"preflight":(PREFLIGHT,True),"nvml":(NVML,False)}
    out={}
    for n,(q,exe) in paths.items():
        private_file(q,n,exe) if q.is_relative_to(DIAG) else trusted_file(q,n,exe)
        out[n+"_sha256"]=sha(q);require(out[n+"_sha256"]==EXPECTED[n],"artifact hash drift "+n)
    trusted_file(SYSTEM_PYTHON,"system python",True)
    for n,h in INPUT_HASHES.items():trusted_file(BUNDLE/n,"sealed input");require(sha(BUNDLE/n)==h,"sealed input drift "+n)
    audit=json_private(STATIC_AUDIT,"static audit")
    require(audit.get("status")=="PASS_STATIC_SOURCE_AND_BUILD_AUDIT" and audit.get("source_manifest_sha256")==EXPECTED["manifest"] and audit.get("static_build",{}).get("object_sha256")==EXPECTED["object"] and audit.get("static_build",{}).get("binary_sha256")==EXPECTED["binary"],"static audit binding")
    verify_exact_source_policy();verify_manifest();verify_build_receipt();verify_control()
    out["guard_sha256"]=sha(Path(__file__).resolve());return out

def run_preflight(admission:Path,logs:Path)->str:
    result=subprocess.run([str(SYSTEM_PYTHON),"-I","-S",str(PREFLIGHT),"--bundle",str(BUNDLE),"--out",str(admission)],cwd=str(ROOT),env={"PATH":"/usr/bin:/bin","HOME":"/nonexistent","CUDA_VISIBLE_DEVICES":"","NVIDIA_VISIBLE_DEVICES":"void"},text=True,capture_output=True,check=False)
    write_new(logs/"preflight.stdout.log",result.stdout);write_new(logs/"preflight.stderr.log",result.stderr)
    require(result.returncode==0,"CPU preflight failed: "+(result.stderr.strip() or result.stdout.strip()))
    values=parse_env(admission);require(values.get("schema")=="e1-frozen-base-knn-projection-admission-v2" and values.get("status")=="PASS" and values.get("bundle_realpath")==str(BUNDLE) and values.get("ops")=="insert,knn" and values.get("excluded_ops")=="delete,range" and values.get("base_immutable")=="true" and values.get("direct_sidecar_allowed")=="false","admission semantics")
    require(sha(admission)==EXPECTED_ADMISSION_SHA,"admission hash drift")
    return sha(admission)

def read_snapshot()->tuple[dict[str,str],str]:
    result=subprocess.run([str(SYSTEM_PYTHON),"-I","-S",str(NVML),"--gpu-ordinal",str(GPU_ORDINAL)],cwd=str(ROOT),env={"PATH":"/usr/bin:/bin","HOME":"/nonexistent"},text=True,capture_output=True,check=False)
    require(result.returncode==0,"NVML snapshot refused GPU1: "+(result.stderr.strip() or result.stdout.strip()))
    raw=result.stdout;rows=raw.splitlines();require(len(rows)==len(SNAPSHOT_KEYS),"NVML snapshot line count")
    d={}
    for want,line in zip(SNAPSHOT_KEYS,rows):
        require(line.count("=")==1,"NVML line syntax");k,v=line.split("=",1);require(k==want and v and not any(x.isspace() for x in v),"NVML key/order");d[k]=v
    require(d["schema"]==SNAPSHOT_SCHEMA and d["gpu_ordinal"]==str(GPU_ORDINAL) and d["gpu_uuid"]==GPU_UUID and UUID_RE.fullmatch(d["gpu_uuid"]) is not None,"NVML GPU binding")
    require(PCI_RE.fullmatch(d["gpu_pci_bus_id"]) is not None and d["compute_process_count"]=="0" and d["graphics_process_count"]=="0","NVML idle status")
    require(SHA_RE.fullmatch(d["python_sha256"]) is not None and SHA_RE.fullmatch(d["nvml_library_sha256"]) is not None,"NVML hash fields")
    return d,raw
def snapshot_entry(i:int,d:dict[str,str],raw:str)->dict[str,Any]:
    return {"sample":i,"boottime_ns":boot_ns(),"fields":d,"raw_sha256":hashlib.sha256(raw.encode("ascii")).hexdigest()}

def validate_args(a:argparse.Namespace)->None:
    require(a.gpu_ordinal==GPU_ORDINAL,"guard pinned to GPU=1")
    require(isinstance(a.run_name,str) and NAME_RE.fullmatch(a.run_name) is not None,"invalid run name")
    require(isinstance(a.approval_id,str) and APPROVAL_RE.fullmatch(a.approval_id) is not None,"invalid approval id")
    require(isinstance(a.nonce,str) and NONCE_RE.fullmatch(a.nonce) is not None,"nonce must be 64 lowercase hex")
    require(isinstance(a.ttl_seconds,int) and 1<=a.ttl_seconds<=MAX_TTL,"TTL must be 1..120")
def runner_command(run:Path,admission:Path,nonce:str)->list[str]:
    return [str(BINARY),"--bundle",str(BUNDLE),"--preflight",str(admission),"--out",str(run/"engine.jsonl"),"--summary",str(run/"summary.json"),"--mode","timing","--warmup-passes",str(WARMUP),"--measured-passes",str(MEASURED),"--timing-guard-nonce",nonce]
def run_binary(run:Path,admission:Path,nonce:str,expiry:int)->dict[str,Any]:
    remain=expiry-boot_ns();require(remain>0,"approval TTL expired before CUDA child")
    timeout=min(MAX_TTL,remain/1_000_000_000);require(timeout>0,"nonpositive child timeout")
    out=run/"logs/runner.stdout.log";err=run/"logs/runner.stderr.log";started=boot_ns();timed=False
    with out.open("x",encoding="utf-8") as so,err.open("x",encoding="utf-8") as se:
        try:r=subprocess.run(runner_command(run,admission,nonce),cwd=str(ROOT),env={"PATH":"/usr/bin:/bin","HOME":"/nonexistent","CUDA_VISIBLE_DEVICES":GPU_UUID,"NVIDIA_VISIBLE_DEVICES":GPU_UUID,"CUDA_DEVICE_ORDER":"PCI_BUS_ID","SAFE_C1_LATENCY_TRADEOFF_SILENT_ONLY_GUARD_NONCE":nonce},stdout=so,stderr=se,timeout=timeout,check=False);code=r.returncode
        except subprocess.TimeoutExpired:code=124;timed=True
    os.chmod(out,0o600);os.chmod(err,0o600);finished=boot_ns()
    return {"exit_code":code,"timed_out":timed,"timeout_seconds":timeout,"started_boottime_ns":started,"finished_boottime_ns":finished,"finished_within_ttl":finished<=expiry,"cuda_visible_devices":GPU_UUID,"nvidia_visible_devices":GPU_UUID,"warmup_passes":WARMUP,"measured_passes":MEASURED,"argv_redacted":[*runner_command(run,admission,"<redacted>")]}
def verify_runner_outputs(run:Path,runner:dict[str,Any])->None:
    events=run/"engine.jsonl";summary=run/"summary.json";stdout=run/"logs/runner.stdout.log"
    private_file(events,"runner events");private_file(summary,"runner summary");private_file(stdout,"runner stdout")
    require(events.stat().st_size>0 and summary.stat().st_size>0,"empty timing output")
    raw=stdout.read_text("utf-8")
    require("Searching..." not in raw and "qnum_l_low:" not in raw,"legacy traversal stdout was not fully suppressed")
    require(raw==EXPECTED_STDOUT,"silent stdout preamble/output contract drift")
    runner["events_sha256"]=sha(events);runner["summary_sha256"]=sha(summary);runner["stdout_sha256"]=sha(stdout)
    runner["stdout_contract"]="PASS_exact_two_line_preamble_zero_legacy_tokens"
def cpu_env()->dict[str,str]:
    return {"PATH":"/usr/bin:/bin","HOME":"/nonexistent","CUDA_VISIBLE_DEVICES":"","NVIDIA_VISIBLE_DEVICES":"void","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
def run_cpu(command:list[str],stdout:Path,stderr:Path)->dict[str,Any]:
    with stdout.open("x",encoding="utf-8") as so,stderr.open("x",encoding="utf-8") as se:r=subprocess.run(command,cwd=str(ROOT),env=cpu_env(),stdout=so,stderr=se,check=False)
    os.chmod(stdout,0o600);os.chmod(stderr,0o600);return {"exit_code":r.returncode}
def run_validator(run:Path,admission:Path)->dict[str,Any]:
    ret=run_cpu([str(SYSTEM_PYTHON),"-I","-S",str(VALIDATOR),"--bundle",str(BUNDLE),"--admission",str(admission),"--events",str(run/"engine.jsonl"),"--summary",str(run/"summary.json"),"--manifest",str(MANIFEST),"--out",str(run/"independent_validation.json")],run/"logs/validator.stdout.log",run/"logs/validator.stderr.log")
    o=run/"independent_validation.json"
    if ret["exit_code"]==0:
        private_file(o,"candidate independent validation");v=json_private(o,"candidate independent validation");require(v.get("status")=="PASS_SEMANTIC_OUTPUT_ONLY" and v.get("variant_manifest_sha256")==EXPECTED["manifest"],"candidate validator status/binding");ret["output_sha256"]=sha(o)
    else:ret["output_sha256"]=None
    return ret
def run_comparator(run:Path)->dict[str,Any]:
    ret=run_cpu([str(SYSTEM_PYTHON),"-I","-S",str(COMPARATOR),"--bundle",str(BUNDLE),"--candidate-run-name",run.name,"--variant-manifest",str(MANIFEST),"--out",str(run/"silent_vs_control_validation.json")],run/"logs/comparator.stdout.log",run/"logs/comparator.stderr.log")
    o=run/"silent_vs_control_validation.json"
    if ret["exit_code"]==0:
        private_file(o,"silent comparator output");v=json_private(o,"silent comparator output");require(v.get("status")=="PASS_SEMANTIC_EQUIVALENCE_COUNTS_ONLY" and v.get("diagnostic_variant")==VARIANT and v.get("publication_eligible") is False and v.get("comparison",{}).get("timing")=="NOT_INTERPRETED" and v.get("leaf_identity_equivalence",{}).get("status")=="NOT_WITNESSED","comparator status/boundary");ret["output_sha256"]=sha(o)
    else:ret["output_sha256"]=None
    return ret

def self_check()->int:
    os.umask(0o077);static=verify_static()
    with tempfile.TemporaryDirectory(prefix="silent-only-guard-selfcheck.") as t:
        d=Path(t);os.chmod(d,0o700);logs=d/"logs";logs.mkdir(mode=0o700)
        admission_sha=run_preflight(d/"admission.env",logs)
        v=run_cpu([str(SYSTEM_PYTHON),"-I","-S",str(VALIDATOR),"--self-check"],logs/"validator.stdout.log",logs/"validator.stderr.log")
        c=run_cpu([str(SYSTEM_PYTHON),"-I","-S",str(COMPARATOR),"--self-check"],logs/"comparator.stdout.log",logs/"comparator.stderr.log")
        require(v["exit_code"]==0 and c["exit_code"]==0,"CPU validator/comparator self-check failed")
        vout=(logs/"validator.stdout.log").read_text("utf-8").strip();cout=(logs/"comparator.stdout.log").read_text("utf-8").strip()
        require(json.loads(vout).get("status")=="PASS_CPU_ONLY_ORACLE_SELFCHECK" and json.loads(cout).get("status")=="PASS_CPU_ONLY_STATIC_SELFCHECK","CPU self-check output status")
    print(json.dumps({"schema":SCHEMA,"status":"PASS_CPU_ONLY_STATIC_SELFCHECK","diagnostic_variant":VARIANT,"publication_eligible":False,"gpu_workload_launched":False,"nvml_used":False,"nvidia_smi_used":False,"guard_sha256":static["guard_sha256"],"artifacts":static,"admission_sha256":admission_sha,"validator_cpu_self_check":True,"comparator_cpu_self_check":True,"scope":"static/source/binary/preflight/oracle checks only; no NVML snapshot and no CUDA binary invocation"},sort_keys=True));return 0

def normal(a:argparse.Namespace)->int:
    validate_args(a);os.umask(0o077);static=verify_static()
    run=RUNS/a.run_name;require(not run.exists() and not run.is_symlink(),"run directory already exists")
    run.mkdir(mode=0o700);logs=run/"logs";logs.mkdir(mode=0o700);admission=run/"admission.env";receipt=run/"guard_receipt.json"
    card={"schema":SCHEMA,"status":"ADMISSION_READY","diagnostic_variant":VARIANT,"publication_eligible":False,"scope":SCOPE,"host":os.uname().nodename,"run_name":a.run_name,"approval_id":a.approval_id,"token":{"nonce_sha256":hashlib.sha256(a.nonce.encode("ascii")).hexdigest(),"issued_boottime_ns":0,"expires_boottime_ns":0,"ttl_seconds":a.ttl_seconds},"gpu":{"requested_ordinal":GPU_ORDINAL,"expected_uuid":GPU_UUID,"launch_cuda_visible_devices":GPU_UUID,"prelaunch_idle_samples":[]},"inputs":{"bundle_path":str(BUNDLE),"admission_path":str(admission),"runner_source_path":str(RUNNER),"compile_helper_path":str(COMPILE_HELPER),"object_path":str(OBJECT),"binary_path":str(BINARY),"validator_path":str(VALIDATOR),"comparator_path":str(COMPARATOR),"variant_manifest_path":str(MANIFEST),"reviewed_diff_path":str(REVIEWED_DIFF),"static_audit_path":str(STATIC_AUDIT),**static,"input_files_sha256":INPUT_HASHES,"control_hashes":CONTROL_HASHES,"guard_sha256":static["guard_sha256"]},"outputs":{"events":str(run/"engine.jsonl"),"summary":str(run/"summary.json"),"runner_stdout":str(logs/"runner.stdout.log"),"validator":str(run/"independent_validation.json"),"comparator":str(run/"silent_vs_control_validation.json")},"do_not_touch":["GPU ordinal other than 1","any pre-existing process","v1b control run","/workspace/legacy_workspace/GTS","/workspace/project/GTS","A800-1","A800-2","A800-3"],"limitations":["optimization_diagnostic_silent_only","not_same_api","timing_not_interpreted_as_speedup","leaf_identity_not_witnessed_by_v1b"]}
    try:card["inputs"]["admission_sha256"]=run_preflight(admission,logs)
    except GateError as e:card["status"]="BLOCKED_CPU_PREFLIGHT";card["error"]=str(e);atomic_json(receipt,card);raise
    # After the third snapshot, do no compile/diff/control preparation: launch immediately.
    pci=None;samples=[]
    try:
        for i in range(1,SAMPLES+1):
            fields,raw=read_snapshot()
            if pci is None:pci=fields["gpu_pci_bus_id"]
            else:require(fields["gpu_pci_bus_id"]==pci,"GPU1 PCI identity drift during prelaunch samples")
            write_new(logs/("prelaunch_nvml_"+str(i)+".txt"),raw,"ascii")
            entry=snapshot_entry(i,fields,raw)
            require(not samples or entry["boottime_ns"]>samples[-1]["boottime_ns"],"prelaunch snapshot freshness/order drift")
            samples.append(entry)
            if i<SAMPLES:time.sleep(SAMPLE_INTERVAL)
    except GateError as e:card["status"]="BLOCKED_PRELAUNCH_NVML";card["gpu"]["prelaunch_idle_samples"]=samples;card["error"]=str(e);atomic_json(receipt,card);raise
    issued=boot_ns();require(issued>samples[-1]["boottime_ns"],"TTL issue time is not after final fresh idle sample");expiry=issued+a.ttl_seconds*1_000_000_000;card["token"]["issued_boottime_ns"]=issued;card["token"]["expires_boottime_ns"]=expiry;card["gpu"]["prelaunch_idle_samples"]=samples;card["status"]="RUNNING";atomic_json(receipt,card)
    runner=run_binary(run,admission,a.nonce,expiry);card["runner"]=runner;ready=False
    if runner["exit_code"]==0 and not runner["timed_out"] and runner["finished_within_ttl"]:
        try:verify_runner_outputs(run,runner);ready=True
        except GateError as e:runner["output_error"]=str(e)
    post=False
    try:
        fields,raw=read_snapshot();require(fields["gpu_pci_bus_id"]==pci,"GPU1 PCI identity drift after runner");write_new(logs/"postrun_nvml.txt",raw,"ascii")
        post_entry={"boottime_ns":boot_ns(),"fields":fields,"raw_sha256":hashlib.sha256(raw.encode("ascii")).hexdigest()}
        require(post_entry["boottime_ns"]>=runner["finished_boottime_ns"] and post_entry["boottime_ns"]>samples[-1]["boottime_ns"],"postrun snapshot freshness/order drift")
        card["gpu"]["postrun_idle_snapshot"]=post_entry;post=True
    except GateError as e:card["gpu"]["postrun_snapshot_error"]=str(e)
    validator={"exit_code":None,"output_sha256":None};comparator={"exit_code":None,"output_sha256":None}
    if ready and post:
        card["status"]="RUNNER_EXITED_PENDING_VALIDATION";atomic_json(receipt,card)
        validator=run_validator(run,admission);card["validator"]=validator
        if validator["exit_code"]==0:
            card["status"]="VALIDATOR_PASSED_PENDING_COMPARISON";atomic_json(receipt,card)
            comparator=run_comparator(run);card["comparator"]=comparator
    card["validator"]=validator;card["comparator"]=comparator
    card["status"]="PASS_OPTIMIZATION_DIAGNOSTIC_SILENT_ONLY_GUARDED_COUNTS_ONLY" if ready and post and validator["exit_code"]==0 and comparator["exit_code"]==0 else "FAILED_OR_BLOCKED"
    atomic_json(receipt,card)
    print(json.dumps({"status":card["status"],"run_dir":str(run),"gpu_ordinal":GPU_ORDINAL,"gpu_uuid":GPU_UUID,"runner_exit":runner["exit_code"],"validator_exit":validator["exit_code"],"comparator_exit":comparator["exit_code"],"claim":"optimization_diagnostic_silent_only_counts_only_not_speedup"},sort_keys=True))
    return 0 if card["status"].startswith("PASS_") else 2

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--self-check",action="store_true");p.add_argument("--gpu-ordinal",type=int);p.add_argument("--run-name");p.add_argument("--approval-id");p.add_argument("--nonce");p.add_argument("--ttl-seconds",type=int);a=p.parse_args()
    if a.self_check:
        require(all(x is None for x in (a.gpu_ordinal,a.run_name,a.approval_id,a.nonce,a.ttl_seconds)),"self-check cannot accept launch arguments");return self_check()
    require(all(x is not None for x in (a.gpu_ordinal,a.run_name,a.approval_id,a.nonce,a.ttl_seconds)),"exact GPU/run-name/approval-id/nonce/TTL required");return normal(a)
if __name__=="__main__":
    try:raise SystemExit(main())
    except GateError as e:print("guard blocked: "+str(e),file=sys.stderr);raise SystemExit(2)

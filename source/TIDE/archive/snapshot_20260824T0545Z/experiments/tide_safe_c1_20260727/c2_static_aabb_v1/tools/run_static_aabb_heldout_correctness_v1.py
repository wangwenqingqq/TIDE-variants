#!/usr/bin/env python3
"""
One-shot executor for the pre-registered Safe-C2 static-AABB held-out correctness gate.

The default mode performs the GPU run only after a CPU-only artifact preflight.
--preflight-only is intentionally CPU-only and does not invoke nvidia-smi.
This tool never retries, deletes, overwrites, kills processes, changes clocks,
or accesses the legacy C2 validation/sealed directories.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
RUNS = ROOT / "runs"
BINARY = ROOT / "bin/static_aabb_topk_heldout_v1"
SOURCE = ROOT / "src/static_aabb_topk_heldout_v1.cu"
BOUND = ROOT / "include/aabb_static_bound_v2_diskguard.cuh"
HEADER = ROOT / "include/search_static_aabb_v2_diskguard.cuh"
CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
BASE_PROTOCOL = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
BASE_PREFLIGHT = ROOT / "provenance/static_aabb_heldout_preflight_v1.json"
SELECTION = ROOT / "provenance/static_aabb_heldout_selection_v1.json"
BUILD = ROOT / "provenance/static_aabb_heldout_correctness_build_v2.json"
EXECUTION_CONTRACT = ROOT / "provenance/static_aabb_heldout_correctness_execution_buildbound_v1.json"
EXECUTION_PREFLIGHT = ROOT / "provenance/static_aabb_heldout_correctness_execution_buildbound_preflight_v1.json"
SOURCE_AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_v1.py"
EXPECTED = {
    "binary": "d10ec9a319927fbd35be27ad55d3cd8cc2cc75e4bd0aa257382e11fc0824a3de",
    "source": "ff807e47189337520439c1b7f9996111d5104f693af020f3f1418b07dc2e4c88",
    "bound": "1747b53494ab9f9ed6632a528e9a7da17edba2b2f2846c9192b5c4347eb69626",
    "header": "3ed53b958b2611c6419376c69741f27fd7f6763b9d41bf1fb96daba98341da7f",
    "contract": "5c2f6b392d3fabffd323057033fbbcffd74604829aed2450094ba3f18757e435",
    "ids": "154d069149a1a6ced802f341db7df7b8dc803dce7a74e65889649a39fd65e633",
    "base": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
    "query": "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
    "base_protocol": "3621befa31a0c6e9a3b424cc6429f848bc6b45f1704f7047c8992b76ca1214a1",
    "base_preflight": "d309ee0195a816d97f216481fa5d940f8bc85ac1b7a5387540d09128e1c60ca7",
    "selection": "3311469a314a803352f9e345ad06d7a59c9d1032d4854838f5573a1af95388eb",
    "build": "0713d1c14558b630872be89490e16dc5caed66c4a4875930a1ed1c84561be2bf",
    "execution_contract": "b72b70e1f743162346e95bec09b77afec732017d1d4360a387843620a7eb6f71",
    "execution_preflight": "912230814062826b5486b2958eae1f4699fcf1d73198b0dfc95370ce7f0a9089",
    "source_audit": "234004d39836aace8531e68dbf08f9991f5537a2b645b9b0c489957f71f0537e",
}
GPU_INDEX = 0
GPU_UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
OMP_THREADS = 8
RESULT_STATUS = "PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1"
SNAPSHOT_SUFFIXES = ("nodes", "empty", "maxd", "ids", "aabb_lo", "aabb_hi", "aabb_dims")

def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}

def json_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)

def cmd_capture(cmd: list[str], output: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.write_text(proc.stdout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc

def ids() -> list[int]:
    value = [int(x) for x in IDS.read_text().splitlines()]
    if value != list(range(5000, 5064)):
        raise RuntimeError("exact64 IDs are not the frozen heldout prefix")
    return value

def cpu_preflight() -> dict[str, Any]:
    files = {
        "binary": BINARY, "source": SOURCE, "bound": BOUND, "header": HEADER,
        "contract": CONTRACT, "ids": IDS, "base": BASE, "query": QUERY,
        "base_protocol": BASE_PROTOCOL, "base_preflight": BASE_PREFLIGHT,
        "selection": SELECTION, "build": BUILD, "execution_contract": EXECUTION_CONTRACT,
        "execution_preflight": EXECUTION_PREFLIGHT, "source_audit": SOURCE_AUDIT,
        "runner": Path(__file__).resolve(),
    }
    artifacts = {name: artifact(path) for name, path in files.items()}
    for name, expected in EXPECTED.items():
        if artifacts[name]["sha256"] != expected:
            raise RuntimeError(f"artifact SHA mismatch: {name}")
    for path in (BASE, QUERY):
        if path.stat().st_size <= 0:
            raise RuntimeError(f"empty input: {path}")
    source_audit = subprocess.run([str(SOURCE_AUDIT)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if source_audit.returncode != 0:
        raise RuntimeError("source audit failed: " + source_audit.stderr.strip())
    audit_json = json.loads(source_audit.stdout)
    if audit_json.get("status") != "PASS":
        raise RuntimeError("source audit PASS status")
    build = json.loads(BUILD.read_text())
    if build.get("status") != "COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION":
        raise RuntimeError("build status")
    if build.get("binary", {}).get("sha256") != EXPECTED["binary"]:
        raise RuntimeError("build binary binding")
    base_protocol = json.loads(BASE_PROTOCOL.read_text())
    if base_protocol.get("status") != "FROZEN_CPU_ONLY_PREPARED":
        raise RuntimeError("base protocol state")
    if base_protocol.get("heldout_input", {}).get("gpu_heldout_correctness", {}).get("ids_file", {}).get("sha256") != EXPECTED["ids"]:
        raise RuntimeError("base protocol exact64 binding")
    old_preflight = json.loads(BASE_PREFLIGHT.read_text())
    if old_preflight.get("status") != "PASS_CPU_ONLY_FROZEN_EXECUTION_PENDING":
        raise RuntimeError("base protocol preflight state")
    execution_contract = json.loads(EXECUTION_CONTRACT.read_text())
    if execution_contract.get("status") != "BUILD_BOUND_CPU_ONLY_EXECUTION_NOT_AUTHORIZED":
        raise RuntimeError("build-bound execution contract state")
    execution_preflight = json.loads(EXECUTION_PREFLIGHT.read_text())
    if execution_preflight.get("status") != "PASS_CPU_ONLY_BUILD_BOUND_EXECUTION_BLOCKED":
        raise RuntimeError("build-bound execution preflight state")
    if execution_preflight.get("execution_contract", {}).get("sha256") != EXPECTED["execution_contract"]:
        raise RuntimeError("build-bound contract binding")
    selection = json.loads(SELECTION.read_text())
    if selection.get("status") != "FROZEN_CPU_ONLY":
        raise RuntimeError("selection state")
    qids = ids()
    return {
        "schema": "safe-c2-static-aabb-heldout-correctness-runner-preflight-v1",
        "status": "PASS_CPU_ONLY_READY_FOR_GPU_GUARD",
        "scope": "CPU artifact/input audit only; no nvidia-smi or CUDA binary execution",
        "nvidia_smi_called": False,
        "gpu_binary_executed": False,
        "formal_claim_eligible": False,
        "expected_gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
        "omp_threads": OMP_THREADS,
        "query_ids": {"count": len(qids), "first": qids[0], "last": qids[-1], "sha256": EXPECTED["ids"]},
        "artifacts": artifacts,
        "source_audit": audit_json,
    }

def nvidia_smi() -> str:
    value = shutil.which("nvidia-smi")
    if not value:
        raise RuntimeError("nvidia-smi unavailable")
    return value

def gpu_snapshot(nvsmi: str, outdir: Path, phase: str) -> dict[str, Any]:
    gpu = outdir / f"gpu_{phase}.csv"
    apps = outdir / f"gpu_processes_{phase}.csv"
    cmd_capture([nvsmi, "--query-gpu=index,uuid,name,driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], gpu)
    cmd_capture([nvsmi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], apps)
    rows = [x.strip() for x in gpu.read_text().splitlines() if x.strip()]
    expected = [x for x in rows if x.split(",", 1)[0].strip() == str(GPU_INDEX)]
    if len(expected) != 1 or GPU_UUID not in expected[0]:
        raise RuntimeError("GPU identity mismatch")
    app_rows = [x.strip() for x in apps.read_text().splitlines() if x.strip()]
    relevant = [x for x in app_rows if x.startswith(GPU_UUID + ",")]
    if relevant:
        raise RuntimeError(f"GPU{GPU_INDEX} has compute applications during {phase}: {relevant}")
    return {"gpu": artifact(gpu), "compute_processes": artifact(apps), "expected_gpu_idle": True}

def validate_result(outdir: Path) -> dict[str, Any]:
    result_path = outdir / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("schema") != "safe-c2-static-aabb-heldout-correctness-v1":
        raise RuntimeError("result schema")
    if result.get("status") != RESULT_STATUS:
        raise RuntimeError("result status")
    if result.get("gpu_executed") is not True or result.get("formal_claim_eligible") is not False:
        raise RuntimeError("result scope flags")
    if result.get("queries", {}).get("count") != 64 or result.get("queries", {}).get("ids") != ids():
        raise RuntimeError("result query binding")
    if result.get("projection", {}).get("dimensions") != 128:
        raise RuntimeError("result projection width")
    for method in ("baseline", "static_aabb"):
        values = result.get(method, {})
        if any(values.get(k) != 0 for k in ("invalid", "duplicate", "oracle_set_mismatch", "distance_mismatch")):
            raise RuntimeError(f"result mismatch: {method}")
    if result.get("host_cover", {}).get("violations") != 0:
        raise RuntimeError("host AABB cover violation")
    if result.get("baseline_and_static_aabb_id_sets_equal") is not True:
        raise RuntimeError("baseline/candidate set inequality")
    if result.get("snapshot_pre_post_byte_equal") is not True:
        raise RuntimeError("binary snapshot mutation")
    snapshot = {}
    for suffix in SNAPSHOT_SUFFIXES:
        before = outdir / f"snapshot_before_{suffix}.bin"
        after = outdir / f"snapshot_after_{suffix}.bin"
        if sha(before) != sha(after):
            raise RuntimeError(f"external snapshot mismatch: {suffix}")
        snapshot[suffix] = artifact(before)
    return {"result": artifact(result_path), "snapshot_before_after_sha256_equal": snapshot}

def manifest(outdir: Path, preflight: dict[str, Any], before: dict[str, Any], after: dict[str, Any], checked: dict[str, Any]) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(outdir.iterdir()):
        if path.is_file() and path.name != "terminal.json":
            artifacts[path.name] = artifact(path)
    return {
        "schema": "safe-c2-static-aabb-heldout-correctness-run-manifest-v1",
        "status": "COMPLETE_HELDOUT_CORRECTNESS_ONLY",
        "scope": "one pre-registered development-held-out static raw-float32 L2 correctness gate; no timing, update, insertion, general-float, or general-metric claim",
        "formal_claim_eligible": False,
        "preflight": preflight,
        "gpu_before": before,
        "gpu_after": after,
        "checked_result": checked,
        "artifacts": artifacts,
    }

def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    preflight = cpu_preflight()
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True, indent=2))
        return 0

    run_id = "static_aabb_heldout_correctness_v1_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)
    outdir = RUNS / run_id
    if outdir.exists():
        raise RuntimeError("run directory collision")
    outdir.mkdir(mode=0o700)
    terminal = outdir / "terminal.json"
    json_atomic(terminal, {
        "schema": "safe-c2-static-aabb-heldout-correctness-terminal-v1",
        "status": "PREPARING",
        "run_id": run_id,
        "started_utc": utc(),
        "formal_claim_eligible": False,
        "scope": "heldout correctness only; no retries/overwrites/clock changes/process kills",
    })
    try:
        json_atomic(outdir / "preflight.json", preflight)
        nvsmi = nvidia_smi()
        before = gpu_snapshot(nvsmi, outdir, "before")
        guard_path = outdir / "gpu_guard.json"
        json_atomic(guard_path, {
            "schema": "safe-c2-static-aabb-heldout-correctness-gpu-guard-v1",
            "status": "PRE_RUN_GPU_IDLE",
            "target_gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
            "before": before,
            "no_process_kill_or_clock_change": True,
        })
        json_atomic(terminal, {
            "schema": "safe-c2-static-aabb-heldout-correctness-terminal-v1",
            "status": "RUNNING",
            "run_id": run_id,
            "started_utc": utc(),
            "formal_claim_eligible": False,
        })
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": GPU_UUID, "OMP_NUM_THREADS": str(OMP_THREADS), "OMP_DYNAMIC": "FALSE"})
        cmd = [shutil.which("nice") or "nice", "-n", "10", str(BINARY), "--base", str(BASE), "--queries", str(QUERY), "--ids", str(IDS), "--outdir", str(outdir)]
        with (outdir / "command.json").open("w") as command_log, (outdir / "stdout.log").open("w") as stdout, (outdir / "stderr.log").open("w") as stderr:
            command_log.write(json.dumps({"command": cmd, "env": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_DYNAMIC")}}, sort_keys=True) + "\n")
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"heldout binary exit code {proc.returncode}")
        after = gpu_snapshot(nvsmi, outdir, "after")
        json_atomic(guard_path, {
            "schema": "safe-c2-static-aabb-heldout-correctness-gpu-guard-v1",
            "status": "PASS_PRE_AND_POST_GPU_IDLE",
            "target_gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
            "before": before,
            "after": after,
            "no_process_kill_or_clock_change": True,
        })
        checked = validate_result(outdir)
        run_manifest = manifest(outdir, preflight, before, after, checked)
        json_atomic(outdir / "manifest.json", run_manifest)
        json_atomic(terminal, {
            "schema": "safe-c2-static-aabb-heldout-correctness-terminal-v1",
            "status": "COMPLETE_HELDOUT_CORRECTNESS_ONLY",
            "run_id": run_id,
            "started_utc": json.loads((outdir / "terminal.json").read_text())["started_utc"],
            "finished_utc": utc(),
            "formal_claim_eligible": False,
            "manifest": artifact(outdir / "manifest.json"),
            "result": artifact(outdir / "result.json"),
        })
        print(json.dumps({"status": "COMPLETE_HELDOUT_CORRECTNESS_ONLY", "run_dir": str(outdir)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "safe-c2-static-aabb-heldout-correctness-terminal-v1",
            "status": "FAILED",
            "run_id": run_id,
            "finished_utc": utc(),
            "formal_claim_eligible": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            json_atomic(terminal, failure)
        finally:
            print(json.dumps({"status": "FAILED", "run_dir": str(outdir), "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2

if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_RUNNER_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)

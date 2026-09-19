#!/usr/bin/env python3
"""Persisted CPU-only fixtures for the C1 v8 external-approval release.

No case invokes nvidia-smi, Nsight, NVCC, a CUDA binary, or a production C1
approval. The replay fixture uses an explicitly unexecutable fixture schema in
a temporary external directory solely to exercise the O_EXCL claim primitive.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True
ROOT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_execution_release_v8_external_one_time_approval")
CONTROL = ROOT / "control"
RECEIPTS = ROOT / "cpu_fixture_receipts"
PYTHON = pathlib.Path("/usr/bin/python3")
BASH = pathlib.Path("/usr/bin/bash")
VERIFIER = CONTROL / "verify_release_v8.py"
GATE = CONTROL / "approval_gate_v8.py"
GUARD = CONTROL / "run_c1_execution_guard_v8.sh"
CAPACITY_CONTRACT = ROOT / "source_only_stage" / "capacity_snapshot_contract_v8.py"
C1_SOURCE = ROOT / "source_only_stage" / "worktree" / "src" / "c1_microbench.cu"


def fail(message: str) -> None:
    raise AssertionError(message)


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_receipt(name: str, body: dict[str, Any]) -> pathlib.Path:
    out = RECEIPTS / name
    body = {"schema": "gtspp-c1-v8-cpu-fixture-receipt-v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **body}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def static_closure_case() -> dict[str, Any]:
    result = subprocess.run([str(PYTHON), "-B", "-I", str(VERIFIER), "--root", str(ROOT)], text=True, capture_output=True, check=False, env={"PATH": "/usr/bin:/bin", "LANG": "C"})
    fail(f"static verifier rc={result.returncode}: {result.stderr}") if result.returncode else None
    payload = json.loads(result.stdout)
    fail("static verifier not pass") if payload.get("pass") is not True else None
    fail("static verifier touched GPU") if payload.get("gpu_tools_invoked") is not False or payload.get("cuda_binary_executed") is not False else None
    return {"fixture": "static_release_closure", "pass": True, "verifier": payload, "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}


def direct_env_bypass_case() -> dict[str, Any]:
    # These values emulate the old v7 launcher-trust environment. The v8 guard
    # must ignore all of them and reject before output/lock/GPU operations.
    before = {"runs": (ROOT / "runs").exists(), "lock": (ROOT / ".c1_v8_gpu0.lock").exists()}
    env = {
        "PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C",
        "C1_ALLOW_GPU0": "YES", "C1_V7_TRUST_ROOT": "YES", "C1_V7_LAUNCHER": "/forged/launcher",
        "C1_V7_TRUST_GUARD_SHA": "forged", "C1_V7_TRUST_PINS_SHA": "forged", "C1_V7_TRUST_PIN_VERIFY_SHA": "forged",
        "C1_RUN_OUT": str(ROOT / "runs" / "forged"), "C1_EXECUTION_APPROVED": "YES",
    }
    result = subprocess.run([str(BASH), str(GUARD)], text=True, capture_output=True, check=False, env=env)
    combined = result.stdout + result.stderr
    fail(f"direct forged env returned {result.returncode}, expected 69") if result.returncode != 69 else None
    fail("direct forged env was not explicitly rejected") if "external --approval-file" not in combined else None
    forbidden = ("nvidia-smi", "nsys", "C1Microbench", "nvcc")
    fail("direct forged env output mentioned a GPU tool") if any(token.lower() in combined.lower() for token in forbidden) else None
    after = {"runs": (ROOT / "runs").exists(), "lock": (ROOT / ".c1_v8_gpu0.lock").exists()}
    fail(f"direct forged env created state before approval: before={before} after={after}") if before != after or after["runs"] or after["lock"] else None
    return {"fixture": "direct_env_bypass_negative", "pass": True, "returncode": result.returncode, "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(), "root_state_before": before, "root_state_after": after, "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}



def bad_approval_case() -> dict[str, Any]:
    # An intentionally schema-invalid external file exercises the production
    # guard/gate path. It must fail before O_EXCL claim, mkdir, telemetry, or
    # binary launch; it is not a production approval.
    before = {"runs": (ROOT / "runs").exists(), "lock": (ROOT / ".c1_v8_gpu0.lock").exists()}
    with tempfile.TemporaryDirectory(prefix="c1_v8_bad_external_", dir="/tmp") as tmp:
        directory = pathlib.Path(tmp)
        os.chmod(directory, 0o700)
        bad = directory / "invalid.json"
        bad.write_text('{"schema":"not-an-approval"}\n')
        os.chmod(bad, 0o600)
        result = subprocess.run([str(BASH), str(GUARD), "--approval-file", str(bad)], text=True, capture_output=True, check=False, env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C", "C1_ALLOW_GPU0": "YES", "C1_EXECUTION_APPROVED": "YES"})
        combined = result.stdout + result.stderr
        fail(f"bad approval returned {result.returncode}, expected 69") if result.returncode != 69 else None
        fail("bad approval did not reject in approval gate") if "C1-V8-APPROVAL BLOCKED" not in combined else None
        fail("bad approval unexpectedly claimed") if bad.with_name(bad.name + ".c1-v8-consumed.json").exists() else None
        fail("bad approval output mentioned GPU tool") if any(token.lower() in combined.lower() for token in ("nvidia-smi", "nsys", "C1Microbench", "nvcc")) else None
    after = {"runs": (ROOT / "runs").exists(), "lock": (ROOT / ".c1_v8_gpu0.lock").exists()}
    fail(f"bad approval created root state: before={before} after={after}") if before != after or after["runs"] or after["lock"] else None
    return {"fixture": "bad_external_approval_negative", "pass": True, "returncode": result.returncode, "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(), "root_state_before": before, "root_state_after": after, "production_approval_created": False, "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}

def replay_claim_case() -> dict[str, Any]:
    # This file is not a production approval: schema/mode make it permanently
    # unexecutable by the production guard. It only exercises the exact O_EXCL
    # claim primitive with two concurrent CPU-only consumers.
    with tempfile.TemporaryDirectory(prefix="c1_v8_fixture_claim_", dir="/tmp") as tmp:
        directory = pathlib.Path(tmp)
        os.chmod(directory, 0o700)
        fixture = directory / "replay_fixture.json"
        fixture.write_text(json.dumps({"schema": "gtspp-c1-v8-approval-claim-fixture-v1", "fixture_id": "fixture_replay_20260728", "fixture_only": True, "mode": "CPU_ONLY_CLAIM_FIXTURE", "not_valid_for_execution": True}, sort_keys=True) + "\n")
        os.chmod(fixture, 0o600)
        cmd = [str(PYTHON), "-B", "-I", str(GATE), "--fixture-claim", "--approval-file", str(fixture)]
        one = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        two = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        outcomes = [one.communicate(), two.communicate()]
        rcs = [one.returncode, two.returncode]
        fail(f"replay fixture expected exactly one O_EXCL winner, rcs={rcs}") if rcs.count(0) != 1 or rcs.count(69) != 1 else None
        loser = outcomes[rcs.index(69)][1]
        fail("replay loser did not report consumed") if "already consumed" not in loser else None
        claim_path = fixture.with_name(fixture.name + ".c1-v8-consumed.json")
        fail("fixture O_EXCL claim missing") if not claim_path.is_file() or claim_path.is_symlink() else None
        claim = json.loads(claim_path.read_text())
        fail("fixture claim accidentally executable") if claim.get("fixture_only") is not True or claim.get("mode") != "CPU_ONLY_CLAIM_FIXTURE" or claim.get("not_valid_for_execution") is not True else None
        return {"fixture": "one_time_replay_o_excl", "pass": True, "producer_returncodes": rcs, "claim_sha256": sha(claim_path), "claim_schema": claim.get("schema"), "synthetic_fixture_only": True, "production_approval_created": False, "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}


def load_contract() -> Any:
    spec = importlib.util.spec_from_file_location("capacity_snapshot_contract_v8", CAPACITY_CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load capacity contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capacity_case() -> dict[str, Any]:
    contract = load_contract()
    positive = {"result_count": 7, "update_result_capacity_slots": 8, "total_result_capacity_slots": 8, "capacity_snapshot_stage": "pre_ephemeral_release"}
    negatives = [
        {"result_count": 0, "update_result_capacity_slots": 0, "total_result_capacity_slots": 0, "capacity_snapshot_stage": "pre_ephemeral_release"},
        {"result_count": 1, "update_result_capacity_slots": 8, "total_result_capacity_slots": 9, "capacity_snapshot_stage": "pre_ephemeral_release"},
        {"result_count": 9, "update_result_capacity_slots": 8, "total_result_capacity_slots": 8, "capacity_snapshot_stage": "pre_ephemeral_release"},
        {"result_count": 1, "update_result_capacity_slots": 8, "total_result_capacity_slots": 8, "capacity_snapshot_stage": "post_release"},
    ]
    fail("capacity positive fixture rejected") if contract.capacity_snapshot_violations(positive) else None
    fail("capacity negative fixture accepted") if any(not contract.capacity_snapshot_violations(x) for x in negatives) else None
    text = C1_SOURCE.read_text(encoding="utf-8")
    snapshot = "const int c1_capacity_snapshot_pre_release = update_result_ws_cap;"
    release = "releaseC1QueryWorkspace(qresult_count, qresult_count_prefix, result_id, result_dis);"
    fields = ["result.update_result_capacity_slots = c1_capacity_snapshot_pre_release;", "result.total_result_capacity_slots = c1_capacity_snapshot_pre_release;", "result.capacity_snapshot_stage = \"pre_ephemeral_release\";"]
    fail("C1 source capacity snapshot was removed") if text.count(snapshot) != 1 or text.count(release) < 1 else None
    fail("capacity snapshot occurs after first release") if text.index(snapshot) > text.index(release) else None
    fail("C1 source capacity fields lost local snapshot") if any(field not in text for field in fields) else None
    return {"fixture": "capacity_snapshot_contract", "pass": True, "positive": 1, "negative": len(negatives), "source_path": str(C1_SOURCE), "source_sha256": sha(C1_SOURCE), "contract_sha256": sha(CAPACITY_CONTRACT), "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}


def main() -> int:
    if not ROOT.is_dir() or ROOT.is_symlink() or not RECEIPTS.is_dir() or RECEIPTS.is_symlink():
        raise SystemExit("canonical release/receipts root required")
    cases = [static_closure_case(), direct_env_bypass_case(), bad_approval_case(), replay_claim_case(), capacity_case()]
    paths = [write_receipt(f"{index:02d}_{case['fixture']}.json", case) for index, case in enumerate(cases, 1)]
    summary = {"fixture": "cpu_fixture_summary", "pass": True, "cases": [case["fixture"] for case in cases], "receipts": [str(path) for path in paths], "production_approval_created": False, "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}
    summary_path = write_receipt("00_summary.json", summary)
    print(json.dumps({"pass": True, "summary": str(summary_path), "receipts": [str(path) for path in paths], "production_approval_created": False, "nvidia_smi_called": False, "gpu_binary_executed": False, "nsys_called": False, "nvcc_called": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

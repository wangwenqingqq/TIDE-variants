#!/usr/bin/env python3
"""CPU-only verifier for the Safe-C2 v3 future execution PINS and fixed plan.

This verifier does not execute the C++ binary and never invokes nvidia-smi/CUDA.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from c2_v3_execution_common import (
    ContractError,
    PLAN_SCHEMA,
    PINS_SCHEMA,
    ROOT_LITERAL,
    STAGE_MODE,
    STAGE_ORDER,
    STAGE_RUNNER_NAME,
    STAGE_TIMED_REPS,
    TRUST_SCHEMA,
    V2_ROOT_LITERAL,
    V2_SEALED_TEST_SHA256,
    WORKLOAD_SHA256,
    load_json,
    load_workload,
    parse_source_manifest,
    require_regular_file,
    root_from_module,
    sha256_file,
)

ROOT = root_from_module(__file__)
PLAN_PATH = ROOT / "protocols/c2_v3_execution_plan_v1.json"
TRUST_PATH = ROOT / "protocols/c2_v3_execution_trust_root_v1.json"
PINS_PATH = ROOT / "provenance/c2_v3_execution_pins_v1.json"
ORIGINAL_PROTOCOL = ROOT / "protocols/safe_c2_speculative_fallback_v3.json"
ORIGINAL_STATIC = ROOT / "tools/verify_v3_static.py"
SOURCE_MANIFEST = ROOT / "source_manifest.sha256"

# The launcher intentionally stays outside PINS because it literally binds the
# PINS/guard hashes; including it would create a self-hash cycle.  The launcher
# is separately hashed in every execution manifest and checked by the guard.
EXPECTED_PINNED_RELATIVE_PATHS = (
    "protocols/safe_c2_speculative_fallback_v3.json",
    "protocols/c2_v3_execution_plan_v1.json",
    "protocols/c2_v3_execution_trust_root_v1.json",
    "source_manifest.sha256",
    "provenance/v2_immutable_source_and_sealed_test.sha256",
    "inputs/sift_learn_compact10k_v1/workload_manifest.json",
    "bin/GTS_safe_c2_speculative_fallback_v3_sift1m",
    "src/gts_speculative_fallback_v3_sift1m.cu",
    "include/config.cuh",
    "include/file.cuh",
    "include/mlp_constant.cuh",
    "include/residual_pruning.cuh",
    "include/search_v3.cuh",
    "include/tree.cuh",
    "tools/verify_v3_static.py",
    "tools/c2_v3_execution_common.py",
    "tools/c2_v3_execution_manifest.py",
    "tools/verify_c2_v3_execution_artifacts.py",
    "tools/verify_c2_v3_execution_pins.py",
    "tools/create_c2_v3_execution_pins.py",
    "tools/run_c2_v3_execution_plan.sh",
    "tools/run_c2_v3_execution_guard.sh",
)


def fail(message: str) -> None:
    raise ContractError(message)


def _descriptor(path: Path, label: str) -> dict[str, Any]:
    require_regular_file(path, label, inside_root=True)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify_plan_contract(plan: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        fail("execution plan schema mismatch")
    if plan.get("canonical_root") != ROOT_LITERAL:
        fail("execution plan canonical root mismatch")
    if plan.get("status") != "IMPLEMENTED_CPU_STATIC_ONLY_NOT_EXECUTED":
        fail("execution plan status must remain CPU-static-only before authorization")
    auth = plan.get("authorization_boundary")
    if not isinstance(auth, dict) or "no GPU binary" not in str(auth.get("current", "")):
        fail("execution plan lacks no-GPU authorization boundary")
    protocol = load_json(ORIGINAL_PROTOCOL, "original v3 protocol")
    if protocol.get("schema") != "safe-c2-speculative-fallback-v3-protocol-v1":
        fail("original v3 protocol schema mismatch")
    if protocol.get("write_boundary") != ROOT_LITERAL:
        fail("original v3 protocol write boundary mismatch")
    if protocol.get("independent_workload", {}).get("final_manifest_sha256") != WORKLOAD_SHA256:
        fail("original protocol manifest binding mismatch")
    if protocol.get("v2_immutability", {}).get("v2_root") != V2_ROOT_LITERAL:
        fail("original protocol v2 root mismatch")
    if protocol.get("v2_immutability", {}).get("v2_sealed_test_ids_sha256") != V2_SEALED_TEST_SHA256:
        fail("original protocol v2 sealed digest mismatch")
    w = plan.get("workload")
    if not isinstance(w, dict):
        fail("execution plan lacks workload object")
    expected_manifest_path = str(ROOT / "inputs/sift_learn_compact10k_v1/workload_manifest.json")
    if w.get("manifest") != expected_manifest_path or w.get("manifest_schema") != "gts-v3-compact-learn-workload-v1" or w.get("manifest_sha256") != WORKLOAD_SHA256:
        fail("execution plan workload binding mismatch")
    if w.get("v2_sealed_test_sha256_forbidden") != V2_SEALED_TEST_SHA256:
        fail("execution plan v2 sealed digest mismatch")
    expected_input_keys = {"base_fvecs", "query_fvecs", "groundtruth_ivecs", "mapping"}
    if set(w.get("inputs", {})) != expected_input_keys:
        fail("execution plan input set mismatch")
    for key, binding in workload["bindings"].items():
        expected = {"path": binding["path"], "sha256": binding["sha256"]}
        if w["inputs"].get(key) != expected:
            fail(f"execution plan input binding mismatch: {key}")
    runner = plan.get("runner")
    if not isinstance(runner, dict):
        fail("execution plan lacks runner")
    if runner.get("binary") != str(ROOT / "bin/GTS_safe_c2_speculative_fallback_v3_sift1m"):
        fail("execution plan binary path mismatch")
    if runner.get("binary_sha256") != sha256_file(ROOT / "bin/GTS_safe_c2_speculative_fallback_v3_sift1m"):
        fail("execution plan binary SHA mismatch")
    if runner.get("source_manifest") != str(SOURCE_MANIFEST):
        fail("execution plan source-manifest mismatch")
    gamma = runner.get("gamma_contract")
    if not isinstance(gamma, dict) or gamma.get("levels") != 8 or gamma.get("frozen_shallow_levels") != [0, 1, 2] or gamma.get("calibration_output_name") != "final_v3_speculative_gamma_vector.txt":
        fail("execution plan gamma contract mismatch")
    resource = plan.get("resource_contract")
    if not isinstance(resource, dict):
        fail("execution plan lacks resource contract")
    if resource.get("host_required") != "CONFIGURE_ARCHIVE_HOST" or resource.get("physical_gpu_index") != 0:
        fail("execution plan host/GPU contract mismatch")
    idle = resource.get("strict_idle")
    if not isinstance(idle, dict) or idle.get("max_memory_used_mib") != 256 or idle.get("required_utilization_percent") != 0 or idle.get("require_no_compute_apps") is not True or idle.get("poll_attempts") != 15 or idle.get("poll_seconds") != 2:
        fail("execution plan strict-idle contract mismatch")
    env = resource.get("child_environment")
    expected_child_environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/root",
        "LANG": "C",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "resolved physical GPU0 UUID",
        "NVIDIA_VISIBLE_DEVICES": "resolved physical GPU0 UUID",
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.1/lib64",
        "required_invocation": "setsid --wait controlled bootstrap; direct exec /usr/bin/env -i",
    }
    if env != expected_child_environment:
        fail("execution plan child environment contract mismatch")
    irrevocable = plan.get("irreversible_consumption")
    if not isinstance(irrevocable, dict) or irrevocable.get("ledger") != str(ROOT / "provenance/c2_v3_execution_consumption_ledger_v1.json") or "before run/output mkdir or GPU telemetry" not in str(irrevocable.get("claim_order", "")) or "Any nonzero exit" not in str(irrevocable.get("rule", "")):
        fail("execution plan irreversible consumption contract mismatch")
    layout = plan.get("run_layout")
    if not isinstance(layout, dict) or layout.get("runs_parent") != str(ROOT / "runs") or layout.get("manifest") != "<run>/execution_manifest.json":
        fail("execution plan run layout mismatch")
    stages = plan.get("stage_sequence")
    if not isinstance(stages, list) or len(stages) != 3:
        fail("execution plan must have exactly three stages")
    found = [entry.get("name") if isinstance(entry, dict) else None for entry in stages]
    if found != list(STAGE_ORDER):
        fail("execution plan stage order must be calibration -> validation -> sealed_test")
    for ordinal, stage_name in enumerate(STAGE_ORDER, start=1):
        entry = stages[ordinal - 1]
        binding = workload["stages"][stage_name]
        if entry.get("ordinal") != ordinal or entry.get("runner_stage") != STAGE_RUNNER_NAME[stage_name] or entry.get("mode") != STAGE_MODE[stage_name] or entry.get("ids_path") != binding["path"] or entry.get("ids_sha256") != binding["sha256"] or entry.get("query_count") != binding["count"] or entry.get("warmup_reps_argument") != 1 or entry.get("timed_reps_argument") != STAGE_TIMED_REPS[stage_name]:
            fail(f"execution plan stage contract mismatch: {stage_name}")
    if stages[0].get("allows_gamma_input") is not False or stages[0].get("must_produce_gamma") is not True or stages[0].get("speed_claim_eligible") is not False:
        fail("calibration plan policy mismatch")
    if stages[1].get("prerequisite") != "calibration PASS plus exact frozen gamma SHA-256" or stages[2].get("prerequisite") != "validation PASS plus the exact calibration gamma SHA-256" or stages[2].get("sealed") is not True:
        fail("held-out prerequisite/sealed-test contract mismatch")
    post = plan.get("postrun_contract")
    if not isinstance(post, dict) or "only guarded_timing.query_pipeline_wall_ms can be a held-out speed metric" not in post.get("required", []):
        fail("execution plan guarded-speed rule mismatch")
    return {"plan_sha256": sha256_file(PLAN_PATH), "stage_counts": {name: workload["stages"][name]["count"] for name in STAGE_ORDER}}


def verify_trust_contract(trust: dict[str, Any]) -> dict[str, Any]:
    if trust.get("schema") != TRUST_SCHEMA or trust.get("canonical_root") != ROOT_LITERAL:
        fail("execution trust-root schema/root mismatch")
    if trust.get("status") != "IMPLEMENTED_CPU_STATIC_ONLY_NOT_EXECUTED":
        fail("execution trust-root status mismatch")
    chain = trust.get("chain")
    if not isinstance(chain, list) or len(chain) != 4 or "irrevocable" not in str(chain[2]).lower() or "UUID" not in str(chain[2]):
        fail("execution trust-root chain incomplete")
    if "does not itself grant permission" not in str(trust.get("non_authorization", "")):
        fail("execution trust-root authorization boundary absent")
    imm = trust.get("immutability")
    if not isinstance(imm, dict) or imm.get("v2_root") != V2_ROOT_LITERAL or imm.get("v2_modification") != "forbidden" or imm.get("v2_sealed_test_sha256_forbidden") != V2_SEALED_TEST_SHA256 or imm.get("no_automatic_reset") is not True:
        fail("execution trust-root immutability contract mismatch")
    return {"trust_root_sha256": sha256_file(TRUST_PATH)}


def verify_future_guard_source_contract(*, require_final_launcher_literals: bool) -> None:
    guard_path = ROOT / "tools/run_c2_v3_execution_guard.sh"
    launcher_path = ROOT / "tools/launch_c2_v3_execution.sh"
    require_regular_file(guard_path, "future outer guard", inside_root=True)
    require_regular_file(launcher_path, "future launcher", inside_root=True)
    guard = guard_path.read_text(encoding="utf-8")
    launcher = launcher_path.read_text(encoding="utf-8")

    # Treat the shell sources as security-sensitive trust-root inputs.  These
    # checks intentionally bind both the execute-state machine and the narrow
    # source-only path; semantic execution remains prohibited here.
    for needle in (
        "#!/bin/bash", "PATH=/usr/bin:/bin",
        "unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH BASH_ENV ENV CDPATH",
        "--verify-source-only", "--execute", "/usr/bin/flock -n", "claim-workflow",
        'claim-stage --run "$OUT" --plan "$PLAN" --pins "$PINS" --stage calibration',
        'init-run --run "$OUT"', "physical_nvidia_smi", "--query-compute-apps", "wait_for_strict_idle",
        "record-runtime-binding", "/usr/bin/setsid --wait /bin/bash -c", "safe-c2-v3-bootstrap", "exec /usr/bin/env -i",
        'CUDA_VISIBLE_DEVICES="$uuid"', 'NVIDIA_VISIBLE_DEVICES="$uuid"', '"$BINARY" "$GPU0_UUID" "$stage"', "LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64",
        "stage-fail", "--all-stages", 'finalize --run "$OUT"', "CHILD_SESSION_PID", "CHILD_ABORT_FILE",
        "adopt_own_child_session_from_ready", "verify_own_child_session", "own_wrapper_is_verified",
        "abort_pre_adoption_and_wait", "ABORT_PRE_ADOPTION", 'kill -TERM -- "-$CHILD_PGID"',
        'kill -KILL -- "-$CHILD_PGID"', 'kill -TERM "$CHILD_WRAPPER_PID"',
        'kill -KILL "$CHILD_WRAPPER_PID"', "record-cleanup", "trap 'exit 130' INT", "trap 'exit 143' TERM HUP",
        "/usr/bin/sha256sum", "/usr/bin/awk", "/usr/bin/readlink", "/usr/bin/setsid",
    ):
        if needle not in guard:
            fail(f"future guard lacks required fail-closed token: {needle}")

    def after(a: str, b: str, label: str) -> None:
        if guard.find(a) < 0 or guard.find(b) < 0 or guard.find(a) >= guard.find(b):
            fail(f"future guard ordering mismatch: {label}")

    after('claim-workflow', 'init-run --run "$OUT"', 'workflow claim before output initialization')
    after('claim-stage --run "$OUT" --plan "$PLAN" --pins "$PINS" --stage calibration', 'init-run --run "$OUT"', 'calibration claim before output initialization')
    after('init-run --run "$OUT"', 'wait_for_strict_idle outer_pre_strict_idle', 'output init before outer telemetry')
    after('adopt_own_child_session_from_ready || die', ': > "$CHILD_GO_FILE"', 'owned session adoption before binary release')
    after('cleanup_own_child "outer_guard_exit_${rc}"', 'mark_current_stage_failed "outer_guard_exit_${rc}"', 'own-child cleanup before failure recording')
    after('CHILD_ABORT_FILE="$OUT/logs/${stage}.child_abort"', "/usr/bin/setsid --wait /bin/bash -c", 'abort path fixed before bootstrap launch')

    # The bootstrap must have an abort gate both while waiting and immediately
    # before exec.  Thus an EXIT cleanup can always stop it before CUDA starts.
    bootstrap_start = guard.find("  /usr/bin/setsid --wait /bin/bash -c '")
    bootstrap_end = guard.find("' safe-c2-v3-bootstrap", bootstrap_start)
    if bootstrap_start < 0 or bootstrap_end < 0:
        fail("future guard lacks bounded controlled-bootstrap body")
    bootstrap = guard[bootstrap_start:bootstrap_end]
    def bootstrap_after(a: str, b: str, label: str) -> None:
        if bootstrap.find(a) < 0 or bootstrap.find(b) < 0 or bootstrap.find(a) >= bootstrap.find(b):
            fail(f"controlled-bootstrap ordering mismatch: {label}")
    bootstrap_after('while [[ ! -e "$go" ]]; do', '[[ ! -e "$abort" ]] || exit 125', 'abort is tested while waiting for GO')
    bootstrap_after('while [[ ! -e "$go" ]]; do', 'exec /usr/bin/env -i', 'GO wait precedes binary exec')
    post_go_abort = 'done\n[[ ! -e "$abort" ]] || exit 125'
    bootstrap_after(post_go_abort, 'exec /usr/bin/env -i', 'post-GO abort gate precedes binary exec')
    for needle in ('CUDA_VISIBLE_DEVICES="$uuid"', 'NVIDIA_VISIBLE_DEVICES="$uuid"', 'LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64'):
        if needle not in bootstrap:
            fail(f"controlled bootstrap lacks env -i child binding: {needle}")

    # In the pre-adoption race, no process-group signal is allowed.  The abort
    # marker is created first; any fallback TERM/KILL may target only the exact
    # /proc-verified PID returned by this guard's own setsid invocation.
    abort_start = guard.find('abort_pre_adoption_and_wait() {')
    abort_end = guard.find('\n}\n', abort_start)
    if abort_start < 0 or abort_end < 0:
        fail("future guard lacks pre-adoption cleanup function")
    abort_body = guard[abort_start:abort_end]
    for needle in (': > "$CHILD_ABORT_FILE"', 'own_wrapper_is_verified; then', 'kill -TERM "$CHILD_WRAPPER_PID"', 'kill -KILL "$CHILD_WRAPPER_PID"', 'for ((attempt=1; attempt<=100; ++attempt))'):
        if needle not in abort_body:
            fail(f"pre-adoption cleanup lacks required token: {needle}")
    if not (abort_body.find(': > "$CHILD_ABORT_FILE"') < abort_body.find('own_wrapper_is_verified; then') < abort_body.find('kill -TERM "$CHILD_WRAPPER_PID"') < abort_body.find('kill -KILL "$CHILD_WRAPPER_PID"')):
        fail("pre-adoption cleanup can signal before abort/own-wrapper proof")

    cleanup_start = guard.find('cleanup_own_child() {')
    cleanup_end = guard.find('\n}\n', cleanup_start)
    if cleanup_start < 0 or cleanup_end < 0:
        fail("future guard lacks own-child cleanup function")
    cleanup_body = guard[cleanup_start:cleanup_end]
    if cleanup_body.find('abort_pre_adoption_and_wait') < 0 or cleanup_body.find('abort_pre_adoption_and_wait') > cleanup_body.find('kill -TERM -- "-$CHILD_PGID"'):
        fail("own-child process-group cleanup is not preceded by pre-adoption abort handling")
    if cleanup_body.find('verify_own_child_session binary || verify_own_child_session bootstrap') > cleanup_body.find('kill -TERM -- "-$CHILD_PGID"'):
        fail("own-child process-group TERM lacks prior exact-session verification")

    static_case_start = guard.find('  --verify-source-only)')
    static_case_end = guard.find('  --execute)', static_case_start)
    if static_case_start < 0 or static_case_end < 0:
        fail("future guard lacks separable source-only branch")
    static_branch = guard[static_case_start:static_case_end]
    if 'physical_nvidia_smi' in static_branch or '"$BINARY"' in static_branch:
        fail("source-only guard branch can reach a GPU-management/binary token")

    for needle in (
        "#!/bin/bash", "PATH=/usr/bin:/bin",
        "unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONOPTIMIZE LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH BASH_ENV ENV CDPATH",
        "EXPECTED_GUARD_SHA=", "EXPECTED_PINS_SHA=", "EXPECTED_PINS_VERIFY_SHA=", "exec /usr/bin/env -i",
        "C2_V3_EXECUTE_ACK=I_CONFIRM_ONE_IRREVOCABLE_C2_V3_WORKFLOW",
        "/usr/bin/sha256sum", "/usr/bin/awk", "/usr/bin/readlink", "/usr/bin/date", "/usr/bin/od", "/usr/bin/tr",
    ):
        if needle not in launcher:
            fail(f"future launcher lacks required token: {needle}")
    if 'nvidia-smi' in launcher:
        fail("launcher must not call nvidia-smi itself")
    if require_final_launcher_literals:
        for name in ("EXPECTED_GUARD_SHA", "EXPECTED_PINS_SHA", "EXPECTED_PINS_VERIFY_SHA"):
            match = re.search(rf"^{name}=([0-9a-f]{{64}})$", launcher, re.MULTILINE)
            if match is None:
                fail(f"launcher {name} is not a finalized SHA-256 literal")


def verify_pins(pins_path: Path, expected_pins_sha: str | None) -> dict[str, Any]:
    require_regular_file(pins_path, "execution PINS", inside_root=True)
    observed_pins_sha = sha256_file(pins_path)
    if expected_pins_sha is not None and observed_pins_sha != expected_pins_sha:
        fail("execution PINS literal SHA-256 mismatch")
    pins = load_json(pins_path, "execution PINS")
    if pins.get("schema") != PINS_SCHEMA or pins.get("canonical_root") != ROOT_LITERAL:
        fail("execution PINS schema/root mismatch")
    if pins.get("pinned_launcher_exclusion") != "launcher is excluded only to avoid its literal PINS/guard self-hash cycle; it is separately literal-bound and run-manifest-hashed":
        fail("execution PINS launcher-exclusion policy mismatch")
    files = pins.get("pinned_files")
    if not isinstance(files, list):
        fail("execution PINS lacks pinned file list")
    seen: set[str] = set()
    actual_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            fail("malformed PINS file entry")
        rel, raw_path, digest, size = item.get("relative_path"), item.get("path"), item.get("sha256"), item.get("bytes")
        if not isinstance(rel, str) or not isinstance(raw_path, str) or not isinstance(digest, str) or not isinstance(size, int):
            fail("malformed PINS file descriptor")
        if rel in seen or Path(rel).is_absolute() or ".." in Path(rel).parts:
            fail("duplicate or unsafe PINS relative path")
        seen.add(rel)
        expected = ROOT / rel
        if str(expected) != raw_path:
            fail("PINS absolute/relative path mismatch")
        require_regular_file(expected, f"PINS file {rel}", inside_root=True)
        if sha256_file(expected) != digest or expected.stat().st_size != size:
            fail(f"PINS content mismatch: {rel}")
        actual_paths.add(rel)
    if actual_paths != set(EXPECTED_PINNED_RELATIVE_PATHS):
        missing = sorted(set(EXPECTED_PINNED_RELATIVE_PATHS) - actual_paths)
        extra = sorted(actual_paths - set(EXPECTED_PINNED_RELATIVE_PATHS))
        fail(f"PINS exact file-set mismatch; missing={missing}, extra={extra}")
    source_desc = pins.get("source_manifest")
    if source_desc != _descriptor(SOURCE_MANIFEST, "source manifest"):
        fail("PINS source-manifest descriptor mismatch")
    source_entries = parse_source_manifest(SOURCE_MANIFEST)
    required_source = {
        str(ROOT / "src/gts_speculative_fallback_v3_sift1m.cu"),
        str(ROOT / "include/config.cuh"), str(ROOT / "include/file.cuh"), str(ROOT / "include/mlp_constant.cuh"),
        str(ROOT / "include/residual_pruning.cuh"), str(ROOT / "include/search_v3.cuh"), str(ROOT / "include/tree.cuh"),
        str(ROOT / "protocols/safe_c2_speculative_fallback_v3.json"), str(ROOT / "tools/verify_v3_static.py"),
        str(ROOT / "bin/GTS_safe_c2_speculative_fallback_v3_sift1m"),
    }
    if not required_source.issubset({str(path) for _, path in source_entries}):
        fail("source manifest lacks a required v3 source/header/static/binary binding")
    workload = load_workload(ROOT, hash_inputs=True)
    if pins.get("workload_manifest") != _descriptor(Path(workload["path"]), "active workload manifest"):
        fail("PINS workload-manifest descriptor mismatch")
    if pins.get("workload_manifest", {}).get("sha256") != WORKLOAD_SHA256:
        fail("PINS active workload SHA mismatch")
    inputs = pins.get("workload_inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"base_fvecs", "query_fvecs", "groundtruth_ivecs", "mapping", "calibration", "validation", "sealed_test"}:
        fail("PINS workload input set mismatch")
    expected_inputs: dict[str, dict[str, Any]] = {}
    for name, binding in workload["bindings"].items():
        path = Path(binding["path"])
        expected_inputs[name] = {"path": str(path), "sha256": binding["sha256"], "bytes": path.stat().st_size}
    for stage, binding in workload["stages"].items():
        path = Path(binding["path"])
        expected_inputs[stage] = {"path": str(path), "sha256": binding["sha256"], "bytes": path.stat().st_size, "count": binding["count"]}
    if inputs != expected_inputs:
        fail("PINS workload input descriptors mismatch")
    v2 = pins.get("v2_immutability")
    if not isinstance(v2, dict) or v2.get("v2_root") != V2_ROOT_LITERAL or v2.get("sealed_test_sha256_forbidden") != V2_SEALED_TEST_SHA256:
        fail("PINS v2 immutability binding mismatch")
    witness = ROOT / "provenance/v2_immutable_source_and_sealed_test.sha256"
    if v2.get("witness") != _descriptor(witness, "v2 immutability witness"):
        fail("PINS v2 witness descriptor mismatch")
    plan = load_json(PLAN_PATH, "execution plan")
    trust = load_json(TRUST_PATH, "execution trust root")
    plan_result = verify_plan_contract(plan, workload)
    trust_result = verify_trust_contract(trust)
    verify_future_guard_source_contract(require_final_launcher_literals=True)
    if pins.get("plan") != _descriptor(PLAN_PATH, "execution plan") or pins.get("trust_root") != _descriptor(TRUST_PATH, "execution trust root"):
        fail("PINS plan/trust descriptor mismatch")
    return {
        "pins_sha256": observed_pins_sha,
        "pin_count": len(files),
        "source_manifest_entry_count": len(source_entries),
        **plan_result,
        **trust_result,
    }


def run_original_static_audit() -> dict[str, Any]:
    require_regular_file(ORIGINAL_STATIC, "original v3 static verifier", inside_root=True)
    completed = subprocess.run([sys.executable, str(ORIGINAL_STATIC)], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ContractError(f"original v3 static audit failed ({completed.returncode}): {completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ContractError(f"original v3 static audit did not emit JSON: {error}") from error
    if result.get("status") != "PASS_STATIC_ONLY" or result.get("gpu_binary_executed") is not False or result.get("nvidia_smi_called") is not False:
        raise ContractError("original v3 static audit result violates CPU-only contract")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="required CPU-only verification mode")
    parser.add_argument("--pins", default=str(PINS_PATH))
    parser.add_argument("--expected-pins-sha", default=None)
    args = parser.parse_args()
    if not args.source_only:
        fail("only --source-only exists; this verifier cannot execute a GPU path")
    if args.expected_pins_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", args.expected_pins_sha):
        fail("expected PINS SHA must be lowercase SHA-256")
    result = verify_pins(Path(args.pins), args.expected_pins_sha)
    original = run_original_static_audit()
    print(json.dumps({
        "schema": "safe-c2-v3-execution-pins-verification-v1",
        "status": "PASS_SOURCE_ONLY",
        "root": ROOT_LITERAL,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "original_static_audit": {"status": original["status"], "v2_witness_files_verified": original.get("v2_witness_files_verified")},
        **result,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"SAFE-C2-V3 EXECUTION PINS FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

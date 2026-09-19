#!/usr/bin/env python3
"""Fail-closed CPU-only source/pin verifier for the C1-v8 profile repair."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any

ROOT_DEFAULT = pathlib.Path(
    "/workspace/experiments/tide_safe_c1_20260727/"
    "c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract"
)
PINS_DEFAULT = ROOT_DEFAULT / "hardened_source_pins_v8.json"
EXTERNALS = {
    pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt"),
    pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/update_workloads/sift_1m_10k_442.txt"),
}
PROTOCOL_SCHEMAS = {
    "formal_supersession": "gtspp-c1-v8-capacity-snapshot-supersession-v1",
    "frozen_protocol": "gtspp-c1-v8-workspace-onequery-microbenchmark-protocol-v1",
    "gpu_uuid_safety": "gtspp-c1-v8-gpu-uuid-safety-v1",
    "idle_poll_safety": "gtspp-c1-v8-gpu-idle-poll-safety-v1",
    "idle_poll_logdir": "gtspp-c1-v8-gpu-idle-poll-logdir-v1",
    "implementation_correction": "gtspp-c1-v8-capacity-snapshot-implementation-correction-v1",
    "execution_plan": "gtspp-c1-v8-measured-execution-plan-v1",
    "profile_execution_plan": "gtspp-c1-v8-profile-execution-plan-v1",
}
HELPERS = {
    "guard", "measured_engine", "profile_engine", "profile_guard", "profile_manifest_utility",
    "strict_semantic_verifier", "measured_analyzer", "profile_verifier", "measured_finalizer",
    "formal_aggregator", "manifest_utility", "runtime_pin_verifier", "runtime_pin_generator",
    "build_script", "build_manifest_creator", "capacity_snapshot_contract", "capacity_snapshot_fixture",
    "source_pin_verifier", "source_pin_generator", "canonical_inspector_source", "guard_session_fixture",
    "profile_child_sessions_fixture",
}
FIXTURES = {
    "positive", "negative_zero", "negative_mismatch", "negative_stage", "negative_result_count",
    "negative_overflow", "profile_manifest_execution", "profile_terminal_execution", "guard_session_execution",
    "profile_child_sessions_execution",
}
INPUTS = {"base", "source_update_trace", "derived_query_only_trace", "derived_trace_provenance"}
REPORTS = ("cuda_api_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum", "um_sum")
UM_NO_PAGE_FAULT_MARKER = "does not contain CUDA Unified Memory CPU page faults data."


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(value: str | pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label}: absolute regular non-symlink file required: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label}: noncanonical/symlink traversal: {path} -> {resolved}")
    return path


def directory(value: str | pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label}: absolute non-symlink directory required: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label}: noncanonical/symlink traversal: {path} -> {resolved}")
    return path


def under(path: pathlib.Path, root: pathlib.Path) -> bool:
    return str(path).startswith(str(root) + "/")


def load(path: pathlib.Path, label: str) -> dict[str, Any]:
    regular(path, label)
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{label}: JSON object required")
    return obj


def entry_file(entry: Any, label: str, root: pathlib.Path, allowed: set[pathlib.Path]) -> pathlib.Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
        raise ValueError(f"{label}: missing path/sha256")
    if re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
        raise ValueError(f"{label}: invalid SHA-256")
    path = regular(entry["path"], label)
    if not under(path, root) and path not in allowed:
        raise ValueError(f"{label}: path outside v8 root / external allowlist: {path}")
    if sha(path) != entry["sha256"]:
        raise ValueError(f"{label}: SHA mismatch")
    return path


def cpp_function(text: str, signature: str) -> str:
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    raise ValueError(f"unterminated function {signature}")


def assert_capacity_snapshot_source_policy(c1: str) -> None:
    """Prove the local snapshot precedes release and global cap is never read after it."""
    body = cpp_function(c1, "QueryResult run_query")
    snapshot = "const int c1_capacity_snapshot_pre_release = update_result_ws_cap;"
    release = "releaseC1QueryWorkspace(qresult_count, qresult_count_prefix, result_id, result_dis);"
    if body.count(snapshot) != 1:
        raise ValueError("capacity snapshot declaration must occur exactly once")
    if body.count(release) != 1:
        raise ValueError("run_query must contain exactly one ephemeral release site")
    snapshot_pos = body.index(snapshot)
    release_pos = body.index(release)
    if snapshot_pos >= release_pos:
        raise ValueError("capacity snapshot must be captured before ephemeral release")
    if re.search(r"\bupdate_result_ws_cap\b", body[release_pos:]):
        raise ValueError("global update_result_ws_cap read after release is forbidden")
    for field in ("update_result_capacity_slots", "total_result_capacity_slots"):
        assignment = f"result.{field} = c1_capacity_snapshot_pre_release;"
        if body.count(assignment) != 1:
            raise ValueError(f"{field} must derive exactly once from the local pre-release snapshot")
    if body.count('result.capacity_snapshot_stage = "pre_ephemeral_release";') != 1:
        raise ValueError("pre-release capacity stage assignment missing")
    if "std::string capacity_snapshot_stage;" not in c1 or r'\"capacity_snapshot_stage\":\"' not in c1:
        raise ValueError("capacity snapshot JSON stage binding missing")


def expected_worktree(root: pathlib.Path) -> set[pathlib.Path]:
    worktree = root / "worktree"
    return {
        worktree / "CMakeLists.txt", worktree / "src" / "main.cu", worktree / "src" / "c1_microbench.cu",
        *[
            worktree / "include" / name
            for name in (
                "c1_alloc_counter.cuh", "config.cuh", "file.cuh", "gpu_timer.cuh", "incremental_insert.cuh",
                "mlp_constant.cuh", "residual_pruning.cuh", "residual_tuner.cuh", "search.cuh",
                "search_naive.cuh", "search_v2.cuh", "tree.cuh", "update.cuh",
            )
        ],
    }


def expected_um_contract() -> dict[str, Any]:
    return {
        "allowed_report": "nsys/reports/profile_um_sum.csv",
        "nonempty_reports": ["nsys/reports/profile_" + report + ".csv" for report in REPORTS[:-1]],
        "zero_byte_requires": {
            "stats_stdout": "nsys/stats_stdout.log",
            "processed_marker": "/um_sum.py] to [{absolute profile_um_sum.csv path}]... PROCESSED (EMPTY RESULTS)",
            "no_page_fault_marker": UM_NO_PAGE_FAULT_MARKER,
        },
    }


def expected_child_session_contract() -> dict:
    return {
        "top_level_directories": ["E_G_c1_off_reference", "P_F_full_C1", "gpu_snapshots", "child_sessions"],
        "directory": "child_sessions",
        "per_variant": {
            "ready_filename_template": "profile_{variant}.ready",
            "go_filename_template": "profile_{variant}.go",
            "ready_exact_format": "pid=<positive decimal>\npgid=<same positive decimal>\nsid=<same positive decimal>\n",
            "go_bytes": 0,
            "cleanup_record_template": "{variant}/child_session_provenance/owned_session_cleanup_v8.json",
            "cleanup_schema": "gtspp-c1-v8-profile-owned-child-cleanup-v1",
            "required_terminal": {
                "phase": "direct",
                "reason": "direct child exited",
                "action": "WAITED_FOR_VERIFIED_OWNED_SESSION",
                "child_returncode": 0,
                "verified_owned_session": True,
                "still_alive_after_cleanup": False,
            },
        },
        "strictness": "The profile verifier rejects any missing, extra, symlinked, non-regular, malformed, or mismatched ready/go/cleanup proof artifact; child_sessions is verified evidence, never a permissive extra directory.",
    }


def verify_protocols(pins: dict[str, Any], root: pathlib.Path) -> None:
    protocols = pins.get("protocols")
    if not isinstance(protocols, dict) or set(protocols) != set(PROTOCOL_SCHEMAS):
        raise ValueError("protocol key set mismatch")
    for key, schema in PROTOCOL_SCHEMAS.items():
        path = entry_file(protocols[key], f"protocol/{key}", root, EXTERNALS)
        obj = load(path, f"protocol/{key}")
        if obj.get("schema") != schema:
            raise ValueError(f"protocol/{key}: v8 schema mismatch")
        if key not in {"execution_plan", "profile_execution_plan"} and obj.get("v8_local_root", obj.get("root")) != str(root):
            raise ValueError(f"protocol/{key}: v8 root binding mismatch")
    capacity = {
        "fields": ["update_result_capacity_slots", "total_result_capacity_slots"],
        "per_operation_gate": "result_count is an integer >= 0 and <= both capacity fields; both capacity fields are integer > 0, exactly equal, and emitted from one pre-release local snapshot",
        "stage": "pre_ephemeral_release",
    }
    for key, label in (("execution_plan", "measured"), ("profile_execution_plan", "profile")):
        plan = load(entry_file(protocols[key], f"{label} execution plan", root, EXTERNALS), f"{label} execution plan")
        if plan.get("capacity_snapshot_contract") != capacity:
            raise ValueError(f"{label} execution plan capacity snapshot contract mismatch")
    gpu_safety = load(entry_file(protocols["gpu_uuid_safety"], "GPU UUID safety", root, EXTERNALS), "GPU UUID safety")
    binding = gpu_safety.get("replacement_safety_binding", {})
    for key, token in (
        ("child_session_ownership", "setsid --wait"),
        ("prepared_manifest_terminalization", "MANIFEST_READY"),
        ("persisted_profile_child_session_contract", "child_sessions"),
    ):
        if not isinstance(binding.get(key), str) or token not in binding[key]:
            raise ValueError(f"GPU safety protocol lacks {key} binding")
    profile = load(entry_file(protocols["profile_execution_plan"], "profile execution plan", root, EXTERNALS), "profile execution plan")
    expected_layout = {
        "nsys_rep": "nsys/profile.nsys-rep",
        "sqlite": "nsys/profile.sqlite",
        "stats_stdout": "nsys/stats_stdout.log",
        "csv_reports": ["nsys/reports/profile_" + report + ".csv" for report in REPORTS],
    }
    if profile.get("nsys", {}).get("artifact_layout") != expected_layout or profile.get("um_zero_byte_contract") != expected_um_contract():
        raise ValueError("profile execution plan UM/artifact contract mismatch")
    if profile.get("child_session_contract") != expected_child_session_contract():
        raise ValueError("profile execution plan child-session contract mismatch")


def verify_trace_provenance(pins: dict[str, Any], root: pathlib.Path) -> None:
    inputs = pins["inputs"]
    if set(inputs) != INPUTS:
        raise ValueError("input key set mismatch")
    files = {name: entry_file(inputs[name], f"input/{name}", root, EXTERNALS) for name in INPUTS}
    provenance = load(files["derived_trace_provenance"], "derived trace provenance")
    if provenance.get("schema") != "gtspp-c1-query-only-trace-provenance-v2" or provenance.get("v8_local_root") != str(root):
        raise ValueError("derived trace provenance v8 binding/schema mismatch")
    for key, pin_name in (("base", "base"), ("source_update_trace", "source_update_trace"), ("derived_trace", "derived_query_only_trace")):
        witness = provenance.get(key)
        if not isinstance(witness, dict) or witness.get("path") != str(files[pin_name]) or witness.get("sha256") != pins["inputs"][pin_name]["sha256"]:
            raise ValueError(f"derived trace provenance pin binding: {key}")
    for key, protocol_key in (("protocol", "frozen_protocol"), ("formal_supersession", "formal_supersession")):
        witness = provenance.get(key)
        expected = pins["protocols"][protocol_key]
        if not isinstance(witness, dict) or witness.get("path") != expected["path"] or witness.get("sha256") != expected["sha256"]:
            raise ValueError(f"derived trace protocol provenance binding: {key}")
    correction = provenance.get("v8_capacity_snapshot_correction")
    expected = pins["protocols"]["implementation_correction"]
    if not isinstance(correction, dict) or correction.get("path") != expected["path"] or correction.get("sha256") != expected["sha256"]:
        raise ValueError("derived trace capacity-correction binding")


def forbid_stale_text(paths: list[pathlib.Path]) -> None:
    # Keep predecessor-root/schema rejection precise: profile-verification-v5 is
    # a deliberate v8 evidence-schema revision, not a stale predecessor root.
    old5 = "v" + "5"
    old6 = "v" + "6"
    stale = (
        "c1_workspace_onequery_microbenchmark_" + old5,
        "c1_workspace_onequery_microbenchmark_" + old6,
        "gtspp-c1-" + old5 + "-", "gtspp-c1-" + old6 + "-",
        "C1-V" + "5", "C1-V" + "6",
        "c1_workspace_onequery_microbenchmark_v8_" + "capacity_snapshot",
    )
    for path in paths:
        if path.suffix not in {".py", ".sh", ".json", ".cu", ".cuh", ".cpp", ".txt"}:
            continue
        text = path.read_text(errors="strict")
        if any(token in text for token in stale):
            raise ValueError(f"stale version/root identity in v8 artifact: {path}")


def run_cpu_fixture(path: pathlib.Path, root: pathlib.Path, label: str) -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-I", str(path), "--root", str(root)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} CPU-only fixture failed: {result.stderr[-800:]}")
    if '"pass": true' not in result.stdout:
        raise ValueError(f"{label} CPU-only fixture lacks pass witness")


def run_review_fail_closed_launcher(path: pathlib.Path) -> None:
    result = subprocess.run(
        ["/usr/bin/bash", str(path)],
        text=True,
        capture_output=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C"},
    )
    combined = result.stdout + result.stderr
    if result.returncode == 0 or "review-required v8 launcher" not in combined:
        raise ValueError(f"launcher did not fail closed before review: {path}")


def static_policy(root: pathlib.Path, pins: dict[str, Any]) -> None:
    c1 = (root / "worktree" / "src" / "c1_microbench.cu").read_text()
    assert_capacity_snapshot_source_policy(c1)
    for token in ("update-result capacity/overflow violation", "query-only result count outside C1 workspace capacity", "setup_query_only_state"):
        if token not in c1:
            raise ValueError(f"missing C1 safety token: {token}")
    semantic = (root / "verify_c1_semantics_v8.py").read_text()
    profile = (root / "verify_c1_profile_artifacts_v8.py").read_text()
    analyzer = (root / "analyze_c1_microbenchmark_v8.py").read_text()
    contract = (root / "capacity_snapshot_contract_v8.py").read_text()
    runtime = (root / "verify_v8_pins.py").read_text()
    profile_manifest = (root / "c1_v8_profile_manifest.py").read_text()
    profile_guard = (root / "run_c1_profile_guard_v8.sh").read_text()
    manifest_fixture = root / "tools" / "test_c1_v8_profile_manifest_fixtures.py"
    terminal_fixture = root / "tools" / "test_c1_v8_profile_terminal_fixture.py"
    session_fixture = root / "tools" / "test_c1_v8_guard_session_fixtures.py"
    child_sessions_fixture = root / "tools" / "test_c1_v8_profile_child_sessions_fixture.py"
    measured_guard = (root / "run_c1_workspace_microbenchmark_guard_v8.sh").read_text()
    for name, text in (("semantic", semantic), ("profile", profile), ("analyzer", analyzer)):
        if "capacity_snapshot_violations" not in text or "capacity_snapshot_contract_v8.py" not in text:
            raise ValueError(f"{name} verifier/analyzer does not bind local capacity contract")
    for token in (
        "CAPACITY_SNAPSHOT_STAGE = \"pre_ephemeral_release\"", "result_count must be an integer >= 0",
        "result_count must not exceed update_result_capacity_slots", "result_count must not exceed total_result_capacity_slots",
        "snapshot must be an integer > 0", "capacity snapshot fields must be equal",
    ):
        if token not in contract:
            raise ValueError(f"capacity contract token missing: {token}")
    for token in ("assert_capacity_snapshot_source_policy", "global update_result_ws_cap read after release", "capacity_snapshot_stage", "profile_manifest_fixture", "profile_terminal_fixture", "profile_child_sessions_fixture"):
        if token not in runtime:
            raise ValueError(f"runtime source verifier lacks required policy token: {token}")
    for token in ("--repfile", "--stats-stdout", "args.repfile", "no_um_events_observed", "UM_NO_PAGE_FAULT_MARKER", "_require_empty_um_witness"):
        if token not in profile_manifest:
            raise ValueError(f"profile manifest lacks strict v8 artifact token: {token}")
    if "pathlib.Path(a.rep)" in profile_manifest:
        raise ValueError("predecessor int replicate Path bug remains")
    for guard_name, guard_text, bootstrap in (
        ("measured", measured_guard, "C1-V8-MEASURED-OWNED-BOOTSTRAP"),
        ("profile", profile_guard, "C1-V8-PROFILE-OWNED-BOOTSTRAP"),
    ):
        for token in (
            '"$SETSID" --wait /bin/bash -c', bootstrap,
            "adopt_own_child_session_from_ready", "allow_and_verify_owned_child_exec",
            "verify_own_child_session", "CHILD_SESSION_PID", "CHILD_PGID", "CHILD_SID",
            "TERM_OWN_VERIFIED_PROCESS_GROUP", "REFUSED_UNVERIFIED_PROCESS_GROUP_HOLDING_LOCK",
            "HOLD_LOCK_UNTIL_SESSION_EXIT", "ensure_manifest_ready_from_disk", "--fixture-first-variant-midfailure",
            "CPU_ONLY_FIXTURE_ACTIVE",
        ):
            if token not in guard_text:
                raise ValueError(f"{guard_name} guard lacks owned-session/manifest-race token: {token}")
        if 'CHILD_PID=' in guard_text or 'kill -TERM -- "-$CHILD_PID"' in guard_text:
            raise ValueError(f"{guard_name} guard retains unsafe background-PID group termination")
    for token in ("--repfile", "--stats-stdout", "PROFILE_MANIFEST_FIXTURE", "PROFILE_TERMINAL_FIXTURE", "PROFILE_CHILD_SESSIONS_FIXTURE", "--fixture-terminal-failure", "MANIFEST_READY", "FINAL_STATUS='FAILED'", "never allocation evidence"):
        if token not in profile_guard:
            raise ValueError(f"profile guard lacks v8 failure/UM policy token: {token}")
    for token in ("nsys_stats_stdout", "no_um_events_observed", "EMPTY RESULTS", "not allocation evidence", "profile-verification-v5", "verify_child_sessions_contract", "CHILD_SESSION_DIRECTORY", "SESSION_READY_PATTERN", "owned_session_cleanup_v8.json", "profile root directory set"):
        if token not in profile:
            raise ValueError(f"profile verifier lacks v8 UM contract token: {token}")
    for name, text in (
        ("capacity fixture", (root / "tools" / "test_c1_v8_capacity_snapshot_fixtures.py").read_text()),
        ("profile manifest fixture", manifest_fixture.read_text()),
        ("profile terminal fixture", terminal_fixture.read_text()),
        ("guard session fixture", session_fixture.read_text()),
        ("profile child-sessions fixture", child_sessions_fixture.read_text()),
        ("semantic", semantic), ("profile", profile), ("analyzer", analyzer),
        ("source verifier", (root / "verify_v8_source_pins.py").read_text()),
        ("source generator", (root / "create_v8_source_pins.py").read_text()),
    ):
        if "sys.dont_write_bytecode = True" not in text:
            raise ValueError(f"{name} lacks no-bytecode policy")
    if manifest_fixture.read_text().index("sys.dont_write_bytecode = True") > manifest_fixture.read_text().index("import argparse"):
        raise ValueError("profile manifest fixture enables no-bytecode too late")
    build_script = (root / "build_v8_variants.sh").read_text()
    if 'python3 -B "$ROOT/create_v8_build_manifest.py"' not in build_script:
        raise ValueError("build manifest creator lacks explicit -B")
    for guard in ((root / "run_c1_workspace_microbenchmark_guard_v8.sh").read_text(), profile_guard):
        if '"$PYTHON" -I' in guard or '"$PYTHON" -B -I' not in guard:
            raise ValueError("guard has an unprotected static Python invocation")
    for launcher_name in ("launch_c1_v8.sh", "launch_c1_profile_v8.sh"):
        launcher = (root / launcher_name).read_text()
        if "REVIEW_REQUIRED_V8" not in launcher or "review-required v8 launcher" not in launcher:
            raise ValueError(f"{launcher_name} is not review-fail-closed")
    expected = expected_worktree(root)
    actual = {path for path in (root / "worktree").rglob("*") if path.is_file() and not path.is_symlink()}
    if actual != expected:
        raise ValueError("active worktree contains stale/extra source")
    for name in ("runs", "profiles", "smoke", "launch_logs", "builds", "build_logs", ".c1_v8_gpu0.lock", "hardened_static_pins_v8.json", "worktree_build_manifest_v8.json"):
        if (root / name).exists() or (root / name).is_symlink():
            raise ValueError(f"source-only root contaminated by runtime/build artifact: {name}")
    bytecode = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if ".git" not in path.parts and (path.name == "__pycache__" or path.suffix == ".pyc")
    )
    if bytecode:
        raise ValueError("source-only root contaminated by Python bytecode: " + ", ".join(bytecode))
    all_pinned: list[pathlib.Path] = list(expected)
    for section in ("protocols", "helpers", "fixtures", "inputs"):
        for value in pins[section].values():
            path = pathlib.Path(value["path"])
            if under(path, root):
                all_pinned.append(path)
    forbid_stale_text(all_pinned)
    # These execute the real artifact handler and real guard EXIT finalizer under
    # temporary directories; they are deliberately before any GPU-capable guard.
    run_cpu_fixture(manifest_fixture, root, "profile-manifest")
    run_cpu_fixture(terminal_fixture, root, "profile-terminal")
    run_cpu_fixture(session_fixture, root, "guard-session-ownership-and-manifest-race")
    run_cpu_fixture(child_sessions_fixture, root, "profile-child-sessions-contract")
    run_review_fail_closed_launcher(root / "launch_c1_v8.sh")
    run_review_fail_closed_launcher(root / "launch_c1_profile_v8.sh")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    parser.add_argument("--pins", type=pathlib.Path, default=PINS_DEFAULT)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if not args.source_only:
        raise SystemExit("only --source-only is permitted; this verifier cannot build or launch")
    try:
        root = directory(args.root, "root")
        pins_path = regular(args.pins, "pins")
        if root != ROOT_DEFAULT or pins_path != PINS_DEFAULT:
            raise ValueError("canonical v8 source root/pins path required")
        pins = load(pins_path, "pins")
        if pins.get("schema") != "gtspp-c1-v8-source-pins-v1" or pins.get("root") != str(root) or pins.get("source_only") is not True:
            raise ValueError("source pins schema/workspace/admin/mode mismatch")
        if pins.get("allowed_external_paths") != [str(path) for path in sorted(EXTERNALS)]:
            raise ValueError("source pins external allowlist mismatch")
        if set(pins.get("helpers", {})) != HELPERS or set(pins.get("fixtures", {})) != FIXTURES:
            raise ValueError("source pins helper/fixture set mismatch")
        for section in ("helpers", "fixtures"):
            for key, value in pins[section].items():
                entry_file(value, f"{section}/{key}", root, EXTERNALS)
        sources = pins.get("worktree_source_files")
        if not isinstance(sources, list) or not sources:
            raise ValueError("source pin worktree list missing")
        expected = expected_worktree(root)
        source_paths = {entry_file(entry, f"worktree/{index}", root, EXTERNALS) for index, entry in enumerate(sources)}
        if source_paths != expected:
            raise ValueError("source pin worktree set mismatch")
        verify_protocols(pins, root)
        verify_trace_provenance(pins, root)
        contract = pins.get("capacity_snapshot_contract")
        if not isinstance(contract, dict) or contract.get("stage") != "pre_ephemeral_release" or contract.get("local_variable") != "c1_capacity_snapshot_pre_release" or contract.get("fields") != ["update_result_capacity_slots", "total_result_capacity_slots"] or contract.get("result_count_field") != "result_count":
            raise ValueError("source pins capacity contract mismatch")
        static_policy(root, pins)
    except Exception as exc:
        raise SystemExit(f"C1-V8-SOURCE-PINS BLOCKED: {type(exc).__name__}: {exc}")
    print(json.dumps({"pass": True, "mode": "SOURCE_ONLY_NO_BUILD_NO_GPU", "pins_sha256": sha(pins_path), "root": str(root), "capacity_snapshot_stage": "pre_ephemeral_release", "profile_fixtures": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

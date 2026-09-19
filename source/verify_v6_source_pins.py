#!/usr/bin/env python3
"""Fail-closed CPU-only source/pin verifier for the C1 v6 capacity repair."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import pathlib
import re
from typing import Any

ROOT_DEFAULT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v6_capacity_snapshot")
PINS_DEFAULT = ROOT_DEFAULT / "hardened_source_pins_v6.json"
EXTERNALS = {
    pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt"),
    pathlib.Path("/workspace/legacy_workspace/GTS/Datasets/update_workloads/sift_1m_10k_442.txt"),
}
PROTOCOL_SCHEMAS = {
    "formal_supersession": "gtspp-c1-v6-capacity-snapshot-supersession-v1",
    "frozen_protocol": "gtspp-c1-v6-workspace-onequery-microbenchmark-protocol-v1",
    "gpu_uuid_safety": "gtspp-c1-v6-gpu-uuid-safety-v1",
    "idle_poll_safety": "gtspp-c1-v6-gpu-idle-poll-safety-v1",
    "idle_poll_logdir": "gtspp-c1-v6-gpu-idle-poll-logdir-v1",
    "implementation_correction": "gtspp-c1-v6-capacity-snapshot-implementation-correction-v1",
    "execution_plan": "gtspp-c1-v6-measured-execution-plan-v1",
    "profile_execution_plan": "gtspp-c1-v6-profile-execution-plan-v1",
}
HELPERS = {
    "guard", "measured_engine", "profile_engine", "profile_guard", "profile_manifest_utility",
    "strict_semantic_verifier", "measured_analyzer", "profile_verifier", "measured_finalizer",
    "formal_aggregator", "manifest_utility", "runtime_pin_verifier", "runtime_pin_generator",
    "build_script", "build_manifest_creator", "capacity_snapshot_contract", "capacity_snapshot_fixture",
    "source_pin_verifier", "source_pin_generator", "canonical_inspector_source",
}
FIXTURES = {"positive", "negative_zero", "negative_mismatch", "negative_stage", "negative_result_count", "negative_overflow"}
INPUTS = {"base", "source_update_trace", "derived_query_only_trace", "derived_trace_provenance"}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
        raise ValueError(f"{label}: path outside v6 root / external allowlist: {path}")
    if sha(path) != entry["sha256"]:
        raise ValueError(f"{label}: SHA mismatch")
    return path


def cpp_function(text: str, signature: str) -> str:
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for pos in range(opening, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
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
    after_release = body[release_pos:]
    if re.search(r"\bupdate_result_ws_cap\b", after_release):
        raise ValueError("global update_result_ws_cap read after release is forbidden")
    for field in ("update_result_capacity_slots", "total_result_capacity_slots"):
        assignment = f"result.{field} = c1_capacity_snapshot_pre_release;"
        if body.count(assignment) != 1:
            raise ValueError(f"{field} must derive exactly once from the local pre-release snapshot")
    if body.count('result.capacity_snapshot_stage = "pre_ephemeral_release";') != 1:
        raise ValueError("pre-release capacity stage assignment missing")
    if "std::string capacity_snapshot_stage;" not in c1:
        raise ValueError("QueryResult stage field missing")
    if r'\"capacity_snapshot_stage\":\"' not in c1:
        raise ValueError("JSON stage serialization missing")


def expected_worktree(root: pathlib.Path) -> set[pathlib.Path]:
    worktree = root / "worktree"
    return {
        worktree / "CMakeLists.txt", worktree / "src" / "main.cu", worktree / "src" / "c1_microbench.cu",
        *[worktree / "include" / name for name in (
            "c1_alloc_counter.cuh", "config.cuh", "file.cuh", "gpu_timer.cuh",
            "incremental_insert.cuh", "mlp_constant.cuh", "residual_pruning.cuh",
            "residual_tuner.cuh", "search.cuh", "search_naive.cuh", "search_v2.cuh",
            "tree.cuh", "update.cuh",
        )],
    }


def verify_protocols(pins: dict[str, Any], root: pathlib.Path) -> None:
    protocols = pins.get("protocols")
    if not isinstance(protocols, dict) or set(protocols) != set(PROTOCOL_SCHEMAS):
        raise ValueError("protocol key set mismatch")
    for key, schema in PROTOCOL_SCHEMAS.items():
        path = entry_file(protocols[key], f"protocol/{key}", root, EXTERNALS)
        obj = load(path, f"protocol/{key}")
        if obj.get("schema") != schema:
            raise ValueError(f"protocol/{key}: v6-local schema mismatch")
        if key not in {"execution_plan", "profile_execution_plan"} and obj.get("v6_local_root", obj.get("root")) != str(root):
            raise ValueError(f"protocol/{key}: v6 root binding mismatch")
    expected_capacity_contract = {
        "fields": ["update_result_capacity_slots", "total_result_capacity_slots"],
        "per_operation_gate": "result_count is an integer >= 0 and <= both capacity fields; both capacity fields are integer > 0, exactly equal, and emitted from one pre-release local snapshot",
        "stage": "pre_ephemeral_release",
    }
    for key, label in (("execution_plan", "measured"), ("profile_execution_plan", "profile")):
        plan = load(entry_file(protocols[key], f"{label} execution plan", root, EXTERNALS), f"{label} execution plan")
        if plan.get("capacity_snapshot_contract") != expected_capacity_contract:
            raise ValueError(f"{label} execution plan capacity snapshot contract mismatch")


def verify_trace_provenance(pins: dict[str, Any], root: pathlib.Path) -> None:
    inputs = pins["inputs"]
    if set(inputs) != INPUTS:
        raise ValueError("input key set mismatch")
    files = {name: entry_file(inputs[name], f"input/{name}", root, EXTERNALS) for name in INPUTS}
    prov = load(files["derived_trace_provenance"], "derived trace provenance")
    if prov.get("schema") != "gtspp-c1-query-only-trace-provenance-v2" or prov.get("v6_local_root") != str(root):
        raise ValueError("derived trace provenance v6 binding/schema mismatch")
    for key, pin_name in (("base", "base"), ("source_update_trace", "source_update_trace"), ("derived_trace", "derived_query_only_trace")):
        witness = prov.get(key)
        if not isinstance(witness, dict) or witness.get("path") != str(files[pin_name]) or witness.get("sha256") != pins["inputs"][pin_name]["sha256"]:
            raise ValueError(f"derived trace provenance pin binding: {key}")
    for key, protocol_key in (("protocol", "frozen_protocol"), ("formal_supersession", "formal_supersession")):
        witness = prov.get(key)
        expected = pins["protocols"][protocol_key]
        if not isinstance(witness, dict) or witness.get("path") != expected["path"] or witness.get("sha256") != expected["sha256"]:
            raise ValueError(f"derived trace protocol provenance binding: {key}")
    correction = prov.get("v6_capacity_snapshot_correction")
    expected = pins["protocols"]["implementation_correction"]
    if not isinstance(correction, dict) or correction.get("path") != expected["path"] or correction.get("sha256") != expected["sha256"]:
        raise ValueError("derived trace capacity-correction binding")


def forbid_stale_text(paths: list[pathlib.Path]) -> None:
    stale_tag = "v" + "5"
    old_root = "c1_workspace_onequery_microbenchmark_" + stale_tag + "_formal"
    for path in paths:
        if path.suffix not in {".py", ".sh", ".json", ".cu", ".cuh", ".cpp", ".txt"}:
            continue
        text = path.read_text(errors="strict")
        if stale_tag in text or stale_tag.upper() in text or old_root in text:
            raise ValueError(f"stale prior-version identity in v6-local artifact: {path}")


def static_policy(root: pathlib.Path, pins: dict[str, Any]) -> None:
    c1 = (root / "worktree" / "src" / "c1_microbench.cu").read_text()
    assert_capacity_snapshot_source_policy(c1)
    for token in ("update-result capacity/overflow violation", "query-only result count outside C1 workspace capacity", "setup_query_only_state"):
        if token not in c1:
            raise ValueError(f"missing C1 safety token: {token}")
    semantic = (root / "verify_c1_semantics_v6.py").read_text()
    profile = (root / "verify_c1_profile_artifacts_v6.py").read_text()
    analyzer = (root / "analyze_c1_microbenchmark_v6.py").read_text()
    contract = (root / "capacity_snapshot_contract_v6.py").read_text()
    runtime = (root / "verify_v6_pins.py").read_text()
    for name, text in (("semantic", semantic), ("profile", profile), ("analyzer", analyzer)):
        if "capacity_snapshot_violations" not in text or "capacity_snapshot_contract_v6.py" not in text:
            raise ValueError(f"{name} verifier/analyzer does not bind local capacity contract")
    for token in ("CAPACITY_SNAPSHOT_STAGE = \"pre_ephemeral_release\"", "result_count must be an integer >= 0", "result_count must not exceed update_result_capacity_slots", "result_count must not exceed total_result_capacity_slots", "snapshot must be an integer > 0", "capacity snapshot fields must be equal"):
        if token not in contract:
            raise ValueError(f"capacity contract token missing: {token}")
    for token in ("assert_capacity_snapshot_source_policy", "global update_result_ws_cap read after release", "capacity_snapshot_stage"):
        if token not in runtime:
            raise ValueError(f"runtime source verifier lacks capacity policy: {token}")
    fixture = (root / "tools" / "test_c1_v6_capacity_snapshot_fixtures.py").read_text()
    source_generator = (root / "create_v6_source_pins.py").read_text()
    build_script = (root / "build_v6_variants.sh").read_text()
    measured_guard = (root / "run_c1_workspace_microbenchmark_guard_v6.sh").read_text()
    profile_guard = (root / "run_c1_profile_guard_v6.sh").read_text()
    for name, text in (("fixture", fixture), ("semantic", semantic), ("profile", profile), ("analyzer", analyzer), ("source verifier", (root / "verify_v6_source_pins.py").read_text()), ("source generator", source_generator)):
        if "sys.dont_write_bytecode = True" not in text:
            raise ValueError(f"{name} lacks no-bytecode policy")
    if fixture.index("sys.dont_write_bytecode = True") > fixture.index("import importlib.util"):
        raise ValueError("fixture enables no-bytecode mode after its first dynamic import")
    if 'python3 -B "$ROOT/create_v6_build_manifest.py"' not in build_script:
        raise ValueError("build manifest creator lacks explicit -B")
    for name, guard in (("measured guard", measured_guard), ("profile guard", profile_guard)):
        if '"$PYTHON" -I' in guard or '"$PYTHON" -B -I' not in guard:
            raise ValueError(f"{name} has an unprotected static Python invocation")
    expected = expected_worktree(root)
    actual = {p for p in (root / "worktree").rglob("*") if p.is_file() and not p.is_symlink()}
    if actual != expected:
        raise ValueError("active worktree contains stale/extra source")
    for name in ("runs", "profiles", "launch_logs", "builds", "build_logs", ".c1_v6_gpu0.lock", "hardened_static_pins_v6.json", "worktree_build_manifest_v6.json"):
        if (root / name).exists() or (root / name).is_symlink():
            raise ValueError(f"source-only root contaminated by runtime/build artifact: {name}")
    bytecode = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if ".git" not in p.parts and (p.name == "__pycache__" or p.suffix == ".pyc")
    )
    if bytecode:
        raise ValueError("source-only root contaminated by Python bytecode: " + ", ".join(bytecode))
    all_pinned = []
    all_pinned.extend(expected)
    for section in ("protocols", "helpers", "fixtures", "inputs"):
        for value in pins[section].values():
            path = pathlib.Path(value["path"])
            if under(path, root):
                all_pinned.append(path)
    forbid_stale_text(all_pinned)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=ROOT_DEFAULT)
    ap.add_argument("--pins", type=pathlib.Path, default=PINS_DEFAULT)
    ap.add_argument("--source-only", action="store_true")
    args = ap.parse_args()
    if not args.source_only:
        raise SystemExit("only --source-only is permitted; this verifier cannot build or launch")
    try:
        root = directory(args.root, "root")
        pins_path = regular(args.pins, "pins")
        if root != ROOT_DEFAULT or pins_path != PINS_DEFAULT:
            raise ValueError("canonical v6 source root/pins path required")
        pins = load(pins_path, "pins")
        if pins.get("schema") != "gtspp-c1-v6-source-pins-v1" or pins.get("root") != str(root) or pins.get("source_only") is not True:
            raise ValueError("source pins schema/workspace/admin/mode mismatch")
        if pins.get("allowed_external_paths") != [str(x) for x in sorted(EXTERNALS)]:
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
        source_paths = {entry_file(item, f"worktree/{idx}", root, EXTERNALS) for idx, item in enumerate(sources)}
        if source_paths != expected:
            raise ValueError("source pin worktree set mismatch")
        verify_protocols(pins, root)
        verify_trace_provenance(pins, root)
        contract = pins.get("capacity_snapshot_contract")
        if not isinstance(contract, dict) or contract.get("stage") != "pre_ephemeral_release" or contract.get("local_variable") != "c1_capacity_snapshot_pre_release" or contract.get("fields") != ["update_result_capacity_slots", "total_result_capacity_slots"] or contract.get("result_count_field") != "result_count":
            raise ValueError("source pins capacity contract mismatch")
        static_policy(root, pins)
    except Exception as exc:
        raise SystemExit(f"C1-V6-SOURCE-PINS BLOCKED: {type(exc).__name__}: {exc}")
    print(json.dumps({"pass": True, "mode": "SOURCE_ONLY_NO_BUILD_NO_GPU", "pins_sha256": sha(pins_path), "root": str(root), "capacity_snapshot_stage": "pre_ephemeral_release"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

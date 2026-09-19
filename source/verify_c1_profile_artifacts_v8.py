#!/usr/bin/env python3
"""Strict CPU-only verifier for the separately outer-guarded v8 Nsight profile.

The output is allocation/provenance evidence only.  In particular, a zero-byte
``profile_um_sum.csv`` documents *no observed UM CPU-page-fault events* only
when Nsight's own stats stdout says so; it is never allocation evidence.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
from datetime import datetime, timezone
from typing import Any

ROOT_DEFAULT = "/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark_v8_profile_child_sessions_contract"
PROFILES_ROOT = pathlib.Path(ROOT_DEFAULT) / "profiles"
PINS_PATH = pathlib.Path(ROOT_DEFAULT) / "hardened_static_pins_v8.json"
PRIMARY = ("E_G_c1_off_reference", "P_F_full_C1")
GATES = {"E_G_c1_off_reference": (0, 0), "P_F_full_C1": (1, 1)}
REPORTS = ("cuda_api_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum", "um_sum")
PHASES = (("cold", 128), ("provision", 1024), ("warmup", 128), ("steady", 1024))
CONTRACT_PATH = pathlib.Path(ROOT_DEFAULT) / "capacity_snapshot_contract_v8.py"
UM_NO_PAGE_FAULT_MARKER = "does not contain CUDA Unified Memory CPU page faults data."
CHILD_SESSION_DIRECTORY = "child_sessions"
SESSION_READY_PATTERN = re.compile(r"\Apid=([1-9][0-9]*)\npgid=([1-9][0-9]*)\nsid=([1-9][0-9]*)\n\Z")
SESSION_CLEANUP_SCHEMA = "gtspp-c1-v8-profile-owned-child-cleanup-v1"
SESSION_CLEANUP_ACTION = "WAITED_FOR_VERIFIED_OWNED_SESSION"
SESSION_CLEANUP_KEYS = {
    "schema", "updated_utc", "label", "expected_kind", "expected_executable",
    "expected_command_path", "wrapper_pid", "session_pid", "pgid", "sid",
    "phase", "reason", "verified_owned_session", "action",
    "still_alive_after_cleanup", "child_returncode",
}
_contract_spec = importlib.util.spec_from_file_location("gtspp_c1_v8_capacity_snapshot_contract", CONTRACT_PATH)
if _contract_spec is None or _contract_spec.loader is None:
    raise RuntimeError("cannot load v8 capacity snapshot contract")
_contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(_contract)
capacity_snapshot_violations = _contract.capacity_snapshot_violations


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_file(value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label}: absolute regular non-symlink file required: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise SystemExit(f"{label}: noncanonical/symlink traversal: {path} -> {resolved}")
    return path


def raw_dir(value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise SystemExit(f"{label}: absolute non-symlink directory required: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise SystemExit(f"{label}: noncanonical/symlink traversal: {path} -> {resolved}")
    return path


def load(path: pathlib.Path, label: str) -> dict[str, Any]:
    raw_file(str(path), label)
    try:
        obj = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"{label}: invalid JSON: {exc}")
    if not isinstance(obj, dict):
        raise SystemExit(f"{label}: JSON object required")
    return obj


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def norm_pci(value: Any) -> str | None:
    match = re.fullmatch(r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", str(value).strip().lower())
    return match.group(1) if match else None


def flatten(value: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if set(("path", "sha256")).issubset(value):
            out.append(value)
        for child in value.values():
            flatten(child, out)
    elif isinstance(value, list):
        for child in value:
            flatten(child, out)


def parse_telemetry(path: pathlib.Path, uuid: str, label: str) -> str:
    raw_file(str(path), label)
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    need(len(lines) == 1, label + ": one telemetry row required")
    fields = [field.strip() for field in lines[0].split(",")]
    need(len(fields) >= 5 and fields[1] == "0" and fields[2] == uuid, label + ": GPU0 UUID telemetry mismatch")
    pci = norm_pci(fields[4])
    need(pci is not None, label + ": malformed PCI bus id")
    return pci


def check_telemetry(root: pathlib.Path, manifest: dict[str, Any], uuid: str) -> str:
    directory = raw_dir(str(root / "gpu_snapshots"), "profile snapshot directory")
    snapshot = pathlib.Path(str(manifest.get("gpu", {}).get("telemetry_snapshot", "")))
    need(str(snapshot).startswith(str(directory) + "/"), "manifest initial profile telemetry boundary")
    outer_pci = parse_telemetry(snapshot, uuid, "outer prelaunch telemetry")
    required = [snapshot, directory / "outer_postrun_telemetry_gpu0.csv", directory / "outer_exit_telemetry_gpu0.csv"]
    for variant in PRIMARY:
        tag = "profile_" + variant
        pre = sorted(directory.glob("pre_" + tag + "_attempt_*_telemetry_gpu0.csv"))
        need(bool(pre), "missing profile prelaunch telemetry " + variant)
        required.extend(pre + [directory / ("post_" + tag + "_telemetry_gpu0.csv")])
        idle = directory / ("pre_" + tag + "_strict_idle_poll.log")
        raw_file(str(idle), "profile strict idle log")
        need("strict_idle=PASS" in idle.read_text(errors="replace"), "no strict idle PASS " + variant)
    for telemetry in required:
        observed = parse_telemetry(telemetry, uuid, "profile telemetry " + telemetry.name)
        need(observed == outer_pci, "profile telemetry PCI differs from outer prelaunch " + telemetry.name)
        prefix = str(telemetry).removesuffix("_telemetry_gpu0.csv")
        for suffix in ("_safety_gpu0.csv", "_compute_gpu0.csv", "_compute_all_visible.csv"):
            component = pathlib.Path(prefix + suffix)
            raw_file(str(component), "profile telemetry component")
            if suffix == "_safety_gpu0.csv":
                need(uuid in component.read_text(errors="replace"), "profile safety UUID mismatch " + component.name)
    return outer_pci


def pin_entry(pins: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    entry = pins.get(section, {}).get(name)
    need(isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str), f"missing pin {section}/{name}")
    path = raw_file(entry["path"], "pin " + section + "/" + name)
    need(sha(path) == entry["sha256"], "pin hash mismatch " + section + "/" + name)
    return entry


def run_pin_verifier(root: pathlib.Path, pins_path: pathlib.Path, pins: dict[str, Any]) -> None:
    verifier = pin_entry(pins, "runtime_helpers", "pin_verifier")
    result = subprocess.run(
        [sys.executable, "-B", "-I", verifier["path"], "--root", str(root), "--pins", str(pins_path), "--static-source"],
        text=True,
        capture_output=True,
        check=False,
    )
    need(result.returncode == 0, "pinned static verifier failed: " + result.stderr[-500:])
    for variant in PRIMARY:
        result = subprocess.run(
            [sys.executable, "-B", "-I", verifier["path"], "--root", str(root), "--pins", str(pins_path), "--variant", variant],
            text=True,
            capture_output=True,
            check=False,
        )
        need(result.returncode == 0, "pinned variant verifier failed " + variant + ": " + result.stderr[-500:])


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


def expected_plan(pins: dict[str, Any]) -> dict[str, Any]:
    entry = pin_entry(pins, "protocols", "profile_execution_plan")
    plan = load(pathlib.Path(entry["path"]), "profile execution plan")
    expected_layout = {
        "nsys_rep": "nsys/profile.nsys-rep",
        "sqlite": "nsys/profile.sqlite",
        "stats_stdout": "nsys/stats_stdout.log",
        "csv_reports": ["nsys/reports/profile_" + report + ".csv" for report in REPORTS],
    }
    expected_um_contract = {
        "allowed_report": "nsys/reports/profile_um_sum.csv",
        "nonempty_reports": ["nsys/reports/profile_" + report + ".csv" for report in REPORTS[:-1]],
        "zero_byte_requires": {
            "stats_stdout": "nsys/stats_stdout.log",
            "processed_marker": "/um_sum.py] to [{absolute profile_um_sum.csv path}]... PROCESSED (EMPTY RESULTS)",
            "no_page_fault_marker": UM_NO_PAGE_FAULT_MARKER,
        },
    }
    need(plan.get("schema") == "gtspp-c1-v8-profile-execution-plan-v1", "profile plan schema")
    need(plan.get("schedule") == [{"replicate": 1, "variant": "E_G_c1_off_reference"}, {"replicate": 1, "variant": "P_F_full_C1"}], "profile plan schedule")
    nsys = plan.get("nsys", {})
    need(nsys.get("executable") == "/usr/local/bin/nsys" and nsys.get("profile_arguments") == ["profile", "--trace=cuda", "--cuda-memory-usage=true", "--force-overwrite=true"], "profile nsys command plan")
    need(nsys.get("stats_arguments") == ["stats", "--force-export=true", "--force-overwrite=true", "--format", "csv"] and nsys.get("reports") == list(REPORTS), "profile stats plan")
    need(nsys.get("artifact_layout") == expected_layout, "profile artifact layout")
    need(plan.get("um_zero_byte_contract") == expected_um_contract, "profile zero-byte UM contract")
    need(plan.get("child_session_contract") == expected_child_session_contract(), "profile child-session contract")
    return plan


def read_stats_stdout(record: dict[str, Any], output: pathlib.Path, variant: str) -> tuple[pathlib.Path, str]:
    expected = output / "nsys" / "stats_stdout.log"
    entry = record.get("nsys_stats_stdout")
    need(isinstance(entry, dict), "missing nsys stats stdout record " + variant)
    path = raw_file(str(entry.get("path", "")), "profile stats stdout " + variant)
    need(path == expected and path.stat().st_size > 0, "profile stats stdout layout/size " + variant)
    need(entry.get("sha256") == sha(path) and entry.get("bytes") == path.stat().st_size, "profile stats stdout hash mismatch " + variant)
    return path, path.read_text(errors="strict")


def check_um_contract(record: dict[str, Any], um_report: pathlib.Path, stats_text: str, zero: bool, variant: str) -> None:
    expected_contract = {
        "report": str(um_report),
        "zero_byte_allowed_only_with": {
            "processed_marker": f"/um_sum.py] to [{um_report}]... PROCESSED (EMPTY RESULTS)",
            "no_page_fault_marker": UM_NO_PAGE_FAULT_MARKER,
        },
    }
    need(record.get("um_zero_byte_contract") == expected_contract, "profile UM zero-byte manifest contract " + variant)
    if zero:
        need(expected_contract["zero_byte_allowed_only_with"]["processed_marker"] in stats_text, "zero-byte UM lacks Nsight EMPTY RESULTS witness " + variant)
        need(UM_NO_PAGE_FAULT_MARKER in stats_text, "zero-byte UM lacks no-page-fault witness " + variant)
    else:
        need("EMPTY RESULTS" not in stats_text, "nonempty UM report conflicts with EMPTY RESULTS witness " + variant)



def verify_profile_root_directory_contract(root: pathlib.Path) -> None:
    expected = set(PRIMARY) | {"gpu_snapshots", CHILD_SESSION_DIRECTORY}
    observed = {entry.name for entry in root.iterdir() if entry.is_dir()}
    need(observed == expected, "profile root directory set")


def canonical_profile_nsys() -> pathlib.Path:
    """Resolve the fixed launcher path once, then reject a non-regular target."""
    link = pathlib.Path("/usr/local/bin/nsys")
    need(link.is_absolute() and link.is_file(), "profile nsys launcher missing")
    resolved = link.resolve(strict=True)
    return raw_file(str(resolved), "canonical profile nsys executable")


def ready_session_id(path: pathlib.Path, variant: str) -> int:
    content = raw_file(str(path), "profile child-session ready " + variant).read_text(errors="strict")
    match = SESSION_READY_PATTERN.fullmatch(content)
    need(match is not None, "profile child-session ready format " + variant)
    values = tuple(int(item) for item in match.groups())
    need(values[0] == values[1] == values[2], "profile child-session ready PID=PGID=SID " + variant)
    return values[0]


def verify_child_sessions_contract(root: pathlib.Path, pins: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify the persisted, guard-created session proof; never merely allow it."""
    session_dir = raw_dir(str(root / CHILD_SESSION_DIRECTORY), "profile child-session directory")
    expected_entries = {
        f"profile_{variant}.ready" for variant in PRIMARY
    } | {
        f"profile_{variant}.go" for variant in PRIMARY
    }
    observed_entries = {entry.name for entry in session_dir.iterdir()}
    need(observed_entries == expected_entries, "profile child-session directory set")
    variant_entries = pins.get("variant_binaries", [])
    need(isinstance(variant_entries, list), "profile child-session binary pin list")
    variant_pins = {entry.get("name"): entry for entry in variant_entries if isinstance(entry, dict)}
    need(all(variant in variant_pins for variant in PRIMARY), "profile child-session primary binary pins")
    nsys_real = canonical_profile_nsys()
    rows: list[dict[str, Any]] = []
    for variant in PRIMARY:
        key = "profile_" + variant
        ready = session_dir / (key + ".ready")
        go = raw_file(str(session_dir / (key + ".go")), "profile child-session GO " + variant)
        need(go.stat().st_size == 0, "profile child-session GO must be empty " + variant)
        session_id = ready_session_id(ready, variant)
        output = raw_dir(str(root / variant), "profile variant directory " + variant)
        cleanup_dir = raw_dir(str(output / "child_session_provenance"), "profile child-session provenance directory " + variant)
        need({entry.name for entry in cleanup_dir.iterdir()} == {"owned_session_cleanup_v8.json"}, "profile child-session provenance set " + variant)
        record = load(cleanup_dir / "owned_session_cleanup_v8.json", "profile child-session cleanup record " + variant)
        need(set(record) == SESSION_CLEANUP_KEYS, "profile child-session cleanup key set " + variant)
        expected_binary = raw_file(str(variant_pins[variant].get("path", "")), "profile child-session expected binary " + variant)
        need(record.get("schema") == SESSION_CLEANUP_SCHEMA, "profile child-session cleanup schema " + variant)
        need(record.get("label") == "profile/" + variant and record.get("expected_kind") == "profile_nsys", "profile child-session cleanup identity " + variant)
        need(record.get("expected_executable") == str(nsys_real), "profile child-session cleanup nsys binding " + variant)
        need(record.get("expected_command_path") == str(expected_binary), "profile child-session cleanup binary binding " + variant)
        ids = [record.get(name) for name in ("wrapper_pid", "session_pid", "pgid", "sid")]
        need(all(type(value) is int and value > 0 for value in ids), "profile child-session cleanup integer IDs " + variant)
        need(ids.count(ids[0]) == len(ids) and ids[0] == session_id, "profile child-session cleanup PID=PGID=SID binding " + variant)
        need(record.get("phase") == "direct" and record.get("reason") == "direct child exited", "profile child-session cleanup phase/reason " + variant)
        need(record.get("verified_owned_session") is True and record.get("still_alive_after_cleanup") is False, "profile child-session cleanup ownership/liveness " + variant)
        need(record.get("action") == SESSION_CLEANUP_ACTION and type(record.get("child_returncode")) is int and record.get("child_returncode") == 0, "profile child-session cleanup terminal result " + variant)
        stamp = record.get("updated_utc")
        need(isinstance(stamp, str), "profile child-session cleanup timestamp type " + variant)
        try:
            datetime.fromisoformat(stamp)
        except ValueError:
            raise SystemExit("profile child-session cleanup timestamp " + variant)
        rows.append({
            "variant": variant,
            "ready_session_pid": session_id,
            "cleanup_record": str(cleanup_dir / "owned_session_cleanup_v8.json"),
            "cleanup_record_sha256": sha(cleanup_dir / "owned_session_cleanup_v8.json"),
            "expected_nsys": str(nsys_real),
            "expected_binary": str(expected_binary),
        })
    return rows


def verify_child(root: pathlib.Path, pins: dict[str, Any], uuid: str, outer_pci: str, child: dict[str, Any], artifact_record: dict[str, Any], expected_variant: str) -> dict[str, Any]:
    variant = expected_variant
    output = raw_dir(str(root / variant), "profile variant directory " + variant)
    need(child.get("replicate") == 1 and child.get("variant") == variant, "profile child schedule identity " + variant)
    binary_pin = {entry.get("name"): entry for entry in pins.get("variant_binaries", [])}.get(variant, {})
    binary = child.get("binary", {})
    cache = child.get("cmake_cache", {})
    need(binary.get("path") == binary_pin.get("path") and binary.get("sha256") == binary_pin.get("sha256"), "profile child binary provenance " + variant)
    need(cache.get("path") == binary_pin.get("cmake_cache", {}).get("path") and cache.get("sha256") == binary_pin.get("cmake_cache", {}).get("sha256"), "profile child CMakeCache provenance " + variant)
    need(child.get("variant_output") == str(output) and child.get("env_i") is True, "profile child output/env-i " + variant)
    environment = child.get("environment", {})
    need(environment == {"CUDA_VISIBLE_DEVICES": uuid, "NVIDIA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "C1_EXECUTION_MODE": "PROFILE"}, "profile child UUID environment " + variant)
    base = "/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt"
    trace = str(pathlib.Path(ROOT_DEFAULT) / "inputs/sift1m_10k_442_first1024_type2_query_only.txt")
    binary_argv = [binary_pin.get("path"), "--base", base, "--trace", trace, "--radius", "500", "--out", str(output), "--replicate", "1", "--variant", variant, "--profile-only"]
    need(child.get("binary_argv") == binary_argv, "profile binary argv " + variant)
    prefix = output / "nsys" / "profile"
    nsys_argv = ["/usr/local/bin/nsys", "profile", "--trace=cuda", "--cuda-memory-usage=true", "--force-overwrite=true", "--output", str(prefix), "--", *binary_argv]
    need(child.get("nsys_argv") == nsys_argv, "outer guarded nsys invocation manifest " + variant)
    card = load(output / "run_card.json", variant + " run card")
    done = load(output / "completion.json", variant + " completion")
    need(card.get("schema") == "gtspp-c1-microbench-run-card-v8" and done.get("schema") == "gtspp-c1-microbench-completion-v8", variant + ": schemas")
    need(card.get("run_mode") == "UNMEASURED_NSYS_PROFILE_DO_NOT_USE" and card.get("profile_only") is True and done.get("run_mode") == "UNMEASURED_NSYS_PROFILE_DO_NOT_USE", variant + ": profile mode")
    need(card.get("requested_variant") == variant == card.get("compiled_variant") == done.get("compiled_variant") and card.get("replicate") == 1, variant + ": variant contract")
    need(card.get("base") == base and card.get("trace") == trace and float(card.get("radius")) == 500.0 and card.get("trace_limit") == 1024 and card.get("C2_residual_mode") == 0, variant + ": profile input contract")
    need((card.get("C1_PERSISTENT_WORKSPACE"), card.get("C1_ONE_QUERY_FASTPATH")) == GATES[variant], variant + ": compile gates")
    need(done.get("tree_invariance_pass") is True and int(card.get("cuda_runtime_version", 0)) > 0 and int(card.get("cuda_driver_version", 0)) > 0, variant + ": profile completion/runtime")
    need(str(card.get("visible_cuda_device_name", "")).startswith("NVIDIA RTX PRO 6000"), variant + ": profile CUDA device")
    need(isinstance(card.get("visible_cuda_device_uuid"), str) and card.get("visible_cuda_device_uuid") == uuid, variant + ": CUDA UUID differs from manifest/session UUID")
    need(norm_pci(card.get("visible_cuda_device_pci_bus_id", "")) == outer_pci, variant + ": CUDA PCI differs from outer telemetry")
    for phase, expected_count in PHASES:
        op_path = raw_file(str(output / ("phase_" + phase + "_ops.jsonl")), variant + " " + phase + " capacity JSONL")
        records = [json.loads(line) for line in op_path.read_text().splitlines() if line.strip()]
        need(len(records) == expected_count, variant + ": " + phase + " capacity record count")
        for ordinal, row in enumerate(records):
            violations = capacity_snapshot_violations(row)
            need(not violations, variant + ": invalid capacity snapshot " + phase + "/" + str(ordinal) + ": " + "; ".join(violations))

    expected = [
        output / "nsys" / "profile.nsys-rep",
        output / "nsys" / "profile.sqlite",
        *[output / "nsys" / "reports" / ("profile_" + report + ".csv") for report in REPORTS],
    ]
    need(set(artifact_record) == {"replicate", "variant", "artifacts", "nsys_stats_stdout", "um_zero_byte_contract"}, "profile artifact record key set " + variant)
    need(artifact_record.get("replicate") == 1 and artifact_record.get("variant") == variant, "profile artifact record identity " + variant)
    artifacts = artifact_record.get("artifacts", [])
    need(isinstance(artifacts, list) and [entry.get("path") for entry in artifacts] == [str(path) for path in expected], "profile artifact manifest layout/order " + variant)
    stats_path, stats_text = read_stats_stdout(artifact_record, output, variant)
    outputs: list[dict[str, Any]] = []
    um_state: dict[str, Any] | None = None
    for index, path in enumerate(expected):
        path = raw_file(str(path), variant + " artifact")
        size = path.stat().st_size
        entry = artifacts[index]
        need(entry.get("sha256") == sha(path) and entry.get("bytes") == size, "profile artifact hash mismatch " + path.name)
        if path != expected[-1]:
            need(size > 0, "empty required non-UM profile artifact " + path.name)
            need("no_um_events_observed" not in entry, "non-UM artifact carries forbidden UM state " + path.name)
        else:
            zero = size == 0
            need(entry.get("no_um_events_observed") is zero, "UM no-events flag mismatches bytes " + variant)
            check_um_contract(artifact_record, path, stats_text, zero, variant)
            um_state = {
                "path": str(path),
                "bytes": size,
                "no_um_events_observed": zero,
                "interpretation": "no UM CPU-page-fault events observed; not allocation evidence" if zero else "nonempty UM report present; profile remains explicitly unmeasured",
                "stats_stdout": str(stats_path),
            }
        if path.suffix == ".sqlite":
            need(path.open("rb").read(16) == b"SQLite format 3\x00", variant + ": invalid Nsight SQLite")
        if path.suffix == ".csv" and path != expected[-1] or (path.suffix == ".csv" and path == expected[-1] and size > 0):
            lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
            need(len(lines) >= 2 and not any("no data" in line.lower() for line in lines), variant + ": empty/no-data Nsight report " + path.name)
        outputs.append({"path": str(path), "sha256": sha(path), "bytes": size})
    need(um_state is not None, "missing UM report state")
    return {
        "variant": variant,
        "run_card_sha256": sha(output / "run_card.json"),
        "completion_sha256": sha(output / "completion.json"),
        "cuda_pci_bus_id_normalized": outer_pci,
        "artifacts": outputs,
        "um_sum": um_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=pathlib.Path, required=True)
    parser.add_argument("--pins", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = raw_dir(str(args.profile_root), "profile root")
    need(root.parent == PROFILES_ROOT and root.name.startswith("c1_v8_profile_"), "profile root must be direct c1_v8_profile_* child of v8 profiles")
    pins_path = raw_file(str(args.pins), "pins")
    need(pins_path == PINS_PATH, "pins path must be canonical v8 hardened pin file")
    pins = load(pins_path, "pins")
    need(pins.get("schema") == "gtspp-c1-v8-pins-v1", "pins schema")
    run_pin_verifier(pathlib.Path(ROOT_DEFAULT), pins_path, pins)
    plan = expected_plan(pins)
    manifest = load(root / "profile_run_manifest_v8.json", "profile run manifest")
    guard = load(root / "profile_guard_card_v8.json", "profile guard card")
    need(manifest.get("schema") == "gtspp-c1-v8-profile-run-manifest-v1" and manifest.get("execution_mode") == "PROFILE" and manifest.get("status") == "COMPLETE", "profile manifest mode/status")
    need(manifest.get("root") == ROOT_DEFAULT and manifest.get("profile_root") == str(root) and manifest.get("pins", {}).get("path") == str(pins_path) and manifest.get("pins", {}).get("sha256") == sha(pins_path), "profile manifest root/pins")
    launcher = raw_file(str(manifest.get("trust_root_launcher", {}).get("path", "")), "profile launcher manifest path")
    need(launcher == pathlib.Path(ROOT_DEFAULT) / "launch_c1_profile_v8.sh" and sha(launcher) == manifest.get("trust_root_launcher", {}).get("sha256"), "profile launcher provenance")
    profile_guard_pin = pin_entry(pins, "runtime_helpers", "profile_guard")
    outer_guard = raw_file(str(manifest.get("outer_profile_guard", {}).get("path", "")), "profile guard manifest path")
    need(outer_guard == pathlib.Path(profile_guard_pin["path"]) and sha(outer_guard) == profile_guard_pin["sha256"] and manifest.get("outer_profile_guard", {}).get("sha256") == profile_guard_pin["sha256"], "outer profile guard provenance")
    uuid = str(manifest.get("gpu", {}).get("uuid", ""))
    need(manifest.get("gpu", {}).get("physical_index") == 0 and uuid.startswith("GPU-"), "profile GPU provenance")
    need(guard.get("schema") == "gtspp-c1-v8-profile-guard-card-v1" and guard.get("status") == "COMPLETE" and guard.get("execution_mode") == "PROFILE" and guard.get("physical_gpu_index") == 0 and guard.get("physical_gpu_uuid") == uuid, "profile guard card provenance")
    expected_entries: list[dict[str, Any]] = []
    flatten(pins, expected_entries)
    expected_map = {entry["path"]: entry["sha256"] for entry in expected_entries}
    observed = {entry.get("path"): entry for entry in manifest.get("pinned_artifacts", [])}
    need(set(observed) == set(expected_map), "profile manifest pinned artifact set")
    for path, digest in expected_map.items():
        need(observed[path].get("sha256") == digest and observed[path].get("actual_sha256") == digest, "profile manifest pin drift " + path)
    verify_profile_root_directory_contract(root)
    child_sessions = verify_child_sessions_contract(root, pins)
    outer_pci = check_telemetry(root, manifest, uuid)
    children = manifest.get("children", [])
    records = manifest.get("profile_artifacts", [])
    need(isinstance(children, list) and isinstance(records, list) and len(children) == 2 and len(records) == 2, "exactly two profile children/artifact records")
    schedule = plan["schedule"]
    need([(item.get("replicate"), item.get("variant")) for item in children] == [(item["replicate"], item["variant"]) for item in schedule], "profile child schedule")
    need([(item.get("replicate"), item.get("variant")) for item in records] == [(item["replicate"], item["variant"]) for item in schedule], "profile artifact schedule")
    events = manifest.get("events", [])
    need(any(item.get("kind") == "outer_prelaunch_pass" for item in events) and any(item.get("kind") == "profile_schedule_verified" for item in events), "profile outer events")
    for variant in PRIMARY:
        need(sum(1 for item in events if item.get("kind") == "profile_variant_prelaunch_pass" and item.get("variant") == variant) == 1, "profile prelaunch event " + variant)
        need(sum(1 for item in events if item.get("kind") == "profile_variant_postlaunch_telemetry" and item.get("variant") == variant) == 1, "profile post telemetry event " + variant)
        need(sum(1 for item in events if item.get("kind") == "profile_nsys_stats_complete" and item.get("variant") == variant) == 1, "profile nsys stats event " + variant)
        need(sum(1 for item in events if item.get("kind") == "profile_variant_completion_contract_pass" and item.get("variant") == variant) == 1, "profile completion event " + variant)
    rows = [verify_child(root, pins, uuid, outer_pci, children[index], records[index], variant) for index, variant in enumerate(PRIMARY)]
    output = {
        "schema": "gtspp-c1-v8-profile-verification-v5",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "profile_root": str(root),
        "pins_sha256": sha(pins_path),
        "profile_run_manifest_sha256": sha(root / "profile_run_manifest_v8.json"),
        "profile_guard_card_sha256": sha(root / "profile_guard_card_v8.json"),
        "gpu_uuid": uuid,
        "gpu_pci_bus_id_normalized": outer_pci,
        "pass": True,
        "formal_profile_eligible": True,
        "profiles": rows,
        "child_sessions": child_sessions,
        "warning": "Explicitly unmeasured Nsight allocation/provenance evidence only; no profile value is timing data. A zero-byte um_sum records no observed UM CPU-page-fault events and is not allocation evidence.",
    }
    target = root / "profile_verification_v8.json"
    if target.is_symlink():
        raise SystemExit("refuse symlink profile verification output")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps({"pass": True, "output": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

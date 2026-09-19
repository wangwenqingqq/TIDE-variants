#!/usr/bin/env python3
"""Create the one frozen CPU-only PINS file for the Safe-C2 v3 trust root.

The command hashes files only.  It never invokes CUDA, nvidia-smi, or a binary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from c2_v3_execution_common import (
    ContractError,
    PINS_SCHEMA,
    ROOT_LITERAL,
    V2_ROOT_LITERAL,
    V2_SEALED_TEST_SHA256,
    WORKLOAD_SHA256,
    atomic_write_json,
    load_json,
    load_workload,
    parse_source_manifest,
    require_regular_file,
    root_from_module,
    sha256_file,
)
from verify_c2_v3_execution_pins import (
    EXPECTED_PINNED_RELATIVE_PATHS,
    PLAN_PATH,
    PINS_PATH,
    TRUST_PATH,
    verify_plan_contract,
    verify_trust_contract,
    verify_future_guard_source_contract,
)

ROOT = root_from_module(__file__)
SOURCE_MANIFEST = ROOT / "source_manifest.sha256"
WITNESS = ROOT / "provenance/v2_immutable_source_and_sealed_test.sha256"
EXCLUSION = "launcher is excluded only to avoid its literal PINS/guard self-hash cycle; it is separately literal-bound and run-manifest-hashed"
ACK = "I_CREATE_OR_REPLACE_FROZEN_C2_V3_EXECUTION_PINS"


def fail(message: str) -> None:
    raise ContractError(message)


def descriptor(path: Path, label: str, *, inside_root: bool = True) -> dict[str, Any]:
    require_regular_file(path, label, inside_root=inside_root)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def make_pins() -> dict[str, Any]:
    workload = load_workload(ROOT, hash_inputs=True)
    plan = load_json(PLAN_PATH, "execution plan")
    trust = load_json(TRUST_PATH, "execution trust root")
    verify_plan_contract(plan, workload)
    verify_trust_contract(trust)
    # Launcher literals are filled only after PINS exists; guard semantics must
    # already be statically valid before this one-time PINS creation.
    verify_future_guard_source_contract(require_final_launcher_literals=False)
    source_entries = parse_source_manifest(SOURCE_MANIFEST)
    required = {str(ROOT / rel) for rel in EXPECTED_PINNED_RELATIVE_PATHS if rel.startswith(("src/", "include/", "bin/", "protocols/safe_", "tools/verify_v3_static.py"))}
    if not required.issubset({str(path) for _, path in source_entries}):
        fail("original source manifest does not cover a required v3 source/header/binary/static file")
    pinned: list[dict[str, Any]] = []
    for rel in EXPECTED_PINNED_RELATIVE_PATHS:
        path = ROOT / rel
        info = descriptor(path, f"pinned file {rel}")
        pinned.append({"relative_path": rel, **info})
    workload_inputs: dict[str, Any] = {}
    for name, binding in workload["bindings"].items():
        path = Path(binding["path"])
        workload_inputs[name] = {"path": str(path), "sha256": binding["sha256"], "bytes": path.stat().st_size}
    for stage, binding in workload["stages"].items():
        path = Path(binding["path"])
        workload_inputs[stage] = {"path": str(path), "sha256": binding["sha256"], "bytes": path.stat().st_size, "count": binding["count"]}
    return {
        "schema": PINS_SCHEMA,
        "canonical_root": ROOT_LITERAL,
        "purpose": "CPU-only frozen trust inputs for a future guarded one-shot C2 v3 execution; this is not GPU authorization.",
        "pinned_launcher_exclusion": EXCLUSION,
        "pinned_files": pinned,
        "source_manifest": descriptor(SOURCE_MANIFEST, "source manifest"),
        "source_manifest_entry_count": len(source_entries),
        "plan": descriptor(PLAN_PATH, "execution plan"),
        "trust_root": descriptor(TRUST_PATH, "execution trust root"),
        "workload_manifest": descriptor(Path(workload["path"]), "active workload manifest"),
        "workload_inputs": workload_inputs,
        "v2_immutability": {
            "v2_root": V2_ROOT_LITERAL,
            "sealed_test_sha256_forbidden": V2_SEALED_TEST_SHA256,
            "witness": descriptor(WITNESS, "v2 immutability witness"),
        },
        "execution_policy": {
            "cpu_static_only_at_creation": True,
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
            "workload_manifest_sha256": WORKLOAD_SHA256,
            "no_automatic_reset_or_retry": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the frozen PINS only with the explicit acknowledgement")
    parser.add_argument("--ack", default="")
    parser.add_argument("--replace", action="store_true", help="requires the same explicit acknowledgement; never use after execution")
    args = parser.parse_args()
    pins = make_pins()
    if not args.write:
        print(json.dumps({"schema": "safe-c2-v3-execution-pins-preview-v1", "status": "PASS_PREVIEW_ONLY", "gpu_binary_executed": False, "nvidia_smi_called": False, "pin_count": len(pins["pinned_files"]), "pins": pins}, sort_keys=True))
        return 0
    if args.ack != ACK:
        fail("refusing to write PINS without exact explicit acknowledgement")
    if PINS_PATH.exists() and not args.replace:
        fail("PINS already exists; refusing replacement without --replace plus acknowledgement")
    if PINS_PATH.is_symlink():
        fail("PINS path is a symlink")
    atomic_write_json(PINS_PATH, pins)
    print(json.dumps({"schema": "safe-c2-v3-execution-pins-creation-v1", "status": "PASS_PINS_WRITTEN", "path": str(PINS_PATH), "sha256": sha256_file(PINS_PATH), "pin_count": len(pins["pinned_files"]), "gpu_binary_executed": False, "nvidia_smi_called": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"SAFE-C2-V3 EXECUTION PINS CREATE FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Audit the synthetic plan more strictly than the upstream parser-only schema."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from generate_synthetic_e2e3_plan import PLAN_RELATIVE_PATH, assert_internal_contract, build_plan, canonical_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_generator_imports_are_local_stdlib(root: Path) -> None:
    source_path = root / "tools/generate_synthetic_e2e3_plan.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    allowed = {"argparse", "hashlib", "json", "pathlib", "typing", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module.split(".")[0]] if node.module else []
        else:
            continue
        if any(module not in allowed for module in modules):
            raise ValueError(f"generator imports nonlocal/non-stdlib module: {modules}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--write-result", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result_path = args.write_result.resolve()
    if result_path.parent != root / "evidence":
        raise ValueError("audit result must be under root/evidence")
    plan_path = root / PLAN_RELATIVE_PATH
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan != build_plan(root):
        raise ValueError("stored plan differs from the deterministic source plan")
    assert_internal_contract(root, plan)
    assert_generator_imports_are_local_stdlib(root)
    result = {
        "schema": "tide.synthetic-e2e3-plan-audit.v1",
        "status": "PASS_SYNTHETIC_SOURCE_PLAN_AUDIT",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": sha256_file(plan_path),
        "e2_repetitions": len(plan["primary_e2"]["repetitions"]),
        "e3_objects": [row["objects"] for row in plan["e3_scale_points"]],
        "nonclaim": "Static/source-plan audit only; no data, trace, GPU, runner, oracle, or performance execution occurred.",
    }
    with result_path.open("xb") as handle:
        handle.write(canonical_bytes(result))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

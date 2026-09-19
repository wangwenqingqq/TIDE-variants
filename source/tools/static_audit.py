#!/usr/bin/env python3
"""Static AST/package audit for the v2 synthetic runner; no runner invocation."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED = {"README.md", "NONCLAIMS.md", "EXECUTION_CONTRACT.md", "input/VALIDATED_SYNTHETIC_E2E3_PLAN.json", "input/PLAN_COPY_PROVENANCE.json", "tools/run_synthetic_campaign.py", "tools/static_audit.py", "tools/create_source_manifest.py"}
BANNED = {"socket", "subprocess", "requests", "urllib", "http", "torch", "cupy", "cuda", "ctypes", "pynvml", "tensorflow"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def imports(tree: ast.AST) -> List[str]:
    values: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.append(node.module.split(".")[0])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--write-result", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); result_path = args.write_result.resolve()
    require(root.is_dir() and result_path.parent == root / "evidence", "invalid root/result path")
    files = []
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"symlink prohibited: {path}")
        if path.is_file(): files.append(path)
    rels = {path.relative_to(root).as_posix() for path in files}
    require(REQUIRED <= rels, f"missing files: {sorted(REQUIRED - rels)}")
    for path in files:
        if path.suffix == ".json": json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            forbidden = set(imports(tree)) & BANNED
            require(not forbidden, f"banned import in {path}: {sorted(forbidden)}")
    source = (root / "tools/run_synthetic_campaign.py").read_text(encoding="utf-8")
    for token in ("class CertifiedSidecarState", "class ExactDeltaState", "receipt_leaf_lower_bound_squared", "full_live_oracle_top_k", "scanned_sidecar", "scanned_delta", "os.geteuid", "os.O_EXCL", "cpu_stdlib_python_only"):
        require(token in source, f"runner lacks required v2 contract token: {token}")
    plan = json.loads((root / "input/VALIDATED_SYNTHETIC_E2E3_PLAN.json").read_text(encoding="utf-8"))
    require(plan.get("synthetic_development_only") is True and plan.get("template_only") is False, "input plan is not synthetic executable plan")
    require(plan.get("common_overlay_budget") == {"expression":"live_sidecars + live_delta","threshold_name":"C_ov","C_ov":4096}, "C_ov mismatch")
    result: Dict[str, Any] = {"schema":"tide.synthetic-e2e3-receipt-sidecar-static-audit.v2", "status":"PASS_SOURCE_ONLY_STATIC_AUDIT", "files":len(files), "nonclaim":"Static AST/package audit only; no GTS/data/trace/GPU/runner/oracle/performance execution occurred."}
    with result_path.open("xb") as handle: handle.write(canonical_bytes(result))
    print(json.dumps(result, sort_keys=True, separators=(",",":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

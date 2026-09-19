#!/usr/bin/env python3
"""Create one transparent, non-executing synthetic E2/E3 plan.

This tool uses only package-local declaration bytes and stdlib hashing.  It
never opens data, a trace, a GTS checkout, a GPU, or a network connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_LABEL = "tide-sigmod2027-e2e3-synthetic-campaign-v1-20260809T185806Z"
PLAN_RELATIVE_PATH = Path("plans/SYNTHETIC_E2E3_CAMPAIGN_PLAN.json")
RECEIPT_RELATIVE_PATH = Path("evidence/SYNTHETIC_PLAN_GENERATION_RECEIPT.json")
MARKERS = {
    "base_state_sha256": Path("fixtures/SYNTHETIC_BASE_STATE_DECLARATION.txt"),
    "catalog_sha256": Path("fixtures/SYNTHETIC_CATALOG_DECLARATION.txt"),
    "full_oracle_impl_sha256": Path("fixtures/SYNTHETIC_FULL_ORACLE_DECLARATION.txt"),
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def identity32(role: str) -> str:
    return sha256_bytes((PACKAGE_LABEL + "|" + role).encode("utf-8"))[:32]


def marker_hashes(root: Path) -> Dict[str, str]:
    return {key: sha256_file(root / relative) for key, relative in MARKERS.items()}


def shared_overlay() -> Dict[str, Any]:
    return {
        "expression": "live_sidecars + live_delta",
        "threshold_name": "C_ov",
        "C_ov": 4096,
    }


def repetition(replication: int, first: str) -> Dict[str, Any]:
    second = "exact_delta" if first == "certified_sidecar" else "certified_sidecar"
    return {
        "replication": replication,
        "order": [first, second],
        "run_ids": [identity32(f"e2-rep-{replication}-{first}"), identity32(f"e2-rep-{replication}-{second}")],
        "status": "SYNTHETIC_PLAN_NOT_EXECUTED",
    }


def scale_point(label: str, objects: int, insert_ratio: float, delete_ratio: float, query_count: int) -> Dict[str, Any]:
    return {
        "label": label,
        "objects": objects,
        "trace_id": identity32(f"e3-{label}-trace"),
        "insert_ratio": insert_ratio,
        "delete_ratio": delete_ratio,
        "query_count": query_count,
        "oracle_coverage": "every_query",
        "turnover_snapshots": [
            "predeclared_initial_state",
            "predeclared_after_insert_phase",
            "predeclared_after_delete_phase",
        ],
        "overlay_accounting": shared_overlay(),
        "sidecar_cap_L": 64,
        "base_delete_barrier": "synchronous",
        "status": "SYNTHETIC_PLAN_NOT_EXECUTED",
    }


def build_plan(root: Path) -> Dict[str, Any]:
    hashes = marker_hashes(root)
    starts = ["certified_sidecar", "exact_delta", "certified_sidecar", "exact_delta", "certified_sidecar"]
    return {
        "schema": "tide.e2e3-campaign-plan.v1",
        "template_only": False,
        "status": "SYNTHETIC_DEVELOPMENT_PLAN_NOT_EXECUTED",
        "synthetic_development_only": True,
        "campaign_id": identity32("campaign"),
        "metric_contract": "integer_l2_squared_v1",
        "k": 10,
        "common_overlay_budget": shared_overlay(),
        "nonclaim": "Parser-only synthetic plan; no data, trace, runner, GPU, oracle, or performance execution occurred.",
        "primary_e2": {
            "variants": ["certified_sidecar", "exact_delta"],
            "trace_id": identity32("e2-trace"),
            **hashes,
            "tie_order": ["distance_key", "stable_id"],
            "base_delete_barrier": "synchronous",
            "overlay_accounting": shared_overlay(),
            "sidecar_cap_L": 64,
            "repetitions": [repetition(index + 1, first) for index, first in enumerate(starts)],
            "status": "SYNTHETIC_PLAN_NOT_EXECUTED",
        },
        "e3_scale_points": [
            scale_point("synthetic_correctness_fixture", 1024, 0.10, 0.10, 128),
            scale_point("synthetic_100k_predeclaration", 100000, 0.10, 0.10, 1000),
            scale_point("synthetic_1m_predeclaration", 1000000, 0.10, 0.10, 1000),
        ],
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assert_internal_contract(root: Path, plan: Dict[str, Any]) -> None:
    require(plan == build_plan(root), "plan differs from deterministic synthetic source")
    e2 = plan["primary_e2"]
    overlay = plan["common_overlay_budget"]
    require(e2["overlay_accounting"] == overlay, "E2 does not use common overlay contract")
    require("delta_only_B" not in e2, "primary E2 must not carry delta_only_B")
    repetitions: List[Dict[str, Any]] = e2["repetitions"]
    require(len(repetitions) >= 5, "requires at least five repetitions")
    starts = [row["order"][0] for row in repetitions]
    require(all(starts[index] != starts[index + 1] for index in range(len(starts) - 1)), "starts are not alternating")
    require(abs(starts.count("certified_sidecar") - starts.count("exact_delta")) <= 1, "starts are not balanced")
    run_ids = [run_id for row in repetitions for run_id in row["run_ids"]]
    require(all(len(row["run_ids"]) == 2 for row in repetitions), "each repetition must declare two variants")
    require(len(run_ids) == len(set(run_ids)), "synthetic run IDs are not globally unique")
    scales = plan["e3_scale_points"]
    require(any(point["objects"] >= 100000 for point in scales), "missing 100K point")
    require(any(point["objects"] >= 1000000 for point in scales), "missing 1M point")
    for point in scales:
        require(point["overlay_accounting"] == overlay, "E3 does not use common overlay contract")
        require(point["sidecar_cap_L"] == e2["sidecar_cap_L"], "E3 L differs from E2 L")
        require(point["base_delete_barrier"] == e2["base_delete_barrier"], "E3 barrier differs from E2 barrier")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--write", action="store_true", help="write plan and receipt once")
    parser.add_argument("--check", action="store_true", help="verify existing plan equals generated plan")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = build_plan(root)
    assert_internal_contract(root, plan)
    plan_path = root / PLAN_RELATIVE_PATH
    receipt_path = root / RECEIPT_RELATIVE_PATH
    if args.write:
        write_new(plan_path, (json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        receipt = {
            "schema": "tide.synthetic-e2e3-plan-generation-receipt.v1",
            "status": "SYNTHETIC_PLAN_GENERATED_NOT_EXECUTED",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": sha256_file(plan_path),
            "marker_sha256": marker_hashes(root),
            "nonclaim": "Generated only a synthetic plan and marker hashes; no workload or performance execution occurred.",
        }
        write_new(receipt_path, canonical_bytes(receipt))
    if args.check:
        require(plan_path.is_file(), "existing plan is absent")
        loaded = json.loads(plan_path.read_text(encoding="utf-8"))
        require(loaded == plan, "existing plan does not equal deterministic generated plan")
    if not args.write and not args.check:
        parser.error("select --write and/or --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

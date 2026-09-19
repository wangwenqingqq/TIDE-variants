#!/usr/bin/env python3
"""Static validator for a concrete, still-preexecution native-GTS E2/E3 plan.

It reads JSON only.  It does not build GTS, access a GPU, read vector data, or
run a trace.  A source-only template is intentionally rejected as nonexecutable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BASELINE = "b589f0c163b20e5344c9984eb58a3d84c37061d6"
VARIANTS = ["certified_sidecar", "exact_delta"]
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_LINEAGE = ("synthetic", "gtsq", "local_cell", "implicit_catalog", "datasets sample")

class PlanError(ValueError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise PlanError(message)

def obj(v: Any, field: str) -> dict[str, Any]:
    require(isinstance(v, dict), f"{field}: object required")
    return v

def arr(v: Any, field: str) -> list[Any]:
    require(isinstance(v, list), f"{field}: list required")
    return v

def text(v: Any, field: str) -> str:
    require(isinstance(v, str) and v, f"{field}: nonempty string required")
    return v

def integer(v: Any, field: str, minimum: int = 0) -> int:
    require(isinstance(v, int) and not isinstance(v, bool) and v >= minimum, f"{field}: integer >= {minimum} required")
    return v

def sha(v: Any, field: str) -> str:
    s = text(v, field)
    require(SHA256.fullmatch(s) is not None, f"{field}: 64 lowercase hex required")
    return s

def runid(v: Any, field: str) -> str:
    s = text(v, field)
    require(RUN_ID.fullmatch(s) is not None, f"{field}: fresh 32 lowercase hex required")
    return s

def no_forbidden_lineage(v: Any, path: str = "root") -> None:
    if isinstance(v, str):
        low = v.lower()
        require(not any(token in low for token in FORBIDDEN_LINEAGE), f"{path}: forbidden synthetic/local-cell lineage")
    elif isinstance(v, dict):
        for k, item in v.items():
            no_forbidden_lineage(item, f"{path}.{k}")
    elif isinstance(v, list):
        for i, item in enumerate(v):
            no_forbidden_lineage(item, f"{path}[{i}]")

def validate_repetitions(rows: Any, field: str) -> None:
    rows = arr(rows, field)
    require(len(rows) >= 5, f"{field}: at least five repetitions")
    starts: list[str] = []
    seen: set[str] = set()
    for expected, row in enumerate(rows, start=1):
        row = obj(row, f"{field}[{expected - 1}]")
        require(row.get("replication") == expected, f"{field}[{expected - 1}]: contiguous replication index")
        order = arr(row.get("order"), f"{field}[{expected - 1}].order")
        require(len(order) == 2 and set(order) == set(VARIANTS), f"{field}[{expected - 1}]: must contain one of each variant")
        starts.append(order[0])
        ids = arr(row.get("run_ids"), f"{field}[{expected - 1}].run_ids")
        require(len(ids) == 2, f"{field}[{expected - 1}]: two fresh run IDs required")
        for i, value in enumerate(ids):
            value = runid(value, f"{field}[{expected - 1}].run_ids[{i}]")
            require(value not in seen, f"{field}: run ID reused")
            seen.add(value)
    require(abs(starts.count(VARIANTS[0]) - starts.count(VARIANTS[1])) <= 1, f"{field}: unbalanced alternating start order")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        plan = obj(plan, "plan")
        require(plan.get("schema") == "tide.native-gts-e2e3-campaign.v1", "schema mismatch")
        require(plan.get("template_only") is False, "source-only template is not executable")
        require(plan.get("status") == "FROZEN_PREEXECUTION", "plan must remain preexecution")
        campaign_id = runid(plan.get("campaign_id"), "campaign_id")

        baseline = obj(plan.get("baseline"), "baseline")
        require(baseline.get("git_commit") == BASELINE, "wrong clean baseline")
        require(baseline.get("dirty_original_gts_permitted") is False, "dirty original GTS is forbidden")
        require(baseline.get("isolated_clean_derivative_required") is True, "isolated derivative required")

        gate1 = obj(plan.get("native_gate1"), "native_gate1")
        require(gate1.get("status") == "QUALIFIED", "native Gate1 qualification required")
        for key in ("gate1_report_sha256", "topology_capture_sha256", "source_sha256", "binary_sha256"):
            sha(gate1.get(key), f"native_gate1.{key}")
        require(gate1.get("actual_post_build_capture") is True, "actual synchronized GTS capture required")
        require(gate1.get("qualified_native_receipt") is True, "actual qualified native receipt required")
        require(gate1.get("legacy_search_paths_not_used") is True, "legacy GTS search path cannot qualify")

        metric = obj(plan.get("metric"), "metric")
        require(metric.get("contract") == "integer_l2_squared_v1", "metric contract")
        require(metric.get("coordinate_domain") == "integer_0_to_255_inclusive", "coordinate domain")
        require(metric.get("tie_order") == ["distance_key", "stable_id"], "tie order")
        integer(metric.get("k"), "metric.k", 1)

        data = obj(plan.get("external_data"), "external_data")
        require(data.get("kind") == "external_integer_vector_catalog", "external catalog required")
        require(data.get("forbid_prior_synthetic_or_local_cell_lineage") is True, "synthetic/local-cell rejection required")
        for key in ("dataset_manifest_sha256", "catalog_sha256", "trace_sha256", "query_manifest_sha256"):
            sha(data.get(key), f"external_data.{key}")
        require(data.get("coordinates_verified_exact_integer_0_255") is True, "exact input verification required")
        require(data.get("loader_preserves_exact_coordinates") is True, "loader bridge proof required")

        policy = obj(plan.get("common_policy"), "common_policy")
        require(policy.get("overlay_accounting_expression") == "live_sidecars + live_delta", "shared accounting expression")
        integer(policy.get("C_ov"), "common_policy.C_ov", 1)
        integer(policy.get("sidecar_cap_L"), "common_policy.sidecar_cap_L", 1)
        require(policy.get("base_delete_barrier") == "synchronous_rebuild_or_fail_stop", "synchronous base-delete barrier")
        require(policy.get("delta_only_B_forbidden") is True and "delta_only_B" not in policy, "Delta-only B forbidden")
        require(policy.get("native_envelopes_immutable_online") is True, "online envelope mutation forbidden")
        text(policy.get("predeclared_trigger_ordering"), "common_policy.predeclared_trigger_ordering")

        require(plan.get("variants") == VARIANTS, "primary variant pair/order")
        cs = obj(plan.get("certified_sidecar_contract"), "certified_sidecar_contract")
        for key in ("strict_all_actual_ancestor_envelopes", "same_native_predicate_as_capture", "actual_receipt_indexed_sidecar_scan", "no_nearest_leaf_fallback", "certificate_rejection_routes_to_exact_delta_or_fail_stop"):
            require(cs.get(key) is True, f"certified_sidecar_contract.{key} must be true")

        e2 = obj(plan.get("e2"), "e2")
        sha(e2.get("trace_sha256"), "e2.trace_sha256")
        require(e2.get("full_live_oracle_coverage") == "every_query", "E2 oracle coverage")
        validate_repetitions(e2.get("repetitions"), "e2.repetitions")

        e3 = arr(plan.get("e3_scale_points"), "e3_scale_points")
        values = [integer(obj(row, f"e3_scale_points[{i}]").get("initial_objects"), f"e3_scale_points[{i}].initial_objects", 1) for i, row in enumerate(e3)]
        require(any(n >= 100_000 for n in values) and any(n >= 1_000_000 for n in values), "E3 requires >=100K and >=1M")
        for i, row in enumerate(e3):
            row = obj(row, f"e3_scale_points[{i}]")
            integer(row.get("query_count_per_snapshot"), f"e3_scale_points[{i}].query_count_per_snapshot", 1000)
            require(row.get("full_live_oracle_coverage") == "every_query", f"e3_scale_points[{i}]: oracle coverage")
            snaps = arr(row.get("turnover_snapshots"), f"e3_scale_points[{i}].turnover_snapshots")
            tags = [obj(x, "snapshot").get("tag") for x in snaps]
            require(tags == ["initial", "insert_turnover", "mixed_overlay_turnover", "post_base_delete_rebuild"], f"e3_scale_points[{i}]: required snapshot order")
            for j, snap in enumerate(snaps):
                snap = obj(snap, f"e3_scale_points[{i}].turnover_snapshots[{j}]")
                for key in ("insert_ratio", "delete_ratio"):
                    value = snap.get(key)
                    require(isinstance(value, (int, float)) and 0 <= value <= 1, f"{key}: ratio in [0,1]")
                require(isinstance(snap.get("amortize_rebuild"), bool), "amortize_rebuild boolean required")
                text(snap.get("amortization_window"), "amortization_window")
            validate_repetitions(row.get("repetitions"), f"e3_scale_points[{i}].repetitions")

        independent = obj(plan.get("independent_validation"), "independent_validation")
        hashes = []
        for key in ("raw_vector_oracle_source_sha256", "topology_certificate_replayer_source_sha256", "receipt_overlay_auditor_source_sha256"):
            hashes.append(sha(independent.get(key), f"independent_validation.{key}"))
        require(len(set(hashes)) == 3, "independent replayer source hashes must differ")
        require(independent.get("independent_reproduction_required") is True, "independent reproduction required")
        reproduction = obj(independent.get("reproduction"), "independent_validation.reproduction")
        require(reproduction.get("fresh_clean_clone") is True and reproduction.get("fresh_binary") is True and reproduction.get("fresh_run_roots") is True, "fresh independent reproduction boundary")

        no_forbidden_lineage(data, "external_data")
        print(json.dumps({"status": "PASS_PLAN_ONLY", "campaign_id": campaign_id, "nonclaim": "No build, GPU, data access, trace execution, timing, or result validation occurred."}, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, PlanError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, sort_keys=True))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())

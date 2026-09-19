#!/usr/bin/env python3
"""CPU-only pre-GPU contract for Fable5 Safe-C1 vs buffer-only.

This is intentionally a trace/oracle parity harness, not a CUDA runner and not
an experiment result.  It fixes one deterministic update trace, replays it
under two policies, and checks that every query is evaluated against the same
full-active-set exact L2 oracle.  A later GPU runner must consume this exact
manifest/trace digest and derive the Safe-C1 certificate from the frozen native
GTS sibling boundaries rather than from this CPU fixture's labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fable5-safe-c1-matched-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def exact_topk(vectors: dict[int, list[float]], query: list[float], active: set[int], k: int) -> list[dict[str, Any]]:
    if len(active) < k:
        fail("fewer_live_ids_than_k")
    rows: list[tuple[float, int]] = []
    for ident in sorted(active):
        value = vectors.get(ident)
        if value is None:
            fail(f"missing_vector:{ident}")
        squared = sum((a - b) ** 2 for a, b in zip(value, query))
        rows.append((squared, ident))
    rows.sort()
    # A GPU result lane must use a tie-free fixture; ordering by id is only a
    # deterministic diagnostic fallback, never a hidden tie-resolution claim.
    for left, right in zip(rows, rows[1:]):
        if math.isclose(left[0], right[0], rel_tol=0.0, abs_tol=1e-12):
            fail(f"fixture_distance_tie:{left[1]}:{right[1]}")
    return [{"id": ident, "squared_l2": round(squared, 12)} for squared, ident in rows[:k]]


class PolicyState:
    def __init__(self, policy: str, base_ids: set[int], sidecar_capacity: int) -> None:
        self.policy = policy
        self.base_ids = set(base_ids)
        self.sidecars: dict[str, list[int]] = {}
        self.delta: list[int] = []
        self.deleted: set[int] = set()
        self.rebuild_pending = False
        self.sidecar_capacity = sidecar_capacity

    def active_ids(self) -> set[int]:
        return set(self.base_ids).union(*[set(ids) for ids in self.sidecars.values()] or [set()]).union(self.delta).difference(self.deleted)

    def placement(self, ident: int) -> str:
        if ident in self.deleted:
            return "deleted"
        if ident in self.base_ids:
            return "base"
        if any(ident in values for values in self.sidecars.values()):
            return "direct"
        if ident in self.delta:
            return "delta"
        return "absent"

    def insert(self, event: dict[str, Any]) -> str:
        if self.rebuild_pending:
            fail("insert_while_rebuild_pending")
        ident = int(event["id"])
        if ident in self.active_ids() or ident in self.deleted:
            fail(f"nonfresh_insert:{ident}")
        certificate = event.get("certificate", {})
        if self.policy == "buffer_only":
            self.delta.append(ident)
            return "delta_buffer_only"
        if not certificate.get("eligible", False):
            self.delta.append(ident)
            return "delta_certificate_reject"
        leaf = certificate.get("leaf")
        if not isinstance(leaf, str) or not leaf:
            fail("eligible_certificate_without_leaf")
        direct = self.sidecars.setdefault(leaf, [])
        if len(direct) >= self.sidecar_capacity:
            self.delta.append(ident)
            return "delta_capacity"
        direct.append(ident)
        return "direct"

    def delete(self, ident: int) -> str:
        if self.rebuild_pending:
            fail("delete_while_rebuild_pending")
        prior = self.placement(ident)
        if prior == "absent" or prior == "deleted":
            fail(f"delete_nonactive:{ident}")
        if prior == "base":
            self.base_ids.remove(ident)
            self.deleted.add(ident)
            self.rebuild_pending = True
            return "base_delete_rebuild_barrier"
        if prior == "direct":
            for values in self.sidecars.values():
                if ident in values:
                    values.remove(ident)
                    break
            self.deleted.add(ident)
            return "direct_delete"
        self.delta.remove(ident)
        self.deleted.add(ident)
        return "delta_delete"

    def rebuild(self) -> str:
        if not self.rebuild_pending:
            fail("rebuild_without_base_delete")
        live = self.active_ids()
        self.base_ids = live
        self.sidecars = {}
        self.delta = []
        self.rebuild_pending = False
        return "immediate_rebuild_complete"

    def query_candidates(self) -> set[int]:
        if self.rebuild_pending:
            fail("query_while_rebuild_pending")
        # The CPU contract uses the full active set to prove oracle parity.  It
        # does NOT claim a native GPU traversal receipt was executed here.
        candidate = set(self.base_ids)
        for values in self.sidecars.values():
            candidate.update(values)
        candidate.update(self.delta)
        candidate.difference_update(self.deleted)
        return candidate


def normalized_vectors(raw: dict[str, list[float]], dimension: int, integer_ids: bool) -> dict[Any, list[float]]:
    parsed: dict[Any, list[float]] = {}
    for key, value in raw.items():
        ident: Any = int(key) if integer_ids else str(key)
        if (integer_ids and ident < 0) or len(value) != dimension or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in value):
            fail(f"invalid_vector:{key}")
        parsed[ident] = [float(x) for x in value]
    return parsed


def replay_policy(manifest: dict[str, Any], policy: str) -> dict[str, Any]:
    trace = manifest["trace"]
    dimension = int(trace["dimension"])
    k = int(trace["k"])
    vectors = normalized_vectors({**trace["base_vectors"], **trace["insert_vectors"]}, dimension, True)
    queries = normalized_vectors(trace["query_vectors"], dimension, False)
    state = PolicyState(policy, set(int(key) for key in trace["base_vectors"]), int(trace["sidecar_capacity"]))
    records: list[dict[str, Any]] = []
    oracle_fingerprints: list[str] = []
    for expected_index, event in enumerate(trace["operations"]):
        if int(event.get("op_index", -1)) != expected_index:
            fail(f"noncontiguous_op_index:{expected_index}")
        operation = event.get("op")
        record: dict[str, Any] = {"op_index": expected_index, "op": operation}
        if operation == "insert":
            record.update({"id": int(event["id"]), "route": state.insert(event)})
        elif operation == "delete":
            record.update({"id": int(event["id"]), "route": state.delete(int(event["id"]))})
        elif operation == "rebuild":
            record["route"] = state.rebuild()
        elif operation == "query":
            query_id = str(event["query_id"])
            if query_id not in queries:
                fail(f"unknown_query:{query_id}")
            candidates = state.query_candidates()
            active = state.active_ids()
            if candidates != active:
                fail(f"candidate_active_set_mismatch:op{expected_index}")
            observed = exact_topk(vectors, queries[query_id], candidates, k)
            oracle = exact_topk(vectors, queries[query_id], active, k)
            if observed != oracle:
                fail(f"oracle_mismatch:op{expected_index}")
            fingerprint = sha256_value(oracle)
            oracle_fingerprints.append(fingerprint)
            record.update({
                "query_id": query_id,
                "active_count": len(active),
                "tier_counts": {"base": len(state.base_ids), "sidecar": sum(len(v) for v in state.sidecars.values()), "delta": len(state.delta)},
                "oracle_full_active_set_checked": True,
                "topk": oracle,
                "oracle_sha256": fingerprint,
            })
        else:
            fail(f"unknown_op:{operation}")
        records.append(record)
    if state.rebuild_pending:
        fail("trace_ended_with_rebuild_pending")
    return {
        "policy": policy,
        "records": records,
        "query_oracle_sha256": oracle_fingerprints,
        "final_active_ids": sorted(state.active_ids()),
        "final_tier_counts": {"base": len(state.base_ids), "sidecar": sum(len(v) for v in state.sidecars.values()), "delta": len(state.delta)},
    }


def static_semantic_checks(root: Path) -> tuple[dict[str, bool], list[str]]:
    source = root / "src/safe_c1_dynamic_gts.cu"
    certificate = root / "include/safe_c1_search_native_routing_certificate.hpp"
    runner = root / "tools/fable5_matched_cpu_contract_v1.py"
    checks: dict[str, bool] = {}
    errors: list[str] = []
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    cert = certificate.read_text(encoding="utf-8") if certificate.is_file() else ""
    own = runner.read_text(encoding="utf-8") if runner.is_file() else ""
    required_source = {
        "external_sidecar_not_legacy_tn": "No method mutates a legacy TN or id_list.",
        "native_base_traversal_receipt": "run_gts_base_topk_with_receipt",
        "receipt_selected_sidecars": "sidecar_candidates_for(base.visited_leaf_ids)",
        "global_exact_delta_merge": "state.delta_ids().begin()",
        "full_active_set_oracle": "exact_active_oracle",
        "base_delete_immediate_rebuild_barrier": "base deletion requires immediate REBUILD",
    }
    required_certificate = {
        "strict_sibling_upper_boundary": "distance < next_min - epsilon_",
        "strict_lower_boundary": "distance > node.min_distance + epsilon_",
        "fail_closed_no_unique_match": "if (match != -1) return -1;",
        "no_max_distance_admission": "Deliberately no max-distance field is used",
    }
    for name, needle in required_source.items():
        ok = needle in text
        checks[name] = ok
        if not ok:
            errors.append("missing_source_semantic:" + name)
    for name, needle in required_certificate.items():
        ok = needle in cert
        checks[name] = ok
        if not ok:
            errors.append("missing_certificate_semantic:" + name)
    # This harness itself must stay CPU-only.  Do not inspect or invoke archival
    # GPU guards.  The only device-selection mechanism that could accidentally
    # turn this script into a GPU launcher must not occur here.
    token = "CUDA_" + "VISIBLE_DEVICES"
    ok = token not in own
    checks["runner_omits_cuda_device_selection"] = ok
    if not ok:
        errors.append("cpu_runner_forbidden_token:" + token)
    return checks, errors


def build_tools() -> dict[str, Any]:
    # Compiler discovery/version only; no compilation and no CUDA application
    # binary execution is performed.
    result: dict[str, Any] = {}
    for executable, args in (("python3", ["--version"]), ("nvcc", ["--version"])):
        path = shutil.which(executable)
        entry: dict[str, Any] = {"path": path, "available": path is not None}
        if path is not None:
            completed = subprocess.run([path, *args], text=True, capture_output=True, check=False)
            entry.update({"returncode": completed.returncode, "version_output": (completed.stdout + completed.stderr).strip().splitlines()[:4]})
        result[executable] = entry
    return result


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append("wrong_schema")
    if manifest.get("execution") != {"gpu_used": False, "cuda_binary_executed": False, "nvidia_smi_called": False}:
        errors.append("execution_boundary_not_cpu_only")
    if manifest.get("semantics", {}).get("rebuild_policy") != "base_delete_immediate":
        errors.append("rebuild_policy_not_explicit_v5_semantics")
    if set(manifest.get("policies", {})) != {"safe_c1", "buffer_only"}:
        errors.append("policy_set_not_matched_pair")
    trace = manifest.get("trace", {})
    required_ops = {"insert", "delete", "query", "rebuild"}
    seen_ops = {entry.get("op") for entry in trace.get("operations", [])}
    if not required_ops.issubset(seen_ops):
        errors.append("required_trace_operation_missing")
    coverage = manifest.get("required_coverage", {})
    for name in ("safe_direct", "capacity_fallback", "certificate_reject", "direct_delete", "delta_delete", "base_delete", "immediate_rebuild", "queries_before_and_after_rebuild"):
        if name not in coverage:
            errors.append("coverage_missing:" + name)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/fable5_matched_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.out.resolve()
    report: dict[str, Any] = {
        "schema": "fable5-safe-c1-matched-cpu-static-audit-v1",
        "root": str(ROOT),
        "manifest": str(manifest_path),
        "gpu_used": False,
        "cuda_binary_executed": False,
        "nvidia_smi_called": False,
        "scope": "CPU-only trace/oracle parity and static semantic audit; no GPU traversal, timing, throughput, recall, or deployment claim.",
        "errors": [],
    }
    try:
        if ROOT.name != "safe_c1_fable5_matched_v1":
            fail("runner_not_in_isolated_fable5_root")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["manifest_sha256"] = sha256_file(manifest_path)
        report["trace_sha256"] = sha256_value(manifest.get("trace"))
        report["source_sha256"] = sha256_file(ROOT / "src/safe_c1_dynamic_gts.cu")
        report["manifest_errors"] = validate_manifest(manifest)
        checks, semantic_errors = static_semantic_checks(ROOT)
        report["static_semantic_checks"] = checks
        report["build_tools"] = build_tools()
        for tool_name, tool_info in report["build_tools"].items():
            if not tool_info.get("available") or tool_info.get("returncode") != 0:
                report["errors"].append("build_tool_unavailable_or_failed:" + tool_name)
        safe = replay_policy(manifest, "safe_c1")
        buffer = replay_policy(manifest, "buffer_only")
        report["policies"] = {"safe_c1": safe, "buffer_only": buffer}
        report["cross_policy_checks"] = {
            "same_trace_sha256": True,
            "same_query_oracle_sha256": safe["query_oracle_sha256"] == buffer["query_oracle_sha256"],
            "same_final_active_ids": safe["final_active_ids"] == buffer["final_active_ids"],
            "safe_c1_has_direct_route": any(row.get("route") == "direct" for row in safe["records"]),
            "buffer_only_forces_all_inserts_to_delta": all(row.get("route", "").startswith("delta_buffer_only") for row in buffer["records"] if row.get("op") == "insert"),
        }
        if not report["cross_policy_checks"]["same_query_oracle_sha256"]:
            report["errors"].append("matched_policies_do_not_share_exact_oracle_answers")
        if not report["cross_policy_checks"]["same_final_active_ids"]:
            report["errors"].append("matched_policies_final_active_set_differs")
        if not report["cross_policy_checks"]["safe_c1_has_direct_route"]:
            report["errors"].append("safe_c1_fixture_has_no_direct_route")
        if not report["cross_policy_checks"]["buffer_only_forces_all_inserts_to_delta"]:
            report["errors"].append("buffer_only_fixture_not_forced_delta")
        report["errors"].extend(report["manifest_errors"])
        report["errors"].extend(semantic_errors)
        report["gpu_preconditions_not_yet_satisfied"] = [
            "The C++ GPU runner does not yet consume this manifest/trace digest.",
            "The CPU fixture labels certificates; the GPU runner must derive them from frozen native sibling boundaries and emit a receipt.",
            "The matched GPU lane must implement the same base-delete/immediate-rebuild policy for both policies, or paper text must be revised before claim expansion.",
            "GPU execution requires a separately reviewed NVML-only guarded launcher and fresh user authorization; this script must never become that launcher.",
        ]
        report["status"] = "PASS_CPU_ONLY_FABLE5_MATCHED_CONTRACT" if not report["errors"] else "FAIL_CPU_ONLY_FABLE5_MATCHED_CONTRACT"
    except Exception as exc:  # fail closed and leave an inspectable report
        report["errors"].append(f"exception:{type(exc).__name__}:{exc}")
        report["status"] = "FAIL_CPU_ONLY_FABLE5_MATCHED_CONTRACT"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

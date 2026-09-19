#!/usr/bin/env python3
"""CPU-only regression fixture for the v2 bundle generator/preflight/guard contract."""
from __future__ import annotations

from array import array
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from safe_c1_v2_bundle_contract import (
    BundleContractError,
    OP_KNN,
    TraceEvent,
    TraceHeader,
    validate_tie_free_witness,
    write_i16,
    write_i32_le,
    write_trace,
)


def expect_tie_abort(pool: array, queries: array, header: TraceHeader, token: str) -> None:
    try:
        validate_tie_free_witness(pool, queries, header, [])
    except BundleContractError as exc:
        if token in str(exc):
            return
        raise RuntimeError(f"wrong static-prefix abort: {exc}") from exc
    raise RuntimeError(f"expected static-prefix abort containing {token}")


def float_collision_pool() -> array:
    dimension = 128
    values = array("h", [0]) * (3 * dimension)
    for axis in range(dimension - 1):
        values[axis] = 32767
        values[dimension + axis] = 32767
    values[2 * dimension - 1] = 1
    return values


def check_static_top_k_plus_one_contract() -> None:
    # Far exact tie after positions [0, k] is permitted.
    far_exact = validate_tie_free_witness(
        array("h", [1, 2, 3, 10, -10]), array("h", [0]),
        TraceHeader(1, 5, 0, 5, 1, 2, 0.0, 0), [],
    )
    if far_exact["static_checks"][0].get("static_prefix_count") != 3 or \
            far_exact["static_base_policy"] != \
            "first min(k+1,base_n) exact squared-L2 and modeled GTS float-L2 keys pairwise distinct; first-k ID order agrees":
        raise RuntimeError("static prefix pass did not record the bounded policy")
    expect_tie_abort(
        array("h", [1, -1, 5, 10]), array("h", [0]),
        TraceHeader(1, 4, 0, 4, 1, 2, 0.0, 0),
        "static_base_top_k_plus_one_exact_squared_l2_tie",
    )
    collision = float_collision_pool()
    query = array("h", [0]) * 128
    # The modeled-float collision is at ranks 1/2, outside k=1's prefix.
    validate_tie_free_witness(collision, query, TraceHeader(128, 3, 0, 3, 1, 1, 0.0, 0), [])
    expect_tie_abort(
        collision, query, TraceHeader(128, 3, 0, 3, 1, 2, 0.0, 0),
        "static_base_top_k_plus_one_modeled_gts_float_key_tie",
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    generator = root / "tools/make_g1_v2_bundle.py"
    preflight = root / "tools/preflight_g1_v2_bundle.py"
    fixture_name = f".cpu_fixture_v2_{uuid.uuid4().hex}"
    fixture_out = root / "bundles" / fixture_name
    work = Path(tempfile.mkdtemp(prefix="safe_c1_v2_bundle_fixture_"))
    try:
        check_static_top_k_plus_one_contract()
        source = work / "source"
        source.mkdir()
        # Base [0,5,11], reservoir [2,8], D=1, K=2. Query pool row 3 has
        # static keys 4,9,81 and dynamic keys 0,4,9,81 after its insert.
        pool = array("h", [0, 5, 11, 2, 8])
        write_i16(source / "pool.i16", pool)
        write_i32_le(source / "initial_base_stable_ids.i32", [0, 1, 2])
        write_i32_le(source / "stable_id_to_pool_row.i32", [0, 1, 2, 3, 4])
        source_header = TraceHeader(1, 3, 2, 5, 1, 2, 0.0, 1)
        write_trace(source / "trace.e1gtrc", source_header, [TraceEvent(0, OP_KNN, 0)])
        spec = {
            "schema": "safe-c1-g1-v2-bundle-spec",
            "query_stable_ids": [3],
            "leaf_capacity": 2,
            "events": [
                {"op": "insert", "argument": 3},
                {"op": "knn", "argument": 0},
            ],
        }
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        outside = work / "outside_v2_root"
        rejected_writer = subprocess.run(
            [sys.executable, str(generator), "--source-bundle", str(source), "--spec", str(spec_path), "--out", str(outside)],
            text=True, capture_output=True,
        )
        if rejected_writer.returncode == 0 or "v2 bundle output must be under this v2 root" not in (rejected_writer.stderr + rejected_writer.stdout):
            raise RuntimeError("v2 bundle generator did not reject an output outside its canonical v2 root")

        make = subprocess.run(
            [sys.executable, str(generator), "--source-bundle", str(source), "--spec", str(spec_path), "--out", str(fixture_out)],
            text=True, capture_output=True,
        )
        if make.returncode != 0:
            raise RuntimeError(f"generator failed: {make.stderr}")

        pending_out = work / "pending.json"
        pending = subprocess.run(
            [sys.executable, str(preflight), "--root", str(root), "--bundle", str(fixture_out), "--out", str(pending_out), "--allow-pending-selection"],
            text=True, capture_output=True,
        )
        pending_doc = json.loads(pending_out.read_text())
        if pending.returncode != 0 or pending_doc.get("status") != "PASS_CPU_ONLY_BUNDLE_PREFLIGHT_PENDING_SELECTION":
            raise RuntimeError(f"pending preflight failed: {pending.stdout} {pending.stderr} {pending_doc}")

        wrong_root_out = work / "wrong_root.json"
        wrong_root = work / "not_the_v2_root"
        wrong_root.mkdir()
        wrong = subprocess.run(
            [sys.executable, str(preflight), "--root", str(wrong_root), "--bundle", str(fixture_out),
             "--out", str(wrong_root_out), "--allow-pending-selection"],
            text=True, capture_output=True,
        )
        wrong_doc = json.loads(wrong_root_out.read_text())
        if wrong.returncode == 0 or "root_must_match_this_v2_preflight_source_root" not in wrong_doc.get("errors", []):
            raise RuntimeError(f"preflight accepted a foreign root: {wrong.stdout} {wrong.stderr} {wrong_doc}")

        normal_out = work / "normal.json"
        normal = subprocess.run(
            [sys.executable, str(preflight), "--root", str(root), "--bundle", str(fixture_out), "--out", str(normal_out)],
            text=True, capture_output=True,
        )
        normal_doc = json.loads(normal_out.read_text())
        if normal.returncode == 0 or normal_doc.get("status") != "FAIL_CPU_ONLY_BUNDLE_PREFLIGHT" or \
                "candidate_selection_pending_not_execution_ready" not in normal_doc.get("errors", []):
            raise RuntimeError(f"normal guard did not refuse pending bundle: {normal.stdout} {normal.stderr} {normal_doc}")

        manifest = json.loads((fixture_out / "manifest.json").read_text())
        if manifest.get("schema") != "safe-c1-g1-v2-tie-free-bundle-manifest" or \
                manifest.get("candidate_selection", {}).get("status") != "PENDING_V2_CERTIFICATE_SELECTION" or \
                manifest.get("tie_free_witness_domain", {}).get("status") != "PASS_CPU_ONLY_BOUNDED_TIE_WITNESS" or \
                manifest.get("stable_id_layout") != "explicit_identity_only_current_v2_runner":
            raise RuntimeError("v2 manifest omitted certificate/tie/pending-selection contract")
        print("PASS v2_static_top_k_plus_one_tie_contract")
        print("PASS v2_bundle_generator_manifest_certificate_and_tie_domain")
        print("PASS v2_bundle_preflight_allows_cpu_pending_but_refuses_execution")
        print("PASS v2_bundle_root_boundary_rejects_foreign_writer_and_preflight_root")
        print("PASS safe_c1_v2_bundle_preflight CPU-only")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(fixture_out, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

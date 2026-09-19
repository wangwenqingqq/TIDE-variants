#!/usr/bin/env python3
"""Read-only static audit for the Safe-C2 v3 speculative-fallback skeleton.

This tool never invokes the CUDA binary, CUDA management commands, or a GPU.
It also verifies the v2 immutable provenance hashes before accepting v3 source.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "safe_c2_speculative_fallback_v3.json"
SOURCE = ROOT / "src" / "gts_speculative_fallback_v3_sift1m.cu"
HEADER = ROOT / "include" / "search_v3.cuh"
V2_WITNESS = ROOT / "provenance" / "v2_immutable_source_and_sealed_test.sha256"
FINAL_MANIFEST = ROOT / "inputs" / "sift_learn_compact10k_v1" / "workload_manifest.json"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle!r}")


def parse_v2_witness() -> int:
    count = 0
    for raw in V2_WITNESS.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            fail(f"malformed v2 witness line: {raw!r}")
        expected, raw_path = parts
        path = Path(raw_path)
        if not path.is_absolute() or "c2_interval_bound_safe_v2_tiefree" not in raw_path:
            fail(f"witness does not point at immutable v2: {raw_path}")
        if not path.is_file():
            fail(f"immutable v2 witness path missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            fail(f"v2 immutability violation: {path}; expected {expected}, saw {observed}")
        count += 1
    if count < 8:
        fail("v2 immutability witness unexpectedly incomplete")
    return count


def actual_vector_path_slices(header: str) -> tuple[str, str]:
    v3_start = header.find("void searchIndexKnnV3Speculative(")
    v2_marker = "void searchIndexKnnV2(short *data_d, TN *node_list, int *id_list, int *max_node_num, float *query_data,"
    v2_start = header.rfind(v2_marker, 0, v3_start)
    if v2_start < 0 or v3_start < 0:
        fail("cannot locate actual v2 fallback / v3 speculative vector+IDs paths")
    return header[v2_start:v3_start], header[v3_start:]


def verify_fail_closed_actual_paths(header: str) -> dict[str, int]:
    baseline, speculative = actual_vector_path_slices(header)
    counts: dict[str, int] = {}
    for label, section in (("v2_gamma_one_fallback", baseline), ("v3_speculative", speculative)):
        if "cudaStatus" in section or "fprintf(stderr" in section:
            fail(f"{label} retains nonfatal CUDA error handling")
        sync_count = section.count("CHECK(cudaDeviceSynchronize())")
        error_count = section.count("CHECK(cudaGetLastError())")
        if sync_count < 12 or error_count < 12 or sync_count != error_count:
            fail(f"{label} lacks complete fail-closed synchronization/error checks")
        if section.count("CHECK(cudaMemGetInfo(&avail, &total))") != 1:
            fail(f"{label} lacks fail-closed cudaMemGetInfo")
        # This catches a bare post-kernel sync that can print-and-continue; the
        # expected form is only CHECK(cudaDeviceSynchronize()).
        if "cudaDeviceSynchronize();" in section:
            fail(f"{label} retains a bare cudaDeviceSynchronize call")
        counts[label] = sync_count
    return counts


def verify_final_workload_manifest() -> dict[str, object]:
    if not FINAL_MANIFEST.is_file():
        return {"present": False}
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(FINAL_MANIFEST) != "72d9b0732784d8f0f5b77d564a65d9a9294feab9781f0a098131fe74129fbf39":
        fail("active final workload manifest SHA-256 mismatch")
    if manifest.get("schema") != "gts-v3-compact-learn-workload-v1":
        fail("final workload manifest schema mismatch")
    if manifest.get("sealed_v2_test_forbidden") is not True:
        fail("final workload manifest does not forbid v2 sealed test")
    sealed_v2 = "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce"
    if manifest.get("v2_sealed_test_ids_sha256_forbidden") != sealed_v2:
        fail("final workload manifest v2 sealed-test digest mismatch")
    bindings = (
        ("base_fvecs_path", "base_fvecs_sha256"),
        ("query_fvecs_path", "query_fvecs_sha256"),
        ("groundtruth_ivecs_path", "groundtruth_ivecs_sha256"),
        ("mapping_path", "mapping_sha256"),
        ("calibration_ids_path", "calibration_ids_sha256"),
        ("validation_ids_path", "validation_ids_sha256"),
        ("test_ids_path", "test_ids_sha256"),
    )
    verified: dict[str, str] = {}
    for path_key, digest_key in bindings:
        raw_path = manifest.get(path_key)
        expected = manifest.get(digest_key)
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            fail(f"final workload binding missing {path_key}/{digest_key}")
        path = Path(raw_path)
        if not path.is_file():
            fail(f"final workload binding path missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"final workload binding digest mismatch: {path_key}")
        verified[digest_key] = actual
    for key in ("calibration_ids_sha256", "validation_ids_sha256", "test_ids_sha256"):
        if verified[key] == sealed_v2:
            fail(f"final workload stage IDs reuse the sealed v2 test: {key}")
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        fail("final workload lacks stage metadata")
    expected_names = {"calibration": "calibration_ids_sha256", "validation": "validation_ids_sha256", "sealed_test": "test_ids_sha256"}
    stage_counts: dict[str, int] = {}
    for stage_name, digest_key in expected_names.items():
        stage = stages.get(stage_name)
        if not isinstance(stage, dict) or stage.get("ids_sha256") != verified[digest_key]:
            fail(f"stage metadata digest mismatch: {stage_name}")
        ids_path = Path(str(stage.get("ids_path", "")))
        lines = [line for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if stage.get("eligible_count") != len(lines) or len(lines) == 0:
            fail(f"stage eligibility count mismatch: {stage_name}")
        values = [int(line) for line in lines]
        if len(set(values)) != len(values) or any(value < 0 or value >= 10000 for value in values):
            fail(f"invalid compact IDs in stage: {stage_name}")
        stage_counts[stage_name] = len(values)
    if manifest.get("status") != "READY_FOR_C2_V3_CALIBRATION_VALIDATION_AND_ONE_SEALED_TEST":
        fail("final workload is not explicitly ready under its frozen protocol")
    return {"present": True, "manifest_sha256": sha256_file(FINAL_MANIFEST), "stage_counts": stage_counts}


def main() -> int:
    if ROOT.name != "c2_speculative_fallback_v3":
        fail(f"unexpected v3 root: {ROOT}")
    for path in (PROTOCOL, SOURCE, HEADER, V2_WITNESS):
        if not path.is_file():
            fail(f"required v3 file missing: {path}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("schema") != "safe-c2-speculative-fallback-v3-protocol-v1":
        fail("unexpected v3 protocol schema")
    if protocol.get("write_boundary") != str(ROOT):
        fail("protocol write boundary is not the v3 root")
    v2 = protocol.get("v2_immutability", {})
    if v2.get("v2_sealed_test_ids_sha256") != "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce":
        fail("protocol v2 sealed test digest mismatch")
    if "forbidden" not in str(v2.get("v2_sealed_test_policy", "")).lower():
        fail("protocol does not forbid v2 sealed test reuse")
    workload = protocol.get("independent_workload", {})
    if workload.get("manifest_schema") != "gts-v3-compact-learn-workload-v1":
        fail("protocol workload manifest schema mismatch")
    required_digest_fields = {
        "base_fvecs_sha256", "query_fvecs_sha256", "groundtruth_ivecs_sha256",
        "mapping_sha256", "calibration_ids_sha256", "validation_ids_sha256", "test_ids_sha256",
    }
    if not required_digest_fields.issubset(set(workload.get("required_runner_verified_digests", []))):
        fail("protocol does not require all compact-workload digests")
    if "mapping_path" not in str(workload.get("mapping_binding", "")):
        fail("protocol does not require an actual local-to-original mapping path binding")
    if workload.get("final_manifest_sha256") != "72d9b0732784d8f0f5b77d564a65d9a9294feab9781f0a098131fe74129fbf39":
        fail("protocol is not bound to the active final workload manifest SHA-256")

    source = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    require(source, '#include "search_v3.cuh"', "v3 header include")
    for needle, label in (
        ("--workload-manifest", "manifest-required CLI"),
        ("validate_workload_binding(args);", "manifest validation before allocation"),
        ("kV2SealedTestIdsSha256", "copied-v2-test digest guard"),
        ("references_v2_sealed_root", "direct-v2-path guard"),
        ("sha256_hex_file", "runner SHA-256 binding"),
        ("mapping_path", "actual local-to-original mapping binding"),
        ("workload local-to-original mapping digest mismatch", "mapping digest fail-closed check"),
        ("v2 sealed-test digest binding mismatch", "manifest v2 digest binding"),
        ("v3 output directory must reside", "v3-only output boundary"),
        ("searchIndexKnnV3Speculative", "speculative traversal call"),
        ("upload_gamma(all_ones_gamma())", "all-gamma=1 fallback"),
        ("cudaMemcpyDeviceToDevice", "device result overwrite"),
        ("write_per_query_jsonl", "per-query JSONL persistence"),
        ("write_json_float", "valid JSON handling for invalid distance diagnostics"),
        ("assert_frozen_timed", "frozen snapshot checks"),
        ("final_guarded_repetitions", "final guarded gate"),
        ("raw_speculative_repetitions", "raw diagnostic separation"),
        ("guarded_query_pipeline_wall_ms", "guarded latency accounting"),
        ("host_final_pair_contract", "explicit final IDs+distances contract"),
        ("run.final_guarded_reference = evaluate_results", "final host pair validation"),
    ):
        require(source, needle, label)
    if '#include "search_v2.cuh"' in source:
        fail("v3 runner accidentally includes v2 header")
    if "PENDING_V2_TEST_IDS_SHA256" in source:
        fail("v3 runner retains an unenforced v2 test digest placeholder")
    for needle, label in (
        ("struct GammaOnlyPruneTrace", "per-query first-event trace"),
        ("unscaled_keep && !scaled_keep", "gamma-only event condition"),
        ("recordGammaOnlyPrune", "gamma-only event recorder"),
        ("searchIndexKnnV3Speculative", "diagnostic traversal overload"),
        ("gamma_only_pruned, gamma_only_trace", "diagnostic kernel parameters"),
    ):
        require(header, needle, label)
    if header.count("unscaled_keep && !scaled_keep") != 2:
        fail("gamma-only detection must occur in both warp and non-warp branches")
    if "should_keep" in header:
        fail("ambiguous old pruning variable remains in v3 header")
    fail_closed_counts = verify_fail_closed_actual_paths(header)

    witness_count = parse_v2_witness()
    workload_result = verify_final_workload_manifest()
    result = {
        "schema": "safe-c2-v3-static-audit-v1",
        "status": "PASS_STATIC_ONLY",
        "root": str(ROOT),
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "v2_witness_files_verified": witness_count,
        "actual_query_paths_fail_closed_sync_checks": fail_closed_counts,
        "v3_source_sha256": sha256_file(SOURCE),
        "v3_header_sha256": sha256_file(HEADER),
        "protocol_sha256": sha256_file(PROTOCOL),
        "final_workload": workload_result,
        "note": "CPU-only audit; no CUDA binary execution or GPU-management command occurred.",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # intentional compact fail-closed CLI
        print(f"SAFE-C2-V3 STATIC AUDIT FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Freeze the Safe-C2 v4 fresh, disjoint SIFT-learn selection before GT.

This script is deliberately CPU/file-I/O only.  It neither imports CUDA nor
opens any C2 result/gamma/trace/GT artifact.  It quarantines all v2 query
payloads and all v3 compact-query payloads/original IDs before ranking a new
pool with a committed pre-GT seed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

DIM = 128
RECORD_BYTES = 4 + DIM * 4
N_LEARN = 100_000
N_V2_QUERY = 10_000
N_V3_COMPACT = 10_000
RESERVED_DRIFT_PREFIX = 3_072
SPLITS = (
    ("qualification_canary", 1_024, "CANARY_QUALIFICATION_ONLY_NOT_FORMAL_EVIDENCE"),
    ("calibration", 2_500, "FORMAL_CALIBRATION_FRESH_ONLY"),
    ("validation", 2_500, "FORMAL_HELDOUT_VALIDATION_FRESH_ONLY"),
    ("sealed_test", 6_500, "FORMAL_SEALED_TEST_FRESH_ONLY"),
)
SEED_ASCII = b"safe-c2-v4-fresh-sift-learn-pre-gt-v1-20260728"
V2_SEALED_TEST_DIGEST_FORBIDDEN = "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce"

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
V3_ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3")
OUT = ROOT / "inputs/fresh_selection_pre_gt_v1"
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
V2_QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
LEARN = Path("/workspace/project/GTS/Datasets/sift1m/sift_learn.fvecs")
V3_MAPPING = V3_ROOT / "inputs/sift_learn_compact10k_v1/local_to_original_learn_id.tsv"
FORENSIC = ROOT / "provenance/v3_execution_forensic_boundary_v1.json"

EXPECTED = {
    "base_sha256": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
    "v2_query_sha256": "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
    "learn_sha256": "331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea",
    "v3_mapping_sha256": "a6794e510245482c4478814511066eaf6eb6b307063caf835b2d534fa5c7e9ed",
    "v3_forensic_sha256": "eb76c7f3955659c80af1a804ba55b1635eaf3f9b56eb8cb2ea199fe8369baa39",
}
TOTAL_SELECTED = sum(n for _, n, _ in SPLITS)
if TOTAL_SELECTED != 12_524:
    raise RuntimeError("split sizes are not the parent-approved 12,524")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def direct_regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"{label} is not a direct regular canonical file: {path}")
    return path


def require_hash(path: Path, expected: str, label: str) -> str:
    direct_regular(path, label)
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected}, got {got}")
    return got


def validate_fvecs(path: Path, rows: int, label: str) -> None:
    expected_bytes = rows * RECORD_BYTES
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{label} bytes {path.stat().st_size}, expected {expected_bytes}")
    with path.open("rb") as handle:
        for row in range(rows):
            record = handle.read(RECORD_BYTES)
            if len(record) != RECORD_BYTES:
                raise RuntimeError(f"{label} truncated at row {row}")
            (dimension,) = struct.unpack("<i", record[:4])
            if dimension != DIM:
                raise RuntimeError(f"{label} dimension {dimension} at row {row}")
        if handle.read(1):
            raise RuntimeError(f"{label} has trailing bytes")


def payloads(path: Path, rows: int):
    with path.open("rb") as handle:
        for row in range(rows):
            record = handle.read(RECORD_BYTES)
            if len(record) != RECORD_BYTES or struct.unpack("<i", record[:4])[0] != DIM:
                raise RuntimeError(f"invalid fvec record {path}:{row}")
            yield record[4:]


def write_text(path: Path, lines) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for line in lines:
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_v3_quarantine() -> tuple[set[int], set[bytes], dict[int, bytes]]:
    require_hash(V3_MAPPING, EXPECTED["v3_mapping_sha256"], "v3 local-to-original mapping")
    originals: set[int] = set()
    payload_hashes: set[bytes] = set()
    expected_by_original: dict[int, bytes] = {}
    lines = V3_MAPPING.read_text(encoding="ascii").splitlines()
    if len(lines) != N_V3_COMPACT:
        raise RuntimeError(f"v3 mapping row count {len(lines)} != {N_V3_COMPACT}")
    for local, line in enumerate(lines):
        fields = line.split("	")
        if len(fields) != 3:
            raise RuntimeError(f"malformed v3 mapping row {local}")
        raw_local, raw_original, raw_hash = fields
        if raw_local != str(local) or not raw_original.isdecimal() or len(raw_hash) != 64:
            raise RuntimeError(f"noncanonical v3 mapping row {local}")
        original = int(raw_original)
        try:
            digest = bytes.fromhex(raw_hash)
        except ValueError as exc:
            raise RuntimeError(f"bad v3 mapping payload hash row {local}") from exc
        if original in originals or digest in payload_hashes:
            raise RuntimeError(f"duplicate v3 quarantine member at row {local}")
        originals.add(original)
        payload_hashes.add(digest)
        expected_by_original[original] = digest
    if len(originals) != N_V3_COMPACT or len(payload_hashes) != N_V3_COMPACT:
        raise RuntimeError("v3 quarantine cardinality mismatch")
    return originals, payload_hashes, expected_by_original


def ensure_parent_and_fresh(out: Path) -> None:
    if str(out.parent) != str(ROOT / "inputs"):
        raise RuntimeError("output parent is not canonical v4 inputs")
    if out.exists() or out.is_symlink():
        raise RuntimeError(f"refusing overwrite/append: {out}")
    if out.parent.is_symlink() or not out.parent.is_dir() or out.parent.resolve() != out.parent:
        raise RuntimeError("v4 inputs parent is not direct canonical directory")


def main() -> int:
    ensure_parent_and_fresh(OUT)
    source_hashes = {
        "base_fvecs_sha256": require_hash(BASE, EXPECTED["base_sha256"], "base fvecs"),
        "v2_query_fvecs_sha256": require_hash(V2_QUERY, EXPECTED["v2_query_sha256"], "v2 query fvecs"),
        "learn_source_fvecs_sha256": require_hash(LEARN, EXPECTED["learn_sha256"], "SIFT learn fvecs"),
        "v3_mapping_sha256": require_hash(V3_MAPPING, EXPECTED["v3_mapping_sha256"], "v3 local-to-original mapping"),
        "v3_forensic_record_sha256": require_hash(FORENSIC, EXPECTED["v3_forensic_sha256"], "v3 forensic boundary"),
    }
    forensic = json.loads(FORENSIC.read_text(encoding="utf-8"))
    if forensic.get("schema") != "safe-c2-v4-v3-forensic-boundary-v1":
        raise RuntimeError("unexpected v3 forensic schema")
    facts = forensic.get("observed_stage_facts", {})
    if facts.get("calibration", {}).get("ledger_state") != "ATTEMPTED":
        raise RuntimeError("v3 calibration forensic state is not ATTEMPTED")
    if facts.get("validation", {}).get("ledger_state") != "UNCLAIMED" or facts.get("sealed_test", {}).get("ledger_state") != "UNCLAIMED":
        raise RuntimeError("v3 validation/test forensic state mismatch")

    validate_fvecs(BASE, 1_000_000, "base")
    validate_fvecs(V2_QUERY, N_V2_QUERY, "v2 query")
    validate_fvecs(LEARN, N_LEARN, "SIFT learn")
    v3_originals, v3_payload_hashes, v3_expected_by_original = load_v3_quarantine()
    v2_payloads = set(payloads(V2_QUERY, N_V2_QUERY))
    if len(v2_payloads) != 9_999:
        raise RuntimeError(f"unexpected v2 payload uniqueness {len(v2_payloads)}")

    seen_payloads: set[bytes] = set()
    candidates: list[tuple[int, bytes, bytes, bytes]] = []
    counts = {
        "input_rows": N_LEARN,
        "reserved_drift_prefix_reject": 0,
        "v2_payload_reject": 0,
        "v3_original_reject": 0,
        "v3_payload_alias_reject": 0,
        "global_duplicate_non_v3_reject": 0,
    }
    observed_v3_members: set[int] = set()
    for original_id, payload in enumerate(payloads(LEARN, N_LEARN)):
        duplicate = payload in seen_payloads
        seen_payloads.add(payload)
        digest = hashlib.sha256(payload).digest()
        if original_id in v3_expected_by_original and digest != v3_expected_by_original[original_id]:
            raise RuntimeError(f"v3 mapping payload mismatch at original ID {original_id}")
        if original_id < RESERVED_DRIFT_PREFIX:
            counts["reserved_drift_prefix_reject"] += 1
            continue
        if payload in v2_payloads:
            counts["v2_payload_reject"] += 1
            continue
        if original_id in v3_originals:
            counts["v3_original_reject"] += 1
            observed_v3_members.add(original_id)
            continue
        if digest in v3_payload_hashes:
            # Quarantine every duplicate payload of a v3 compact query, even
            # when this later occurrence would also be a generic duplicate.
            counts["v3_payload_alias_reject"] += 1
            continue
        if duplicate:
            counts["global_duplicate_non_v3_reject"] += 1
            continue
        rank = hashlib.sha256(
            SEED_ASCII + bytes((0,)) + str(original_id).encode("ascii") + bytes((0,)) + digest
        ).digest()
        candidates.append((original_id, digest, payload, rank))
    if observed_v3_members != v3_originals:
        raise RuntimeError("not every v3 original ID was observed/quarantined")
    counts["learn_global_unique"] = len(seen_payloads)
    counts["fresh_clean_candidates"] = len(candidates)
    expected_counts = {
        "input_rows": 100_000,
        "reserved_drift_prefix_reject": 3_072,
        "v2_payload_reject": 9_670,
        "v3_original_reject": 10_000,
        "v3_payload_alias_reject": 21,
        "global_duplicate_non_v3_reject": 174,
        "learn_global_unique": 99_766,
        "fresh_clean_candidates": 77_063,
    }
    if counts != expected_counts:
        raise RuntimeError(f"fresh-pool filtering count mismatch: expected {expected_counts}, got {counts}")
    if len(candidates) < TOTAL_SELECTED:
        raise RuntimeError("insufficient fresh candidate pool")

    selected = sorted(candidates, key=lambda row: (row[3], row[0]))[:TOTAL_SELECTED]
    originals = [row[0] for row in selected]
    digests = [row[1] for row in selected]
    if len(set(originals)) != TOTAL_SELECTED or len(set(digests)) != TOTAL_SELECTED:
        raise RuntimeError("selected fresh IDs/payloads are not unique")
    if any(oid in v3_originals for oid in originals) or any(d in v3_payload_hashes for d in digests):
        raise RuntimeError("v3 compact query leaked into v4 selection")
    if any(oid < RESERVED_DRIFT_PREFIX for oid in originals):
        raise RuntimeError("reserved drift original ID leaked into v4 selection")
    if any(row[2] in v2_payloads for row in selected):
        raise RuntimeError("v2 payload leaked into v4 selection")

    temporary = OUT.parent / f"{OUT.name}.tmp.{os.urandom(12).hex()}"
    temporary.mkdir(mode=0o750)
    try:
        compact = temporary / "sift_learn_fresh12524.fvecs"
        with compact.open("wb") as handle:
            for _, _, payload, _ in selected:
                handle.write(struct.pack("<i", DIM))
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        mapping = temporary / "local_to_original_learn_id.tsv"
        write_text(mapping, (
            f"{local}\t{original}\t{digest.hex()}\n"
            for local, (original, digest, _, _) in enumerate(selected)
        ))
        stage_metadata: dict[str, object] = {}
        offset = 0
        for stage, size, purpose in SPLITS:
            members = selected[offset:offset + size]
            if len(members) != size:
                raise RuntimeError(f"stage underflow: {stage}")
            local_ids = temporary / f"{stage}.candidate_local_ids.txt"
            original_ids = temporary / f"{stage}.candidate_original_learn_ids.txt"
            stage_mapping = temporary / f"{stage}.candidate_mapping.tsv"
            write_text(local_ids, (f"{local}\n" for local in range(offset, offset + size)))
            write_text(original_ids, (f"{member[0]}\n" for member in members))
            write_text(stage_mapping, (
                f"{local}\t{member[0]}\t{member[1].hex()}\n"
                for local, member in zip(range(offset, offset + size), members)
            ))
            stage_metadata[stage] = {
                "candidate_local_range": [offset, offset + size],
                "candidate_count": size,
                "candidate_local_ids_path": str(OUT / local_ids.name),
                "candidate_local_ids_sha256": sha256_file(local_ids),
                "candidate_original_ids_path": str(OUT / original_ids.name),
                "candidate_original_ids_sha256": sha256_file(original_ids),
                "candidate_mapping_path": str(OUT / stage_mapping.name),
                "candidate_mapping_sha256": sha256_file(stage_mapping),
                "purpose": purpose,
                "tie_admission": "PENDING_INDEPENDENT_EXACT_FP32_INT64_ORACLE_NO_REPLACEMENT",
            }
            offset += size
        if offset != TOTAL_SELECTED:
            raise RuntimeError("stage allocation sum mismatch")

        manifest = {
            "schema": "gts-v4-fresh-sift-learn-selection-pre-gt-v1",
            "status": "FRESH_IDS_FROZEN_PRE_GT_NO_C2_NO_GPU",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "scope": "Fresh v4 query selection only. No GT, C2 binary, CUDA, GPU management command, gamma, timing, trace, recall, or result artifact is read or produced.",
            "selector": {
                "path": str(ROOT / "tools/prepare_v4_fresh_selection_pre_gt.py"),
                "sha256": sha256_file(Path(__file__).resolve()),
                "seed_ascii": SEED_ASCII.decode("ascii"),
                "seed_sha256": hashlib.sha256(SEED_ASCII).hexdigest(),
                "ranking": "SHA256(seed_ascii || NUL || decimal_original_id || NUL || SHA256(payload)).digest ascending, then original_id ascending",
            },
            "source": {
                "base_fvecs_path": str(BASE),
                "v2_query_fvecs_path": str(V2_QUERY),
                "learn_source_fvecs_path": str(LEARN),
                "v3_quarantined_mapping_path": str(V3_MAPPING),
                "v3_forensic_record_path": str(FORENSIC),
                **source_hashes,
            },
            "forbidden_or_quarantined": {
                "v2_sealed_test_ids_sha256_forbidden": V2_SEALED_TEST_DIGEST_FORBIDDEN,
                "v2_query_payload_denylist_count": len(v2_payloads),
                "v3_compact_original_id_quarantine_count": len(v3_originals),
                "v3_compact_payload_quarantine_count": len(v3_payload_hashes),
                "reserved_drift_original_id_interval": [0, RESERVED_DRIFT_PREFIX],
                "global_first_occurrence_dedup": True,
                "v3_calibration_attempted_must_not_be_reused": True,
                "v3_validation_and_sealed_test_unclaimed_forensic_only_not_tuning": True,
            },
            "filtering_counts": counts,
            "derived": {
                "compact_query_fvecs_path": str(OUT / compact.name),
                "compact_query_fvecs_sha256": sha256_file(compact),
                "compact_query_fvecs_bytes": compact.stat().st_size,
                "compact_query_fvecs_count": TOTAL_SELECTED,
                "local_to_original_mapping_path": str(OUT / mapping.name),
                "local_to_original_mapping_sha256": sha256_file(mapping),
                "local_to_original_mapping_rows": TOTAL_SELECTED,
                "selected_original_id_min": min(originals),
                "selected_original_id_max": max(originals),
            },
            "candidate_splits": stage_metadata,
            "canary_boundary": {
                "stage": "qualification_canary",
                "allowed_conclusion": "binary/guard/parser/artifact qualification only",
                "forbidden_conclusions": [
                    "speed result", "recall result", "gamma selection", "parameter tuning",
                    "formal calibration reuse", "formal validation/test reuse"
                ],
                "failure_policy": "A failed canary invalidates that v4 build/protocol pin; do not tune on the canary. A future replacement requires a new disjoint canary selection and new pins.",
            },
            "next_required_step": "Run an independent exact FP32+int64 oracle and deterministic tie admission without replacement. It may create a final v4 workload manifest only after independently proving split disjointness, no v2/v3 leakage, and predeclared minimum eligible counts. It must not run C2/GPU.",
        }
        manifest_path = temporary / "selection_manifest_pre_gt.json"
        write_json(manifest_path, manifest)
        artifact_hashes = {
            child.name: sha256_file(child)
            for child in sorted(temporary.iterdir()) if child.is_file() and child.name != "selection_artifacts.sha256"
        }
        with (temporary / "selection_artifacts.sha256").open("w", encoding="ascii", newline="\n") as handle:
            for name, digest in sorted(artifact_hashes.items()):
                handle.write(f"{digest}  {name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUT)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result = {
        "status": "PASS_PRE_GT_SELECTION_FROZEN",
        "out": str(OUT),
        "selection_manifest_sha256": sha256_file(OUT / "selection_manifest_pre_gt.json"),
        "compact_query_sha256": sha256_file(OUT / "sift_learn_fresh12524.fvecs"),
        "mapping_sha256": sha256_file(OUT / "local_to_original_learn_id.tsv"),
        "selected": TOTAL_SELECTED,
        "gpu_or_c2_executed": False,
        "groundtruth_generated": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PREPARE_V4_FRESH_SELECTION_PRE_GT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)

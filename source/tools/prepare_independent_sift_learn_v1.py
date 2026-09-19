#!/usr/bin/env python3
"""Freeze the independent Safe-C2 v3 SIFT-learn compact-query input.

CPU-only.  It deliberately performs no ground-truth calculation and never
reads C2 output, gamma, or any sealed v2 test result.  It creates the frozen
pre-GT query selection; the separate exact oracle will perform tie admission
without replacement and write the final runnable workload manifest.
"""
from __future__ import annotations

import argparse
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
N_QUERY = 10_000
RESERVED_LEARN_PREFIX = 3_072
SEED_ASCII = b"safe-c2-v3-sift-learn-clean-v1-2026-07-27"

BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
CONSUMED_V2_QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
LEARN = Path("/workspace/project/GTS/Datasets/sift1m/sift_learn.fvecs")
DRIFT_PROTOCOL = Path("/workspace/experiments/tide_safe_c1_20260727/c2_controlled_drift_recalibration/protocol_sift1m_query_to_learn_controlled_drift_v1.json")
EXPECTED = {
    "base_fvecs_sha256": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
    "consumed_v2_query_fvecs_sha256": "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
    "learn_source_fvecs_sha256": "331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea",
    "compact_query_fvecs_sha256": "318e5085dfb4f50831d8f6620cffc736da9c11fb44454b507d2f5e7019d2e11e",
    "clean_candidates_sha256": "9a63e7511cc940e57013128d48ce48bee8c4acde1b80903482b6d9ff1ebc9c38",
    "local_to_original_mapping_sha256": "a6794e510245482c4478814511066eaf6eb6b307063caf835b2d534fa5c7e9ed",
    "calibration_original_ids_sha256": "0b4212068a0578502c1c9d384e463c7974f5069a60c72ce69ba900bb2bb21499",
    "calibration_mapping_sha256": "1e1dfffc2edcb2a03bd05643465fb5e2f85b4efc8bce0b75fd5b5ff44bf7279a",
    "validation_original_ids_sha256": "df60c9ba9f6cb6116efbd83a81bd65200ccbede2888d86fa8e641907a6d17d31",
    "validation_mapping_sha256": "0337a76e024200ce9daf3a084bf87d55bbb057e503b879ebdf9b49bbd99f15ea",
    "sealed_test_original_ids_sha256": "533cc05a9012887696c3192b9513e7e2bfc627a102af6d10b537752dd63600f0",
    "sealed_test_mapping_sha256": "23e85a8c5c17d45f54a2460b08b968a05f241f1adff6e4e092c165da285d8a1b",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or a symlink: {path}")
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected}, got {got}")
    return got


def validate_fvec_file(path: Path, rows: int, label: str):
    size = path.stat().st_size
    if size != rows * RECORD_BYTES:
        raise RuntimeError(f"{label} unexpected size {size}; expected {rows * RECORD_BYTES}")
    with path.open("rb") as f:
        for i in range(rows):
            record = f.read(RECORD_BYTES)
            if len(record) != RECORD_BYTES:
                raise RuntimeError(f"{label} truncated at row {i}")
            (d,) = struct.unpack("<i", record[:4])
            if d != DIM:
                raise RuntimeError(f"{label} dimension {d} at row {i}, expected {DIM}")
        if f.read(1):
            raise RuntimeError(f"{label} has trailing bytes")


def read_payloads(path: Path, rows: int):
    with path.open("rb") as f:
        for i in range(rows):
            record = f.read(RECORD_BYTES)
            if len(record) != RECORD_BYTES or struct.unpack("<i", record[:4])[0] != DIM:
                raise RuntimeError(f"invalid fvec record at {path} row {i}")
            yield record[4:]


def write_text(path: Path, lines):
    with path.open("w", encoding="ascii", newline="\n") as f:
        for line in lines:
            f.write(line)
        f.flush()
        os.fsync(f.fileno())


def assert_expected(name: str, path: Path) -> str:
    got = sha256_file(path)
    expected = EXPECTED[name]
    if got != expected:
        raise RuntimeError(f"derived {name} SHA mismatch: expected {expected}, got {got}")
    return got


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory; must not already exist")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or out.is_symlink():
        raise RuntimeError(f"refusing overwrite/append: {out}")
    if out.parent.exists() and out.parent.is_symlink():
        raise RuntimeError(f"output parent is symlink: {out.parent}")

    source_hashes = {
        "base_fvecs_sha256": require_hash(BASE, EXPECTED["base_fvecs_sha256"], "base"),
        "consumed_v2_query_fvecs_sha256": require_hash(CONSUMED_V2_QUERY, EXPECTED["consumed_v2_query_fvecs_sha256"], "consumed v2 query"),
        "learn_source_fvecs_sha256": require_hash(LEARN, EXPECTED["learn_source_fvecs_sha256"], "learn source"),
    }
    if not DRIFT_PROTOCOL.is_file() or DRIFT_PROTOCOL.is_symlink():
        raise RuntimeError(f"reserved drift protocol missing/symlink: {DRIFT_PROTOCOL}")
    source_hashes["reserved_drift_protocol_sha256"] = sha256_file(DRIFT_PROTOCOL)
    validate_fvec_file(BASE, 1_000_000, "base")
    validate_fvec_file(CONSUMED_V2_QUERY, 10_000, "consumed v2 query")
    validate_fvec_file(LEARN, N_LEARN, "learn")

    # The denylist uses payload bytes only, not any v2 GT, C2 response, gamma,
    # timing, trace, or test-ID membership.
    deny = set(read_payloads(CONSUMED_V2_QUERY, N_QUERY))
    if len(deny) != 9_999:
        raise RuntimeError(f"unexpected consumed-query payload uniqueness: {len(deny)}")

    seen_global = set()
    clean = []  # (original_id, payload_sha_hex, payload, score_bytes)
    counts = {"input_rows": N_LEARN, "reserved_prefix_reject": 0,
              "v2_content_reject": 0, "global_duplicate_reject": 0}
    for original_id, payload in enumerate(read_payloads(LEARN, N_LEARN)):
        duplicate = payload in seen_global
        # Crucially, every row (including reserved/deny rows) enters the global
        # seen set before filtering, preventing a later duplicate from escaping.
        seen_global.add(payload)
        if original_id < RESERVED_LEARN_PREFIX:
            counts["reserved_prefix_reject"] += 1
            continue
        if payload in deny:
            counts["v2_content_reject"] += 1
            continue
        if duplicate:
            counts["global_duplicate_reject"] += 1
            continue
        p_hash = hashlib.sha256(payload).digest()
        score = hashlib.sha256(SEED_ASCII + b"\0" + str(original_id).encode("ascii") + b"\0" + p_hash).digest()
        clean.append((original_id, p_hash.hex(), payload, score))
    counts["learn_global_unique"] = len(seen_global)
    counts["clean_unique"] = len(clean)
    expected_counts = {"input_rows": 100000, "reserved_prefix_reject": 3072,
                       "v2_content_reject": 9670, "global_duplicate_reject": 195,
                       "learn_global_unique": 99766, "clean_unique": 87063}
    if counts != expected_counts:
        raise RuntimeError(f"cleaning count mismatch: expected {expected_counts}, got {counts}")

    # Canonical audit is original-id order, not the random selection rank.
    clean_by_id = sorted(clean, key=lambda x: x[0])
    out_tmp = out.parent / (out.name + ".tmp." + os.urandom(8).hex())
    out_tmp.mkdir(mode=0o700, parents=True)
    try:
        clean_path = out_tmp / "clean_candidates_original_id.tsv"
        write_text(clean_path, (f"{oid}\t{payload_hex}\n" for oid, payload_hex, _, _ in clean_by_id))
        clean_sha = assert_expected("clean_candidates_sha256", clean_path)

        selected = sorted(clean, key=lambda x: (x[3], x[0]))[:N_QUERY]
        if len(selected) != N_QUERY:
            raise RuntimeError("insufficient clean candidates")
        if len({x[0] for x in selected}) != N_QUERY or len({x[1] for x in selected}) != N_QUERY:
            raise RuntimeError("selected original IDs or payloads are not unique")
        if any(x[0] < RESERVED_LEARN_PREFIX for x in selected):
            raise RuntimeError("selected reserved original ID")
        if any(x[2] in deny for x in selected):
            raise RuntimeError("selected payload leaked from v2 query")

        compact_path = out_tmp / "sift_learn_clean10k.fvecs"
        with compact_path.open("wb") as f:
            for _, _, payload, _ in selected:
                f.write(struct.pack("<i", DIM)); f.write(payload)
            f.flush(); os.fsync(f.fileno())
        compact_sha = assert_expected("compact_query_fvecs_sha256", compact_path)

        mapping_path = out_tmp / "local_to_original_learn_id.tsv"
        write_text(mapping_path, (f"{local}\t{oid}\t{payload_hex}\n"
                                  for local, (oid, payload_hex, _, _) in enumerate(selected)))
        mapping_sha = assert_expected("local_to_original_mapping_sha256", mapping_path)

        stage_ranges = {"calibration": (0, 2000), "validation": (2000, 4000), "sealed_test": (4000, 10000)}
        stage_meta = {}
        for stage, (lo, hi) in stage_ranges.items():
            rows = [(local, *selected[local][:2]) for local in range(lo, hi)]
            local_ids_path = out_tmp / f"{stage}.unfiltered_local_ids.txt"
            original_ids_path = out_tmp / f"{stage}.original_learn_ids.txt"
            stage_map_path = out_tmp / f"{stage}.mapping.tsv"
            write_text(local_ids_path, (f"{local}\n" for local, _, _ in rows))
            write_text(original_ids_path, (f"{oid}\n" for _, oid, _ in rows))
            write_text(stage_map_path, (f"{local}\t{oid}\t{payload_hex}\n" for local, oid, payload_hex in rows))
            key = "calibration" if stage == "calibration" else ("validation" if stage == "validation" else "sealed_test")
            original_sha = assert_expected(f"{key}_original_ids_sha256", original_ids_path)
            map_sha = assert_expected(f"{key}_mapping_sha256", stage_map_path)
            stage_meta[stage] = {
                "local_range": [lo, hi], "count_unfiltered": hi-lo,
                "unfiltered_local_ids_path": str(local_ids_path), "unfiltered_local_ids_sha256": sha256_file(local_ids_path),
                "original_learn_ids_path": str(original_ids_path), "original_learn_ids_sha256": original_sha,
                "mapping_path": str(stage_map_path), "mapping_sha256": map_sha,
                "tie_admission": "PENDING_EXACT_FP32_AND_INT64_ORACLE_NO_REPLACEMENT",
            }

        manifest = {
            "schema": "gts-v3-compact-learn-selection-pre-gt-v1",
            "status": "INPUT_SELECTION_FROZEN_GT_AND_TIE_ADMISSION_PENDING",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "scope": "Independent SIFT-learn query selection only; no GPU, GT, C2 traversal, gamma, recall, timing, or v2 test decisions.",
            "seed_ascii": SEED_ASCII.decode("ascii"),
            "record_format": {"fvec_dimension": DIM, "record_bytes": RECORD_BYTES},
            "source": {
                "base_fvecs_path": str(BASE),
                "consumed_v2_query_fvecs_path": str(CONSUMED_V2_QUERY),
                "learn_source_fvecs_path": str(LEARN),
                "reserved_drift_protocol_path": str(DRIFT_PROTOCOL),
                **source_hashes,
            },
            "leakage_controls": {
                "sealed_v2_test_forbidden": True,
                "v2_query_payload_denylist_count": len(deny),
                "reserved_original_learn_ids": [0, RESERVED_LEARN_PREFIX],
                "global_first_occurrence_dedup": True,
                "ordering": "SHA256(seed_ascii || NUL || decimal_original_id || NUL || SHA256(payload)).digest ascending, then original_id ascending",
            },
            "counts": counts,
            "derived": {
                "compact_query_fvecs_path": str(compact_path),
                "compact_query_fvecs_sha256": compact_sha,
                "compact_query_fvecs_bytes": compact_path.stat().st_size,
                "clean_candidates_path": str(clean_path),
                "clean_candidates_sha256": clean_sha,
                "local_to_original_mapping_path": str(mapping_path),
                "local_to_original_mapping_sha256": mapping_sha,
                "local_to_original_mapping_rows": N_QUERY,
                "selected_original_id_min": min(x[0] for x in selected),
                "selected_original_id_max": max(x[0] for x in selected),
            },
            "unfiltered_stages": stage_meta,
            "next_required_step": "Run independent CPU exact FP32+int64 oracle and tie admission without replacement; only it may write final runnable workload manifest and filtered stage IDs.",
        }
        manifest_path = out_tmp / "selection_manifest_pre_gt.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        # A source-independent hash manifest makes any later accidental mutation visible.
        hashes = {}
        for p in sorted(out_tmp.iterdir()):
            if p.is_file() and p.name != "artifacts.sha256": hashes[p.name] = sha256_file(p)
        with (out_tmp / "artifacts.sha256").open("w", encoding="ascii", newline="\n") as f:
            for name, h in sorted(hashes.items()): f.write(f"{h}  {name}\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(out_tmp, out)
    except Exception:
        shutil.rmtree(out_tmp, ignore_errors=True)
        raise
    print(json.dumps({"status": "PASS", "output": str(out), "compact_query_sha256": compact_sha,
                      "mapping_sha256": mapping_sha, "clean_count": len(clean), "selected_count": len(selected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PREPARE_INDEPENDENT_SIFT_LEARN_V1_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)

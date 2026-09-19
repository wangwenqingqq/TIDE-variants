#!/usr/bin/python3.12
"""Derive non-overlapping strict-C2 partition seals from the existing sealed split.

This CPU-only stdlib tool is intentionally a one-time provenance transformation.
The emitted calibration seal carries only calibration membership and calibration
Native-GTS target aggregates.  The emitted held-out seal carries only held-out
membership and deliberately contains no quality target.  Strict-C2 runners and
validators consume those partition-local seals; they never open the historical
combined split containing both partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "safe-c1-hnsw-calibration-test-split-v1"
SEAL_SCHEMA = "safe-c1-hnsw-strict-c2-partition-seal-v1"
SOURCE_SPLIT_FILE_SHA256 = "92394b5aa7cee61d6f2583b08061d30a1a287ef5776c7dbe5c9a6a79ddadd585"
SOURCE_SPLIT_CANONICAL_SHA256 = "c825f1e29f80fcb93b0d302e8a3da3273de0de578a839bd72d86eceb5b532355"
OFFICIAL_MEMBER_SCOPE = "SHA-256 of LF-delimited ASCII lines in KNN-ordinal order: knn_ordinal<TAB>op_index<TAB>query_id<LF>."
CALIBRATION_MEMBER_SHA256 = "eae0763680a0f71862a3b63e3c4429ffa467e5f9e7df2a03cc5607cbd552658d"
HELDOUT_MEMBER_SHA256 = "e96b8f45d71d3aa566e09c7b92a4844d664b30774c0f04850d7d0f8cea4a7aba"


class Fail(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def is_uint(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def uint(value: Any, label: str) -> int:
    need(is_uint(value), f"{label} must be a nonnegative integer")
    return int(value)


def sha(value: Any, label: str) -> str:
    need(isinstance(value, str) and len(value) == 64 and
         all(ch in "0123456789abcdef" for ch in value),
         f"{label} must be a lowercase SHA-256")
    return value


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def regular(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
         f"{label} must be a real regular file")
    return path.resolve(strict=True)


def real_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Fail(f"missing {label}: {path}") from exc
    need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
         f"{label} must be a real directory")
    return path.resolve(strict=True)


def read_object(path: Path, label: str) -> dict[str, Any]:
    regular(path, label)
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Fail(f"invalid {label} JSON: {exc}") from exc
    need(isinstance(item, dict), f"{label} must be an object")
    return item


def official_member_digest(members: list[dict[str, int]]) -> str:
    """Official sealed-split invariant; do not substitute an alternate digest."""
    return hashlib.sha256("".join(
        f"{item['knn_ordinal']}\t{item['op_index']}\t{item['query_id']}\n"
        for item in members).encode("ascii")).hexdigest()


def members_for(split: dict[str, Any], name: str) -> tuple[list[dict[str, int]], str]:
    splits = split.get("splits")
    need(isinstance(splits, dict) and set(splits) == {"calibration", "held_out"},
         "source split partition keys drift")
    item = splits.get(name)
    need(isinstance(item, dict), f"source split lacks {name} object")
    raw = item.get("members")
    need(isinstance(raw, list) and len(raw) == 68 and item.get("member_count") == 68,
         f"source {name} membership cardinality drift")
    members: list[dict[str, int]] = []
    previous = -1
    for index, raw_member in enumerate(raw):
        need(isinstance(raw_member, dict) and set(raw_member) ==
             {"knn_ordinal", "op_index", "query_id", "global_delta_live"},
             f"source {name} member {index} shape drift")
        member = {key: uint(raw_member.get(key), f"source {name} member {index}.{key}")
                  for key in ("knn_ordinal", "op_index", "query_id", "global_delta_live")}
        need(member["knn_ordinal"] > previous, f"source {name} ordinal order drift")
        parity = 0 if name == "calibration" else 1
        need(member["knn_ordinal"] % 2 == parity,
             f"source {name} membership violates pre-registered ordinal parity")
        previous = member["knn_ordinal"]
        members.append(member)
    digest = official_member_digest(members)
    need(item.get("member_sha256") == digest, f"source {name} official member hash drift")
    expected_official = CALIBRATION_MEMBER_SHA256 if name == "calibration" else HELDOUT_MEMBER_SHA256
    need(digest == expected_official,
         f"source {name} official tab/ordinal member hash drift")
    return members, digest


def source_binding(split: dict[str, Any], bundle: Path) -> dict[str, Any]:
    need(split.get("schema") == SOURCE_SCHEMA, "source split schema drift")
    raw_without_hash = dict(split)
    supplied = sha(raw_without_hash.pop("split_sha256", None), "source split_sha256")
    need(hashlib.sha256(canonical(raw_without_hash)).hexdigest() == supplied,
         "source split self-hash mismatch")
    need(supplied == SOURCE_SPLIT_CANONICAL_SHA256,
         "source split canonical SHA differs from the established sealed split")
    paths = split.get("input_paths")
    hashes = split.get("input_sha256")
    projection = split.get("projection")
    need(isinstance(paths, dict) and isinstance(hashes, dict) and isinstance(projection, dict),
         "source split binding objects missing")
    need(isinstance(paths.get("bundle"), str) and Path(paths["bundle"]).resolve(strict=True) == bundle,
         "source split bundle path does not match --bundle")
    fields = {
        "bundle_manifest_json_sha256": sha(hashes.get("bundle_manifest_json_sha256"), "manifest SHA"),
        "bundle_metadata_json_sha256": sha(hashes.get("bundle_metadata_json_sha256"), "metadata SHA"),
        "projection_trace_sha256": sha(hashes.get("projection_trace_sha256"), "trace SHA"),
        "source_trace_sha256": sha(hashes.get("source_trace_sha256"), "source trace SHA"),
    }
    for key in ("event_count", "k", "knn_count"):
        uint(projection.get(key), f"source projection.{key}")
    need(projection.get("event_count") == 305 and projection.get("k") == 10 and
         projection.get("knn_count") == 136, "source projection dimensions drift")
    need(projection.get("metadata_sha256") == fields["bundle_metadata_json_sha256"] and
         projection.get("trace_sha256") == fields["projection_trace_sha256"],
         "source projection hash binding drift")
    return {
        "bundle_path": str(bundle),
        "manifest_sha256": fields["bundle_manifest_json_sha256"],
        "metadata_sha256": fields["bundle_metadata_json_sha256"],
        "trace_sha256": fields["projection_trace_sha256"],
        "source_trace_sha256": fields["source_trace_sha256"],
        "event_count": 305,
        "insert_count": 169,
        "knn_count": 136,
        "k": 10,
    }


def calibration_target(split: dict[str, Any]) -> dict[str, int]:
    targets = split.get("native_gts_overlap_targets")
    need(isinstance(targets, dict) and isinstance(targets.get("calibration"), dict),
         "source split calibration target missing")
    value = targets["calibration"]
    merged_overlap = value.get("merged_overlap")
    merged_exact = value.get("merged_exact_result_sets")
    need(isinstance(merged_overlap, dict) and isinstance(merged_exact, dict),
         "source calibration target shape drift")
    result = {
        "merged_overlap_numerator": uint(merged_overlap.get("numerator"), "calibration overlap numerator"),
        "merged_overlap_denominator": uint(merged_overlap.get("denominator"), "calibration overlap denominator"),
        "merged_exact_match_numerator": uint(merged_exact.get("numerator"), "calibration exact numerator"),
        "merged_exact_match_denominator": uint(merged_exact.get("denominator"), "calibration exact denominator"),
        "knn_count": uint(value.get("knn_count"), "calibration target KNN count"),
        "k": uint(value.get("k"), "calibration target K"),
    }
    need(tuple(result.values()) == (651, 680, 53, 68, 68, 10),
         "source calibration target differs from sealed 651/680 and 53/68")
    return result


def make_seal(name: str, binding: dict[str, Any], source_file_sha: str, source_canonical_sha: str,
              members: list[dict[str, int]], digest: str,
              target: dict[str, int] | None) -> dict[str, Any]:
    common: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "scope": ("Strict C2 partition-local frozen immutable-base, insertion-only, KNN-only "
                  "quality protocol input. This seal carries exactly one presealed partition and "
                  "is not a timing, full-E1, deletion, range, rebuild, direct-sidecar, GPU, or "
                  "universal exactness claim."),
        "partition": name,
        "source_split_schema": SOURCE_SCHEMA,
        "source_split_file_sha256": source_file_sha,
        "source_split_canonical_sha256": source_canonical_sha,
        "projection": binding,
        "member_sha256_scope": OFFICIAL_MEMBER_SCOPE,
        "member_count": len(members),
        "member_sha256": digest,
        "members": members,
        "cross_partition_content_absent": True,
        "seal_sha256_scope": "SHA-256 of canonical UTF-8 JSON for this object with seal_sha256 omitted; sort_keys=true, separators=(',', ':'), trailing LF.",
    }
    if name == "calibration":
        need(target is not None, "calibration target omitted")
        common["calibration_native_gts_target"] = target
    else:
        need(target is None, "held-out seal must not embed any target")
        common["held_out_quality_target_embedded"] = False
    common["seal_sha256"] = hashlib.sha256(canonical(common)).hexdigest()
    return common


def write_new(path: Path, payload: bytes) -> None:
    parent = real_directory(path.parent, "output parent")
    del parent
    need(not os.path.lexists(path), f"refusing to overwrite existing output: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            need(count > 0, f"short write to {path}")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chown(path, 0, 0)
    os.chmod(path, 0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive strict-C2 partition-local seals")
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--calibration-out", required=True, type=Path)
    parser.add_argument("--heldout-out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        bundle = real_directory(args.bundle, "bundle")
        split = read_object(args.split, "source split")
        binding = source_binding(split, bundle)
        calibration_members, calibration_digest = members_for(split, "calibration")
        heldout_members, heldout_digest = members_for(split, "held_out")
        need({item["knn_ordinal"] for item in calibration_members}.isdisjoint(
             {item["knn_ordinal"] for item in heldout_members}),
             "source split partitions overlap")
        need(len(calibration_members) + len(heldout_members) == binding["knn_count"],
             "source split does not cover all KNNs")
        source_canonical_sha = sha(split.get("split_sha256"), "source split_sha256")
        source_file_sha = hashlib.sha256(args.split.read_bytes()).hexdigest()
        need(source_file_sha == SOURCE_SPLIT_FILE_SHA256,
             "source split file SHA differs from the official sealed split")
        calibration = make_seal("calibration", binding, source_file_sha, source_canonical_sha,
                                calibration_members, calibration_digest, calibration_target(split))
        heldout = make_seal("held_out", binding, source_file_sha, source_canonical_sha,
                            heldout_members, heldout_digest, None)
        need(args.calibration_out.resolve(strict=False) != args.heldout_out.resolve(strict=False),
             "partition seal output paths alias")
        write_new(args.calibration_out, canonical(calibration))
        try:
            write_new(args.heldout_out, canonical(heldout))
        except Exception:
            # Do not silently leave a partial pair: the existing calibration seal
            # remains evidence of a failed derivation and the command fail-stops.
            raise
        print(json.dumps({
            "schema": "safe-c1-hnsw-strict-c2-partition-seal-derivation-receipt-v1",
            "status": "PASS",
            "source_split_file_sha256": source_file_sha,
            "source_split_canonical_sha256": source_canonical_sha,
            "calibration_out": str(args.calibration_out.resolve()),
            "calibration_seal_sha256": calibration["seal_sha256"],
            "heldout_out": str(args.heldout_out.resolve()),
            "heldout_seal_sha256": heldout["seal_sha256"],
            "cross_partition_seal_separation": True,
            "calibration_embeds_only_calibration_target": True,
            "heldout_embeds_no_quality_target": True,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Fail as exc:
        print(f"strict_c2_seal_derivation fail-stop: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

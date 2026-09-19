#!/usr/bin/env python3
"""Validate every ChEMBL 37 HDF5 fingerprint against checksum-listed FPS."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

EXPECTED_FPS_SHA256 = "c33bfac42abfea96840279ec35eeb3364c1063aa3e35c02369618beaf6389529"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_fields(dtype: np.dtype) -> list[str]:
    names = list(dtype.names or ())
    result = [name for name in names if name.startswith("f") and name[1:].isdigit()]
    return sorted(result, key=lambda name: int(name[1:]))


def sorted_digest_summary(values: np.ndarray) -> tuple[str, bytes, bytes]:
    values.sort(kind="quicksort")
    digest = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
    first = bytes(values[0]).hex() if len(values) else ""
    last = bytes(values[-1]).hex() if len(values) else ""
    return digest, first, last


def multiset_difference(left: np.ndarray, right: np.ndarray, sample_limit: int = 100) -> dict[str, object]:
    """Merge two sorted fixed-width SHA-256 arrays without positional overcount."""
    left_bytes = memoryview(left).cast("B")
    right_bytes = memoryview(right).cast("B")

    def item(buffer: memoryview, index: int) -> bytes:
        start = index * 32
        return bytes(buffer[start : start + 32])

    i = 0
    j = 0
    common = 0
    left_only = 0
    right_only = 0
    left_samples: list[str] = []
    right_samples: list[str] = []
    while i < len(left) and j < len(right):
        lv = item(left_bytes, i)
        rv = item(right_bytes, j)
        if lv == rv:
            common += 1
            i += 1
            j += 1
        elif lv < rv:
            left_only += 1
            if len(left_samples) < sample_limit:
                left_samples.append(lv.hex())
            i += 1
        else:
            right_only += 1
            if len(right_samples) < sample_limit:
                right_samples.append(rv.hex())
            j += 1
    while i < len(left):
        lv = item(left_bytes, i)
        left_only += 1
        if len(left_samples) < sample_limit:
            left_samples.append(lv.hex())
        i += 1
    while j < len(right):
        rv = item(right_bytes, j)
        right_only += 1
        if len(right_samples) < sample_limit:
            right_samples.append(rv.hex())
        j += 1
    return {
        "common_with_multiplicity": common,
        "h5_only_with_multiplicity": left_only,
        "fps_only_with_multiplicity": right_only,
        "h5_only_digest_samples": left_samples,
        "fps_only_digest_samples": right_samples,
    }


def read_h5(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    popcnt_lut = np.fromiter((value.bit_count() for value in range(256)), dtype=np.uint8, count=256)
    bit_reverse_lut = np.fromiter(
        (int(f"{value:08b}"[::-1], 2) for value in range(256)),
        dtype=np.uint8,
        count=256,
    )
    with h5py.File(path, "r") as handle:
        dataset = handle["fps"]
        names = fingerprint_fields(dataset.dtype)
        if len(names) != 32:
            raise RuntimeError(f"expected 32 u64 fields (2048 bits), got {names}")
        required = {"fp_id", "popcnt"}
        if not required.issubset(dataset.dtype.names or ()):
            raise RuntimeError(f"missing required fields {required}: {dataset.dtype}")
        count = int(dataset.shape[0])
        digests = np.empty(count, dtype="S32")
        stored_recomputed_mismatches = 0
        first_mismatch: dict[str, int] | None = None
        chunk_rows = 100_000
        for start in range(0, count, chunk_rows):
            block = dataset[start : start + chunk_rows]
            packed = np.empty((len(block), len(names)), dtype="<u8")
            for word, name in enumerate(names):
                packed[:, word] = block[name]
            row_bytes = len(names) * 8
            # FPSim2 process_fp first takes RDKit ToBitString(), then
            # BitStrToIntList interprets every 64 characters MSB-first. FPS
            # stores the corresponding RDKit binary bytes, where bit index 0
            # is the least-significant bit of its byte. Convert each numeric
            # word to big-endian bytes and reverse bits within each byte.
            big_endian_words = packed.astype(">u8", copy=True)
            normalized = bit_reverse_lut[big_endian_words.view(np.uint8)]
            raw = memoryview(normalized).cast("B")
            for local in range(len(block)):
                lo = local * row_bytes
                digests[start + local] = hashlib.sha256(raw[lo : lo + row_bytes]).digest()
            recomputed = popcnt_lut[packed.view(np.uint8)].reshape(len(block), row_bytes).sum(axis=1)
            mismatch = np.flatnonzero(recomputed != block["popcnt"])
            stored_recomputed_mismatches += int(len(mismatch))
            if first_mismatch is None and len(mismatch):
                local = int(mismatch[0])
                first_mismatch = {
                    "row": start + local,
                    "fp_id": int(block["fp_id"][local]),
                    "stored": int(block["popcnt"][local]),
                    "recomputed": int(recomputed[local]),
                }
            print(json.dumps({"source": "h5", "rows_processed": start + len(block), "rows_total": count}), flush=True)
        info = {
            "rows": count,
            "word_count": len(names),
            "fingerprint_bits": len(names) * 64,
            "dtype": str(dataset.dtype),
            "fingerprint_fields": names,
            "stored_recomputed_popcount_mismatches": stored_recomputed_mismatches,
            "first_popcount_mismatch": first_mismatch,
            "normalization": (
                f"concatenate {names[0]}..{names[-1]} in numeric-suffix order; "
                "each numeric uint64 -> big-endian bytes -> reverse bits within each byte"
            ),
            "normalization_evidence": "CONTENT_NORMALIZATION_AMENDMENT_20260829.md and encoding probe",
        }
    return digests, info


def read_fps(path: Path, expected_rows: int) -> tuple[np.ndarray, dict[str, object]]:
    digests = np.empty(expected_rows, dtype="S32")
    count = 0
    num_bits: int | None = None
    type_header: str | None = None
    invalid_width = 0
    first_invalid_line: int | None = None
    popcount_min: int | None = None
    popcount_max: int | None = None
    with gzip.open(path, "rt", encoding="ascii", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.startswith("#"):
                text = line.rstrip("\r\n")
                if text.startswith("#num_bits="):
                    num_bits = int(text.split("=", 1)[1])
                elif text.startswith("#type="):
                    type_header = text.split("=", 1)[1]
                continue
            text = line.rstrip("\r\n")
            if not text:
                continue
            try:
                fp_hex, _identifier = text.split("\t", 1)
                raw = bytes.fromhex(fp_hex)
            except Exception as error:
                raise RuntimeError(f"invalid FPS row at line {line_number}: {error}") from error
            if len(raw) != 256:
                invalid_width += 1
                if first_invalid_line is None:
                    first_invalid_line = line_number
            if count >= expected_rows:
                raise RuntimeError(f"FPS has more than HDF5 row count {expected_rows}")
            digests[count] = hashlib.sha256(raw).digest()
            popcount = sum(value.bit_count() for value in raw)
            popcount_min = popcount if popcount_min is None else min(popcount_min, popcount)
            popcount_max = popcount if popcount_max is None else max(popcount_max, popcount)
            count += 1
            if count % 100_000 == 0:
                print(json.dumps({"source": "fps", "rows_processed": count, "rows_expected": expected_rows}), flush=True)
    if count != expected_rows:
        raise RuntimeError(f"row-count mismatch FPS={count}, HDF5={expected_rows}")
    info = {
        "rows": count,
        "num_bits_header": num_bits,
        "type_header": type_header,
        "invalid_width_rows": invalid_width,
        "first_invalid_width_line": first_invalid_line,
        "population_count_min": popcount_min,
        "population_count_max": popcount_max,
        "normalization": "bytes.fromhex of the complete FPS fingerprint column",
    }
    return digests, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--fps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begin = time.monotonic()

    fps_sha = sha256_file(args.fps)
    h5_sha = sha256_file(args.h5)
    print(json.dumps({"fps_sha256": fps_sha, "expected_fps_sha256": EXPECTED_FPS_SHA256, "h5_sha256": h5_sha}), flush=True)
    if fps_sha != EXPECTED_FPS_SHA256:
        raise RuntimeError(f"official FPS checksum mismatch: {fps_sha}")

    h5_digests, h5_info = read_h5(args.h5)
    fps_digests, fps_info = read_fps(args.fps, int(h5_info["rows"]))
    h5_multiset_sha, h5_first, h5_last = sorted_digest_summary(h5_digests)
    fps_multiset_sha, fps_first, fps_last = sorted_digest_summary(fps_digests)
    unequal = np.flatnonzero(h5_digests != fps_digests)
    difference = multiset_difference(h5_digests, fps_digests)
    content_equal = (
        difference["h5_only_with_multiplicity"] == 0
        and difference["fps_only_with_multiplicity"] == 0
    )
    result = {
        "experiment_id": "tide_20260829_gate6_chembl37_fps_h5_content_integrity_v3",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_h5": str(args.h5),
        "source_h5_sha256": h5_sha,
        "source_fps": str(args.fps),
        "source_fps_sha256": fps_sha,
        "published_fps_sha256": EXPECTED_FPS_SHA256,
        "published_fps_checksum_match": fps_sha == EXPECTED_FPS_SHA256,
        "h5": h5_info,
        "fps": fps_info,
        "h5_sorted_fingerprint_digest_sha256": h5_multiset_sha,
        "fps_sorted_fingerprint_digest_sha256": fps_multiset_sha,
        "sorted_digest_first": {"h5": h5_first, "fps": fps_first},
        "sorted_digest_last": {"h5": h5_last, "fps": fps_last},
        "unequal_sorted_digest_count": int(len(unequal)),
        "first_unequal_sorted_digest_index": int(unequal[0]) if len(unequal) else None,
        "multiset_difference": difference,
        "complete_fingerprint_multiset_equal": content_equal,
        "direct_published_h5_checksum_available": False,
        "source_integrity_pass": bool(
            fps_sha == EXPECTED_FPS_SHA256
            and h5_info["fingerprint_bits"] == 2048
            and fps_info["num_bits_header"] == 2048
            and fps_info["invalid_width_rows"] == 0
            and h5_info["stored_recomputed_popcount_mismatches"] == 0
            and content_equal
        ),
        "elapsed_s": time.monotonic() - begin,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["source_integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

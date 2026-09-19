#!/usr/bin/env python3
"""Prepare one-source ChEMBL 37 FPS data for static WORDS32 validation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np


ID_PATTERN = re.compile(r"CHEMBL([0-9]+)\Z")
WORDS = 32
ROW_WORDS = WORDS + 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_memmap(path: Path, dtype: str, shape: tuple[int, ...]) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if path.exists() or part.exists():
        raise RuntimeError(f"refusing to replace existing output: {path}")
    return np.memmap(part, mode="w+", dtype=dtype, shape=shape), part


def close_and_publish(array: np.memmap, part: Path, final: Path) -> None:
    array.flush()
    del array
    os.replace(part, final)


def keyed_u64(prefix: str, numeric_id: int) -> int:
    digest = hashlib.sha256(f"{prefix}:{numeric_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=2_897_819)
    parser.add_argument("--delta-rows", type=int, default=20_760)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()
    begin = time.monotonic()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {args.output_root}")
    if not 0 < args.delta_rows < args.expected_rows:
        raise RuntimeError("delta row count must be between zero and the corpus size")
    if not 0 < args.queries <= args.delta_rows:
        raise RuntimeError("query count must be positive and no larger than the delta")

    source_sha256 = sha256(args.fps)
    if source_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"official FPS checksum mismatch: {source_sha256} != {args.expected_sha256}"
        )

    root = args.output_root
    scratch = root / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    unsorted_fp = np.memmap(
        scratch / "fp_u64x32.unsorted.bin",
        mode="w+",
        dtype="<u8",
        shape=(args.expected_rows, WORDS),
    )
    unsorted_ids = np.memmap(
        scratch / "id_i64.unsorted.bin",
        mode="w+",
        dtype="<i8",
        shape=(args.expected_rows,),
    )
    unsorted_pc = np.memmap(
        scratch / "popcnt_u16.unsorted.bin",
        mode="w+",
        dtype="<u2",
        shape=(args.expected_rows,),
    )
    split_keys = np.memmap(
        scratch / "split_key_u64.unsorted.bin",
        mode="w+",
        dtype="<u8",
        shape=(args.expected_rows,),
    )

    headers: dict[str, str] = {}
    seen_ids: set[int] = set()
    rows = 0
    with gzip.open(args.fps, "rt", encoding="ascii", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.rstrip("\r\n")
            if not text:
                continue
            if text.startswith("#"):
                if "=" in text:
                    key, value = text[1:].split("=", 1)
                    headers[key] = value
                continue
            if rows >= args.expected_rows:
                raise RuntimeError("FPS contains more rows than the frozen contract")
            columns = text.split("\t")
            if len(columns) != 2:
                raise RuntimeError(f"line {line_number}: invalid FPS column count")
            fingerprint_hex, identifier = columns
            match = ID_PATTERN.fullmatch(identifier)
            if not match:
                raise RuntimeError(f"line {line_number}: invalid identifier {identifier!r}")
            numeric_id = int(match.group(1))
            if numeric_id in seen_ids:
                raise RuntimeError(f"duplicate identifier {identifier}")
            seen_ids.add(numeric_id)
            try:
                fingerprint = bytes.fromhex(fingerprint_hex)
            except ValueError as error:
                raise RuntimeError(f"line {line_number}: invalid fingerprint hex") from error
            if len(fingerprint) != WORDS * 8:
                raise RuntimeError(
                    f"line {line_number}: expected {WORDS * 8} fingerprint bytes"
                )
            unsorted_fp[rows] = np.frombuffer(fingerprint, dtype="<u8")
            unsorted_ids[rows] = numeric_id
            split_keys[rows] = keyed_u64("20260903:chembl37-static", numeric_id)
            rows += 1
            if rows % 250_000 == 0:
                print(json.dumps({"phase": "parse", "rows": rows}), flush=True)

    if rows != args.expected_rows or len(seen_ids) != args.expected_rows:
        raise RuntimeError(
            f"frozen row/ID contract failed: rows={rows}, unique={len(seen_ids)}"
        )
    if headers.get("num_bits") != "2048":
        raise RuntimeError(f"unexpected num_bits header: {headers.get('num_bits')}")
    expected_type = (
        "RDKit-Morgan/1 radius=2 fpSize=2048 useFeatures=0 "
        "useChirality=0 useBondTypes=1"
    )
    if headers.get("type") != expected_type or headers.get("software") != "RDKit/2022.09.4":
        raise RuntimeError(f"unexpected fingerprint contract: {headers}")
    popcount_lut = np.fromiter(
        (bin(value).count("1") for value in range(256)),
        dtype=np.uint8,
        count=256,
    )
    fingerprint_bytes = unsorted_fp.view(np.uint8).reshape(args.expected_rows, WORDS * 8)
    for start in range(0, args.expected_rows, 200_000):
        stop = min(args.expected_rows, start + 200_000)
        unsorted_pc[start:stop] = popcount_lut[fingerprint_bytes[start:stop]].sum(
            axis=1, dtype=np.uint16
        )
        print(json.dumps({"phase": "popcount", "rows": stop}), flush=True)
    unsorted_fp.flush()
    unsorted_ids.flush()
    unsorted_pc.flush()
    split_keys.flush()
    del seen_ids

    selected = np.argpartition(split_keys, args.delta_rows - 1)[: args.delta_rows]
    delta_mask = np.zeros(args.expected_rows, dtype=np.bool_)
    delta_mask[selected] = True
    selected_ids = np.asarray(unsorted_ids[selected], dtype="<i8")
    selected_id_sha256 = hashlib.sha256(np.sort(selected_ids).tobytes()).hexdigest()

    order = np.lexsort((unsorted_ids, unsorted_pc))
    sorted_delta_mask = delta_mask[order]
    base_rows = args.expected_rows - args.delta_rows
    gate0 = root / "data/gate0_prepared"
    stage_a = root / "data/stage_a"
    union_fp_path = gate0 / "union_fp_u64x32.bin"
    union_id_path = gate0 / "union_id_i64.bin"
    union_pc_path = gate0 / "union_popcnt_u16.bin"
    base_fp_path = gate0 / "base_fp_u64x32.bin"
    base_id_path = gate0 / "base_id_i64.bin"
    base_pc_path = gate0 / "base_popcnt_u16.bin"
    union_fp, union_fp_part = atomic_memmap(union_fp_path, "<u8", (args.expected_rows, WORDS))
    union_ids, union_id_part = atomic_memmap(union_id_path, "<i8", (args.expected_rows,))
    union_pc, union_pc_part = atomic_memmap(union_pc_path, "<u2", (args.expected_rows,))
    base_fp, base_fp_part = atomic_memmap(base_fp_path, "<u8", (base_rows, WORDS))
    base_ids, base_id_part = atomic_memmap(base_id_path, "<i8", (base_rows,))
    base_pc, base_pc_part = atomic_memmap(base_pc_path, "<u2", (base_rows,))

    delta_fp = np.empty((args.delta_rows, WORDS), dtype="<u8")
    delta_ids = np.empty(args.delta_rows, dtype="<i8")
    delta_pc = np.empty(args.delta_rows, dtype="<u2")
    union_offset = 0
    base_offset = 0
    delta_offset = 0
    chunk_rows = 200_000
    for start in range(0, args.expected_rows, chunk_rows):
        indexes = order[start : start + chunk_rows]
        count = len(indexes)
        stop = union_offset + count
        union_fp[union_offset:stop] = unsorted_fp[indexes]
        union_ids[union_offset:stop] = unsorted_ids[indexes]
        union_pc[union_offset:stop] = unsorted_pc[indexes]
        mask = sorted_delta_mask[start : start + count]
        base_count = int(np.count_nonzero(~mask))
        delta_count = int(np.count_nonzero(mask))
        if base_count:
            base_fp[base_offset : base_offset + base_count] = unsorted_fp[indexes[~mask]]
            base_ids[base_offset : base_offset + base_count] = unsorted_ids[indexes[~mask]]
            base_pc[base_offset : base_offset + base_count] = unsorted_pc[indexes[~mask]]
            base_offset += base_count
        if delta_count:
            delta_fp[delta_offset : delta_offset + delta_count] = unsorted_fp[indexes[mask]]
            delta_ids[delta_offset : delta_offset + delta_count] = unsorted_ids[indexes[mask]]
            delta_pc[delta_offset : delta_offset + delta_count] = unsorted_pc[indexes[mask]]
            delta_offset += delta_count
        union_offset = stop
        print(json.dumps({"phase": "sort-write", "rows": union_offset}), flush=True)

    if union_offset != args.expected_rows or base_offset != base_rows or delta_offset != args.delta_rows:
        raise RuntimeError(
            f"derived row balance failed: union={union_offset}, base={base_offset}, delta={delta_offset}"
        )
    if np.any(union_pc[1:] < union_pc[:-1]) or np.any(base_pc[1:] < base_pc[:-1]):
        raise RuntimeError("population-count order validation failed")

    close_and_publish(union_fp, union_fp_part, union_fp_path)
    close_and_publish(union_ids, union_id_part, union_id_path)
    close_and_publish(union_pc, union_pc_part, union_pc_path)
    close_and_publish(base_fp, base_fp_part, base_fp_path)
    close_and_publish(base_ids, base_id_part, base_id_path)
    close_and_publish(base_pc, base_pc_part, base_pc_path)

    delta_path = stage_a / "delta_u64x34.bin"
    queries_path = stage_a / "queries_u64x34.bin"
    delta_packed, delta_part = atomic_memmap(
        delta_path, "<u8", (args.delta_rows, ROW_WORDS)
    )
    delta_packed[:, 0] = delta_ids.astype("<u8", copy=False)
    delta_packed[:, 1 : 1 + WORDS] = delta_fp
    delta_packed[:, -1] = delta_pc.astype("<u8", copy=False)
    close_and_publish(delta_packed, delta_part, delta_path)

    query_order = sorted(
        range(args.delta_rows),
        key=lambda index: hashlib.sha256(
            f"20260903:chembl37-static-query:{int(delta_ids[index])}".encode()
        ).digest(),
    )[: args.queries]
    query_packed, query_part = atomic_memmap(
        queries_path, "<u8", (args.queries, ROW_WORDS)
    )
    chosen = np.asarray(query_order, dtype=np.int64)
    query_packed[:, 0] = delta_ids[chosen].astype("<u8", copy=False)
    query_packed[:, 1 : 1 + WORDS] = delta_fp[chosen]
    query_packed[:, -1] = delta_pc[chosen].astype("<u8", copy=False)
    query_id_sha256 = hashlib.sha256(
        delta_ids[chosen].astype("<i8", copy=False).tobytes()
    ).hexdigest()
    close_and_publish(query_packed, query_part, queries_path)

    artifacts = [
        union_fp_path,
        union_id_path,
        union_pc_path,
        base_fp_path,
        base_id_path,
        base_pc_path,
        delta_path,
        queries_path,
    ]
    artifact_records = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in artifacts
    }
    manifest = {
        "experiment_id": "tide_20260903_chembl37_static_single_source_preparation_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.fps),
        "source_sha256": source_sha256,
        "published_source_sha256": args.expected_sha256,
        "headers": headers,
        "word_count": WORDS,
        "fingerprint_bits": WORDS * 64,
        "rows": args.expected_rows,
        "base_rows": base_rows,
        "delta_rows": args.delta_rows,
        "query_rows": args.queries,
        "population_count_min": int(unsorted_pc.min()),
        "population_count_max": int(unsorted_pc.max()),
        "layout_order": "population count, then numeric ChEMBL ID",
        "split_key": "little-endian uint64(SHA256('20260903:chembl37-static:<numeric-ID>')[:8])",
        "query_key": "SHA256('20260903:chembl37-static-query:<numeric-ID>') byte order",
        "selected_id_sha256": selected_id_sha256,
        "query_id_sha256": query_id_sha256,
        "artifacts": artifact_records,
        "elapsed_s": time.monotonic() - begin,
    }
    manifest_path = root / "PREPARATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    del unsorted_fp, unsorted_ids, unsorted_pc, split_keys
    if not args.keep_scratch:
        shutil.rmtree(scratch)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

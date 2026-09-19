#!/usr/bin/env python3
"""Derive the narrow Safe-C1 frozen-base KNN projection from a frozen E1 trace.

This is intentionally a new workload, not a silent repair of the full E1 trace:
base deletes and range events are excluded because this first Safe-C1 gate has an
immutable native base and supports KNN only. Relative order of retained events is
preserved and all provenance is sealed in the generated manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from collections import Counter
from pathlib import Path

HEADER = struct.Struct("<8sI6IfQ")
EVENT = struct.Struct("<IB3xi")
MAGIC = b"E1GTRC01"
VERSION = 1
INSERT, DELETE, KNN, RANGE = 1, 2, 3, 4
COPY_FILES = (
    "pool.i16",
    "queries.i16",
    "stable_id_to_pool_row.i32",
    "initial_base_stable_ids.i32",
    "quantized_oracle_expected.jsonl",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def active_hash(active: set[int]) -> str:
    return hashlib.sha256("".join(f"{sid}\n" for sid in sorted(active)).encode("ascii")).hexdigest()


def load_i32_exact(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{count}i", raw))


def parse_trace(path: Path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated trace header")
    header = HEADER.unpack_from(raw)
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, event_count = header
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported E1 trace magic/version")
    if not all(v > 0 for v in (dim, base_n, pool_n, query_n, k)) or reservoir_n != pool_n - base_n:
        raise ValueError("invalid E1 trace dimensions")
    expected = HEADER.size + event_count * EVENT.size
    if len(raw) != expected:
        raise ValueError(f"trace byte size mismatch: expected {expected}, got {len(raw)}")
    events = []
    for position in range(event_count):
        op_index, op, argument = EVENT.unpack_from(raw, HEADER.size + position * EVENT.size)
        if op_index != position or op not in (INSERT, DELETE, KNN, RANGE):
            raise ValueError(f"invalid E1 event at position {position}")
        events.append((op_index, op, argument))
    return {
        "dimension": dim,
        "base_n": base_n,
        "reservoir_n": reservoir_n,
        "pool_n": pool_n,
        "query_n": query_n,
        "k": k,
        "radius": radius,
        "event_count": event_count,
    }, events, raw


def validate_source(header: dict, events: list[tuple[int, int, int]], base_ids: list[int]) -> Counter:
    if len(base_ids) != header["base_n"] or len(set(base_ids)) != len(base_ids):
        raise ValueError("initial base IDs do not form a distinct base_n-sized set")
    if any(sid < 0 or sid >= header["pool_n"] for sid in base_ids):
        raise ValueError("initial base ID outside pool")
    active = set(base_ids)
    inserted_ever: set[int] = set()
    counts: Counter = Counter()
    for op_index, op, arg in events:
        if op == INSERT:
            if not (header["base_n"] <= arg < header["pool_n"]):
                raise ValueError(f"source insert {op_index} is not in reservoir")
            if arg in active or arg in inserted_ever:
                raise ValueError(f"source insert {op_index} repeats a stable ID")
            active.add(arg)
            inserted_ever.add(arg)
        elif op == DELETE:
            if arg not in active:
                raise ValueError(f"source delete {op_index} targets an inactive ID")
            if arg not in set(base_ids):
                raise ValueError(
                    f"source delete {op_index} targets mutable ID {arg}; projection policy would be unsound"
                )
            active.remove(arg)
        else:
            if not (0 <= arg < header["query_n"]):
                raise ValueError(f"source query {op_index} has invalid query id")
        counts[{INSERT: "insert", DELETE: "delete", KNN: "knn", RANGE: "range"}[op]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    if not source.is_dir():
        raise SystemExit(f"source bundle is not a directory: {source}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    for name in (*COPY_FILES, "trace.e1gtrc", "manifest.json", "metadata.json"):
        if not (source / name).is_file():
            raise SystemExit(f"source bundle missing {name}")

    header, source_events, source_trace = parse_trace(source / "trace.e1gtrc")
    base_ids = load_i32_exact(source / "initial_base_stable_ids.i32", header["base_n"])
    source_counts = validate_source(header, source_events, base_ids)
    if source_counts["delete"] == 0:
        raise SystemExit("projection expects at least one excluded base delete for provenance")

    mapping = load_i32_exact(source / "stable_id_to_pool_row.i32", header["pool_n"])
    pool_rows = (source / "pool.i16").stat().st_size // (2 * header["dimension"])
    if (source / "pool.i16").stat().st_size != pool_rows * 2 * header["dimension"]:
        raise SystemExit("pool.i16 is not a whole int16 row-major matrix")
    if any(row < 0 or row >= pool_rows for row in mapping) or len(set(mapping)) != len(mapping):
        raise SystemExit("stable_id_to_pool_row is not an injective in-range mapping")
    if (source / "queries.i16").stat().st_size != header["query_n"] * header["dimension"] * 2:
        raise SystemExit("queries.i16 shape disagrees with trace header")

    projected_source_indices: list[int] = []
    projected_events: list[tuple[int, int, int]] = []
    for old_index, op, argument in source_events:
        if op not in (INSERT, KNN):
            continue
        projected_source_indices.append(old_index)
        projected_events.append((len(projected_events), op, argument))
    projection_counts = Counter({"insert": 0, "knn": 0})
    active = set(base_ids)
    for _, op, argument in projected_events:
        if op == INSERT:
            if argument in active:
                raise SystemExit("projection insert duplicates an active stable ID")
            active.add(argument)
            projection_counts["insert"] += 1
        else:
            projection_counts["knn"] += 1
    if not projected_events or projection_counts["knn"] == 0 or projection_counts["insert"] == 0:
        raise SystemExit("projection must retain both inserts and KNN queries")

    out.mkdir(parents=True, mode=0o700)
    try:
        for name in COPY_FILES:
            shutil.copyfile(source / name, out / name)
        with (out / "trace.e1gtrc").open("wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, header["dimension"], header["base_n"],
                                header["reservoir_n"], header["pool_n"], header["query_n"],
                                header["k"], header["radius"], len(projected_events)))
            for event in projected_events:
                f.write(EVENT.pack(*event))
        trace_path = out / "trace.e1gtrc"
        event_bytes = trace_path.read_bytes()[HEADER.size:]
        metadata = {
            "schema": "e1-frozen-base-knn-projection-bundle-v1",
            "title": "Frozen-base, insertion-only KNN projection of E1",
            "scope": (
                "Safe-C1 immutable-base KNN gate only; not the full E1 workload, not a base-delete, "
                "range, rebuild, direct-sidecar, or performance claim."
            ),
            "source_bundle": str(source),
            "source_trace_sha256": sha256_file(source / "trace.e1gtrc"),
            "source_event_stream_sha256": hashlib.sha256(source_trace[HEADER.size:]).hexdigest(),
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "source_metadata_sha256": sha256_file(source / "metadata.json"),
            "source_header": {**header, "magic": MAGIC.decode("ascii"), "version": VERSION},
            "header": {**header, "magic": MAGIC.decode("ascii"), "version": VERSION,
                       "event_count": len(projected_events)},
            "trace_counts": {
                "source": dict(source_counts),
                "projection": {"insert": projection_counts["insert"], "knn": projection_counts["knn"]},
            },
            "projection": {
                "allowed_ops": ["insert", "knn"],
                "excluded_ops": ["delete", "range"],
                "filter_policy": (
                    "Retain only source insert/KNN events in source order; exclude all source deletes "
                    "because they are immutable-base deletions, and exclude range because this is KNN-only."
                ),
                "source_indices_sha256": hashlib.sha256(
                    "".join(f"{idx}\n" for idx in projected_source_indices).encode("ascii")
                ).hexdigest(),
                "source_indices_count": len(projected_source_indices),
                "first_source_index": projected_source_indices[0],
                "last_source_index": projected_source_indices[-1],
                "final_active_count": len(active),
                "final_active_set_sha256": active_hash(active),
                "projection_trace_sha256": sha256_file(trace_path),
                "projection_event_stream_sha256": hashlib.sha256(event_bytes).hexdigest(),
                "stable_id_mapping_sha256": sha256_file(out / "stable_id_to_pool_row.i32"),
                "initial_base_stable_ids_sha256": sha256_file(out / "initial_base_stable_ids.i32"),
                "source_quantized_oracle_role": "source provenance only; never a projection oracle",
            },
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = {name: sha256_file(out / name) for name in (*COPY_FILES, "trace.e1gtrc", "metadata.json")}
        manifest = {
            "schema": "e1-frozen-base-knn-projection-manifest-v1",
            "bundle": "e1_frozen_base_knn_projection_v1",
            "source_trace_sha256": metadata["source_trace_sha256"],
            "projection_trace_sha256": metadata["projection"]["projection_trace_sha256"],
            "files_sha256": files,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "PASS",
            "bundle": str(out),
            "source_counts": dict(source_counts),
            "projection_counts": dict(projection_counts),
            "projection_trace_sha256": metadata["projection"]["projection_trace_sha256"],
            "final_active_count": len(active),
            "final_active_set_sha256": active_hash(active),
        }, sort_keys=True))
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

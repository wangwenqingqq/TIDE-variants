"""Read-only source sampling. Writes only a new, explicitly selected output root."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

DATES = ("2026-06-01", "2026-06-15", "2026-07-01", "2026-07-17",
         "2026-08-04", "2026-08-18", "2026-08-25")
SEED = 20260913


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def selection(ids, modulus):
    with np.errstate(over="ignore"):
        value = (ids.astype(np.uint64) ^ np.uint64(SEED)) + np.uint64(0x9e3779b97f4a7c15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        value ^= value >> np.uint64(31)
    return value % np.uint64(modulus) == 0


def verify_rows(rows):
    if rows.ndim != 2 or rows.shape[1] != 6 or not len(rows):
        raise ValueError("expected a nonempty interleaved six-word run")
    lut = np.array([int(x).bit_count() for x in range(256)], dtype=np.uint8)
    fp_bytes = np.ascontiguousarray(rows[:, 1:5]).view(np.uint8).reshape(-1, 32)
    actual = lut[fp_bytes].sum(axis=1)
    if not np.array_equal(actual, rows[:, 5]):
        raise ValueError("stored/recomputed population count mismatch")
    if np.any(rows[1:, 5] < rows[:-1, 5]):
        raise ValueError("run not popcount sorted")
    if len(np.unique(rows[:, 0])) != len(rows):
        raise ValueError("duplicate source IDs within release")


def write_run(path, rows):
    with path.open("xb") as stream:
        rows.astype("<u8", copy=False).tofile(stream)
    return {"file": path.name, "rows": len(rows), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def source_record(path, root):
    before = path.stat()
    sha = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("source changed while hashing")
    return {"relative_path": str(path.relative_to(root)), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": sha}


def choose_queries(runs):
    chosen = []
    labels = []
    for epoch, rows in enumerate(runs):
        nonempty = rows[rows[:, 5] > 0]
        count = 16 if epoch == 0 else (6 if epoch == 6 else 2)
        if len(nonempty) < count:
            raise ValueError("too few nonempty queries in an arrival cohort")
        indices = np.linspace(0, len(nonempty) - 1, count, dtype=np.int64)
        chosen.extend(nonempty[indices])
        labels.extend([{"arrival_epoch": epoch, "source": DATES[epoch]}] * count)
    rows = np.array(chosen, dtype="<u8")
    if len(rows) != 32 or len(np.unique(rows[:, 0])) != 32:
        raise ValueError("query count/uniqueness mismatch")
    return rows, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modulus", type=int, default=38)
    args = parser.parse_args()
    if args.modulus != 38:
        parser.error("this registered pilot admits only modulus 38")
    root = args.source_root.resolve()
    output = args.output.resolve()
    if output == root or output.is_relative_to(root):
        parser.error("output must be outside the historical source root")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    prepared = root / "data/prepared"
    snapshot = prepared / "snapshots" / DATES[0]
    source_paths = [snapshot / name for name in
                    ("id_i64.bin", "fp_u64x4.bin", "popcnt_u16.bin")]
    source_paths += [prepared / "transition_queries" / f"{a}_to_{b}" / "delta_u64x6.bin"
                     for a, b in zip(DATES, DATES[1:])]
    sources = [source_record(path, root) for path in source_paths]
    audit_path = root / "results/TRANSITION_REALITY.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("insert_only_transition_count") != 6 or not audit.get("transition_audit_component_pass"):
        raise ValueError("historical insert-only audit absent or failed")
    for item, a, b in zip(audit["transitions"], DATES, DATES[1:]):
        if item["transition"] != f"{a}_to_{b}":
            raise ValueError("transition order mismatch")
    ids = np.memmap(source_paths[0], mode="r", dtype="<u8")
    fp = np.memmap(source_paths[1], mode="r", dtype="<u8", shape=(len(ids), 4))
    pc = np.memmap(source_paths[2], mode="r", dtype="<u2")
    if len(ids) != 37849270 or len(pc) != len(ids):
        raise ValueError("initial snapshot shape mismatch")
    blocks = []
    for first in range(0, len(ids), 1_000_000):
        stop = min(first + 1_000_000, len(ids))
        selected = np.flatnonzero(selection(ids[first:stop], args.modulus)) + first
        rows = np.empty((len(selected), 6), dtype="<u8")
        rows[:, 0], rows[:, 1:5], rows[:, 5] = ids[selected], fp[selected], pc[selected]
        blocks.append(rows)
    runs = [np.concatenate(blocks)]
    del blocks, ids, fp, pc
    for path in source_paths[3:]:
        if path.stat().st_size % 48:
            raise ValueError("invalid delta file size")
        rows = np.memmap(path, mode="r", dtype="<u8").reshape(-1, 6)
        runs.append(np.array(rows[selection(rows[:, 0], args.modulus)], copy=True))
    all_ids = np.concatenate([rows[:, 0] for rows in runs])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("duplicate stable ID across arrival cohorts")
    for rows in runs:
        verify_rows(rows)
    queries, labels = choose_queries(runs)
    manifest = {"scope": "consistent-ID sample of six real insert-only transitions; finite history",
                "seed": SEED, "sample_modulus": args.modulus, "fingerprint_bits": 256,
                "row_format": "little-endian uint64: stable_id, fp[4], popcount",
                "dates": DATES, "sources": sources,
                "historical_audit_sha256": digest(audit_path), "runs": [], "query_labels": labels}
    for epoch, rows in enumerate(runs):
        manifest["runs"].append(write_run(output / f"run{epoch}.bin", rows))
    manifest["queries"] = write_run(output / "queries.bin", queries)
    # A small, labelled correctness fixture, capped per arrival cohort. It is not
    # another natural trace and never appears in the performance table.
    fixture = output / "fixture"
    fixture.mkdir()
    fixture_runs = []
    for epoch, rows in enumerate(runs):
        count = min(len(rows), 4096 if epoch == 0 else 64)
        fixture_runs.append(rows[np.linspace(0, len(rows) - 1, count, dtype=np.int64)])
        write_run(fixture / f"run{epoch}.bin", fixture_runs[-1])
    fixture_queries, _ = choose_queries(fixture_runs)
    write_run(fixture / "queries.bin", fixture_queries)
    manifest["fixture_scope"] = "real-data correctness fixture; capped 4096 base/64 per delta"
    manifest["prepare_wall_s"] = time.perf_counter() - start
    for path, record in zip(source_paths, sources):
        if path.stat().st_mtime_ns != record["mtime_ns"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError("historical source changed during preparation")
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows_by_arrival": [len(x) for x in runs], "total_rows": len(all_ids),
                      "queries": len(queries), "prepare_wall_s": manifest["prepare_wall_s"]}), flush=True)


if __name__ == "__main__":
    main()

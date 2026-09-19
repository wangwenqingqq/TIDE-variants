"""Prepare a separate 1/4 real-history sample and 64 queries; never alter Gate 0."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate0.prepare_history import DATES, SEED, selection, verify_rows, source_record, write_run


def queries(runs):
    rows, labels = [], []
    for epoch, run in enumerate(runs):
        valid = run[run[:, 5] > 0]
        n = 32 if epoch == 0 else (12 if epoch == 6 else 4)
        if len(valid) < n:
            raise ValueError("not enough nonempty cohort rows")
        indices = np.linspace(0, len(valid) - 1, n, dtype=np.int64)
        rows.extend(valid[indices])
        labels.extend([{"arrival_epoch": epoch, "date": DATES[epoch]}] * n)
    result = np.asarray(rows, dtype="<u8")
    if len(result) != 64 or len(np.unique(result[:, 0])) != 64:
        raise ValueError("query count/uniqueness error")
    return result, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modulus", type=int, choices=(4, 38), default=4)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root):
        parser.error("output must be a fresh independent directory outside source")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    prep = root / "data/prepared"
    initial = prep / "snapshots" / DATES[0]
    paths = [initial / name for name in ("id_i64.bin", "fp_u64x4.bin", "popcnt_u16.bin")]
    paths += [prep / "transition_queries" / f"{a}_to_{b}" / "delta_u64x6.bin"
              for a, b in zip(DATES, DATES[1:])]
    records = [source_record(p, root) for p in paths]
    ids = np.memmap(paths[0], mode="r", dtype="<u8")
    fp = np.memmap(paths[1], mode="r", dtype="<u8", shape=(len(ids), 4))
    pc = np.memmap(paths[2], mode="r", dtype="<u2")
    if len(ids) != 37849270 or len(pc) != len(ids):
        raise ValueError("source row contract mismatch")
    parts = []
    for begin in range(0, len(ids), 1_000_000):
        end = min(begin + 1_000_000, len(ids))
        selected = np.flatnonzero(selection(ids[begin:end], args.modulus)) + begin
        part = np.empty((len(selected), 6), dtype="<u8")
        part[:, 0], part[:, 1:5], part[:, 5] = ids[selected], fp[selected], pc[selected]
        parts.append(part)
    runs = [np.concatenate(parts)]
    del parts, ids, fp, pc
    for path in paths[3:]:
        raw = np.memmap(path, mode="r", dtype="<u8").reshape(-1, 6)
        runs.append(np.asarray(raw[selection(raw[:, 0], args.modulus)]).copy())
    all_ids = np.concatenate([run[:, 0] for run in runs])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("duplicate IDs across releases")
    del all_ids
    for run in runs: verify_rows(run)
    q, labels = queries(runs)
    manifest = {"scope": "Gate1 consistent-ID finite six-transition sample, not steady state",
                "seed": SEED, "modulus": args.modulus, "fingerprint_bits": 256,
                "sources": records, "dates": DATES, "runs": [], "query_labels": labels}
    for epoch, run in enumerate(runs):
        manifest["runs"].append(write_run(output / f"run{epoch}.bin", run))
    manifest["queries"] = write_run(output / "queries.bin", q)
    fixture = output / "fixture"
    fixture.mkdir()
    small = []
    for epoch, run in enumerate(runs):
        n = min(len(run), 4096 if epoch == 0 else 64)
        small.append(run[np.linspace(0, len(run) - 1, n, dtype=np.int64)])
        write_run(fixture / f"run{epoch}.bin", small[-1])
    fq, _ = queries(small)
    write_run(fixture / "queries.bin", fq)
    manifest["fixture_scope"] = "capped real-data correctness fixture; no timing claim"
    manifest["prepare_wall_s"] = time.perf_counter() - started
    for path, record in zip(paths, records):
        now = path.stat()
        if now.st_mtime_ns != record["mtime_ns"] or now.st_size != record["bytes"]:
            raise RuntimeError("source changed during preparation")
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": [len(x) for x in runs], "total": sum(map(len, runs)),
                      "queries": len(q), "prepare_wall_s": manifest["prepare_wall_s"]}), flush=True)


if __name__ == "__main__":
    main()

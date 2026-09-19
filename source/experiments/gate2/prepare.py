"""ID-disjoint development quarter; reuse immutable Gate 1 preparation helpers."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate0.prepare_history import DATES, SEED, verify_rows, source_record, write_run
from experiments.gate1.prepare import queries


def residue(ids):
    with np.errstate(over="ignore"):
        x = (ids.astype(np.uint64) ^ np.uint64(SEED)) + np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        x ^= x >> np.uint64(31)
    return x % np.uint64(4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root, out, test = a.source_root.resolve(), a.output.resolve(), a.test.resolve()
    if out == root or out.is_relative_to(root) or out == test or out.is_relative_to(test):
        p.error("development output must be a fresh independent directory")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    prep = root / "data/prepared"
    paths = [prep / "snapshots" / DATES[0] / n for n in
             ("id_i64.bin", "fp_u64x4.bin", "popcnt_u16.bin")]
    paths += [prep / "transition_queries" / f"{a}_to_{b}" / "delta_u64x6.bin"
              for a, b in zip(DATES, DATES[1:])]
    sources = [source_record(p, root) for p in paths]
    original = json.loads((test / "MANIFEST.json").read_text())
    if sources != original["sources"]:
        raise ValueError("source provenance differs from immutable test manifest")
    ids = np.memmap(paths[0], mode="r", dtype="<u8")
    fp = np.memmap(paths[1], mode="r", dtype="<u8", shape=(len(ids), 4))
    pc = np.memmap(paths[2], mode="r", dtype="<u2")
    if len(ids) != 37849270 or len(pc) != len(ids):
        raise ValueError("source row contract mismatch")
    parts = []
    for begin in range(0, len(ids), 1_000_000):
        end = min(begin + 1_000_000, len(ids))
        chosen = np.flatnonzero(residue(ids[begin:end]) == 1) + begin
        part = np.empty((len(chosen), 6), dtype="<u8")
        part[:, 0], part[:, 1:5], part[:, 5] = ids[chosen], fp[chosen], pc[chosen]
        parts.append(part)
    runs = [np.concatenate(parts)]
    del parts, ids, fp, pc
    for path in paths[3:]:
        raw = np.memmap(path, mode="r", dtype="<u8").reshape(-1, 6)
        runs.append(np.asarray(raw[residue(raw[:, 0]) == 1]).copy())
    all_ids = np.concatenate([r[:, 0] for r in runs])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("duplicate development IDs")
    del all_ids
    for epoch, run in enumerate(runs):
        verify_rows(run)
        held = np.memmap(test / f"run{epoch}.bin", mode="r", dtype="<u8").reshape(-1, 6)
        if not np.all(residue(run[:, 0]) == 1) or not np.all(residue(held[:, 0]) == 0):
            raise ValueError("development/test residue disjointness failed")
    q, labels = queries(runs)
    manifest = {"scope": "Gate2 ID-disjoint development; same six release dates as previously observed test",
                "seed": SEED, "modulus": 4, "remainder": 1, "fingerprint_bits": 256,
                "sources": sources, "dates": DATES, "runs": [], "query_labels": labels,
                "test_manifest": source_record(test / "MANIFEST.json", test), "ID_disjoint": True}
    for e, r in enumerate(runs):
        manifest["runs"].append(write_run(out / f"run{e}.bin", r))
    manifest["queries"] = write_run(out / "queries.bin", q)
    fixture = out / "fixture"
    fixture.mkdir()
    small = [r[np.linspace(0, len(r)-1, min(len(r), 4096 if e == 0 else 64), dtype=np.int64)]
             for e, r in enumerate(runs)]
    for e, r in enumerate(small):
        write_run(fixture / f"run{e}.bin", r)
    fq, _ = queries(small)
    write_run(fixture / "queries.bin", fq)
    for path, record in zip(paths, sources):
        if path.stat().st_mtime_ns != record["mtime_ns"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError("source changed during preparation")
    manifest["wall_s"] = time.monotonic()-start
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"rows": [len(r) for r in runs], "total": sum(map(len, runs)),
                      "ID_disjoint": True, "wall_s": manifest["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()

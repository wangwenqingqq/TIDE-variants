#!/usr/bin/env python3
"""Create deterministic entropy/boundary correctness data, never performance data."""
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

root = Path(sys.argv[1])
rng = random.Random(20260904)
rows = []
for i in range(4353):
    if i < 10:
        bits = (1 << i) - 1
    elif i % 7 == 0:
        count = rng.choice([0, 1, 2, 7, 8, 10, 31, 63, 127, 128, 255, 256])
        bits = sum(1 << b for b in rng.sample(range(256), count))
    else:
        bits = rng.getrandbits(256)
    words = [(bits >> (64 * w)) & ((1 << 64) - 1) for w in range(4)]
    rows.append((i + 1, words, bits.bit_count() if hasattr(int, "bit_count") else bin(bits).count("1")))
base = sorted(rows[:4096], key=lambda r: (r[2], r[0]))
delta = sorted(rows[4096:], key=lambda r: (r[2], r[0]))
union = sorted(rows, key=lambda r: (r[2], r[0]))
queries = [(999999, [1023, 0, 0, 0], 10), (999998, [(1 << 64) - 1] * 4, 256),
           (999997, [1, 0, 0, 0], 1)]
queries += [r for r in rows if r[2] > 0][10:39]
assert len(queries) == 32 and all(q[2] for q in queries)
blobs = {}
for name, data in (("base", base), ("union", union)):
    blobs[f"data/gate0_prepared/{name}_fp_u64x4.bin"] = b"".join(struct.pack("<4Q", *r[1]) for r in data)
    blobs[f"data/gate0_prepared/{name}_id_i64.bin"] = b"".join(struct.pack("<Q", r[0]) for r in data)
    blobs[f"data/gate0_prepared/{name}_popcnt_u16.bin"] = b"".join(struct.pack("<H", r[2]) for r in data)
for name, data in (("delta", delta), ("queries", queries)):
    blobs[f"data/stage_a/{name}_u64x6.bin"] = b"".join(struct.pack("<6Q", r[0], *r[1], r[2]) for r in data)
manifest = {"type": "synthetic_correctness_only", "seed": 20260904,
            "base_rows": len(base), "delta_rows": len(delta), "queries": len(queries),
            "checks": ["nonzero_queries", "zero_and_full_database_fingerprints", "7/10_and_4/5_exact_boundaries", "ragged_rows", "64_delta_fragments"],
            "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in blobs.items()}}
for name, data in blobs.items():
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_bytes() == data, "refuse to overwrite different fixture"
    else:
        p.write_bytes(data)
(root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"root": str(root), "rows": len(rows), "queries": len(queries)}))

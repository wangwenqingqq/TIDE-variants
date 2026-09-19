#!/usr/bin/env python3
"""Prepare the frozen normalized Words/Birkbeck Gate-0 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SEED_TAG = "certigraph-20260827-birkbeck-v1"
NCAL = 256
NTEST = 1024
MAX_LEN = 40


def sha256(path: Path, block: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def normalize(text: str) -> str:
    return text.lower().replace("_", " ")


def read_base(path: Path) -> tuple[list[str], dict]:
    lines = path.read_text(encoding="latin-1").splitlines()
    if not lines:
        raise ValueError("empty base")
    header = lines[0].split()
    if len(header) != 3 or int(header[2]) != 6:
        raise ValueError(f"unexpected base header: {lines[0]!r}")
    declared = int(header[1])
    if len(lines) - 1 != declared:
        raise ValueError((declared, len(lines) - 1))
    seen: set[str] = set()
    base: list[str] = []
    for raw in lines[1:]:
        item = normalize(raw)
        if not item or len(item.encode("latin-1")) > MAX_LEN:
            continue
        if item not in seen:
            seen.add(item)
            base.append(item)
    return base, {
        "declared_rows": declared,
        "normalized_unique_rows": len(base),
        "declared_max_len": int(header[0]),
        "normalized_max_len": max(map(len, base)),
    }


def read_birkbeck(path: Path) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = defaultdict(set)
    target: str | None = None
    for raw in path.read_text(encoding="latin-1").splitlines():
        if raw.startswith("$"):
            target = normalize(raw[1:])
        elif raw and target is not None:
            pairs[normalize(raw)].add(target)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--birkbeck", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("words_normalized.txt", "queries.txt", "query_manifest.json"):
        if (args.out / name).exists():
            raise FileExistsError(args.out / name)

    base, base_stats = read_base(args.base)
    base_to_id = {word: i for i, word in enumerate(base)}
    raw_queries = read_birkbeck(args.birkbeck)
    eligible = []
    for query, targets in raw_queries.items():
        encoded = query.encode("latin-1")
        if not query or len(encoded) > MAX_LEN or query in base_to_id:
            continue
        target_ids = sorted(base_to_id[t] for t in targets if t in base_to_id)
        if not target_ids:
            continue
        digest = hashlib.sha256(encoded).hexdigest()
        eligible.append((digest, query, target_ids))
    eligible.sort(key=lambda x: (x[0], x[1]))
    if len(eligible) < NCAL + NTEST:
        raise RuntimeError(f"only {len(eligible)} eligible queries")
    selected = eligible[: NCAL + NTEST]

    base_path = args.out / "words_normalized.txt"
    with base_path.open("w", encoding="latin-1", newline="\n") as f:
        f.write(f"{MAX_LEN} {len(base)} 6\n")
        for word in base:
            f.write(word + "\n")

    query_path = args.out / "queries.txt"
    with query_path.open("w", encoding="latin-1", newline="\n") as f:
        f.write(f"{MAX_LEN} {len(selected)} 6\n")
        for _, query, _ in selected:
            f.write(query + "\n")

    records = []
    for qid, (digest, query, target_ids) in enumerate(selected):
        records.append(
            {
                "query_id": qid,
                "split": "calibration" if qid < NCAL else "test",
                "sha256": digest,
                "query_latin1_hex": query.encode("latin-1").hex(),
                "query_display": query,
                "target_ids": target_ids,
                "target_display": [base[i] for i in target_ids],
            }
        )
    manifest = {
        "schema": "certigraph-birkbeck-query-manifest-v1",
        "seed_tag": SEED_TAG,
        "normalization": "Latin-1 decode; lowercase; underscore to space",
        "max_len": MAX_LEN,
        "ncal": NCAL,
        "ntest": NTEST,
        "eligible_unique_queries": len(eligible),
        "raw_unique_queries": len(raw_queries),
        "base_stats": base_stats,
        "inputs": {
            "base": {"path": str(args.base), "sha256": sha256(args.base)},
            "birkbeck": {
                "path": str(args.birkbeck),
                "sha256": sha256(args.birkbeck),
            },
        },
        "outputs": {
            "base": str(base_path),
            "queries": str(query_path),
        },
        "records": records,
    }
    (args.out / "query_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "base_rows": len(base),
                "eligible_queries": len(eligible),
                "selected_calibration": NCAL,
                "selected_test": NTEST,
                "base_sha256": sha256(base_path),
                "queries_sha256": sha256(query_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


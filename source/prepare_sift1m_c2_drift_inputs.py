#!/usr/bin/env python3
"""CPU-only provenance/input preparer for the pre-registered C2 drift protocol.

This program never imports CUDA or executes GTS.  It validates the archived fvecs
read-only and writes only below the protocol's isolation.write_root.  With
--materialize it writes selected *query* partitions as .npy files; it never
copies or modifies the 1M-vector base file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

try:
    import numpy as np
except ImportError:  # Check-only validation intentionally has no NumPy dependency.
    np = None

RANGE_RE = re.compile(r"^\[(\d+),(\d+)\)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_range(text: str) -> tuple[int, int]:
    match = RANGE_RE.fullmatch(text)
    if not match:
        raise ValueError(f"expected half-open range '[begin,end)', got {text!r}")
    begin, end = (int(match.group(1)), int(match.group(2)))
    if end <= begin:
        raise ValueError(f"empty/reversed range {text}")
    return begin, end


def fvec_meta(path: Path) -> dict:
    size = path.stat().st_size
    if size < 4:
        raise ValueError(f"fvecs file too small: {path}")
    with path.open("rb") as fh:
        raw_dim = fh.read(4)
    dim = int.from_bytes(raw_dim, byteorder="little", signed=True)
    if dim <= 0:
        raise ValueError(f"invalid first fvecs dimension {dim}: {path}")
    record_bytes = 4 + 4 * dim
    if size % record_bytes:
        raise ValueError(f"nonintegral fvecs row count for {path}")
    return {"path": str(path), "bytes": size, "dimension": dim,
            "rows": size // record_bytes, "record_bytes": record_bytes}


def ensure_under(out: Path, root: Path) -> None:
    try:
        out.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"refusing to write outside isolation root: {out}") from exc


def load_rows(path: Path, meta: dict, begin: int, end: int):
    # Each fvecs record is int32 dimension followed by dim IEEE-754 float32 bits.
    words = np.memmap(path, mode="r", dtype="<i4",
                      shape=(meta["rows"], meta["dimension"] + 1))
    dims = np.asarray(words[begin:end, 0])
    if not np.all(dims == meta["dimension"]):
        bad = int(np.flatnonzero(dims != meta["dimension"])[0])
        raise ValueError(f"inconsistent fvecs dimension at row {begin + bad}: {path}")
    # Reinterpret coordinates rather than casting integer bit patterns.
    return np.asarray(words[begin:end, 1:].view("<f4"), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("protocol_sift1m_query_to_learn_controlled_drift_v1.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="validate metadata/ranges and write provenance only")
    parser.add_argument("--materialize", action="store_true",
                        help="write each selected query partition as a .npy file below --out-dir")
    parser.add_argument("--verify-sha256", action="store_true",
                        help="read source files and verify the pre-registered full SHA-256 values")
    args = parser.parse_args()
    if args.check_only and args.materialize:
        raise SystemExit("--check-only and --materialize are mutually exclusive")
    if args.materialize and np is None:
        raise SystemExit("numpy is required for --materialize; --check-only needs no NumPy")

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") != "gtspp-c2-controlled-drift-recalibration-protocol-v1":
        raise SystemExit("unexpected protocol schema")
    write_root = Path(protocol["isolation"]["write_root"])
    out = args.out_dir
    ensure_under(out, write_root)
    out.mkdir(parents=True, exist_ok=True)

    sources = {"base": protocol["data_contract"]["base"],
               "pre_shift_source": protocol["data_contract"]["pre_shift_source"],
               "shifted_source": protocol["data_contract"]["shifted_source"]}
    observed_sources = {}
    for name, spec in sources.items():
        path = Path(spec["path"])
        if not path.is_file():
            raise SystemExit(f"missing required source {name}: {path}")
        observed = fvec_meta(path)
        for key in ("dimension", "rows"):
            if observed[key] != spec[key]:
                raise SystemExit(f"{name} {key} mismatch: expected {spec[key]}, observed {observed[key]}")
        observed["expected_sha256"] = spec["sha256"]
        if args.verify_sha256:
            observed["observed_sha256"] = sha256(path)
            observed["sha256_match"] = observed["observed_sha256"] == spec["sha256"]
            if not observed["sha256_match"]:
                raise SystemExit(f"SHA-256 mismatch for {name}: {path}")
        observed_sources[name] = observed

    partitions = []
    for name, spec in protocol["partitions"].items():
        source_name = spec["source"]
        if source_name not in sources:
            raise SystemExit(f"unknown source {source_name} for partition {name}")
        begin, end = parse_range(spec["range"])
        if end > observed_sources[source_name]["rows"]:
            raise SystemExit(f"partition {name} is outside {source_name}")
        if end - begin != spec["n"]:
            raise SystemExit(f"partition {name} n disagrees with range")
        item = {"name": name, "source": source_name, "range": spec["range"], "n": spec["n"],
                "purpose": spec["purpose"]}
        if args.materialize:
            arr = load_rows(Path(sources[source_name]["path"]), observed_sources[source_name], begin, end)
            dest = out / f"{name}.npy"
            np.save(dest, arr, allow_pickle=False)
            item["materialized_path"] = str(dest)
            item["shape"] = list(arr.shape)
            item["sha256"] = sha256(dest)
        partitions.append(item)

    report = {
        "schema": "gtspp-c2-controlled-drift-input-validation-v1",
        "status": "PASS_CPU_ONLY_INPUT_VALIDATION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "metadata/range/provenance check only; no CUDA, GTS, gamma selection, detector, oracle, or performance execution",
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
        "source_metadata": observed_sources,
        "partitions": partitions,
        "materialized": bool(args.materialize),
        "verify_sha256": bool(args.verify_sha256)
    }
    result = out / "input_validation.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

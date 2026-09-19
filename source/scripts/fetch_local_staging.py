#!/usr/bin/env python3
"""Concurrent local staging fetch for the Gate-6 acquisition amendment."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

SURE_ROOT = "https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data"
CHEMBL_ROOT = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37"
MISSING_DATES = ["2026-06-01", "2026-06-15", "2026-07-01", "2026-07-17", "2026-08-04"]


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def header(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", "--head", url],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def fetch_one(record: dict[str, str], log_dir: Path) -> dict[str, object]:
    destination = Path(record["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{record['name']}.curl.log"
    receipt = log_dir / f"{record['name']}.headers.txt"
    header(record["url"], receipt)
    start = time.monotonic()
    if not destination.exists():
        part = destination.with_suffix(destination.suffix + ".part")
        with log.open("ab") as stream:
            command = [
                "curl", "--fail", "--location", "--show-error", "--retry", "8",
                "--retry-all-errors", "--continue-at", "-", "--output", str(part),
                record["url"],
            ]
            stream.write(("COMMAND " + " ".join(command) + "\n").encode())
            stream.flush()
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        os.replace(part, destination)
    return {
        **record,
        "bytes": destination.stat().st_size,
        "sha256": digest(destination, "sha256"),
        "elapsed_s": time.monotonic() - start,
        "headers": str(receipt),
        "log": str(log),
    }


def parse_published_checksum(path: Path, filename: str) -> tuple[str, str] | None:
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or filename not in stripped:
            continue
        fields = stripped.replace("*", " ").split()
        hashes = [field.lower() for field in fields if all(c in "0123456789abcdefABCDEF" for c in field)]
        for value in hashes:
            if len(value) == 32:
                return "md5", value
            if len(value) == 64:
                return "sha256", value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    staging = root / "data" / "local_staging"
    run = root / "results" / "raw" / "20260828T150000Z_local_source_staging_v1"
    run.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for date in MISSING_DATES:
        records.append({
            "kind": "surechembl",
            "release": date,
            "name": f"surechembl_{date}",
            "url": f"{SURE_ROOT}/{date}/fpsim2_fingerprints.h5",
            "path": str(staging / "surechembl" / date / "fpsim2_fingerprints.h5"),
        })
    records.append({
        "kind": "chembl",
        "release": "37",
        "name": "chembl_37_h5",
        "url": f"{CHEMBL_ROOT}/chembl_37.h5",
        "path": str(staging / "chembl_37" / "chembl_37.h5"),
    })
    small = ["README", "LICENSE", "REQUIRED.ATTRIBUTION", "checksums.txt"]
    for name in small:
        record = {
            "kind": "chembl-metadata",
            "release": "37",
            "name": f"chembl_37_{name.replace('.', '_')}",
            "url": f"{CHEMBL_ROOT}/{name}",
            "path": str(staging / "chembl_37" / name),
        }
        fetch_one(record, run)

    with (run / "command.json").open("w") as stream:
        json.dump({"workers": args.workers, "records": records}, stream, indent=2)
        stream.write("\n")
    (run / "launcher_pid.txt").write_text(f"{os.getpid()}\n")
    (run / "start_utc.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, record, run) for record in records]
        completed = [future.result() for future in futures]

    checksum_file = staging / "chembl_37" / "checksums.txt"
    published = parse_published_checksum(checksum_file, "chembl_37.h5")
    chembl = next(record for record in completed if record["name"] == "chembl_37_h5")
    checksum_validation: dict[str, object] = {"found": published is not None}
    if published:
        algorithm, expected = published
        observed = digest(Path(chembl["path"]), algorithm)
        checksum_validation.update({
            "algorithm": algorithm,
            "expected": expected,
            "observed": observed,
            "match": observed == expected,
        })
    manifest = {
        "experiment_id": "tide_20260828_gate6_local_source_staging",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route_amendment": str(root / "ACQUISITION_ROUTE_AMENDMENT_20260828.md"),
        "sources": completed,
        "chembl_published_checksum": checksum_validation,
    }
    (staging / "LOCAL_STAGING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "exit_code.txt").write_text("0\n")
    print(json.dumps(manifest, indent=2))
    if not checksum_validation.get("match", False):
        raise RuntimeError("ChEMBL 37 published checksum missing or mismatched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

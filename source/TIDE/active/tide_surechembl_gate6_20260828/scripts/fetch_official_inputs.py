#!/usr/bin/env python3
"""Fetch immutable official Gate-6 inputs with resumable downloads and receipts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

SURE_ROOT = "https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data"
CHEMBL_ROOT = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37"
DATES = [
    "2026-06-01",
    "2026-06-15",
    "2026-07-01",
    "2026-07-17",
    "2026-08-04",
    "2026-08-18",
    "2026-08-25",
]
REUSE = {
    "2026-08-18": "/workspace/TIDE/active/tide_surechembl_gate0_20260827/data/2026-08-18/fpsim2_fingerprints.h5",
    "2026-08-25": "/workspace/TIDE/active/tide_surechembl_gate0_20260827/data/2026-08-25/fpsim2_fingerprints.h5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, check=True)
    else:
        with stdout.open("wb") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def fetch(url: str, destination: Path, receipt: Path) -> None:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "--fail", "--location", "--silent", "--show-error", "--head", url], receipt)
    if destination.exists():
        return
    part = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl", "--fail", "--location", "--show-error", "--retry", "8",
            "--retry-all-errors", "--continue-at", "-", "--output", str(part), url,
        ],
        check=True,
    )
    os.replace(part, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "data" / "official"
    receipts = root / "receipts" / "sources"
    data.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    lock_path = Path("/tmp/tide_gate6_data_fetch.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records: list[dict[str, object]] = []
        for date in DATES:
            destination = data / "surechembl" / date / "fpsim2_fingerprints.h5"
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"{SURE_ROOT}/{date}/fpsim2_fingerprints.h5"
            header = receipts / f"surechembl_{date}_headers.txt"
            run(["curl", "--fail", "--location", "--silent", "--show-error", "--head", url], header)
            reused = date in REUSE
            if reused:
                source = Path(REUSE[date])
                if not source.is_file():
                    raise FileNotFoundError(source)
                if destination.exists() or destination.is_symlink():
                    if destination.resolve() != source.resolve():
                        raise RuntimeError(f"unexpected existing target: {destination}")
                else:
                    destination.symlink_to(source)
            else:
                fetch(url, destination, header)
            records.append({
                "kind": "surechembl",
                "release": date,
                "url": url,
                "path": str(destination),
                "resolved_path": str(destination.resolve()),
                "reused_gate0": reused,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "headers": str(header),
            })

        small = ["README", "LICENSE", "REQUIRED.ATTRIBUTION", "checksums.txt"]
        for name in small:
            destination = data / "chembl_37" / name
            fetch(f"{CHEMBL_ROOT}/{name}", destination, receipts / f"chembl37_{name}_headers.txt")
        destination = data / "chembl_37" / "chembl_37.h5"
        url = f"{CHEMBL_ROOT}/chembl_37.h5"
        fetch(url, destination, receipts / "chembl37_h5_headers.txt")
        records.append({
            "kind": "chembl",
            "release": "37",
            "url": url,
            "path": str(destination),
            "resolved_path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "headers": str(receipts / "chembl37_h5_headers.txt"),
            "published_checksums": str(data / "chembl_37" / "checksums.txt"),
        })
        manifest = {
            "experiment_id": "tide_20260828_gate6_source_acquisition",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lock": str(lock_path),
            "sources": records,
        }
        output = data / "SOURCE_MANIFEST.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

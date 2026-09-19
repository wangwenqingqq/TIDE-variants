#!/usr/bin/env python3
"""Validate and seal a C1 minimal CUDA semantic-closure receipt."""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

KEY_FILES = [
    "CMakeLists.txt",
    "include/update.cuh",
    "include/search.cuh",
    "src/main.cu",
    "src/c1_minimal_closure.cu",
]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: pathlib.Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lease", required=True)
    parser.add_argument("--command", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    required = {
        "schema": "tide-c1-minimal-gpu-v1",
        "status": "pass",
        "c2_enabled": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise SystemExit(f"invalid summary: {key}={summary.get(key)!r}, expected {expected!r}")
    if summary.get("queries_checked", 0) <= 0:
        raise SystemExit("invalid summary: no queries were oracle-checked")

    files = {}
    for rel in KEY_FILES:
        path = args.source / rel
        if not path.is_file():
            raise SystemExit(f"missing source file: {path}")
        files[rel] = sha256(path)

    receipt = {
        "schema": "tide-c1-minimal-gpu-receipt-v1",
        "status": "pass",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_uuid": args.gpu_uuid,
        "gpu_lease": args.lease,
        "command_sha256": sha256(args.command),
        "summary_sha256": sha256(args.summary),
        "source_git_head": git_head(args.source),
        "source_files_sha256": files,
        "semantic_summary": summary,
        "claims_not_made": [
            "native GTS traversal integration",
            "query latency or speedup",
            "C2 pruning benefit",
            "concurrent mutation correctness",
        ],
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "receipt": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # Evidence must fail closed.
        print(f"receipt validation failed: {exc}", file=sys.stderr)
        raise

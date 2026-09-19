#!/usr/bin/env python3
"""Create or verify a content manifest for this source-only package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

MANIFEST_NAME = "PACKAGE_MANIFEST.json"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink prohibited: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            raise ValueError(f"bytecode artifact prohibited: {path}")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": "tide.synthetic-e2e3-sourceonly-manifest.v1",
        "status": "SOURCE_ONLY_SYNTHETIC_DEVELOPMENT_PACKAGE",
        "payload_files": records,
        "nonclaim": "File hashes provide only package integrity; no experiment, performance, or runtime claim is made.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / MANIFEST_NAME
    built = build_manifest(root)
    if args.write:
        with manifest_path.open("xb") as handle:
            handle.write(canonical_bytes(built))
    if args.verify:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored != build_manifest(root):
            raise ValueError("manifest does not match package payload")
    if not args.write and not args.verify:
        parser.error("select --write and/or --verify")
    print(json.dumps({"status": "PASS_MANIFEST", "files": len(built["payload_files"])}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

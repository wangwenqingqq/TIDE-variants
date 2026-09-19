#!/usr/bin/env python3
"""CPU-only source/binary provenance manifest for the isolated C1 runner."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text_command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def cache_value(cache: Path, key: str) -> str | None:
    if not cache.is_file():
        return None
    pattern = re.compile(rf'^{re.escape(key)}(?::[^=]+)?=(.*)$')
    for line in cache.read_text(errors='replace').splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    wt = root / 'worktree'
    builds = root / 'builds'
    if not wt.is_dir() or not builds.is_dir():
        raise SystemExit('missing isolated worktree or builds')
    sources = []
    for p in sorted(wt.rglob('*')):
        if p.is_file():
            sources.append({'path': str(p.relative_to(wt)), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    variants = []
    for build in sorted(p for p in builds.iterdir() if p.is_dir()):
        binary = build / 'bin' / 'C1Microbench'
        if not binary.is_file():
            continue
        cache = build / 'CMakeCache.txt'
        variants.append({
            'name': build.name,
            'binary': str(binary),
            'binary_sha256': sha256(binary),
            'binary_bytes': binary.stat().st_size,
            'cmake_cache': str(cache),
            'C1_PERSISTENT_WORKSPACE': cache_value(cache, 'C1_PERSISTENT_WORKSPACE'),
            'C1_ONE_QUERY_FASTPATH': cache_value(cache, 'C1_ONE_QUERY_FASTPATH'),
            'CMAKE_CUDA_COMPILER': cache_value(cache, 'CMAKE_CUDA_COMPILER'),
            'CMAKE_BUILD_TYPE': cache_value(cache, 'CMAKE_BUILD_TYPE'),
        })
    out = {
        'schema': 'gtspp-c1-worktree-build-manifest-v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU-only manifest creation; no GPU binary execution',
        'host': platform.node(),
        'worktree': str(wt),
        'source_files': sources,
        'variants': variants,
        'toolchain': {
            'cmake_version': text_command('cmake', '--version'),
            'nvcc_version': text_command('/usr/local/cuda-13.1/bin/nvcc', '--version'),
        },
    }
    path = root / 'worktree_build_manifest.json'
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

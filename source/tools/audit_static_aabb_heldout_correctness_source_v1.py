#!/usr/bin/env python3
"""CPU-only source audit for the frozen 64-query static-AABB held-out checker."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
SOURCE = ROOT / "src/static_aabb_topk_heldout_v1.cu"
BOUND = ROOT / "include/aabb_static_bound_v2_diskguard.cuh"
SEARCH = ROOT / "include/search_static_aabb_v2_diskguard.cuh"
TREE = ROOT / "include/tree.cuh"
CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
LEGACY_AUDIT = ROOT / "tools/audit_diskguard_v2_1_sources.py"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
FULL_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
EXPECTED = {
    "source": "ff807e47189337520439c1b7f9996111d5104f693af020f3f1418b07dc2e4c88",
    "bound": "1747b53494ab9f9ed6632a528e9a7da17edba2b2f2846c9192b5c4347eb69626",
    "search": "3ed53b958b2611c6419376c69741f27fd7f6763b9d41bf1fb96daba98341da7f",
    "contract": "5c2f6b392d3fabffd323057033fbbcffd74604829aed2450094ba3f18757e435",
}


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path: Path) -> list[int]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid IDs file: {path}")
    try:
        ids = [int(line) for line in path.read_text().splitlines()]
    except ValueError as exc:
        raise RuntimeError(f"non-integer ID: {path}") from exc
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"IDs not strictly sorted/distinct: {path}")
    return ids


def need(text: str, marker: str) -> None:
    if marker not in text:
        raise RuntimeError(f"source marker absent: {marker}")


def main() -> int:
    for path in (SOURCE, BOUND, SEARCH, TREE, CONTRACT, LEGACY_AUDIT, EXACT_IDS, FULL_IDS):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid path: {path}")
    actual = {name: sha256(path) for name, path in {
        "source": SOURCE, "bound": BOUND, "search": SEARCH, "contract": CONTRACT,
    }.items()}
    for name, expected in EXPECTED.items():
        if actual[name] != expected:
            raise RuntimeError(f"frozen SHA mismatch: {name}")

    source = SOURCE.read_text()
    for marker in (
        '#include "tree.cuh"',
        '#include "search_static_aabb_v2_diskguard.cuh"',
        "constexpr int kDim=128;",
        "constexpr int kBaseN=1000000;",
        "if((int)v.size()!=64)",
        "held-out correctness requires exactly 64 pre-registered IDs",
        "std::vector<int> all_dimensions()",
        "for(int d=0; d<kDim; ++d) out[d]=d;",
        "uint64_t raw_l2_sift_integer_sq",
        "#pragma omp parallel for schedule(static)",
        "std::vector<std::array<int,kK>> exact_oracle",
        "searchIndexKnnStaticAabbV2DiskGuard",
        "PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1",
        "formal_claim_eligible\\\":false",
        "snapshot_pre_post_byte_equal",
    ):
        need(source, marker)
    for prohibited in ("--groundtruth", "groundtruth", "cudaEvent", "timed_", "rp_predict(", "recordGammaOnlyPrune"):
        if prohibited in source:
            raise RuntimeError(f"prohibited source marker: {prohibited}")

    exact = read_ids(EXACT_IDS)
    full = read_ids(FULL_IDS)
    if len(exact) != 64 or len(full) != 256 or exact != full[:64]:
        raise RuntimeError("frozen exact64/full256 relation")
    if any(qid < 5000 or qid >= 10000 or qid <= 31 or 1000 <= qid <= 1259 for qid in full):
        raise RuntimeError("heldout exclusion violation")

    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "PASS_CPU_ONLY":
        raise RuntimeError("disk contract not passing")
    legacy = json.loads(subprocess.check_output([str(LEGACY_AUDIT)], text=True))
    if legacy.get("status") != "PASS":
        raise RuntimeError("diskguard dependency source audit failed")
    result = {
        "schema": "safe-c2-static-aabb-heldout-correctness-source-audit-v1",
        "status": "PASS_CPU_ONLY_SOURCE_AUDIT",
        "scope": (
            "source-only audit of the 64-query independent-CPU-oracle held-out correctness "
            "checker; no compile, GPU execution, nvidia-smi, timing, old-C2 validation, or sealed access"
        ),
        "correctness_source": {"path": str(SOURCE), "sha256": actual["source"], "bytes": SOURCE.stat().st_size},
        "dependencies": {
            "diskguard_bound_header": {"path": str(BOUND), "sha256": actual["bound"]},
            "diskguard_search_header": {"path": str(SEARCH), "sha256": actual["search"]},
            "tree_header": {"path": str(TREE), "sha256": sha256(TREE)},
            "sift_integer_disk_contract": {"path": str(CONTRACT), "sha256": actual["contract"], "status": contract["status"]},
            "existing_diskguard_dependency_audit": {
                "path": str(LEGACY_AUDIT), "sha256": sha256(LEGACY_AUDIT), "result": legacy
            },
        },
        "fixed_inputs": {
            "exact64_ids": {"path": str(EXACT_IDS), "sha256": sha256(EXACT_IDS), "count": len(exact)},
            "full256_ids": {"path": str(FULL_IDS), "sha256": sha256(FULL_IDS), "count": len(full)},
            "exact64_is_full256_prefix": True,
        },
        "source_contract": {
            "dimension": 128,
            "coordinate_policy": "all raw dimensions in ascending order",
            "correctness_query_count": 64,
            "oracle": "CPU uint64 integer-squared exact top-10 over 1,000,000 base vectors; no GT input",
            "openmp_pragma_present": True,
            "openmp_compile_flag_required": "-Xcompiler=-fopenmp",
            "no_timing_or_gt_input_in_source": True,
        },
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_SOURCE_AUDIT_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)

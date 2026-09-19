#!/usr/bin/env python3
"""CPU-only static audit for the Safe-C2 static-AABB held-out correctness runner."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
SOURCE = ROOT / "src/static_aabb_topk_heldout_v1.cu"
BOUND = ROOT / "include/aabb_static_bound_v2_diskguard.cuh"
HEADER = ROOT / "include/search_static_aabb_v2_diskguard.cuh"
CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
EXPECTED = {
    "source": "ff807e47189337520439c1b7f9996111d5104f693af020f3f1418b07dc2e4c88",
    "bound": "1747b53494ab9f9ed6632a528e9a7da17edba2b2f2846c9192b5c4347eb69626",
    "header": "3ed53b958b2611c6419376c69741f27fd7f6763b9d41bf1fb96daba98341da7f",
    "contract": "5c2f6b392d3fabffd323057033fbbcffd74604829aed2450094ba3f18757e435",
    "ids": "154d069149a1a6ced802f341db7df7b8dc803dce7a74e65889649a39fd65e633",
}

def sha(p: Path) -> str:
    if not p.is_file() or p.is_symlink():
        raise RuntimeError(f"invalid regular file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def need(text: str, token: str, name: str) -> None:
    if token not in text:
        raise RuntimeError(f"missing {name}: {token}")

def main() -> int:
    files = {"source": SOURCE, "bound": BOUND, "header": HEADER, "contract": CONTRACT, "ids": IDS}
    got = {name: sha(path) for name, path in files.items()}
    for name, digest in EXPECTED.items():
        if got[name] != digest:
            raise RuntimeError(f"frozen {name} SHA mismatch")

    s = SOURCE.read_text()
    for token, name in [
        ("if((int)v.size()!=64)", "exact heldout64 guard"),
        ("all_dimensions()", "all-coordinate selector"),
        ("for(int d=0; d<kDim; ++d) out[d]=d;", "all 128 dimensions"),
        ("raw_l2_sift_integer_sq", "integer exact CPU oracle"),
        ("uint64_t", "integer exact accumulator"),
        ("#pragma omp parallel for schedule(static)", "parallel CPU oracle"),
        ("searchIndexKnnV2(", "radial baseline route"),
        ("searchIndexKnnStaticAabbV2DiskGuard(", "static AABB route"),
        ("PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1", "terminal pass schema"),
        ("snapshot_pre_post_byte_equal", "snapshot invariant"),
        ("base violates SIFT integer-float [0,255] contract", "base contract"),
        ("query violates SIFT integer-float [0,255] contract", "query contract"),
    ]:
        need(s, token, name)
    for forbidden in ("--projection-dims", "top_variance_dims", "rp_predict(", "GammaOnlyPruneTrace"):
        if forbidden in s:
            raise RuntimeError(f"forbidden heldout-source token: {forbidden}")

    b = BOUND.read_text()
    for token in ("__fsub_rd", "__fmul_rd", "__fadd_rd", "static_aabb_next_up_nonnegative", "__fmul_ru", "static_aabb_strict_prune"):
        need(b, token, "diskguard bound")
    h = HEADER.read_text()
    start = h.find("__global__ void nodeProcessKnnStaticAabb(")
    if start < 0:
        raise RuntimeError("static candidate kernel missing")
    candidate = h[start:]
    for token in ("static_aabb_strict_prune", "searchIndexKnnStaticAabbV2DiskGuard"):
        need(candidate, token, "candidate route")
    for forbidden in ("rp_predict(", "recordGammaOnlyPrune", "GammaOnlyPruneTrace"):
        if forbidden in candidate:
            raise RuntimeError(f"legacy gamma reachable in candidate: {forbidden}")

    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "PASS_CPU_ONLY":
        raise RuntimeError("SIFT disk contract status")
    ids = [int(line) for line in IDS.read_text().splitlines()]
    if len(ids) != 64 or ids != sorted(ids) or len(set(ids)) != 64:
        raise RuntimeError("heldout exact64 shape")
    if ids != list(range(5000, 5064)):
        raise RuntimeError("heldout exact64 frozen prefix")
    result = {
        "schema": "safe-c2-static-aabb-heldout-correctness-source-audit-v1",
        "status": "PASS",
        "scope": "CPU static audit only; source is not compiled or GPU-executed by this audit",
        "source_sha256": got["source"],
        "bound_sha256": got["bound"],
        "header_sha256": got["header"],
        "contract_sha256": got["contract"],
        "ids_sha256": got["ids"],
        "query_count": 64,
        "projection": "all 128 raw SIFT coordinates",
        "numeric_scope": "SIFT integer-float disk contract only",
        "formal_claim_eligible": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_SOURCE_AUDIT_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""CPU-only audit for the frozen static-AABB development-held-out protocol."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
PROTOCOL = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
SELECTION = ROOT / "provenance/static_aabb_heldout_selection_v1.json"
IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
BASE = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs")
QUERY = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs")
GT = Path("/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs")
CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
AUDIT = ROOT / "tools/audit_diskguard_v2_1_sources.py"
RUNNER = ROOT / "tools/run_diskguard_v2_1_m128.sh"
DIM = 128
BASE_COUNT = 1_000_000
QUERY_COUNT = 10_000
GT_WIDTH = 100
SCAN_START = 5000
SCAN_STOP_EXCLUSIVE = 10_000
TARGET_COUNT = 256
EXPECTED_HASHES = {
    "bound_header": "1747b53494ab9f9ed6632a528e9a7da17edba2b2f2846c9192b5c4347eb69626",
    "search_header": "3ed53b958b2611c6419376c69741f27fd7f6763b9d41bf1fb96daba98341da7f",
    "topk_source": "42398aa18e34091dcc96c05cc4c7edea23c85ae6dd3dc830da22877186729af3",
    "perf_source": "5dc92a9660bc168e1325edae7a1c9d41f59a93e036e83f57f2ccd9d55fa6819f",
    "topk_binary": "d33e0fb3ba7757195d376bc6a7a5b579d9578bf0e8a6928cfa2670ae640bcf19",
    "perf_binary": "e587ad822de007880e1caae8ea89072c07a5b786dcb5d15357ecde61ce577cde",
}


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_fvec_at(path: Path, row: int, count: int) -> tuple[float, ...]:
    if not 0 <= row < count:
        raise RuntimeError(f"fvec row out of range: {row}")
    stride = 4 + 4 * DIM
    with path.open("rb") as f:
        f.seek(row * stride)
        raw_dim = f.read(4)
        if len(raw_dim) != 4 or struct.unpack("<i", raw_dim)[0] != DIM:
            raise RuntimeError(f"bad fvec header at row {row}")
        raw = f.read(4 * DIM)
        if len(raw) != 4 * DIM:
            raise RuntimeError(f"short fvec at row {row}")
        return struct.unpack("<128f", raw)


def read_gt_at(path: Path, row: int) -> tuple[int, ...]:
    stride = 4 + 4 * GT_WIDTH
    with path.open("rb") as f:
        f.seek(row * stride)
        raw_dim = f.read(4)
        if len(raw_dim) != 4 or struct.unpack("<i", raw_dim)[0] != GT_WIDTH:
            raise RuntimeError(f"bad GT header at row {row}")
        raw = f.read(4 * GT_WIDTH)
        if len(raw) != 4 * GT_WIDTH:
            raise RuntimeError(f"short GT row {row}")
        ids = struct.unpack("<100i", raw)
    if any(index < 0 or index >= BASE_COUNT for index in ids[:11]):
        raise RuntimeError(f"invalid GT ID at row {row}")
    return ids


def sq_l2(q: tuple[float, ...], x: tuple[float, ...]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(q, x))


def expected_selection() -> tuple[list[int], list[dict[str, object]]]:
    selected: list[int] = []
    witness: list[dict[str, object]] = []
    for qid in range(SCAN_START, SCAN_STOP_EXCLUSIVE):
        q = read_fvec_at(QUERY, qid, QUERY_COUNT)
        gt = read_gt_at(GT, qid)
        d10 = sq_l2(q, read_fvec_at(BASE, gt[9], BASE_COUNT))
        d11 = sq_l2(q, read_fvec_at(BASE, gt[10], BASE_COUNT))
        if d10 < d11:
            selected.append(qid)
            witness.append({"qid": qid, "d10_sq_float64": d10, "d11_sq_float64": d11})
            if len(selected) == TARGET_COUNT:
                break
    if len(selected) != TARGET_COUNT:
        raise RuntimeError("short deterministic heldout selection")
    return selected, witness


def main() -> int:
    for path in (PROTOCOL, SELECTION, IDS, EXACT_IDS, BASE, QUERY, GT, CONTRACT, AUDIT, RUNNER):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid input path: {path}")
    if BASE.stat().st_size != BASE_COUNT * (4 + 4 * DIM):
        raise RuntimeError("base size")
    if QUERY.stat().st_size != QUERY_COUNT * (4 + 4 * DIM):
        raise RuntimeError("query size")
    if GT.stat().st_size != QUERY_COUNT * (4 + 4 * GT_WIDTH):
        raise RuntimeError("GT size")

    protocol = json.loads(PROTOCOL.read_text())
    selection = json.loads(SELECTION.read_text())
    if protocol.get("schema") != "safe-c2-static-aabb-heldout-protocol-v1":
        raise RuntimeError("protocol schema")
    if protocol.get("status") != "FROZEN_CPU_ONLY_PREPARED":
        raise RuntimeError("protocol state")
    if protocol.get("hard_boundaries", {}).get("gpu_executed_for_this_protocol") is not False:
        raise RuntimeError("protocol claims GPU execution")
    if protocol.get("algorithm_freeze", {}).get("projection_dimensions") != 128:
        raise RuntimeError("not all-coordinate protocol")
    if protocol["heldout_input"]["selection_metadata"]["sha256"] != sha256(SELECTION):
        raise RuntimeError("selection provenance hash")
    if protocol["heldout_input"]["ids_file"]["sha256"] != sha256(IDS):
        raise RuntimeError("IDs hash")
    if selection["output_ids"]["sha256"] != sha256(IDS):
        raise RuntimeError("selection ID hash")
    if selection["gpu_heldout_correctness_subset"]["ids_sha256"] != sha256(EXACT_IDS):
        raise RuntimeError("exact subset ID hash")
    if selection["inputs"][str(BASE)]["sha256"] != sha256(BASE):
        raise RuntimeError("base input hash")
    if selection["inputs"][str(QUERY)]["sha256"] != sha256(QUERY):
        raise RuntimeError("query input hash")
    if selection["inputs"][str(GT)]["sha256"] != sha256(GT):
        raise RuntimeError("GT input hash")

    file_ids = [int(line) for line in IDS.read_text().splitlines()]
    recomputed_ids, recomputed_witness = expected_selection()
    if file_ids != recomputed_ids or selection["selected_ids"] != recomputed_ids:
        raise RuntimeError("deterministic selection mismatch")
    if selection["boundary_witness"] != recomputed_witness:
        raise RuntimeError("boundary witness mismatch")
    if len(file_ids) != TARGET_COUNT or len(set(file_ids)) != TARGET_COUNT or file_ids != sorted(file_ids):
        raise RuntimeError("heldout ID uniqueness/order")
    if any(qid < SCAN_START or qid >= SCAN_STOP_EXCLUSIVE or qid <= 31 or 1000 <= qid <= 1259 for qid in file_ids):
        raise RuntimeError("heldout disjointness")
    exact_ids = [int(line) for line in EXACT_IDS.read_text().splitlines()]
    expected_exact = file_ids[:64]
    if exact_ids != expected_exact or selection["gpu_heldout_correctness_subset"]["selected_ids"] != expected_exact:
        raise RuntimeError("exact correctness subset")
    exact_input = protocol["heldout_input"]["gpu_heldout_correctness"]
    if exact_input["selected_ids"] != expected_exact or exact_input["ids_file"]["sha256"] != sha256(EXACT_IDS):
        raise RuntimeError("protocol exact correctness subset")
    perf_input = protocol["heldout_input"]["gt_gated_heldout_performance"]
    if perf_input["ids_file"]["sha256"] != sha256(IDS) or perf_input["query_count"] != 256:
        raise RuntimeError("protocol GT-gated performance set")

    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "PASS_CPU_ONLY":
        raise RuntimeError("disk contract")
    for path in (BASE, QUERY):
        if contract[str(path).split("/")[-1] if False else "base" if path == BASE else "query"]["sha256"] != sha256(path):
            raise RuntimeError("disk contract hash")
    source_audit = json.loads(subprocess.check_output([str(AUDIT)], text=True))
    if source_audit.get("status") != "PASS":
        raise RuntimeError("source audit")
    for name, expected in EXPECTED_HASHES.items():
        artifact = protocol["algorithm_freeze"]["source_and_binary_artifacts"][name]
        path = Path(artifact["path"])
        if artifact["sha256"] != expected or sha256(path) != expected:
            raise RuntimeError(f"frozen artifact mismatch: {name}")
    if protocol["algorithm_freeze"]["source_audit"]["sha256"] != sha256(AUDIT):
        raise RuntimeError("source audit tool hash")
    if protocol["algorithm_freeze"]["source_audit"]["result"] != source_audit:
        raise RuntimeError("source audit result changed")

    runner_text = RUNNER.read_text()
    disposition = protocol["current_runner_disposition"]
    if disposition["sha256"] != sha256(RUNNER) or disposition["reuse_for_heldout_formal_execution"] is not False:
        raise RuntimeError("exploratory runner disposition")
    if "EXPLORATORY" not in runner_text or "formal_claim_eligible" not in runner_text:
        raise RuntimeError("runner marker")
    result = {
        "schema": "safe-c2-static-aabb-heldout-preflight-v1",
        "status": "PASS_CPU_ONLY_FROZEN_EXECUTION_PENDING",
        "scope": "audit of frozen static-AABB development-held-out protocol; no GPU or old-C2 validation/sealed access",
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "heldout_ids": {"path": str(IDS), "sha256": sha256(IDS), "count": len(file_ids), "first": file_ids[0], "last": file_ids[-1]},
        "gpu_heldout_correctness_exact64": {
            "path": str(EXACT_IDS), "sha256": sha256(EXACT_IDS), "count": len(exact_ids),
            "same_frozen_heldout256_prefix": exact_ids == file_ids[:64],
            "oracle": "required independent CPU exact top-10; not executed by this CPU-only preflight",
        },
        "gt_gated_heldout_performance": {
            "path": str(IDS), "sha256": sha256(IDS), "count": len(file_ids),
            "GT_role": "fixed standard label only; not a calibration signal",
        },
        "selection_recomputed_cpu_only": True,
        "source_audit": source_audit,
        "current_v2_1_runner_reuse_for_heldout_formal_execution": False,
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
        "formal_claim_eligible": False,
        "next_required_action": (
            "freeze and audit a separate heldout execution runner with preflight/guard/terminal/manifest "
            "statuses before any GPU run; only then can a heldout run be assessed for paper eligibility"
        ),
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_PREFLIGHT_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)

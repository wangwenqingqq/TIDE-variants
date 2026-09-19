#!/usr/bin/env python3
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "gts_safe_c3_leaf_envelope.cu"
REF = ROOT / "reference" / "gts_c3_native_direct_insert_probe.cu"
PROTO = ROOT / "protocol" / "safe_c3_leaf_envelope_v1.json"
CANON = Path("/workspace/experiments/tide_safe_c1_20260727/source_gts_incremental/include")

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fail(msg: str) -> None:
    raise SystemExit(f"STATIC_AUDIT_FAIL: {msg}")

for p in (SRC, REF, PROTO, CANON / "tree.cuh", CANON / "search_v2.cuh", CANON / "residual_pruning.cuh"):
    if not p.is_file() or p.is_symlink():
        fail(f"unsafe/missing prerequisite {p}")

source = SRC.read_text(encoding="utf-8")
protocol = json.loads(PROTO.read_text(encoding="utf-8"))
if protocol.get("candidate", {}).get("stable_id") != 4102:
    fail("protocol candidate binding")
if protocol.get("candidate", {}).get("expected_route") != [0, 5, 60, 604]:
    fail("protocol route binding")
if protocol.get("mutation_whitelist", {}).get("tn_size_increment_nodes") != [0, 5, 60, 604]:
    fail("protocol size whitelist")
if protocol.get("mutation_whitelist", {}).get("max_dis_update", {}).get("node") != 604:
    fail("protocol max whitelist")

required = [
    "#define main c3_archived_leaf_only_probe_main_disabled",
    "safe_c3_commit_once_kernel",
    "cuMemGetAddressRange",
    "kSafeCandidateStableId = 4102",
    "kSafeExpectedLeaf = 604",
    "kSafeExpectedRoute{{0, 5, 60, 604}}",
    "id_list[write_slot] = stable_id",
    "node_list[route_nodes[i]].size = expected_sizes[i] + 1",
    "max_distance[max_nodes[i]] = next_max[i]",
    "validate_parent_cardinality",
    "validate_cover_and_sibling_fences",
    "run_real_static_topk",
    "searchIndexKnnV2",
    "exact_full_live_range_fallback_no_gts_receipt",
    "native_range_correctness_claim",
    "write_atomic_no_overwrite",
]
for token in required:
    if token not in source:
        fail(f"missing required source token: {token}")

forbidden = ["incrementalInsert(", "findTargetLeaf(", "searchIndexRnn", "nvidia-smi", "--max-accepted"]
for token in forbidden:
    if token in source:
        fail(f"forbidden source token: {token}")

# The old entry point is renamed solely by macro and must never be called.
if re.search(r"\bc3_archived_leaf_only_probe_main_disabled\s*\(", source):
    fail("archived leaf-only entry point is called")

header_hashes = {name: sha(CANON / name) for name in ("tree.cuh", "search_v2.cuh", "residual_pruning.cuh")}
expected_headers = {
    "tree.cuh": "f812c385d254d77e84bbd515e4055e94cc8163ee7a7d57216f3659c09f569c2b",
    "search_v2.cuh": "3847a244787dd8d9f3a21d055dc08ba3f396bb9bd79a3281c8579a45b87eb89d",
    "residual_pruning.cuh": "745a4bd5564b8085c40a48abac625f24dcbacb8a996d29e4cba311dd936e33b2",
}
if header_hashes != expected_headers:
    fail(f"archival header hash drift: {header_hashes}")

report = {
    "schema": "safe-c3-leaf-static-audit-v1",
    "status": "PASS_STATIC_AUDIT",
    "source_sha256": sha(SRC),
    "reference_probe_sha256": sha(REF),
    "protocol_sha256": sha(PROTO),
    "archival_header_sha256": header_hashes,
    "assertions": {
        "fixed_candidate_and_route": True,
        "full_path_sizes": True,
        "required_ancestor_max_update": True,
        "measured_id_list_capacity": True,
        "full_d2h_whitelist_audit": True,
        "real_native_vector_knn": True,
        "no_native_range_claim": True,
        "no_legacy_incremental_fallback": True,
        "no_nvidia_smi": True,
    },
}
print(json.dumps(report, sort_keys=True, separators=(",", ":")))


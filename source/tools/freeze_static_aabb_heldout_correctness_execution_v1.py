#!/usr/bin/env python3
"""Freeze the source-bound execution contract for held-out GPU correctness only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1")
BASE_PROTOCOL = ROOT / "provenance/static_aabb_heldout_protocol_v1.json"
OUT = ROOT / "provenance/static_aabb_heldout_correctness_execution_v1.json"
AUDIT = ROOT / "tools/audit_static_aabb_heldout_correctness_source_v1.py"
SOURCE = ROOT / "src/static_aabb_topk_heldout_v1.cu"
BOUND = ROOT / "include/aabb_static_bound_v2_diskguard.cuh"
SEARCH = ROOT / "include/search_static_aabb_v2_diskguard.cuh"
CONTRACT = ROOT / "provenance/sift_integer_disk_contract_v1.json"
EXACT_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout_exact64_v1.ids"
FULL_IDS = ROOT / "inputs/standard_sift_static_aabb_heldout256_v1.ids"
PLANNED_BINARY = ROOT / "bin/static_aabb_topk_heldout_v1"
EXPECTED_SOURCE_SHA = "ff807e47189337520439c1b7f9996111d5104f693af020f3f1418b07dc2e4c88"
EXPECTED_BOUND_SHA = "1747b53494ab9f9ed6632a528e9a7da17edba2b2f2846c9192b5c4347eb69626"
EXPECTED_SEARCH_SHA = "3ed53b958b2611c6419376c69741f27fd7f6763b9d41bf1fb96daba98341da7f"
EXPECTED_CONTRACT_SHA = "5c2f6b392d3fabffd323057033fbbcffd74604829aed2450094ba3f18757e435"


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing pre-existing output: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    for path in (BASE_PROTOCOL, AUDIT, SOURCE, BOUND, SEARCH, CONTRACT, EXACT_IDS, FULL_IDS):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid required input: {path}")
    if OUT.exists() or OUT.is_symlink():
        raise RuntimeError("execution protocol already exists")
    if PLANNED_BINARY.exists() or PLANNED_BINARY.is_symlink():
        raise RuntimeError("planned correctness binary already exists; this source-only freeze is no longer applicable")

    base = json.loads(BASE_PROTOCOL.read_text())
    if base.get("schema") != "safe-c2-static-aabb-heldout-protocol-v1" or base.get("status") != "FROZEN_CPU_ONLY_PREPARED":
        raise RuntimeError("base heldout protocol status")
    if base.get("formal_claim_eligible") is not False:
        raise RuntimeError("base protocol formal status")
    exact_binding = base["heldout_input"]["gpu_heldout_correctness"]
    perf_binding = base["heldout_input"]["gt_gated_heldout_performance"]
    if (exact_binding["query_count"] != 64 or exact_binding["ids_file"]["sha256"] != sha256(EXACT_IDS) or
            perf_binding["query_count"] != 256 or perf_binding["ids_file"]["sha256"] != sha256(FULL_IDS)):
        raise RuntimeError("base heldout split binding")

    audit = json.loads(subprocess.check_output([str(AUDIT)], text=True))
    if audit.get("status") != "PASS_CPU_ONLY_SOURCE_AUDIT":
        raise RuntimeError("correctness source audit")
    if audit["correctness_source"]["sha256"] != EXPECTED_SOURCE_SHA:
        raise RuntimeError("new source SHA")
    if audit["dependencies"]["diskguard_bound_header"]["sha256"] != EXPECTED_BOUND_SHA:
        raise RuntimeError("bound header SHA")
    if audit["dependencies"]["diskguard_search_header"]["sha256"] != EXPECTED_SEARCH_SHA:
        raise RuntimeError("search header SHA")
    if audit["dependencies"]["sift_integer_disk_contract"]["sha256"] != EXPECTED_CONTRACT_SHA:
        raise RuntimeError("disk contract SHA")

    compile_argv = [
        "/usr/local/cuda-13.1/bin/nvcc", "-std=c++17", "-O3", "--ftz=false",
        "--generate-code=arch=compute_120,code=[compute_120,sm_120]",
        "-Xcompiler=-fopenmp", f"-I{ROOT / 'include'}", str(SOURCE), "-o", str(PLANNED_BINARY),
    ]
    execution = {
        "schema": "safe-c2-static-aabb-heldout-correctness-execution-v1",
        "status": "FROZEN_CPU_ONLY_SOURCE_BOUND_NOT_BUILT",
        "scope": (
            "successor execution contract for only the pre-registered 64-query static-AABB held-out "
            "GPU correctness check on standard SIFT1M; no timing or performance claim"
        ),
        "inherits_heldout_protocol": {
            "path": str(BASE_PROTOCOL),
            "sha256": sha256(BASE_PROTOCOL),
            "formal_claim_eligible": False,
            "old_v2_1_runner_reuse_allowed": False,
        },
        "fixed_input_split": {
            "gpu_heldout_correctness": {
                "ids_file": exact_binding["ids_file"],
                "query_count": 64,
                "ids": exact_binding["selected_ids"],
                "oracle": "independent CPU uint64 integer-squared exact top-10 across all 1,000,000 base vectors; no GT input",
            },
            "gt_gated_heldout_performance": {
                "ids_file": perf_binding["ids_file"],
                "query_count": 256,
                "status": "NOT_IMPLEMENTED_BY_THIS_EXECUTION_CONTRACT",
                "GT_role": perf_binding["GT_role"],
            },
            "immutability": (
                "the selector created these new inputs with refuse-to-overwrite behavior; this execution "
                "contract requires the recorded SHA-256 values and forbids replacement"
            ),
        },
        "implementation_freeze": {
            "source": audit["correctness_source"],
            "source_audit": {"path": str(AUDIT), "sha256": sha256(AUDIT), "result": audit},
            "headers_and_contract": audit["dependencies"],
            "projection": {
                "dimensions": 128,
                "policy": "all raw SIFT coordinates in ascending order",
                "query_or_GT_calibration": "none",
            },
        },
        "build_contract": {
            "status": "NOT_BUILT_NO_BINARY_EXECUTION",
            "compiler_path": "/usr/local/cuda-13.1/bin/nvcc",
            "required_command_argv": compile_argv,
            "required_flags": ["-std=c++17", "-O3", "--ftz=false", "compute_120/sm_120", "-Xcompiler=-fopenmp"],
            "why_openmp_flag_is_required": "the independent CPU exact oracle contains #pragma omp parallel for",
            "planned_binary": {"path": str(PLANNED_BINARY), "sha256": None, "bytes": None},
            "required_build_artifacts_before_gpu": [
                "build log",
                "build JSON binding compiler version, exact command, source/header/contract hashes, binary SHA-256 and byte count",
                "CPU-only source audit pass",
            ],
            "gpu_binary_executed": False,
            "nvidia_smi_called": False,
        },
        "required_successor_runner": {
            "status": "NOT_IMPLEMENTED",
            "must_not_reuse": str(ROOT / "tools/run_diskguard_v2_1_m128.sh"),
            "preflight_requirements": [
                "verify dataset base/query/GT hashes inherited from the base protocol",
                "verify exact64 ID hash, source/header/contract hashes, and build-card binary SHA before launch",
                "verify output directory is fresh and non-symlinked",
                "write preflight.json before GPU invocation",
            ],
            "gpu_guard_requirements": [
                "record GPU index, UUID, model, driver/CUDA version, memory, clocks/power telemetry before and after",
                "require no foreign compute processes; do not kill or disrupt any process",
                "write gpu_guard.json and refuse execution on a nonempty process guard",
            ],
            "terminal_requirements": [
                "always write terminal.json with exit_code and PASS/FAILED terminal state",
                "PASS only if result status is PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1, both baseline and static-AABB error counters are zero, ID sets equal, host-cover violations are zero, and snapshots are byte-identical",
                "write a manifest binding all artifacts by SHA-256",
            ],
        },
        "nonclaims": [
            "No GPU run, compile, binary, performance result, or paper-evidence claim exists yet.",
            "This contract proves neither C3 direct insertion/update safety nor general-metric behavior.",
            "The one-ULP disk guard remains SIFT-integer-contract-specific.",
            "A separate source and runner are still required for the fixed-256 GT-gated performance protocol.",
        ],
        "formal_claim_eligible": False,
    }
    atomic_write(OUT, json.dumps(execution, sort_keys=True, indent=2) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_EXECUTION_FREEZE_V1: {exc}", file=sys.stderr)
        raise SystemExit(2)

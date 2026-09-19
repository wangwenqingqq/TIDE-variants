#!/usr/bin/env python3
"""Independent CPU-only verifier for a future C2 v4 pre-ledger receipt.

It never invokes the binary, CUDA, nvidia-smi, nsys, or a GPU.  The receipt
must already have been produced by the final pinned binary's
--validate-workload-only path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
sys.path.insert(0, str(ROOT / "tools"))
from c2_v4_strict_json_contract import StrictJsonError, require_exact_top_object, strict_loads  # noqa: E402

WORKLOAD_FIELDS = {
    "schema", "status", "selection_pre_gt_path", "selection_pre_gt_sha256",
    "v3_forensic_record_path", "v3_forensic_record_sha256", "formal_protocol_path",
    "formal_protocol_sha256", "sealed_v2_test_forbidden", "v2_sealed_test_ids_sha256_forbidden",
    "base_fvecs_path", "base_fvecs_sha256", "query_fvecs_path", "query_fvecs_sha256",
    "query_fvecs_count", "groundtruth_ivecs_path", "groundtruth_ivecs_sha256",
    "groundtruth_width", "mapping_path", "mapping_sha256", "calibration_ids_path",
    "calibration_ids_sha256", "calibration_query_count", "validation_ids_path",
    "validation_ids_sha256", "validation_query_count", "sealed_test_ids_path",
    "sealed_test_ids_sha256", "sealed_test_query_count",
}
FILE_PAIRS = (
    ("selection_pre_gt", "selection_pre_gt_path", "selection_pre_gt_sha256"),
    ("v3_forensic_record", "v3_forensic_record_path", "v3_forensic_record_sha256"),
    ("formal_protocol", "formal_protocol_path", "formal_protocol_sha256"),
    ("base_fvecs", "base_fvecs_path", "base_fvecs_sha256"),
    ("query_fvecs", "query_fvecs_path", "query_fvecs_sha256"),
    ("groundtruth_ivecs", "groundtruth_ivecs_path", "groundtruth_ivecs_sha256"),
    ("mapping", "mapping_path", "mapping_sha256"),
    ("calibration_ids", "calibration_ids_path", "calibration_ids_sha256"),
    ("validation_ids", "validation_ids_path", "validation_ids_sha256"),
    ("sealed_test_ids", "sealed_test_ids_path", "sealed_test_ids_sha256"),
)
V2_FORBIDDEN = "50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce"


class VerifyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def direct_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise VerifyError(f"{label} is not a direct canonical regular file: {path}")
    return path


def direct_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise VerifyError(f"{label} is not a direct canonical directory: {path}")
    return path


def load_strict(path: Path, label: str) -> dict[str, Any]:
    direct_file(path, label)
    try:
        value = strict_loads(path.read_text(encoding="utf-8"))
    except StrictJsonError as exc:
        raise VerifyError(f"{label} strict JSON failure: {exc}") from exc
    if not isinstance(value, dict):
        raise VerifyError(f"{label} root is not object")
    return value


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise VerifyError(f"invalid SHA-256 {label}")
    return value


def canonical_ids(path: Path, count: int) -> None:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != count:
        raise VerifyError(f"ID count mismatch {path}: {len(lines)} != {count}")
    seen: set[int] = set()
    for line in lines:
        if not line or (len(line) > 1 and line[0] == "0") or not line.isdecimal():
            raise VerifyError(f"noncanonical ID in {path}")
        value = int(line)
        if not 0 <= value < 12524 or value in seen:
            raise VerifyError(f"out-of-range/duplicate ID in {path}")
        seen.add(value)


def verify(*, receipt_path: Path, manifest_path: Path, binary_path: Path) -> dict[str, Any]:
    receipt = load_strict(receipt_path, "receipt")
    manifest = load_strict(manifest_path, "final workload manifest")
    try:
        require_exact_top_object(
            manifest_path.read_text(encoding="utf-8"),
            expected_fields=WORKLOAD_FIELDS,
            expected_strings={
                "schema": "gts-v4-fresh-sift-learn-workload-v1",
                "status": "READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE",
                "v2_sealed_test_ids_sha256_forbidden": V2_FORBIDDEN,
            },
            expected_booleans={"sealed_v2_test_forbidden": True},
            expected_positive_integers={
                "query_fvecs_count", "groundtruth_width", "calibration_query_count",
                "validation_query_count", "sealed_test_query_count",
            },
        )
    except StrictJsonError as exc:
        raise VerifyError(f"final manifest strict top-level contract failure: {exc}") from exc
    if manifest["query_fvecs_count"] != 12524 or manifest["groundtruth_width"] != 100:
        raise VerifyError("final manifest runtime constants mismatch")

    expected_bindings: list[dict[str, Any]] = []
    for label, path_key, sha_key in FILE_PAIRS:
        path = direct_file(Path(manifest[path_key]), f"manifest {label}")
        digest = require_sha(manifest[sha_key], f"manifest {label}")
        observed = sha256_file(path)
        if observed != digest:
            raise VerifyError(f"manifest binding hash mismatch: {label}")
        expected_bindings.append({"label": label, "path": str(path), "sha256": digest, "bytes": path.stat().st_size})

    for stage in ("calibration", "validation", "sealed_test"):
        canonical_ids(Path(manifest[f"{stage}_ids_path"]), manifest[f"{stage}_query_count"])

    direct_file(binary_path, "pinned final binary")
    expected_binary_sha = sha256_file(binary_path)
    if receipt.get("schema") != "safe-c2-v4-workload-validation-receipt-v1":
        raise VerifyError("receipt schema mismatch")
    if receipt.get("status") != "PASS_NO_CUDA_OR_DATASET_LOAD" or receipt.get("invocation") != "--validate-workload-only":
        raise VerifyError("receipt status/invocation mismatch")
    if receipt.get("strict_top_level_parser") is not True:
        raise VerifyError("receipt did not attest strict top-level parser")
    binary = receipt.get("binary")
    manifest_receipt = receipt.get("manifest")
    args = receipt.get("arguments")
    if not isinstance(binary, dict) or binary.get("path") != str(binary_path) or binary.get("sha256") != expected_binary_sha:
        raise VerifyError("receipt binary binding mismatch")
    if not isinstance(manifest_receipt, dict) or manifest_receipt.get("path") != str(manifest_path) or manifest_receipt.get("sha256") != sha256_file(manifest_path):
        raise VerifyError("receipt manifest binding mismatch")
    if args != {"mode": "calibrate", "stage": "calibration"}:
        raise VerifyError("receipt parser-gate argv identity mismatch")
    bindings = receipt.get("input_bindings")
    if not isinstance(bindings, list) or bindings != expected_bindings:
        raise VerifyError("receipt input binding set/order/bytes mismatch")
    return {
        "status": "PASS_PRELEDGER_RECEIPT_CPU_ONLY",
        "receipt": str(receipt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "binary_sha256": expected_binary_sha,
        "bindings": len(expected_bindings),
        "gpu_binary_executed": False,
        "nvidia_smi_called": False,
    }


def self_test() -> dict[str, Any]:
    # Synthetic byte files only: this tests verifier semantics, not the CUDA
    # binary. No user/experiment artifact is touched.
    with tempfile.TemporaryDirectory(prefix="c2_v4_preledger_receipt_test_") as raw:
        root = Path(raw)
        def make(name: str, data: bytes) -> Path:
            p = root / name
            p.write_bytes(data)
            return p
        selection = make("selection.json", b"selection")
        forensic = make("forensic.json", b"forensic")
        protocol = make("protocol.json", b"protocol")
        base = make("base.fvecs", b"base")
        query = make("query.fvecs", b"query")
        gt = make("gt.ivecs", b"gt")
        mapping = make("map.tsv", b"map")
        stages = {name: make(f"{name}.ids", b"0\n") for name in ("calibration", "validation", "sealed_test")}
        binary = make("GTS_safe_c2_v4", b"not-executed-synthetic-binary")
        os.chmod(binary, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        manifest: dict[str, Any] = {
            "schema": "gts-v4-fresh-sift-learn-workload-v1",
            "status": "READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE",
            "selection_pre_gt_path": str(selection), "selection_pre_gt_sha256": sha256_file(selection),
            "v3_forensic_record_path": str(forensic), "v3_forensic_record_sha256": sha256_file(forensic),
            "formal_protocol_path": str(protocol), "formal_protocol_sha256": sha256_file(protocol),
            "sealed_v2_test_forbidden": True,
            "v2_sealed_test_ids_sha256_forbidden": V2_FORBIDDEN,
            "base_fvecs_path": str(base), "base_fvecs_sha256": sha256_file(base),
            "query_fvecs_path": str(query), "query_fvecs_sha256": sha256_file(query),
            "query_fvecs_count": 12524,
            "groundtruth_ivecs_path": str(gt), "groundtruth_ivecs_sha256": sha256_file(gt),
            "groundtruth_width": 100,
            "mapping_path": str(mapping), "mapping_sha256": sha256_file(mapping),
        }
        for stage, p in stages.items():
            manifest[f"{stage}_ids_path"] = str(p)
            manifest[f"{stage}_ids_sha256"] = sha256_file(p)
            manifest[f"{stage}_query_count"] = 1
        manifest_path = root / "workload_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        bindings = []
        for label, path_key, sha_key in FILE_PAIRS:
            p = Path(manifest[path_key])
            bindings.append({"label": label, "path": str(p), "sha256": manifest[sha_key], "bytes": p.stat().st_size})
        receipt = {
            "schema": "safe-c2-v4-workload-validation-receipt-v1",
            "status": "PASS_NO_CUDA_OR_DATASET_LOAD",
            "invocation": "--validate-workload-only",
            "binary": {"path": str(binary), "sha256": sha256_file(binary)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "arguments": {"mode": "calibrate", "stage": "calibration"},
            "strict_top_level_parser": True,
            "input_bindings": bindings,
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        return verify(receipt_path=receipt_path, manifest_path=manifest_path, binary_path=binary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.receipt, args.manifest, args.binary)):
            raise VerifyError("--self-test accepts no file arguments")
        result = self_test()
    else:
        if not all(value is not None for value in (args.receipt, args.manifest, args.binary)):
            raise VerifyError("--receipt --manifest --binary are required")
        result = verify(receipt_path=args.receipt, manifest_path=args.manifest, binary_path=args.binary)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VERIFY_C2_V4_PRELEDGER_RECEIPT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)

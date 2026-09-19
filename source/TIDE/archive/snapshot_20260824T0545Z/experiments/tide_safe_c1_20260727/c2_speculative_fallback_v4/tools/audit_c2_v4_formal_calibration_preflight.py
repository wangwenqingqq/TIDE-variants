#!/usr/bin/env python3
"""CPU-only fail-closed preflight for one Safe-C2 v4 formal calibration run."""
import hashlib, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
PINS = ROOT / "provenance/c2_v4_formal_calibration_gpu_guard_pins_v1.json"
OUT = ROOT / "formal_runs/c2_v4_calibration_v1"

def fail(message):
    raise SystemExit("FAIL_FORMAL_CALIBRATION_PREFLIGHT: " + message)

def regular(path):
    path = pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        fail("required regular file missing/symlink: " + str(path))
    return path

def sha(path):
    h = hashlib.sha256()
    with open(regular(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    with open(regular(path), encoding="utf-8") as f:
        return json.load(f)

def ids(path, label, expected_count):
    vals = [int(line.strip()) for line in regular(path).read_text(encoding="ascii").splitlines() if line.strip()]
    if len(vals) != expected_count or len(set(vals)) != expected_count:
        fail(label + " count/uniqueness")
    if min(vals, default=-1) < 0 or max(vals, default=12524) >= 12524:
        fail(label + " range")
    return set(vals)

pins = load(PINS)
if pins.get("schema") != "safe-c2-v4-formal-calibration-gpu-guard-pins-v1":
    fail("pins schema")
if pins.get("status") != "PINNED_READY_FOR_FORMAL_CALIBRATION_GPU_PREFLIGHT":
    fail("pins status")
for name, record in pins.get("bindings", {}).items():
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        fail("pin record shape: " + name)
    if sha(record["path"]) != record["sha256"]:
        fail("pin hash mismatch: " + name)

protocol = load(pins["bindings"]["protocol"]["path"])
if protocol.get("schema") != "safe-c2-v4-formal-calibration-execution-protocol-v1":
    fail("protocol schema")
if protocol.get("status") != "FROZEN_PRE_FORMAL_CALIBRATION_GPU_EXECUTION":
    fail("protocol status")
if protocol.get("canonical_root") != str(ROOT):
    fail("protocol root")
if protocol.get("write_boundary") != str(OUT):
    fail("protocol write boundary")
runtime = protocol.get("runtime", {})
if runtime != {"mode": "calibrate", "stage_argument": "calibration", "warmup_reps": 0, "timed_reps": 1}:
    fail("runtime contract")
gpu = protocol.get("gpu", {})
if gpu.get("physical_index") != 0 or gpu.get("cuda_visible_devices") != "0":
    fail("GPU index contract")
if gpu.get("uuid") != "GPU-CONFIGURE-ARCHIVE-DEVICE" or gpu.get("pci_bus_id") != "00000000:16:00.0":
    fail("GPU identity contract")

manifest_path = pathlib.Path(pins["bindings"]["workload_manifest"]["path"])
manifest = load(manifest_path)
if manifest.get("schema") != "gts-v4-fresh-sift-learn-workload-v1":
    fail("workload schema")
if manifest.get("status") != "READY_FOR_C2_V4_FORMAL_AFTER_PRELEDGER_PARSER_GATE":
    fail("workload state")
if manifest.get("query_fvecs_count") != 12524 or manifest.get("groundtruth_width") != 100:
    fail("workload dimensions")
splits = {}
for label, count in (("calibration",2458),("validation",2463),("sealed_test",6403)):
    path = manifest.get(label + "_ids_path")
    expected_sha = manifest.get(label + "_ids_sha256")
    if not isinstance(path,str) or sha(path) != expected_sha:
        fail(label + " IDs checksum")
    splits[label] = ids(path,label,count)
if splits["calibration"] & splits["validation"] or splits["calibration"] & splits["sealed_test"] or splits["validation"] & splits["sealed_test"]:
    fail("formal split overlap")
canary_path = ROOT / "inputs/canary_workload_v1/qualification_canary.ids"
if sha(canary_path) != "12975b282b19cc530ba1951b7c58f2efcfe447d2520e707eb0a5d409b88239c0":
    fail("canary witness checksum")
canary = ids(canary_path,"canary",1010)
if any(canary & values for values in splits.values()):
    fail("canary/formal overlap")

for path_key, sha_key in (
    ("base_fvecs_path","base_fvecs_sha256"),
    ("query_fvecs_path","query_fvecs_sha256"),
    ("groundtruth_ivecs_path","groundtruth_ivecs_sha256"),
    ("mapping_path","mapping_sha256"),
    ("selection_pre_gt_path","selection_pre_gt_sha256"),
    ("formal_protocol_path","formal_protocol_sha256"),
    ("v3_forensic_record_path","v3_forensic_record_sha256"),
):
    if sha(manifest[path_key]) != manifest[sha_key]:
        fail("manifest input checksum: " + path_key)
if manifest.get("sealed_v2_test_forbidden") is not True:
    fail("v2 sealed-test forbid marker")

expected_cal = str(manifest["calibration_ids_path"])
if protocol.get("calibration_ids") != {"path": expected_cal, "sha256": manifest["calibration_ids_sha256"], "count":2458}:
    fail("protocol calibration binding")
if protocol.get("canary_output_is_forbidden_input") is not True:
    fail("canary output forbid marker")
if OUT.exists() or OUT.is_symlink():
    fail("formal calibration output already exists")
guard = pathlib.Path(pins["bindings"]["guard"]["path"])
if subprocess.run(["bash","-n",str(guard)], check=False).returncode != 0:
    fail("guard syntax")
for p in (guard, pathlib.Path(pins["bindings"]["protocol"]["path"])):
    if "canary_runs" in p.read_text(encoding="utf-8"):
        fail("canary output path referenced by formal preflight artifact")
pre = load(pins["bindings"]["preledger_receipt"]["path"])
if pre.get("status") != "PASS_NO_CUDA_OR_DATASET_LOAD":
    fail("preledger receipt status")
pre_audit = load(pins["bindings"]["preledger_audit"]["path"])
if pre_audit.get("status") != "PASS_PRELEDGER_GATE_V2_STRICT_RECEIPT_VERIFIED":
    fail("preledger audit status")
print("PASS_FORMAL_CALIBRATION_CPU_ONLY_PREFLIGHT")

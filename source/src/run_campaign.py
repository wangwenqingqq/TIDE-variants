#!/usr/bin/env python3
"""Run only on the admitted original GPU 2; retain every attempt and fail closed."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
UUID = "GPU-CONFIGURE-ARCHIVE-DEVICE"
SOURCE_SHA = "8c0add9b2204fe75b93786bce6293cc9dea7ce9db146b00b7c2bb9730f5d867d"
BINARY_SHA = "82e2d40ad59998fa1f2b1c61e9eed28db23940cc32c3eb31530aa1e14c90ad00"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def capture(command):
    return sp.check_output(command, text=True).strip()


def state():
    fields = "index,uuid,name,driver_version,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.limit"
    gpu_text = capture(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
    apps_text = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    gpu = [r for r in csv.reader(gpu_text.splitlines()) if r[1].strip() == UUID]
    apps = [r for r in csv.reader(apps_text.splitlines()) if r and r[0].strip() == UUID]
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu_text": gpu_text,
            "apps_text": apps_text, "selected_gpu": gpu, "selected_apps": apps}


def admit(snapshot, check_util=True):
    gpu = snapshot["selected_gpu"]
    assert len(gpu) == 1 and gpu[0][0].strip() == "2", "GPU UUID/index mismatch"
    assert not snapshot["selected_apps"], "foreign compute process present; do not disturb it"
    assert int(gpu[0][4].strip()) < 128, "GPU memory not idle"
    if check_util:
        assert int(gpu[0][5].strip()) <= 1, "GPU utilization not idle"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-confirmed-gpu2", action="store_true")
    args = parser.parse_args()
    if not args.allocation_confirmed_gpu2:
        parser.error("explicit original-GPU-2 admission is required")
    assert capture(["hostname"]) == "CONFIGURE_ARCHIVE_HOST", "wrong host"
    assert digest(EXP / "src/fair_dispatch.cu") == SOURCE_SHA
    binary = EXP / "bin/fair_dispatch"
    assert digest(binary) == BINARY_SHA
    locks = []
    for path in ("/tmp/tide_surechembl_gpu0123.lock", "/tmp/tide_surechembl_gate6_gpu2.lock"):
        lock = open(path, "a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = EXP / "raw" / ("admitted_gpu2_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    print(str(out), flush=True)
    before = state()
    (out / "00_state.json").write_text(json.dumps(before, indent=2) + "\n")
    admit(before)
    fixture = ROOT / "data/prepared/sanitizer_fixture_v1"
    latest = ROOT / "data/prepared/transition_roots/2026-08-18_to_2026-08-25"
    names = [f"data/gate0_prepared/{part}_{suffix}.bin"
             for part in ("base", "union")
             for suffix in ("fp_u64x4", "id_i64", "popcnt_u16")]
    names += ["data/stage_a/delta_u64x6.bin", "data/stage_a/queries_u64x6.bin"]
    paths = [root / name for root in (fixture, latest) for name in names]
    inputs = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths]
    manifest = {"host": capture(["hostname"]), "user": capture(["id"]), "gpu_uuid": UUID,
                "physical_gpu": 2, "runtime_gpu_ordinal": 0, "source_sha256": SOURCE_SHA,
                "binary_sha256": BINARY_SHA, "wrapper_pid": os.getpid(),
                "wrapper_sha256": digest(Path(__file__)), "inputs": inputs}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID)

    def invoke(name, root, mode, extra=(), prefix=()):
        # Let the monitoring window age past our previous invocation's activity.
        time.sleep(2)
        pre = state()
        (out / (name + ".pre.json")).write_text(json.dumps(pre, indent=2) + "\n")
        admit(pre)
        cmd = [*prefix, str(binary), "--root", str(root), "--mode", mode,
               "--output", str(out / (name + ".csv")), "--gpu", "0", *extra]
        record = {"command": cmd, "environment_override": {"CUDA_VISIBLE_DEVICES": UUID}, "start_utc": pre["utc"]}
        path = out / (name + ".process.json")
        with (out / (name + ".log")).open("w") as log:
            proc = sp.Popen(cmd, stdout=log, stderr=sp.STDOUT, env=env)
            record["pid"] = proc.pid
            path.write_text(json.dumps(record, indent=2) + "\n")
            record["returncode"] = proc.wait()
        post = state()
        record["end_utc"] = post["utc"]
        path.write_text(json.dumps(record, indent=2) + "\n")
        (out / (name + ".post.json")).write_text(json.dumps(post, indent=2) + "\n")
        assert record["returncode"] == 0, "failed process retained: " + name
        admit(post, check_util=False)

    synthetic = EXP / "fixtures/high_entropy_v1"
    sp.run([sys.executable, str(EXP / "src/prepare_synthetic_fixture.py"), str(synthetic)], check=True)
    invoke("01_fixture_complete", fixture, "faircheck", ["--cycles", "3"])
    invoke("01_synthetic_complete", synthetic, "faircheck", ["--cycles", "3"])
    for tool in ("memcheck", "synccheck"):
        invoke("02_" + tool, fixture, "faircheck", ["--cycles", "1"],
               ["/usr/local/bin/compute-sanitizer", "--tool", tool, "--error-exitcode", "99"])
    invoke("02_synthetic_memcheck", synthetic, "faircheck", ["--cycles", "1"],
           ["/usr/local/bin/compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99"])
    invoke("03_pointer_churn", fixture, "faircheck", ["--cycles", "50"])
    schedules = [("fused,queued", "1,64", "70,80"), ("queued,fused", "64,1", "80,70"),
                 ("queued,fused", "1,64", "70,80"), ("fused,queued", "64,1", "80,70")]
    for i, (order, geometry, threshold) in enumerate(schedules, 1):
        invoke("pair" + str(i), latest, "fairbench", ["--order", order,
               "--geometry-order", geometry, "--threshold-order", threshold])
    (out / "COLLECTION_COMPLETE.json").write_text(json.dumps({
        "state": "collection_complete_analysis_pending", "gpu_uuid": UUID,
        "not_admitted": ["novelty", "production_value", "full_sustained_promotion", "Graph"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

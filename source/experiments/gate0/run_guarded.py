"""Bounded, locked, fail-closed GPU pilot. Never touches foreign processes."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess as sp
import time

from build_and_oracle import ROOT, digest, input_hashes

UUID = None
GPU_INDEX = None


def state():
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.limit"
    rows = list(csv.reader(sp.check_output(["nvidia-smi", "--query-gpu=" + fields,
                                           "--format=csv,noheader,nounits"], text=True).splitlines()))
    selected = [[v.strip() for v in row] for row in rows if row[1].strip() == UUID]
    apps = list(csv.reader(sp.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                           "--format=csv,noheader,nounits"], text=True).splitlines()))
    apps = [[v.strip() for v in row] for row in apps if row and row[0].strip() == UUID]
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu": selected, "apps": apps}


def admit(snapshot):
    gpu = snapshot["gpu"]
    if len(gpu) != 1 or gpu[0][0] != str(GPU_INDEX):
        raise RuntimeError("selected GPU UUID/index mismatch")
    if snapshot["apps"] or int(gpu[0][5]) >= 256 or int(gpu[0][6]) > 1:
        raise RuntimeError("selected GPU not idle; no process will be disturbed")


def stop_own_process(proc):
    if proc.poll() is not None:
        return
    # start_new_session=True makes this exactly the child process group we own.
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except sp.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def main():
    global UUID, GPU_INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-index", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    UUID, GPU_INDEX = args.gpu_uuid, args.gpu_index
    data, out = args.input.resolve(), args.output.resolve()
    if not out.is_relative_to(ROOT / "raw_logs"):
        parser.error("output must be a new child of this project's raw_logs")
    out.mkdir(parents=True, exist_ok=False)
    ready = json.loads((ROOT / "build/CPU_READY.json").read_text())
    if str(data) != ready["input"] or input_hashes(data) != ready["input_hashes"]:
        raise RuntimeError("input hash mismatch")
    for relative, sha in ready["sources"].items():
        if digest(ROOT / relative) != sha:
            raise RuntimeError("source changed after CPU verification: " + relative)
    for relative, sha in ready["oracle_hashes"].items():
        if digest(data / relative) != sha:
            raise RuntimeError("CPU oracle changed")
    binary = ROOT / "build/maintenance_replay"
    if digest(binary) != ready["binary_sha256"]:
        raise RuntimeError("binary changed")
    locks = []
    try:
        for path in ("/tmp/tide_surechembl_gpu0123.lock", f"/tmp/tide_surechembl_gate6_gpu{GPU_INDEX}.lock"):
            # No O_CREAT on an existing shared lock: compatible with Linux
            # protected_regular in sticky /tmp. Do not chmod, delete, or replace it.
            handle = open(path, "r+" if Path(path).exists() else "x+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        admit(state())
    except Exception as exc:
        (out / "ADMISSION_FAILED.json").write_text(json.dumps({"reason": str(exc), "gpu_uuid": UUID,
            "gpu_index": GPU_INDEX, "gpu_process_started": False}, indent=2) + "\n")
        raise
    manifest = {"gpu_uuid": UUID, "physical_gpu": GPU_INDEX, "runtime_ordinal": 0,
                "cpu_ready_sha256": digest(ROOT / "build/CPU_READY.json"),
                "runner_sha256": digest(Path(__file__)), "binary_sha256": ready["binary_sha256"],
                "host": sp.check_output(["hostname"], text=True).strip(), "wrapper_pid": os.getpid(),
                "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "commands": []}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    campaign_start = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID)

    def invoke(name, extra, prefix=()):
        time.sleep(2)
        pre = state()
        (out / f"{name}.pre.json").write_text(json.dumps(pre, indent=2) + "\n")
        admit(pre)
        cmd = [*prefix, str(binary), *extra]
        record = {"command": cmd, "start_utc": pre["utc"], "environment_override": {"CUDA_VISIBLE_DEVICES": UUID},
                  "samples": [], "failure": None}
        proc = None
        try:
            with (out / f"{name}.log").open("x") as log:
                proc = sp.Popen(cmd, stdout=log, stderr=sp.STDOUT, env=env, start_new_session=True)
                record["pid"] = proc.pid
                started = time.monotonic()
                while proc.poll() is None:
                    if time.monotonic() - started > 900 or time.monotonic() - campaign_start > 1800:
                        raise RuntimeError("registered time budget exceeded")
                    sample = state()
                    record["samples"].append(sample)
                    for app in sample["apps"]:
                        try:
                            own = os.getpgid(int(app[1])) == proc.pid
                        except ProcessLookupError:
                            continue
                        if not own:
                            raise RuntimeError("foreign GPU process appeared; stop only this campaign")
                    time.sleep(1)
                record["returncode"] = proc.returncode
                if proc.returncode:
                    raise RuntimeError(f"GPU process failed with code {proc.returncode}")
            post = state()
            record["post"] = post
            if post["apps"]:
                raise RuntimeError("GPU still has active compute processes after completion")
            # Sanitizer process exit status is not sufficient without its summary.
            if prefix:
                log_text = (out / f"{name}.log").read_text()
                if "ERROR SUMMARY: 0 errors" not in log_text:
                    raise RuntimeError("sanitizer zero-error summary absent")
        except BaseException as exc:
            record["failure"] = str(exc)
            if proc is not None:
                stop_own_process(proc)
                record["returncode"] = proc.returncode
            raise
        finally:
            record["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            (out / f"{name}.process.json").write_text(json.dumps(record, indent=2) + "\n")
            manifest["commands"].append(name)
            (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(name + " PASS", flush=True)

    fixture = data / "fixture"
    sanitizer = "/usr/local/cuda-13.1/bin/compute-sanitizer"
    invoke("00_guards", ["--mode", "guards"])
    invoke("01_fixture", ["--root", str(fixture), "--mode", "check", "--output", str(out / "01_fixture")])
    for name in ("memcheck", "synccheck"):
        invoke("02_" + name,
               ["--root", str(fixture), "--mode", "check", "--output", str(out / ("02_" + name))],
               [sanitizer, "--tool", name, "--error-exitcode", "99"])
    (out / "GPU_CORRECTNESS_PASS.json").write_text(json.dumps({"binary_sha256": ready["binary_sha256"],
        "fixture_complete_requests_per_run": 1792, "old_epoch_checks_per_run": 48,
        "sanitizers": ["memcheck", "synccheck"], "concurrent_reclamation_tested": False}, indent=2) + "\n")
    for rotation in range(4):
        name = f"pilot{rotation}"
        invoke(name, ["--root", str(data), "--mode", "bench", "--rotation", str(rotation),
                      "--output", str(out / name)])
    (out / "COLLECTION_COMPLETE.json").write_text(json.dumps({"status": "collection_complete_analysis_pending",
        "fresh_processes": 4, "measured_requests": 7168, "warmup_requests": 1792,
        "budget_wall_s": time.monotonic() - campaign_start, "scope": "sampled history, Q=1, sequential only"}, indent=2) + "\n")
    print("COLLECTION_COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()

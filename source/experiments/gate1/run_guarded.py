"""One idle GPU, nonblocking shared locks, whole-card cap observation, fail closed."""
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

from build import ROOT, digest, inputs


def state(uuid):
    fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,pstate,clocks.sm,clocks.mem,power.limit"
    rows = list(csv.reader(sp.check_output(["nvidia-smi", "--query-gpu=" + fields,
                                           "--format=csv,noheader,nounits"], text=True).splitlines()))
    gpu = [[v.strip() for v in r] for r in rows if r[1].strip() == uuid]
    rows = list(csv.reader(sp.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                           "--format=csv,noheader,nounits"], text=True).splitlines()))
    apps = [[v.strip() for v in r] for r in rows if r and r[0].strip() == uuid]
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu": gpu, "apps": apps}


def admit(snapshot, index):
    if len(snapshot["gpu"]) != 1 or int(snapshot["gpu"][0][0]) != index:
        raise RuntimeError("GPU UUID/index mismatch")
    if snapshot["apps"] or int(snapshot["gpu"][0][5]) >= 256 or int(snapshot["gpu"][0][6]) > 1:
        raise RuntimeError("selected GPU is not idle; leave all foreign work untouched")


def stop_owned(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except sp.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-index", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    data, out = args.input.resolve(), args.output.resolve()
    if not out.is_relative_to(ROOT / "raw_logs"):
        parser.error("output must be a new project raw_logs child")
    out.mkdir(parents=True, exist_ok=False)
    ready_path = ROOT / "build/CPU_READY.json"
    ready = json.loads(ready_path.read_text())
    if ready["input"] != str(data) or inputs(data) != ready["input_hashes"]:
        raise RuntimeError("input identity mismatch")
    for rel, sha in ready["sources"].items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError("source changed after CPU_READY: " + rel)
    for rel, sha in ready["oracle_hashes"].items():
        if digest(data / rel) != sha:
            raise RuntimeError("oracle identity mismatch")
    binary = ROOT / "build/replay"
    if digest(binary) != ready["binary_sha256"]:
        raise RuntimeError("binary identity mismatch")
    supplemental_path = ROOT / "build/CPU_GATE1M.json"
    supplemental = json.loads(supplemental_path.read_text())
    gate_data = Path(supplemental["input"])
    if supplemental["binary_sha256"] != ready["binary_sha256"] or inputs(gate_data) != supplemental["input_hashes"]:
        raise RuntimeError("million-row gate identity mismatch")
    if digest(ROOT / "experiments/gate1/supplement.py") != supplemental["script_sha256"]:
        raise RuntimeError("supplemental preparation script changed")
    for rel, sha in supplemental["oracle_hashes"].items():
        if digest(gate_data / rel) != sha:
            raise RuntimeError("million-row oracle changed")
    locks = []
    try:
        for path in ("/tmp/tide_surechembl_gpu0123.lock", f"/tmp/tide_surechembl_gate6_gpu{args.gpu_index}.lock"):
            handle = open(path, "r+" if Path(path).exists() else "x+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        admit(state(args.gpu_uuid), args.gpu_index)
    except Exception as exc:
        (out / "ADMISSION_FAILED.json").write_text(json.dumps({"reason": str(exc), "gpu_process_started": False}) + "\n")
        raise
    manifest = {"gpu_uuid": args.gpu_uuid, "physical_gpu": args.gpu_index,
                "cpu_ready_sha256": digest(ready_path), "runner_sha256": digest(Path(__file__)),
                "cpu_gate1m_sha256": digest(supplemental_path),
                "binary_sha256": ready["binary_sha256"], "host": sp.check_output(["hostname"], text=True).strip(),
                "wrapper_pid": os.getpid(), "commands": [], "start_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_MODULE_LOADING="EAGER", OMP_NUM_THREADS="4")
    campaign_start = time.monotonic()

    def invoke(name, mode, root, cap=2048, rotation=0, sanitizer=None):
        # NVML utilization is a trailing sample and may outlive the exited child.
        # This inter-process quiescence is outside every C++ measurement interval.
        time.sleep(2)
        pre = state(args.gpu_uuid)
        (out / f"{name}.pre.json").write_text(json.dumps(pre, indent=2) + "\n")
        try:
            admit(pre, args.gpu_index)
        except Exception as exc:
            (out / f"{name}.ADMISSION_FAILED.json").write_text(json.dumps(
                {"reason": str(exc), "pre": pre, "gpu_process_started": False}, indent=2) + "\n")
            raise
        command = [str(binary), "--mode", mode, "--root", str(root), "--output", str(out / name),
                   "--cap-mib", str(cap), "--rotation", str(rotation)]
        if sanitizer:
            prefix = ["/usr/local/cuda-13.1/bin/compute-sanitizer", "--tool", sanitizer, "--error-exitcode", "99"]
            if sanitizer == "memcheck":
                prefix += ["--leak-check", "full"]
            command = prefix + command
        record = {"command": command, "pre": pre, "environment_override":
                  {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "OMP_NUM_THREADS")},
                  "samples": [], "failure": None, "total_cap_mib": cap}
        proc = None
        try:
            with (out / f"{name}.log").open("x") as log:
                proc = sp.Popen(command, env=env, stdout=log, stderr=sp.STDOUT, start_new_session=True)
                record["pid"] = proc.pid
                start = time.monotonic()
                while proc.poll() is None:
                    if time.monotonic() - start > 600 or time.monotonic() - campaign_start > 1800:
                        raise RuntimeError("registered execution time limit exceeded")
                    sample = state(args.gpu_uuid)
                    record["samples"].append(sample)
                    if int(sample["gpu"][0][5]) > cap:
                        raise RuntimeError("NVML observed whole-card VRAM exceeds cap; invalidate attempt")
                    for app in sample["apps"]:
                        try:
                            if os.getpgid(int(app[1])) != proc.pid:
                                raise RuntimeError("foreign GPU job appeared; stop only owned child")
                        except ProcessLookupError:
                            pass
                    time.sleep(0.5)
                record["returncode"] = proc.returncode
                if proc.returncode:
                    raise RuntimeError(f"GPU child failed: {proc.returncode}")
            record["post"] = state(args.gpu_uuid)
            if record["post"]["apps"]:
                raise RuntimeError("GPU still has live processes after child completion")
            if sanitizer:
                expected = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                            if sanitizer == "racecheck" else "ERROR SUMMARY: 0 errors")
                if expected not in (out / f"{name}.log").read_text():
                    raise RuntimeError("sanitizer zero-error summary missing")
                if sanitizer == "memcheck" and "LEAK SUMMARY: 0 bytes leaked in 0 allocations" not in (out / f"{name}.log").read_text():
                    raise RuntimeError("zero-leak summary missing")
        except BaseException as exc:
            record["failure"] = str(exc)
            if proc is not None:
                stop_owned(proc)
                record["returncode"] = proc.returncode
            raise
        finally:
            record["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            (out / f"{name}.process.json").write_text(json.dumps(record, indent=2) + "\n")
            manifest["commands"].append(name)
            (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(name + " PASS", flush=True)

    fixture = data / "fixture"
    invoke("00_guards", "guards", fixture)
    invoke("01_fixture", "sequential", fixture)
    for sanitizer in ("memcheck", "synccheck", "racecheck"):
        invoke("02_" + sanitizer + "_guards", "guards", fixture, sanitizer=sanitizer)
        invoke("03_" + sanitizer + "_fixture", "sequential", fixture, sanitizer=sanitizer)
    invoke("04_million_native_gate", "sequential", gate_data)
    (out / "GPU_CORRECTNESS_PASS.json").write_text(json.dumps({"binary_sha256": ready["binary_sha256"],
        "gates": ["allocation_and_copy_rollback", "before_publication_rollback", "budget_rejection", "overflow",
                  "pending_GPU_old_owner", "cancellation", "all_epochs_native_batch_complete_oracle"],
        "sanitizers": ["memcheck_with_full_leak_check", "synccheck", "racecheck"],
        "racecheck_scope": "CUDA shared-memory hazards; not a host race proof"}, indent=2) + "\n")
    for rotation in range(4):
        for cap in ((2048, 1280) if rotation % 2 == 0 else (1280, 2048)):
            invoke(f"seq_r{rotation}_cap{cap}", "sequential", data, cap, rotation)
        invoke(f"concurrent_r{rotation}", "concurrent", data, 2048, rotation)
    (out / "COLLECTION_COMPLETE.json").write_text(json.dumps({"status": "collection_complete_analysis_pending",
        "sequential_fresh_processes": 8, "concurrent_fresh_processes": 4,
        "wrapper_wall_s": time.monotonic() - campaign_start,
        "scope": "10M finite six-release real history, native Q=1/8/64, 256-bit fingerprints, tau=.7/.8"}, indent=2) + "\n")
    print("COLLECTION_COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()

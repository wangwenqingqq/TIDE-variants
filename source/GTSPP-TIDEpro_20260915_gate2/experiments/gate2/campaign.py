"""Guarded single-GPU development -> immutable selection -> fixed test campaign."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/gate1"))
from run_guarded import state, admit, stop_owned
from experiments.gate1.build import inputs
from experiments.gate2.metrics import digest, summarize, interval_for, choose, passes


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--gpu-index", type=int, choices=range(8), required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT / "raw_logs"):
        p.error("output must be a new raw_logs child")
    out.mkdir(parents=True, exist_ok=False)
    ready_path = ROOT/"build/CPU_READY.json"
    ready = json.loads(ready_path.read_text())
    for rel, sha in ready["sources"].items():
        if digest(ROOT/rel) != sha:
            raise RuntimeError("source changed after CPU freeze: "+rel)
    for data in ready["datasets"].values():
        root = Path(data["root"])
        if inputs(root) != data["input_hashes"]:
            raise RuntimeError("input identity mismatch")
        for rel, sha in data["oracle_hashes"].items():
            if digest(root/rel) != sha:
                raise RuntimeError("oracle identity mismatch")
    binary = ROOT/"build/replay"
    if digest(binary) != ready["binary_sha256"]:
        raise RuntimeError("binary identity mismatch")
    locks = []
    try:
        for path in ("/tmp/tide_surechembl_gpu0123.lock", f"/tmp/tide_surechembl_gate6_gpu{a.gpu_index}.lock"):
            handle = open(path, "r+" if Path(path).exists() else "x+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        admit(state(a.gpu_uuid), a.gpu_index)
    except Exception as exc:
        save(out/"ADMISSION_FAILED.json", {"reason": str(exc), "gpu_process_started": False})
        raise
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu_uuid, CUDA_MODULE_LOADING="EAGER", OMP_NUM_THREADS="4")
    start = time.monotonic()
    manifest = {"gpu_uuid": a.gpu_uuid, "physical_gpu": a.gpu_index, "wrapper_pid": os.getpid(),
                "cpu_ready_sha256": digest(ready_path), "binary_sha256": digest(binary),
                "host": sp.check_output(["hostname"], text=True).strip(),
                "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "commands": []}

    def invoke(name, mode, data, cap=2048, policy="all_delta", interval=2., rotation=0, sanitizer=False):
        time.sleep(2)  # outside measured intervals; trailing NVML utilization can outlive child
        pre = state(a.gpu_uuid)
        save(out/(name+".pre.json"), pre)
        admit(pre, a.gpu_index)
        command = [str(binary), "--mode", mode, "--root", str(data), "--output", str(out/name),
                   "--cap-mib", str(cap), "--rotation", str(rotation)]
        if mode != "guards":
            command += ["--policy", policy, "--interval-ms", str(interval), "--window-ms", "8000"]
        if sanitizer:
            command = ["/usr/local/cuda-13.1/bin/compute-sanitizer", "--tool", "memcheck", "--leak-check", "full",
                       "--error-exitcode", "99"]+command
        record = {"command": command, "pre": pre, "samples": [], "failure": None,
                  "cap_mib": cap, "environment_override": {k: env[k] for k in
                    ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "OMP_NUM_THREADS")}}
        proc = None
        try:
            with (out/(name+".log")).open("x") as log:
                proc = sp.Popen(command, env=env, stdout=log, stderr=sp.STDOUT, start_new_session=True)
                record["pid"] = proc.pid
                begin = time.monotonic()
                while proc.poll() is None:
                    if time.monotonic()-begin > 180 or time.monotonic()-start > 2400:
                        raise RuntimeError("registered execution deadline exceeded")
                    sample = state(a.gpu_uuid)
                    record["samples"].append(sample)
                    if len(sample["gpu"]) != 1 or int(sample["gpu"][0][5]) > cap:
                        raise RuntimeError("whole-card cap exceeded or GPU missing")
                    for app in sample["apps"]:
                        try:
                            if os.getpgid(int(app[1])) != proc.pid:
                                raise RuntimeError("foreign GPU process; stop only owned child")
                        except ProcessLookupError:
                            pass
                    time.sleep(.5)
                record["returncode"] = proc.returncode
                if proc.returncode:
                    raise RuntimeError("GPU child failed; retain complete failed attempt")
            record["post"] = state(a.gpu_uuid)
            if record["post"]["apps"]:
                raise RuntimeError("GPU process remained after child exit")
            if sanitizer:
                log = (out/(name+".log")).read_text()
                if "ERROR SUMMARY: 0 errors" not in log or "LEAK SUMMARY: 0 bytes leaked in 0 allocations" not in log:
                    raise RuntimeError("memcheck did not report zero errors/leaks")
        except BaseException as exc:
            record["failure"] = str(exc)
            if proc is not None:
                stop_owned(proc)
                record["returncode"] = proc.returncode
            raise
        finally:
            record["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            save(out/(name+".process.json"), record)
            manifest["commands"].append(name)
            # Append-only journal survives failures without replacing evidence.
            with (out/"journal.jsonl").open("a") as f:
                f.write(json.dumps({"name": name, "process_sha256": digest(out/(name+".process.json"))})+"\n")
        if mode == "guards":
            print(name+" PASS", flush=True)
            return None
        summary = summarize(out/name)
        save(out/(name+".summary.json"), summary)
        print(name+f" PASS coverage={summary['coverage_pass']} p99={summary['p99_response_ms']:.4f}ms completion={summary['completion_fraction']:.5f}", flush=True)
        return summary

    dev, test = (Path(ready["datasets"][n]["root"]) for n in ("dev", "test"))
    invoke("gate_guards", "guards", dev/"fixture")
    for sanitizer in (False, True):
        for mode in ("growth", "shadow"):
            result = invoke(f"gate_{mode}_memcheck{int(sanitizer)}", mode, dev/"fixture", policy="tier4", sanitizer=sanitizer)
            if not result["coverage_pass"]:
                raise RuntimeError("new protocol fixture coverage failed")
    save(out/"GPU_CORRECTNESS_PASS.json", {"binary_sha256": digest(binary),
         "inherited_guards": True, "new_protocol_modes": ["growth", "shadow"], "memcheck_errors": 0,
         "scope": "same inherited kernel and ownership; new fixed-clock host protocol with full-output oracle"})
    frozen = {"registered_contract_sha256": ready["sources"]["docs/GATE2_CONTRACT.md"],
              "cpu_ready_sha256": digest(ready_path), "configuration": {}, "development_cases": []}
    for cap in (2048, 1280):
        calibration = invoke(f"dev_calibrate_cap{cap}", "calibrate", dev, cap)
        interval = interval_for(calibration["mean_service_ms"])
        baseline = invoke(f"dev_static_final_cap{cap}", "static_final", dev, cap, interval=interval)
        deadline = 2*baseline["p99_response_ms"]
        cases = []
        for policy in ("periodic2", "periodic3", "periodic6", "tier2", "tier4"):
            name = f"dev_{policy}_cap{cap}"
            cases.append(invoke(name, "growth", dev, cap, policy, interval))
            frozen["development_cases"].append({"name": name, "sha256": digest(out/(name+".summary.json"))})
        frozen["configuration"][str(cap)] = {"interval_ms": interval, "offered_queries_per_s": 8000/interval,
             "p99_deadline_ms": deadline, "completion_fraction_min": .99, "publication_lag_max_ms": 1000,
             "periodic": choose(cases, deadline, "periodic"), "tier": choose(cases, deadline, "tier"),
             "dev_static_pass": passes(baseline, deadline), "calibration": calibration, "baseline": baseline}
    frozen["frozen_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(out/"FROZEN.json", frozen)
    frozen_sha = digest(out/"FROZEN.json")
    print("FROZEN "+frozen_sha+" "+json.dumps({c: {k: v[k] for k in ("interval_ms", "p99_deadline_ms", "periodic", "tier")} for c, v in frozen["configuration"].items()}), flush=True)
    for rotation in range(4):
        for cap in ((2048, 1280) if rotation%2 == 0 else (1280, 2048)):
            config = frozen["configuration"][str(cap)]
            interval = config["interval_ms"]
            for mode in ("static_base", "static_final"):
                invoke(f"test_r{rotation}_cap{cap}_{mode}", mode, test, cap, interval=interval, rotation=rotation)
            policies = ["all_delta", config["periodic"], config["tier"], "compact"]
            policies = policies[rotation:]+policies[:rotation]
            for policy in policies:
                for mode in (("shadow", "growth") if rotation%2 == 0 else ("growth", "shadow")):
                    if digest(out/"FROZEN.json") != frozen_sha:
                        raise RuntimeError("selection changed during test")
                    invoke(f"test_r{rotation}_cap{cap}_{policy}_{mode}", mode, test, cap, policy, interval, rotation)
    save(out/"MANIFEST.json", manifest)
    save(out/"COLLECTION_COMPLETE.json", {"status": "complete_analysis_pending", "test_cases": 80,
         "frozen_sha256": frozen_sha, "wrapper_wall_s": time.monotonic()-start,
         "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print("COLLECTION_COMPLETE "+str(out), flush=True)


if __name__ == "__main__":
    main()

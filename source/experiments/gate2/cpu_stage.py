"""Compile new host protocol, generate development oracle, freeze all inputs."""
import argparse
import json
from pathlib import Path
import subprocess as sp
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gate1.build import inputs
from experiments.gate2.metrics import digest

SOURCES = ["docs/GATE2_CONTRACT.md", "experiments/gate2/prepare.py", "experiments/gate2/replay.cu",
           "experiments/gate2/cpu_stage.py", "experiments/gate2/campaign.py", "experiments/gate2/metrics.py",
           "experiments/gate1/prepare.py", "experiments/gate1/replay.cu", "experiments/gate1/runtime.cuh",
           "experiments/gate1/build.py", "experiments/gate1/run_guarded.py",
           "experiments/gate0/prepare_history.py", "vendor/tide/frozen_query_runtime.cuh"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    a = p.parse_args()
    dev, test = a.dev.resolve(), a.test.resolve()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    if (build / "CPU_READY.json").exists() or (build / "replay").exists():
        raise RuntimeError("refusing previous build evidence")
    record = {"cpu_only": True, "sources": {n: digest(ROOT/n) for n in SOURCES},
              "datasets": {n: {"root": str(d), "input_hashes": inputs(d)} for n, d in (("dev", dev), ("test", test))},
              "commands": []}
    commands = [["/usr/local/cuda-13.1/bin/nvcc", "--version"],
                ["/usr/local/cuda-13.1/bin/nvcc", "-std=c++17", "-O3", "-arch=sm_120", "-lineinfo",
                 "-Xcompiler=-O3,-march=native,-fopenmp,-pthread", "--ptxas-options=-v",
                 "experiments/gate2/replay.cu", "-o", "build/replay"],
                [str(build/"replay"), "--mode", "oracle", "--root", str(dev/"fixture")],
                [str(build/"replay"), "--mode", "oracle", "--root", str(dev)]]
    for i, command in enumerate(commands):
        t = time.monotonic()
        result = sp.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
        log = build/f"cpu_{i}.log"
        log.write_text(result.stdout+result.stderr)
        record["commands"].append({"argv": command, "returncode": result.returncode,
                                   "wall_s": time.monotonic()-t, "log_sha256": digest(log)})
        print(result.stdout+result.stderr, flush=True)
        if result.returncode:
            (build/"CPU_FAILED.json").write_text(json.dumps(record, indent=2)+"\n")
            raise RuntimeError("CPU stage failed; evidence retained")
    for label, data in (("dev", dev), ("test", test)):
        oracles = sorted(data.glob("oracle/*.bin"))+sorted(data.glob("fixture/oracle/*.bin"))
        if len(oracles) != 1792:
            raise RuntimeError("missing complete oracle vectors")
        record["datasets"][label]["oracle_hashes"] = {str(f.relative_to(data)): digest(f) for f in oracles}
    record["binary_sha256"] = digest(build/"replay")
    record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (build/"CPU_READY.json").write_text(json.dumps(record, indent=2)+"\n")
    print("CPU_READY independent development oracle and immutable test oracle", flush=True)


if __name__ == "__main__":
    main()

"""Freeze compiler, inputs and 1,792 independent complete oracle vectors. CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs(data):
    paths = [data / "MANIFEST.json"]
    for root in (data, data / "fixture"):
        paths.extend([root / f"run{i}.bin" for i in range(7)] + [root / "queries.bin"])
    return {str(p.relative_to(data)): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    data = args.input.resolve()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    ready = build / "CPU_READY.json"
    if ready.exists() or (data / "oracle").exists() or (data / "fixture/oracle").exists():
        raise RuntimeError("refusing existing CPU evidence")
    sources = ["experiments/gate1/prepare.py", "experiments/gate1/build.py",
               "experiments/gate1/runtime.cuh", "experiments/gate1/replay.cu",
               "experiments/gate0/prepare_history.py", "vendor/tide/frozen_query_runtime.cuh"]
    record = {"sources": {p: digest(ROOT / p) for p in sources}, "input": str(data),
              "input_hashes": inputs(data), "commands": [], "cpu_only": True}
    commands = [
        ["/usr/local/cuda-13.1/bin/nvcc", "--version"],
        ["/usr/local/cuda-13.1/bin/nvcc", "-std=c++17", "-O3", "-arch=sm_120", "-lineinfo",
         "-Xcompiler=-O3,-march=native,-fopenmp,-pthread", "--ptxas-options=-v",
         "experiments/gate1/replay.cu", "-o", "build/replay"],
        [str(build / "replay"), "--mode", "oracle", "--root", str(data / "fixture")],
        [str(build / "replay"), "--mode", "oracle", "--root", str(data)],
    ]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for i, command in enumerate(commands):
        start = time.monotonic()
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
        log = build / f"cpu_{stamp}_{i}.log"
        log.write_text(proc.stdout + proc.stderr)
        record["commands"].append({"argv": command, "returncode": proc.returncode,
                                  "wall_s": time.monotonic() - start, "log": str(log.relative_to(ROOT)),
                                  "log_sha256": digest(log)})
        print(proc.stdout + proc.stderr, flush=True)
        if proc.returncode:
            (build / f"CPU_FAILED_{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError("CPU stage failed; evidence retained")
    record["binary_sha256"] = digest(build / "replay")
    paths = sorted(data.glob("oracle/*.bin")) + sorted(data.glob("fixture/oracle/*.bin"))
    if len(paths) != 1792:
        raise RuntimeError("wrong oracle vector count")
    record["oracle_hashes"] = {str(p.relative_to(data)): digest(p) for p in paths}
    record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ready.write_text(json.dumps(record, indent=2) + "\n")
    print("CPU_READY 1792 complete oracle vectors, no GPU call", flush=True)


if __name__ == "__main__":
    main()

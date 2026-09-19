"""CPU-only compilation and independent complete-result oracle preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SHA = "7ccf4416f08ec653f8d74c32562a4ee75b9ede251f953e3ec8619e8b63523302"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(data):
    paths = [data / "MANIFEST.json"]
    for base in (data, data / "fixture"):
        paths += [base / f"run{i}.bin" for i in range(7)] + [base / "queries.bin"]
    return {str(p.relative_to(data)): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda-13.1/bin/nvcc"))
    args = parser.parse_args()
    data = args.input.resolve()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    ready = build / "CPU_READY.json"
    if ready.exists() or (data / "oracle").exists() or (data / "fixture/oracle").exists():
        raise RuntimeError("existing CPU evidence found; do not overwrite")
    if digest(ROOT / "vendor/tide/frozen_query_runtime.cuh") != RUNTIME_SHA:
        raise RuntimeError("frozen donor runtime differs")
    sources = ["experiments/gate0/maintenance_replay.cu", "vendor/tide/frozen_query_runtime.cuh",
               "experiments/gate0/prepare_history.py", "experiments/gate0/build_and_oracle.py"]
    record = {"sources": {p: digest(ROOT / p) for p in sources}, "input": str(data),
              "input_hashes": input_hashes(data), "commands": [], "cpu_only": True}
    commands = [
        [str(args.nvcc), "--version"],
        [str(args.nvcc), "-std=c++17", "-O3", "-arch=sm_120", "-lineinfo",
         "-Xcompiler=-O3,-march=native", "--ptxas-options=-v",
         "experiments/gate0/maintenance_replay.cu", "-o", "build/maintenance_replay"],
        [str(build / "maintenance_replay"), "--mode", "oracle", "--root", str(data / "fixture")],
        [str(build / "maintenance_replay"), "--mode", "oracle", "--root", str(data)],
    ]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for number, command in enumerate(commands):
        start = time.time()
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
        log = build / f"cpu_{stamp}_{number}.log"
        log.write_text(proc.stdout + proc.stderr)
        record["commands"].append({"argv": command, "returncode": proc.returncode,
                                   "wall_s": time.time() - start, "log": str(log.relative_to(ROOT)),
                                   "log_sha256": digest(log)})
        print(proc.stdout + proc.stderr, flush=True)
        if proc.returncode:
            (build / f"CPU_FAILED_{stamp}.json").write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError("CPU stage failed; evidence retained")
    record["binary_sha256"] = digest(build / "maintenance_replay")
    oracle_paths = sorted(data.glob("oracle/*.bin")) + sorted(data.glob("fixture/oracle/*.bin"))
    if len(oracle_paths) != 896:
        raise RuntimeError("expected 448 oracle vectors for each dataset")
    record["oracle_hashes"] = {str(p.relative_to(data)): digest(p) for p in oracle_paths}
    record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ready.write_text(json.dumps(record, indent=2) + "\n")
    print("CPU_READY: compiler and 896 complete oracle vectors verified; no GPU call executed", flush=True)


if __name__ == "__main__":
    main()

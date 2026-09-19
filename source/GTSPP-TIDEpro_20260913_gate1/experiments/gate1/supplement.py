"""Prepare and attest a million-row native-batch gate, retaining initial CPU_READY."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from build import ROOT, digest, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    data = ROOT / "raw_data/history_mod38"
    receipt = ROOT / "build/CPU_GATE1M.json"
    if data.exists() or receipt.exists():
        raise RuntimeError("refusing existing supplemental evidence")
    ready = json.loads((ROOT / "build/CPU_READY.json").read_text())
    binary = ROOT / "build/replay"
    if digest(binary) != ready["binary_sha256"]:
        raise RuntimeError("binary identity changed")
    commands = [
        ["python3", str(ROOT / "experiments/gate1/prepare.py"), "--source-root", str(args.source_root),
         "--output", str(data), "--modulus", "38"],
        [str(binary), "--mode", "oracle", "--root", str(data)],
    ]
    record = {"binary_sha256": ready["binary_sha256"], "input": str(data),
              "script_sha256": digest(Path(__file__)), "cpu_only": True, "commands": []}
    for i, command in enumerate(commands):
        start = time.monotonic()
        proc = subprocess.run(command, text=True, capture_output=True, timeout=600)
        log = ROOT / f"build/supplement_{i}.log"
        log.write_text(proc.stdout + proc.stderr)
        record["commands"].append({"argv": command, "returncode": proc.returncode,
                                  "wall_s": time.monotonic() - start, "log_sha256": digest(log)})
        print(proc.stdout + proc.stderr, flush=True)
        if proc.returncode:
            (ROOT / "build/CPU_GATE1M_FAILED.json").write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError("supplemental CPU stage failed")
    record["input_hashes"] = inputs(data)
    paths = sorted(data.glob("oracle/*.bin"))
    if len(paths) != 896:
        raise RuntimeError("supplemental oracle count mismatch")
    record["oracle_hashes"] = {str(p.relative_to(data)): digest(p) for p in paths}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print("CPU_GATE1M_READY 896 additional complete vectors", flush=True)


if __name__ == "__main__":
    main()

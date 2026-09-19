#!/usr/bin/env python3
"""Run one command with bounded wall time and retain Linux resource evidence."""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def read_status(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return values
    for line in lines:
        if line.startswith(("VmRSS:", "VmHWM:", "VmPeak:", "VmSize:")):
            name, value, unit = line.split()
            if unit != "kB":
                raise RuntimeError(f"unexpected /proc unit: {unit}")
            values[name.rstrip(":")] = int(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing command after --")

    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    maxima = {"VmRSS": 0, "VmHWM": 0, "VmPeak": 0, "VmSize": 0}
    timed_out = False
    start = time.monotonic()
    with args.stdout.open("xb") as stdout, args.stderr.open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        while process.poll() is None:
            for name, value in read_status(process.pid).items():
                maxima[name] = max(maxima[name], value)
            if time.monotonic() - start > args.timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.02)
        return_code = process.wait()
    elapsed = time.monotonic() - start
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "command": command,
        "elapsed_wall_seconds": elapsed,
        "exit_code": return_code,
        "max_proc_kib": maxima,
        "ru_maxrss_kib": usage_after.ru_maxrss,
        "user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "timed_out": timed_out,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if return_code != 0 or timed_out:
        raise SystemExit(124 if timed_out else return_code)


if __name__ == "__main__":
    main()

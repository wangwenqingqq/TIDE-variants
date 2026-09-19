#!/usr/bin/env python3
"""Read-only GPU telemetry; flag foreign processes, never signal them."""
import sys
sys.dont_write_bytecode = True
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(sys.argv[1])
owner = int(sys.argv[2])


def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == owner:
            return True
        seen.add(pid)
        try:
            info = Path(f'/proc/{pid}/status').read_text()
        except FileNotFoundError:
            return True  # Already finished before ancestry could be read.
        pid = int(next(line.split()[1] for line in info.splitlines() if line.startswith('PPid:')))
    return False


with (root / 'raw/TELEMETRY.jsonl').open('x', buffering=1) as stream:
    while not (root / 'raw/MONITOR_STOP').exists():
        proc = subprocess.run(['nvidia-smi', '-i', '2', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'], text=True, capture_output=True)
        sample = {'unix_ns': time.time_ns(), 'returncode': proc.returncode,
                  'stdout': proc.stdout, 'stderr': proc.stderr}
        foreign = []
        if proc.returncode == 0:
            for row in proc.stdout.splitlines():
                if row.strip():
                    pid = int(row.split(',')[0])
                    if not descendant(pid):
                        foreign.append(pid)
        sample['foreign_pids'] = foreign
        stream.write(json.dumps(sample) + '\n')
        if foreign or proc.returncode:
            with (root / 'raw/CONTAMINATION.json').open('x') as out:
                json.dump(sample, out, indent=2)
            break
        time.sleep(1)

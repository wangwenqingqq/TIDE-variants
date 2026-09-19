#!/usr/bin/env python3
"""Build the diagnostic Gate-6 NCU dynamic instruction ledger.

NCU duration is profiler evidence only. It is intentionally kept separate from
the formal complete-service sustained denominator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_V1 = ROOT / "results/raw/20260829T042000Z_mechanism_ncu_v1"
RAW = ROOT / "results/raw/20260829T042300Z_mechanism_ncu_v2"
OUT = ROOT / "results/mechanism_ncu/formal_surechembl_latest_v2"

METRICS = (
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    "smsp__inst_executed_op_branch.sum",
    "smsp__inst_executed_op_generic_atom.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_sectors_op_read.sum",
    "profiler__replayer_passes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path) -> tuple[dict[str, int], str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 or rows[0].get("ID") != "" or rows[1].get("ID") != "0":
        raise ValueError(f"unexpected NCU raw CSV structure: {path}")
    data = rows[1]
    values = {metric: int(data[metric]) for metric in METRICS}
    return values, data["Kernel Name"]


def opcode(line: str) -> str:
    tokens = line.strip().split()
    if not tokens:
        return ""
    if tokens[0].startswith("@"):
        tokens = tokens[1:]
    return tokens[0].split(".", 1)[0] if tokens else ""


def load_source(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or rows[0][0] != "Kernel Name" or rows[1][0] != "Address":
        raise ValueError(f"unexpected NCU source CSV structure: {path}")
    header = rows[1]
    records = [dict(zip(header, row)) for row in rows[2:] if len(row) == len(header)]
    selected = {"POPC": [], "atomic": []}
    for record in records:
        op = opcode(record["Source"])
        if op == "POPC":
            selected["POPC"].append(record)
        if op in {"ATOM", "ATOMG", "ATOMS", "RED", "REDG", "REDS"}:
            selected["atomic"].append(record)

    def summarize(records: list[dict[str, str]]) -> dict[str, object]:
        return {
            "static_pc_count": len(records),
            "warp_instructions_executed": sum(int(r["Instructions Executed"]) for r in records),
            "thread_instructions_executed": sum(int(r["Thread Instructions Executed"]) for r in records),
            "predicated_on_thread_instructions_executed": sum(int(r["Predicated-On Thread Instructions Executed"]) for r in records),
            "pcs": [
                {
                    "address": r["Address"],
                    "sass": r["Source"].strip(),
                    "warp_instructions_executed": int(r["Instructions Executed"]),
                    "thread_instructions_executed": int(r["Thread Instructions Executed"]),
                    "predicated_on_thread_instructions_executed": int(r["Predicated-On Thread Instructions Executed"]),
                }
                for r in records
            ],
        }

    return {
        "source_rows": len(records),
        "popc": summarize(selected["POPC"]),
        "atomic": summarize(selected["atomic"]),
    }


def load_app(path: Path) -> dict[tuple[int, int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            (int(row["threshold_num"]), int(row["threshold_den"]), int(row["query"])): row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    variants: dict[str, object] = {}
    for variant in ("logical64", "unbounded64"):
        metrics, kernel_name = load_raw(RAW / f"{variant}.raw_fixed.csv")
        source = load_source(RAW / f"{variant}.source_sass_fixed.csv")
        stdout = (RAW / f"{variant}.stdout.log").read_text()
        stderr = (RAW / f"{variant}.stderr.log").read_text()
        report = RAW / f"{variant}.ncu-rep"
        variants[variant] = {
            "kernel_name": kernel_name,
            "profiled_launches": len(re.findall(r"Profiling ", stdout)),
            "reported_replay_passes": int(re.search(r"- (\d+) passes", stdout).group(1)),
            "exit_code": int((RAW / f"{variant}.exit_code.txt").read_text().strip()),
            "stderr_empty": stderr == "",
            "report_sha256": sha256(report),
            "metrics": metrics,
            "duration_us": metrics["gpu__time_duration.sum"] / 1000.0,
            "source_counters": source,
        }

    logical_app = load_app(OUT / "logical64.csv")
    unbounded_app = load_app(OUT / "unbounded64.csv")
    key = (7, 10, 0)
    left, right = logical_app[key], unbounded_app[key]
    exact_equal = all(left[field] == right[field] for field in ("hits", "result_hash", "overflow"))
    candidate_ratio = int(right["candidate_rows"]) / int(left["candidate_rows"])
    metric_ratios = {
        metric: variants["unbounded64"]["metrics"][metric] / variants["logical64"]["metrics"][metric]
        for metric in METRICS
        if variants["logical64"]["metrics"][metric] != 0
    }
    popc_ratio = (
        variants["unbounded64"]["source_counters"]["popc"]["thread_instructions_executed"]
        / variants["logical64"]["source_counters"]["popc"]["thread_instructions_executed"]
    )
    atomic_equal = (
        variants["logical64"]["source_counters"]["atomic"]["thread_instructions_executed"]
        == variants["unbounded64"]["source_counters"]["atomic"]["thread_instructions_executed"]
        == 1
    )
    collection_valid = all(
        variants[v]["profiled_launches"] == 1
        and variants[v]["reported_replay_passes"] == 6
        and variants[v]["metrics"]["profiler__replayer_passes"] == 6
        and variants[v]["exit_code"] == 0
        and variants[v]["stderr_empty"]
        for v in variants
    )
    mechanism_consistent = bool(
        collection_valid
        and exact_equal
        and candidate_ratio > 1.0
        and metric_ratios["gpu__time_duration.sum"] > 1.0
        and metric_ratios["smsp__inst_executed.sum"] > 1.0
        and metric_ratios["dram__bytes_op_read.sum"] > 1.0
        and popc_ratio > 1.0
        and atomic_equal
    )

    invalid_v1 = {
        "path": str(RAW_V1.relative_to(ROOT)),
        "state": "invalid_no_kernel_profiled",
        "reason": "demangled kernel filter required exact_runs_kernel<4>, but the live name was exact_runs_kernel<(int)4>; NCU returned zero while reporting no kernels profiled",
        "logical64_stdout_sha256": sha256(RAW_V1 / "logical64.stdout.log"),
        "unbounded64_stdout_sha256": sha256(RAW_V1 / "unbounded64.stdout.log"),
    }
    summary = {
        "experiment_id": "tide_20260829_gate6_mechanism_ncu_surechembl_latest_v2",
        "evidence_state": "measured_diagnostic_only",
        "scope": "latest SureChEMBL, physical GPU2, deterministic query0 at threshold 7/10, exact WORDS4 kernel, one retained launch after 16 matching warmups",
        "not_a_performance_denominator": "NCU used six kernel-replay passes; formal complete-service sustained performance is reported separately",
        "ncu_version": (RAW / "ncu_version.txt").read_text().strip(),
        "driver_counter_policy": (RAW / "driver_counter_policy.txt").read_text().strip(),
        "invalid_predecessor": invalid_v1,
        "collection_valid": collection_valid,
        "profiled_result_observables_equal": exact_equal,
        "profiled_request": {
            "threshold": "7/10",
            "query": 0,
            "logical64_candidate_rows": int(left["candidate_rows"]),
            "unbounded64_candidate_rows": int(right["candidate_rows"]),
            "unbounded64_over_logical64_candidate_ratio": candidate_ratio,
            "hits": int(left["hits"]),
            "result_hash": int(left["result_hash"]),
            "overflow": int(left["overflow"]),
        },
        "variants": variants,
        "unbounded64_over_logical64": {
            "metrics": metric_ratios,
            "popc_thread_instruction_ratio": popc_ratio,
            "atomic_thread_instruction_count_equal_at_one": atomic_equal,
        },
        "mechanism_evidence_consistent": mechanism_consistent,
        "interpretation": "Disabling the exact population-count bound admitted more candidates and proportionally increased executed POPC work, total instructions, DRAM reads, L2 read sectors, and profiler kernel duration, while final results and the executed atomic count remained unchanged.",
        "aggregate_G6_MECHANISM": "PARTIAL_SURECHEMBL_ONLY" if mechanism_consistent else "UNVALIDATED",
        "aggregate_caveat": "This is one deterministic diagnostic request, not a latency estimator, and ChEMBL 37 remains blocked by source integrity.",
    }
    out_path = OUT / "NCU_DYNAMIC_LEDGER.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    raw_pointer = {
        "primary_summary": str(out_path.relative_to(ROOT)),
        "primary_summary_sha256": sha256(out_path),
        "decision": "MECHANISM_DIAGNOSTIC_CONSISTENT" if mechanism_consistent else "UNVALIDATED",
    }
    (RAW / "NCU_AGGREGATE.json").write_text(json.dumps(raw_pointer, indent=2) + "\n")
    print(json.dumps({
        "collection_valid": collection_valid,
        "exact_equal": exact_equal,
        "candidate_ratio": candidate_ratio,
        "duration_ratio": metric_ratios["gpu__time_duration.sum"],
        "instruction_ratio": metric_ratios["smsp__inst_executed.sum"],
        "dram_read_ratio": metric_ratios["dram__bytes_op_read.sum"],
        "l2_read_sector_ratio": metric_ratios["lts__t_sectors_op_read.sum"],
        "popc_thread_instruction_ratio": popc_ratio,
        "atomic_equal_at_one": atomic_equal,
        "mechanism_consistent": mechanism_consistent,
        "summary": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()

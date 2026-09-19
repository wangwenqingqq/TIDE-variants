#!/usr/bin/env python3
"""Create additive final evidence for the four Gate-5 blocker closures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--compaction", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    exact = json.loads(args.exact.read_text())
    recovery = json.loads(args.recovery.read_text())
    compaction = json.loads(args.compaction.read_text())
    memory = json.loads(args.memory.read_text())
    storage_reported = all(
        key in recovery
        for key in ("max_store_bytes", "max_trial_bytes", "delta_payload_bytes")
    )
    host_memory_reported = all(
        key in memory
        for key in (
            "cuda_context_baseline_used_bytes",
            "mapped_input_file_bytes",
            "owned_host_payload_bytes",
            "host_auxiliary_ledger",
            "measured_host_rss_peak_kib",
            "measured_host_hwm_peak_kib",
        )
    ) and memory.get("all_cuda_malloc_sites_audited") is True
    memory_gate = (
        memory.get("B5_MEMORY_DEVICE_CLOSURE") is True
        and storage_reported
        and host_memory_reported
    )
    gates = {
        "B5_EXACT_CLOSURE": exact.get("B5_EXACT_CLOSURE") is True,
        "B5_RECOVERY_CLOSURE": recovery.get("B5_RECOVERY_CLOSURE") is True,
        "B5_COMPACTION_CLOSURE": compaction.get("B5_COMPACTION_CLOSURE") is True,
        "B5_MEMORY_CLOSURE": memory_gate,
    }
    all_pass = all(gates.values())
    result = {
        "experiment_id": "tide_20260828_gate5_four_blocker_closure",
        "protocol": str(args.root / "BLOCKER_CLOSURE_PROTOCOL_20260828.md"),
        "implementation_scope": (
            "immutable host snapshot owning immutable device runs; "
            "query-specific DeviceSlice materialization per reader"
        ),
        "gates": gates,
        "all_four_blockers_closed": all_pass,
        "decision": (
            "BLOCKERS_CLOSED__ADMIT_WITH_ACTUAL_HOST_SNAPSHOT_SCOPE"
            if all_pass
            else "HOLD__BLOCKER_CLOSURE_INCOMPLETE"
        ),
        "exactness": exact,
        "recovery": {key: value for key, value in recovery.items() if key != "records"},
        "compaction": {
            key: value for key, value in compaction.items() if key != "process_records"
        },
        "memory": {
            **memory,
            "durable_storage": {
                "max_store_bytes": recovery.get("max_store_bytes"),
                "max_trial_bytes": recovery.get("max_trial_bytes"),
                "delta_payload_bytes": recovery.get("delta_payload_bytes"),
            },
            "B5_MEMORY_CLOSURE": memory_gate,
            "host_memory_reported": host_memory_reported,
        },
        "original_gate5_artifacts_immutable": True,
        "original_full_racecheck": "aborted/unvalidated; not rerun and not a pass",
        "evidence_paths": {
            "exact": str(args.exact),
            "recovery": str(args.recovery),
            "compaction": str(args.compaction),
            "memory": str(args.memory),
        },
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    md = f"""# Gate-5 four-blocker closure

Decision: **{result['decision']}**

| Gate | Result |
|---|---|
| Exact CPU-oracle matrix | {gates['B5_EXACT_CLOSURE']} |
| End-to-end crash recovery | {gates['B5_RECOVERY_CLOSURE']} |
| Canonical compaction | {gates['B5_COMPACTION_CLOSURE']} |
| Complete memory/storage report | {gates['B5_MEMORY_CLOSURE']} |

## Exactness

- State checks: {exact.get('state_checks')}; mismatches: {exact.get('state_mismatches')}.
- Post-compaction checks: {exact.get('post_compaction_checks')}; mismatches: {exact.get('post_compaction_mismatches')}.
- Canonical comparison: {exact.get('compaction_row_mismatches')} row and {exact.get('compaction_byte_mismatches')} byte mismatches.

## Recovery

- Deterministic/randomized trials: {recovery.get('deterministic_trials')}/{recovery.get('randomized_trials')}.
- Failures: {recovery.get('failures')}; old/new epochs: {recovery.get('old_epoch_recoveries')}/{recovery.get('new_epoch_recoveries')}.

## Compaction

- Paired p99 ratio: {compaction.get('paired_geomean_p99_ratio'):.6f}.
- Process-cluster bootstrap 95% interval: {compaction.get('process_cluster_bootstrap_95')}.
- Byte mismatches: {compaction.get('byte_mismatches')}.
- Minimum captures during compaction: {compaction.get('minimum_captures_during_compaction')}.

## Memory

- Requested/measured production peak bytes: {memory.get('requested_production_peak_bytes')}/{memory.get('measured_production_peak_bytes')}.
- Runtime/allocator excess bytes: {memory.get('allocator_runtime_excess_bytes')}.
- Cleanup residual bytes: {memory.get('cleanup_residual_bytes')}.
- Maximum durable store/trial bytes: {recovery.get('max_store_bytes')}/{recovery.get('max_trial_bytes')}.

The implementation scope is the actual host-published immutable snapshot with
per-query device-slice materialization. The original full Racecheck remains
aborted/unvalidated and was not rerun. Existing Gate-5 final artifacts were not
overwritten.
"""
    args.output_md.write_text(md)
    manifest = args.output_json.parent / "BLOCKER_CLOSURE_ARTIFACTS.sha256"
    manifest.write_text(
        f"{sha256_file(args.output_json)}  {args.output_json.name}\n"
        f"{sha256_file(args.output_md)}  {args.output_md.name}\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pure-CPU contract for C1 v8 per-operation capacity snapshots.

The C++ runner records both capacity fields from one local value captured before
an ephemeral workspace may reset its global capacity.  This module deliberately
contains no CUDA, subprocess, or filesystem side effects so measured and profile
verifiers can enforce exactly the same JSON contract.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

CAPACITY_SNAPSHOT_STAGE = "pre_ephemeral_release"
CAPACITY_FIELDS = ("update_result_capacity_slots", "total_result_capacity_slots")


def _strict_int(value: Any) -> int | None:
    # bool is an int subclass but is never a valid capacity.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def capacity_snapshot_violations(record: Mapping[str, Any]) -> list[str]:
    """Return deterministic, human-readable violations for one JSONL record.

    ``result_count`` is part of the same per-operation contract as the two
    capacity snapshots: it must be a non-negative integer and fit in each
    snapshot.  Keeping this here prevents the measured and profile verifiers
    from drifting into different interpretations of a valid record.
    """
    if not isinstance(record, Mapping):
        return ["record is not an object"]
    result_count = _strict_int(record.get("result_count"))
    update = _strict_int(record.get(CAPACITY_FIELDS[0]))
    total = _strict_int(record.get(CAPACITY_FIELDS[1]))
    failures: list[str] = []
    if result_count is None or result_count < 0:
        failures.append("result_count must be an integer >= 0")
    if update is None or update <= 0:
        failures.append("update_result_capacity_slots snapshot must be an integer > 0")
    if total is None or total <= 0:
        failures.append("total_result_capacity_slots snapshot must be an integer > 0")
    if update is not None and total is not None and update != total:
        failures.append("capacity snapshot fields must be equal")
    if result_count is not None and update is not None and result_count > update:
        failures.append("result_count must not exceed update_result_capacity_slots")
    if result_count is not None and total is not None and result_count > total:
        failures.append("result_count must not exceed total_result_capacity_slots")
    if record.get("capacity_snapshot_stage") != CAPACITY_SNAPSHOT_STAGE:
        failures.append(f"capacity_snapshot_stage must be {CAPACITY_SNAPSHOT_STAGE!r}")
    return failures

#!/usr/bin/env python3
"""CPU-only checker for the pre-registered C2 controlled-drift monitor.

It does not run GTS and cannot generate oracle answers.  A future isolated C2
runner must emit anchor features and independently verified exact-shadow records;
this checker then applies the frozen threshold and enforces next-query gamma=1
fallback semantics from the protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"refusing to write outside isolation root: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_no}: {exc}")
            if not isinstance(value, dict):
                raise SystemExit(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(value)
    return rows


def features(rows: list[dict[str, Any]], expected_n: int | None = None) -> list[list[float]]:
    if expected_n is not None and len(rows) != expected_n:
        raise SystemExit(f"expected {expected_n} rows, found {len(rows)}")
    values: list[list[float]] = []
    width: int | None = None
    for index, row in enumerate(rows):
        if row.get("local_index") != index:
            raise SystemExit(f"local_index must be contiguous from 0; row {index} has {row.get('local_index')!r}")
        x = row.get("anchor_distances")
        if not isinstance(x, list) or not x:
            raise SystemExit(f"row {index} has no anchor_distances array")
        try:
            f = [float(v) for v in x]
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"non-numeric anchor_distances at row {index}") from exc
        if any(not math.isfinite(v) for v in f):
            raise SystemExit(f"nonfinite anchor distance at row {index}")
        if width is None:
            width = len(f)
        if len(f) != width:
            raise SystemExit("inconsistent anchor-distance width")
        values.append(f)
    return values


def fit_model(protocol: dict[str, Any], ref_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = protocol["detector"]
    width = 16
    reference = features(ref_rows, protocol["partitions"]["detector_reference"]["n"])
    threshold = features(threshold_rows, protocol["partitions"]["detector_threshold"]["n"])
    if not all(len(row) == width for row in reference + threshold):
        raise SystemExit("protocol requires exactly 16 anchor-distance features per row")
    n = len(reference)
    mu = [math.fsum(row[j] for row in reference) / n for j in range(width)]
    sigma = [math.sqrt(math.fsum((row[j] - mu[j]) ** 2 for row in reference) / (n - 1)) for j in range(width)]
    if any((not math.isfinite(s)) or s <= 0.0 for s in sigma):
        raise SystemExit("zero/nonfinite anchor sigma; abort per protocol")
    window = int(cfg["window_size"])
    if len(threshold) % window:
        raise SystemExit("threshold records must divide into full fixed windows")

    def score(chunk: list[list[float]]) -> float:
        means = [math.fsum(row[j] for row in chunk) / len(chunk) for j in range(width)]
        return max(abs(means[j] - mu[j]) / (sigma[j] / math.sqrt(window) + EPS) for j in range(width))

    scores = [score(threshold[i:i + window]) for i in range(0, len(threshold), window)]
    return {
        "schema": "gtspp-c2-anchor-drift-model-v1",
        "scope": "fixed pre-registered covariate-shift detector; not a recall certificate",
        "protocol_schema": protocol["schema"],
        "anchor_feature_count": width,
        "window_size": window,
        "mu": mu,
        "sigma": sigma,
        "threshold_window_scores": scores,
        "tau": max(scores),
        "comparison": "strictly_greater_than",
        "consecutive_windows_required": 2
    }


def score_window(chunk: list[list[float]], model: dict[str, Any]) -> float:
    w = int(model["window_size"])
    if len(chunk) != w:
        raise ValueError("partial window")
    mu, sigma = model["mu"], model["sigma"]
    means = [math.fsum(row[j] for row in chunk) / w for j in range(len(mu))]
    return max(abs(means[j] - mu[j]) / (sigma[j] / math.sqrt(w) + EPS) for j in range(len(mu)))


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-7 * max(1.0, abs(a), abs(b))


def replay_phase(protocol: dict[str, Any], model: dict[str, Any], rows: list[dict[str, Any]], phase: str, initial_gamma: float) -> dict[str, Any]:
    if phase not in ("pre_shift_live", "shifted_live", "shifted_recovery_test"):
        raise SystemExit(f"not a live/recovery phase: {phase}")
    expected_n = int(protocol["partitions"][phase]["n"])
    xs = features(rows, expected_n)
    if len(model.get("mu", [])) != 16 or len(model.get("sigma", [])) != 16:
        raise SystemExit("unexpected detector model width")
    if any(len(row) != 16 for row in xs):
        raise SystemExit("live event width must equal 16")
    w = int(protocol["detector"]["window_size"])
    stride, offset = 8, 7
    if expected_n % w:
        raise SystemExit("protocol live length must divide into fixed detector windows")

    current_gamma = float(initial_gamma)
    fallback_from: int | None = None
    fallback_reason: str | None = None
    consecutive = 0
    scores: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    breaches: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    baseline_failure = False

    for i, (row, x) in enumerate(zip(rows, xs)):
        if row.get("phase") != phase:
            raise SystemExit(f"event {i} phase mismatch: {row.get('phase')!r} != {phase!r}")
        expected_shadow = (i % stride) == offset
        if row.get("shadow") is not expected_shadow:
            raise SystemExit(f"event {i} shadow flag violates fixed stride=8/offset=7")
        reported_gamma = float(row.get("gamma"))
        if not close(reported_gamma, current_gamma):
            raise SystemExit(f"event {i} gamma {reported_gamma} violates controller state {current_gamma}")
        if expected_shadow:
            oracle = row.get("oracle")
            if not isinstance(oracle, dict):
                raise SystemExit(f"shadow event {i} lacks oracle object")
            ok_membership = oracle.get("membership_ok") is True
            ok_distance = oracle.get("distance_ok") is True
            boundary_tie = oracle.get("k_boundary_tie") is True
            if boundary_tie:
                ambiguous.append({"local_index": i, "gamma": reported_gamma})
            elif not (ok_membership and ok_distance):
                breach = {"local_index": i, "gamma": reported_gamma,
                          "membership_ok": ok_membership, "distance_ok": ok_distance}
                breaches.append(breach)
                if reported_gamma > 1.0 + 1e-7:
                    if fallback_from is None:
                        fallback_from, fallback_reason = i + 1, "EXACT_SHADOW_BREACH"
                        current_gamma = 1.0
                else:
                    baseline_failure = True
            # A K-boundary tie cannot certify the profile.  If it occurs under gamma>1,
            # conservatively quarantine it for the next request rather than treating it as pass.
            if boundary_tie and reported_gamma > 1.0 + 1e-7 and fallback_from is None:
                fallback_from, fallback_reason = i + 1, "EXACT_SHADOW_K_BOUNDARY_AMBIGUITY"
                current_gamma = 1.0

        if (i + 1) % w == 0:
            s = score_window(xs[i + 1 - w:i + 1], model)
            exceeds = s > float(model["tau"])
            consecutive = consecutive + 1 if exceeds else 0
            item = {"window_index": i // w, "end_local_index": i,
                    "score": s, "exceeds_tau": exceeds,
                    "consecutive_exceedances": consecutive}
            scores.append(item)
            if consecutive >= 2:
                alert = {"local_index": i, "window_index": i // w, "score": s,
                         "tau": float(model["tau"])}
                alerts.append(alert)
                if fallback_from is None:
                    fallback_from, fallback_reason = i + 1, "DRIFT_ALERT_TWO_CONSECUTIVE_WINDOWS"
                    current_gamma = 1.0

    return {
        "schema": "gtspp-c2-drift-phase-check-v1",
        "phase": phase,
        "initial_gamma": initial_gamma,
        "final_gamma": current_gamma,
        "fallback_from_next_local_index": fallback_from,
        "fallback_reason": fallback_reason,
        "window_scores": scores,
        "alerts": alerts,
        "exact_shadow_breaches": breaches,
        "k_boundary_ambiguities": ambiguous,
        "baseline_failure": baseline_failure,
        "result": "ABORT_BASELINE_FAILURE" if baseline_failure else "CHECKED_NOT_A_CORRECTNESS_CERTIFICATE"
    }


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("schema") != "gtspp-c2-controlled-drift-recalibration-protocol-v1":
        raise SystemExit("unexpected protocol schema")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path(__file__).with_name("protocol_sift1m_query_to_learn_controlled_drift_v1.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--reference-jsonl", type=Path, required=True)
    fit.add_argument("--threshold-jsonl", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--events-jsonl", type=Path, required=True)
    replay.add_argument("--phase", required=True)
    replay.add_argument("--initial-gamma", type=float, required=True)
    replay.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    root = Path(protocol["isolation"]["write_root"])
    ensure_under(args.out, root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        model = fit_model(protocol, read_jsonl(args.reference_jsonl), read_jsonl(args.threshold_jsonl))
        model["generated_utc"] = datetime.now(timezone.utc).isoformat()
        model["protocol"] = {"path": str(args.protocol), "sha256": sha256(args.protocol)}
        model["input"] = {"reference_jsonl": str(args.reference_jsonl), "threshold_jsonl": str(args.threshold_jsonl)}
        args.out.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    else:
        model = json.loads(args.model.read_text())
        result = replay_phase(protocol, model, read_jsonl(args.events_jsonl), args.phase, args.initial_gamma)
        result["generated_utc"] = datetime.now(timezone.utc).isoformat()
        result["protocol"] = {"path": str(args.protocol), "sha256": sha256(args.protocol)}
        result["model"] = {"path": str(args.model), "sha256": sha256(args.model)}
        result["events_jsonl"] = str(args.events_jsonl)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

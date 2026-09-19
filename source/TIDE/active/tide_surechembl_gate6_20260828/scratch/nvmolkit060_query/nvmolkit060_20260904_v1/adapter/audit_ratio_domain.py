#!/usr/bin/env python3
"""Exhaustively audit nvMolKit thresholds over all 256-bit count ratios."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from nvmolkit.similarity import crossTanimotoSimilarity


def prefix_fingerprint(bit_count: int) -> list[int]:
    words = []
    remaining = bit_count
    for _ in range(8):
        take = min(32, remaining)
        words.append(0 if take == 0 else (1 << take) - 1)
        remaining -= take
    return words


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    packed = np.asarray([prefix_fingerprint(i) for i in range(257)], dtype=np.uint32)
    device = torch.from_numpy(packed).to("cuda")
    scores = crossTanimotoSimilarity(device, device).torch()
    torch.cuda.synchronize()
    values = scores.cpu().numpy()

    result = {
        "format": "nvmolkit060_exhaustive_256bit_ratio_domain_v1",
        "construction": "prefix bit sets with populations 0..256; every nonzero intersection/union ratio is represented",
        "score_dtype": str(scores.dtype),
        "zero_zero_score": float(values[0, 0]),
        "thresholds": {},
    }
    for numerator, denominator in ((7, 10), (4, 5)):
        base = np.float32(numerator / denominator)
        down1 = np.nextafter(base, np.float32(-np.inf))
        down2 = np.nextafter(down1, np.float32(-np.inf))
        candidates = {
            "python_double": float(numerator / denominator),
            "float32_canonical": float(base),
            "float32_nextafter_down_1": float(down1),
            "float32_nextafter_down_2": float(down2),
        }
        rules = {
            name: {"threshold": threshold, "threshold_hex": threshold.hex(), "false_negative": [], "false_positive": []}
            for name, threshold in candidates.items()
        }
        pass_scores = []
        fail_scores = []
        for lhs in range(257):
            for rhs in range(257):
                intersection = min(lhs, rhs)
                union = max(lhs, rhs)
                if union == 0:
                    continue
                exact = denominator * intersection >= numerator * union
                score = float(values[lhs, rhs])
                (pass_scores if exact else fail_scores).append((score, lhs, rhs, intersection, union))
                for rule in rules.values():
                    observed = score >= rule["threshold"]
                    if exact and not observed:
                        rule["false_negative"].append((lhs, rhs, intersection, union, score))
                    elif observed and not exact:
                        rule["false_positive"].append((lhs, rhs, intersection, union, score))
        min_pass = min(pass_scores)
        max_fail = max(fail_scores)
        result["thresholds"][f"{numerator}_{denominator}"] = {
            "minimum_passing_score": min_pass,
            "maximum_failing_score": max_fail,
            "strict_score_separation": max_fail[0] < min_pass[0],
            "rules": {
                name: {
                    "threshold": rule["threshold"],
                    "threshold_hex": rule["threshold_hex"],
                    "false_negative_count": len(rule["false_negative"]),
                    "false_positive_count": len(rule["false_positive"]),
                    "first_false_negative": rule["false_negative"][:3],
                    "first_false_positive": rule["false_positive"][:3],
                }
                for name, rule in rules.items()
            },
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

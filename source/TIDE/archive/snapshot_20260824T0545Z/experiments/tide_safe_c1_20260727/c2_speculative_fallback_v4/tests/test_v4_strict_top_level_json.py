#!/usr/bin/env python3
"""CPU-only adversarial tests for the v4 manifest parser contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4")
sys.path.insert(0, str(ROOT / "tools"))
from c2_v4_strict_json_contract import StrictJsonError, require_exact_top_object, strict_loads  # noqa: E402

FIELDS = {"schema", "sealed_v2_test_forbidden", "query_fvecs_count", "payload"}
EXPECTED_STRINGS = {"schema": "gts-v4"}
EXPECTED_BOOLEANS = {"sealed_v2_test_forbidden": True}
POSITIVE = {"query_fvecs_count"}


def good(payload: object) -> dict[str, object]:
    return {
        "schema": "gts-v4",
        "sealed_v2_test_forbidden": True,
        "query_fvecs_count": 12524,
        "payload": payload,
    }


def accepts(text: str) -> None:
    require_exact_top_object(
        text,
        expected_fields=FIELDS,
        expected_strings=EXPECTED_STRINGS,
        expected_booleans=EXPECTED_BOOLEANS,
        expected_positive_integers=POSITIVE,
    )


def rejects(text: str) -> None:
    try:
        accepts(text)
    except StrictJsonError:
        return
    raise AssertionError(f"expected rejection: {text!r}")


def main() -> int:
    # (1) Arbitrary root ordering is accepted; nested same-named keys are not
    # allowed to shadow the root schema.
    reordered = (
        '{"payload":{"schema":"nested-evil","meta":{"schema":"nested-again"}},'
        '"query_fvecs_count":12524,"sealed_v2_test_forbidden":true,"schema":"gts-v4"}'
    )
    accepts(reordered)

    # (2) Escapes decode before exact type/value checks; escaped slash and
    # escaped hyphen must not change root lookup semantics.
    escaped = (
        '{"payload":{"path":"\\/tmp\\/x","schema":"nested"},'
        '"schema":"gts\\u002dv4","sealed_v2_test_forbidden":true,"query_fvecs_count":12524}'
    )
    accepts(escaped)

    # (3) Nested key alone cannot satisfy a missing top-level field.
    rejects('{"payload":{"schema":"gts-v4"},"sealed_v2_test_forbidden":true,"query_fvecs_count":12524}')

    # (4) Duplicate root and duplicate nested keys are both rejected before
    # any field lookup, regardless of order.
    rejects('{"schema":"gts-v4","schema":"evil","sealed_v2_test_forbidden":true,"query_fvecs_count":12524,"payload":{}}')
    rejects('{"schema":"gts-v4","sealed_v2_test_forbidden":true,"query_fvecs_count":12524,"payload":{"x":1,"x":2}}')

    # (5) Type coercion is forbidden.
    rejects(json.dumps(good({}) | {"sealed_v2_test_forbidden": "true"}))
    rejects(json.dumps(good({}) | {"query_fvecs_count": "12524"}))
    rejects(json.dumps(good({}) | {"query_fvecs_count": 12524.0}))

    # (6) Unknown/missing keys, malformed escape syntax, and invalid UTF-16
    # surrogate must all fail closed.
    rejects(json.dumps(good({}) | {"extra": 1}))
    rejects('{"schema":"gts-v4","sealed_v2_test_forbidden":true,"query_fvecs_count":12524,"payload":"\\uD800"}')
    rejects('{"schema":"gts-v4","sealed_v2_test_forbidden":true,"query_fvecs_count":12524,"payload":"\\q"}')
    rejects('{"schema":"gts-v4","sealed_v2_test_forbidden":true,"query_fvecs_count":12524,"payload":"\\u0000"}')

    # (7) The reference parser itself accepts root ordering and returns the
    # root value, not a nested similarly named field.
    loaded = strict_loads(reordered)
    assert loaded["schema"] == "gts-v4"
    assert loaded["payload"]["schema"] == "nested-evil"
    print("PASS strict_top_level_json_cases=10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

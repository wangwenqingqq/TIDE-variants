#!/usr/bin/env python3
"""CPU-only reference contract for the v4 strict top-level JSON parser.

This is a test oracle for the C++ source; it does not execute or compile the
CUDA binary. It rejects duplicate JSON keys at every object scope and makes
top-level field lookup explicit.
"""
from __future__ import annotations

import json
from typing import Any, Iterable


class StrictJsonError(ValueError):
    pass


def _reject_constant(token: str) -> None:
    raise StrictJsonError(f"non-standard JSON numeric constant: {token}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_strings(value: Any) -> None:
    if isinstance(value, str):
        if "\x00" in value:
            raise StrictJsonError("NUL is forbidden in runner-visible JSON strings")
        if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
            raise StrictJsonError("unpaired surrogate in JSON string")
    elif isinstance(value, list):
        for item in value:
            _validate_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_strings(key)
            _validate_strings(item)


def strict_loads(text: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except StrictJsonError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrictJsonError(f"invalid JSON: {exc}") from exc
    _validate_strings(value)
    return value


def require_exact_top_object(
    text: str,
    *,
    expected_fields: Iterable[str],
    expected_strings: dict[str, str] | None = None,
    expected_booleans: dict[str, bool] | None = None,
    expected_positive_integers: Iterable[str] = (),
) -> dict[str, Any]:
    value = strict_loads(text)
    if not isinstance(value, dict):
        raise StrictJsonError("root must be an object")
    expected = set(expected_fields)
    if set(value) != expected:
        raise StrictJsonError(f"top-level field set mismatch: got {sorted(value)}, expected {sorted(expected)}")
    for key, target in (expected_strings or {}).items():
        if not isinstance(value.get(key), str) or value[key] != target:
            raise StrictJsonError(f"top-level string mismatch: {key}")
    for key, target in (expected_booleans or {}).items():
        # bool must not be accepted through an integer-like coercion.
        if type(value.get(key)) is not bool or value[key] is not target:
            raise StrictJsonError(f"top-level boolean mismatch: {key}")
    for key in expected_positive_integers:
        item = value.get(key)
        if type(item) is not int or item <= 0:
            raise StrictJsonError(f"top-level positive integer mismatch: {key}")
    return value

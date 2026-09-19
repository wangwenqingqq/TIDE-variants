#!/usr/bin/env python3
"""Audit only this source-only contract package; never invoke it as a benchmark."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REQUIRED = {
    "README.md", "NONCLAIMS.md", "NATIVE_GTS_E2E3_CONTRACT.md", "CHECKLIST.md",
    "templates/NATIVE_GTS_TOPOLOGY_CAPTURE_TEMPLATE.json",
    "templates/NATIVE_GTS_E2E3_CAMPAIGN_PLAN_TEMPLATE.json",
    "schemas/native_gts_e2e3_campaign.schema.json",
    "schemas/native_gts_raw_run_record.schema.json",
    "docs/INDEPENDENT_REPLAY_CONTRACT.md", "docs/PROVENANCE_AND_REPRODUCTION.md",
    "tools/validate_native_gts_e2e3_plan.py", "tools/static_source_audit.py"
}
BASELINE = "b589f0c163b20e5344c9984eb58a3d84c37061d6"

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = sorted(p for p in REQUIRED if not (root / p).is_file())
    try:
        template = json.loads((root / "templates/NATIVE_GTS_E2E3_CAMPAIGN_PLAN_TEMPLATE.json").read_text())
        capture = json.loads((root / "templates/NATIVE_GTS_TOPOLOGY_CAPTURE_TEMPLATE.json").read_text())
        text = (root / "NATIVE_GTS_E2E3_CONTRACT.md").read_text()
        ok = not missing and template.get("template_only") is True and template.get("status") == "SOURCE_ONLY_NOT_EXECUTED" and capture.get("template_only") is True and BASELINE in text
        result = {"schema": "tide.native-gts-e2e3-static-audit.v1", "status": "PASS_SOURCE_ONLY_STATIC_AUDIT" if ok else "FAIL_CLOSED", "missing": missing, "nonclaim": "This audit checks package structure only; it does not build/run GTS, access GPU/data, or establish a campaign result."}
    except Exception as exc:
        result = {"schema": "tide.native-gts-e2e3-static-audit.v1", "status": "FAIL_CLOSED", "error": str(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2

if __name__ == "__main__":
    raise SystemExit(main())

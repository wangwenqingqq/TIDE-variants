# Validation results

## Bound artifact

Checks used a clean `git archive` of aggregate commit
`34b1f901c30d294ae8c82e099ff17c0d50d86d68`, not an uncommitted source tree.
All 2068 captured source/build/support instances matched their committed SHA-256.

## Static checks

- 634 Python sources passed AST parsing (no imports or program execution).
- 321 shell sources passed Bash 5.2 syntax checks (no script execution).
- 10 source schemas/templates passed JSON parsing.
- Full reachable history passed the publication leak scan, including commit
  metadata and branch names. Original upstream license notices are retained.

## Representative CUDA compilation

CUDA 13.1.115, C++17, `sm_120`; each command generated an object file with `-c`.
These are **translation-unit compile checks**, not linked executable checks,
GPU correctness tests, sanitizer runs, benchmarks, or sustained-service validation.
No archived launcher or GPU workload was executed.

| Translation unit | Compile-only result |
|---|---|
| `source/TIDE/active/tide_surechembl_gate0_20260827/src/exact_scan.cu` | PASS |
| `source/TIDE/active/tide_surechembl_gate6_20260828/src/gate6_query_campaign_sustained_unbounded_v1.cu` | PASS |
| `source/GTSPP-TIDEpro_20260915_gate2/experiments/gate2/replay.cu` | PASS |
| `source/RT-TIDE/experiments/large_batch_20260825/cublas_exact_scan_probe.cu` | PASS |
| `source/TIDE/active/tide_surechembl_gate6_20260828/scratch/query_study_p2_20260905_v1/src/tide_batch_bridge.cu` | PASS |

Exact compiler arguments, object/log hashes, and the checked commit are in
`VALIDATION.json`. All other historical translation units are uncompiled in this
archive campaign; their presence is preservation, not endorsement.

## Archive integrity

Run `python3 verify_archive.py` from a clone with all catalog branches available
locally. It checks every declared committed file hash and Git mode. The aggregate
branch preserves all captured instances even when individual branches share an
identical payload. Privacy substitutions and unavailable runtime inputs/authority
records can invalidate legacy pinned execution checks; no new execution approval
is implied. See `COVERAGE.md` for omissions and permission gaps.

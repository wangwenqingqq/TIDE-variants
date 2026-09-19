# TIDE variant archive

Private, source-only preservation snapshot dated 2026-09-19. **This is not a
validated release, a performance ranking, or a claim of 138 independent algorithms.**

## Navigation

- `main`: catalog, scope, dependencies, and validation boundary.
- `variants/active/*`: TIDE gate stages, including SureChEMBL gates 0–6.
- `variants/experiments/*`: query studies, read-while-update, asynchronous update,
  service variants, static-query experiments, and fusion fair controls.
- `variants/rt-tide/*`: RT distance/delta/fused/PCA probes and follow-up experiments.
- `variants/tidepro/*`: later maintenance/replay/service stages.
- `variants/legacy/*`: July shared GTS/TIDE C1/C2/C3 history, including intermediate
  fixes, diagnostics, and rejected or unqualified implementations.
- `variants/protocol/*`: synthetic lifecycle protocols and control-plane source,
  **not GPU search algorithms**.
- `variants/baselines/*`, `variants/companions/*`, `variants/adapters/*`:
  external comparator adapters, related GTS/CertiGraph work, and Page2Tile intent
  code. Their presence does not make them TIDE contributions.
- `archive/all-sources`: the complete **captured** source forest, preserving
  cross-stage relationships for inspection. This is not an extra variant.

138 stage branches represent 137 distinct source/build/support payloads;
2068 original file instances are preserved in the aggregate branch.
See [CATALOG.md](CATALOG.md) and [VARIANTS.json](VARIANTS.json). Byte-identical stage
payloads share a commit; different names alone do not establish different mechanisms.
For a shared commit, file-level source labels describe the representative alias;
all aliases are listed, and the aggregate branch preserves each original instance.

## Restore boundary

Check out a branch and inspect `SNAPSHOT.json`. Individual stages live under
`source/`; RT probes also carry captured dependency context under `context/`.
The aggregate branch retains workspace-relative paths under `source/` with a
neutral historical snapshot label. `SNAPSHOT.json` records original and archived
SHA-256, Git mode, and every file-level privacy adjustment.

**Do not execute historical launchers blindly.** Machine-specific roots are
replaced with `/workspace` placeholders; host/device identity guards are replaced
with explicit configuration placeholders. Private network-storage and external
environment roots are also replaced with neutral placeholders. Original data, authority records,
software environments, binaries, and source-pin manifests are not shipped.
Legacy embedded source hashes are preserved as historical evidence, not recomputed
to authorize a sanitized launcher. Paths, guards, inputs, and pins need a fresh
human-reviewed execution contract before reuse. No safety check is disabled here.

See [COVERAGE.md](COVERAGE.md), [DEPENDENCIES.md](DEPENDENCIES.md), and the latest
`main` version of [VALIDATION.md](VALIDATION.md). Archive integrity, syntax,
compilation, GPU correctness, and performance are distinct gates.

## Rights

No new blanket license is applied. Upstream copyrights and captured license
notices remain intact, including RTSpatial and GPUSim. Shared GTS-derived source
retains its upstream rights; absence of a license is not permission to relicense.
This private archive does not grant public redistribution rights.

# Dependencies and historical limitations

- CUDA/C++ stages need a compatible CUDA toolkit, host compiler, and architecture.
  Historical flags and numeric semantics are retained, except declared publication
  path/identity substitutions. Archive compilation does not exercise a GPU.
- RT-TIDE requires OptiX headers, a configured RTSpatial include root, generated
  PTX, and a suitable GPU at runtime. RTSpatial source and its original MIT notice
  are captured; vendor SDK/environment installations are not. The patched header
  fork carries the same license notice. Some GTSPP example files already reference
  absent optimized headers and must not be treated as a complete build target.
- GPUSim pinned source copies retain their license and Doxygen input template.
  LES3 captured search headers include the referenced `roaring.hh`; CRoaring and
  other toolchain/library dependencies still need configuration. These comparator
  snapshots are not claimed as original TIDE work or freshly validated baselines.
- `UPSTREAMS.json` records inspected upstream checkout state where available; a
  dirty RTSpatial checkout is not represented as pristine upstream. Archived file
  hashes, rather than the upstream HEAD alone, identify the actual payload.
- Historical shell launchers require a modern Linux Bash, not macOS Bash 3.2.
- FPSim2, chemfp, nvmolkit, NumPy/RDKit, FAISS, HNSW, and related external packages
  are represented by selected adapter sources, not copied installations. Names
  or directory-version labels are historical provenance, not a new compatibility
  guarantee. External packages must be obtained under their own licenses.
- Protocol JSON schemas/templates are source support, not live authorization.
  Operational issuers, nonce ledgers, approval receipts, and dataset manifests
  are intentionally not bundled. Historical fixture/hash checks can consequently
  fail until a new execution contract is prepared.
- Archive privacy substitutions preserve kernel math/control flow. They change
  path/identity defaults and some descriptive identifiers. Original and published
  hashes, plus adjustment categories, are listed per file in `SNAPSHOT.json`.
  Do not treat sanitized launchers as byte-identical to the original execution.

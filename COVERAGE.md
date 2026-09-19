# Scope and exclusions

The live source scan covered the readable user workspace on the selected
multi-GPU server, including `TIDE`, `RT-TIDE`, all three discovered `GTSPP-TIDEpro`
stage roots, and source-content hits outside those roots. The scan was read-only;
source snapshots were hash-checked while copying. No GPU workload was launched.

## Included

- All captured source-bearing active TIDE stage roots, including nested gate6
  scratch experiments. Parent gates do not silently stand in for scratch versions.
- RT-TIDE probes, historical patched RTSpatial headers, and three experiment roots.
- TIDEpro gate0, gate1, and gate2 trees, including frozen runtime headers.
- July shared safety experiments and TIDE-named synthetic protocol source stages.
- Related external-baseline adapters, legacy ANNS adapters, and three selected
  Page2Tile TIDE-intent source/test files. Page2Tile itself is not fully archived.
- Source/build/script files and specifically reviewed headers, templates, schemas,
  and license notices. Two fixture-builder stage roots missed by a broad initial
  build-name filter were explicitly rescanned and added before publication.

## Deliberately not included

- Datasets, raw measurements, binaries, generated CMake/build trees, installed
  Python/CUDA environments, downloaded wheels, caches, and runtime authority or
  credential ledgers. Existing source backups inside a captured stage are kept.
- Managed runtime/skill installations and internal research/status prose.
- Third-party package installations and FPSim2 prior-art source extracts; relevant
  adapters remain. These packages must be supplied independently.
- Non-TIDE-named legacy root-artifact controls and GTS-only stages. Some such
  controls mention TIDE and are shared dependencies (selection-plan generators,
  lifecycle host controls); this is an explicit omission, not proof of irrelevance.
- Synthetic execution directories containing only receipts, rather than source.
- Paths outside the scanned workspace, skipped data/result/environment directories,
  symlink aliases as independent copies, and inaccessible legacy areas.

## Known inaccessible or unresolved areas

Legacy ANNS native CPU self-test/stage routes and the persistent-kNN, turnover,
and long-stream source areas had permission failures. They are not represented as
complete. The main TIDE, RT-TIDE and three TIDEpro roots did not show source read
permission failures in this scan. These facts do not establish whole-host coverage.

The separate GTS archive has its own wider-coverage backlog. Publishing this TIDE
archive does not fill or certify that backlog. A clean source archive also does
not establish the scientific validity of historical code or result claims.

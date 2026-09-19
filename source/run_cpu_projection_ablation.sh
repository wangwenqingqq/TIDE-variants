#!/usr/bin/env bash
# Immutable one-shot runner for the nonformal CPU-only v4 projection ablation.
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_multilandmark_certificate_probe_v4_projection
SRC="$ROOT/cpu_projection_ablation.cpp"
BIN="$ROOT/cpu_projection_ablation"
OUT="$ROOT/result.json"
V3=/workspace/experiments/tide_safe_c1_20260727/c2_multilandmark_certificate_probe_v3_aabb/result.json
CMP="$ROOT/v3_full_aabb_comparison.json"
PROV="$ROOT/provenance.json"
[[ -f "$SRC" && ! -L "$SRC" && -f "$V3" && ! -L "$V3" ]] || exit 65
[[ ! -e "$BIN" && ! -L "$BIN" && ! -e "$OUT" && ! -L "$OUT" && ! -e "$CMP" && ! -L "$CMP" && ! -e "$PROV" && ! -L "$PROV" ]] || exit 66
g++ -std=c++17 -O3 -Wall -Wextra -Werror "$SRC" -o "$BIN"
"$BIN" >"$OUT"
python3 - "$OUT" "$V3" "$CMP" <<'PY'
import json, sys
out_path, v3_path, cmp_path = sys.argv[1:]
with open(out_path, "r", encoding="utf-8") as f:
    out = json.load(f)
with open(v3_path, "r", encoding="utf-8") as f:
    v3 = json.load(f)
m128 = next(x for x in out["modes"] if x["m"] == 128)
expected = v3["work"]["radial_plus_full_aabb"]
observed = m128["work"]
keys = ("point_distances", "nodes", "pruned_nodes")
matches = all(observed[k] == expected[k] for k in keys)
record = {
    "schema": "c2-projection-aabb-v3-full-aabb-crosscheck-v1",
    "scope": "crosscheck only; neither result is formal performance evidence",
    "v4_m": 128,
    "expected_v3_full_aabb_work": {k: expected[k] for k in keys},
    "observed_v4_m128_work": {k: observed[k] for k in keys},
    "exact_work_match": matches,
    "v4_m128_certificate_violations": m128["certificate"]["violations"],
    "v4_m128_output_mismatches_vs_bruteforce": m128["exactness"]["output_mismatches_vs_bruteforce"],
}
with open(cmp_path, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write("\n")
if not matches:
    raise SystemExit("m=128 does not reproduce v3 full-AABB work counters")
PY
python3 - "$ROOT" <<'PY'
import datetime, hashlib, json, os, platform, subprocess, sys
root = sys.argv[1]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def item(role, path):
    return {"role": role, "path": path, "bytes": os.path.getsize(path), "sha256": digest(path)}
base="/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs"
queries="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs"
calids="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/final_workload_v2/calibration.ids"
v3="/workspace/experiments/tide_safe_c1_20260727/c2_multilandmark_certificate_probe_v3_aabb/result.json"
gpp=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
record={
  "schema":"c2-projection-aabb-provenance-v1",
  "status":"COMPLETE_CPU_ONLY_NONFORMAL",
  "scope":"CPU-only exploratory feasibility ablation; not heldout and not GTS performance evidence",
  "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "host":platform.node(),
  "platform":platform.platform(),
  "compiler":gpp,
  "commands":{
    "compile":"g++ -std=c++17 -O3 -Wall -Wextra -Werror cpu_projection_ablation.cpp -o cpu_projection_ablation",
    "run":"./cpu_projection_ablation > result.json",
    "crosscheck":"python3 inline JSON crosscheck: v4 m=128 counters == v3 radial_plus_full_aabb counters"
  },
  "execution_constraints":{
    "gpu_executed":False,
    "validation_or_sealed_inputs_accessed":False,
    "query_selection":"first 256 IDs from existing C2 calibration.ids, sorted; no validation/sealed IDs",
    "projection_selection":"computed before query file is read, only from fixed base sample coordinate population variances"
  },
  "inputs":[item("fixed_base_sample_source",base),item("calibration_query_fvec_source",queries),item("calibration_id_list",calids),item("v3_full_aabb_crosscheck_reference",v3)],
  "artifacts":[item("source",os.path.join(root,"cpu_projection_ablation.cpp")),item("runner",os.path.join(root,"run_cpu_projection_ablation.sh")),item("binary",os.path.join(root,"cpu_projection_ablation")),item("result",os.path.join(root,"result.json")),item("v3_crosscheck",os.path.join(root,"v3_full_aabb_comparison.json"))]
}
with open(os.path.join(root,"provenance.json"),"w",encoding="utf-8") as f:
    json.dump(record,f,indent=2,sort_keys=True)
    f.write("\n")
PY
cat "$OUT"
printf '\n=== v3 m=128 crosscheck ===\n'
cat "$CMP"
printf '\n=== provenance ===\n'
cat "$PROV"

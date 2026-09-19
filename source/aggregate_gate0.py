#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORDS = ROOT / "raw/words_main_20260825T084757Z/result.json"
PROTEIN = ROOT / "raw/protein_transfer_20260825T084855Z/result.json"
NATIVE = ROOT / "raw/native_words_main_20260825T085332Z/result.json"
UPDATE = ROOT / "raw/flat_update_words_20260825T085652Z/result.json"
TIDE = ROOT.parent / "TIDE_gate12_20260825/FINAL_VERDICT.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def row(doc, method, batch, radius, pivots=None):
    for x in doc["results"]:
        if x["method"] != method or x["batch"] != batch or x["radius"] != radius:
            continue
        if pivots is not None and x["pivots"] != pivots:
            continue
        return x
    raise KeyError((method, batch, radius, pivots))


def flat_case(doc, radius, batch, pivots=64):
    scan = row(doc, "exact_scan", batch, radius)
    flat = row(doc, "flat_bound_verify", batch, radius, pivots)
    return {
        "radius": radius,
        "batch": batch,
        "pivots": pivots,
        "scan_wall_median_ms": scan["wall_median_ms"],
        "scan_wall_p95_ms": scan["wall_p95_ms"],
        "flat_wall_median_ms": flat["wall_median_ms"],
        "flat_wall_p95_ms": flat["wall_p95_ms"],
        "speedup": scan["wall_median_ms"] / flat["wall_median_ms"],
        "candidate_ratio": flat["candidate_ratio"],
        "exact_call_reduction": 1.0 - flat["candidate_ratio"],
        "result_count_mean": sum(scan["result_counts"]) / len(scan["result_counts"]),
        "oracle_mismatches": scan["result_count_mismatches"] + flat["result_count_mismatches"],
        "worst_repeat_speedup_lower_bound": min(scan["wall_ms"]) / max(flat["wall_ms"]),
    }


def native_case(doc, radius, batch):
    x = next(x for x in doc["results"] if x["radius"] == radius and x["batch"] == batch)
    return {
        "radius": radius,
        "batch": batch,
        "tree_wall_median_ms": x["tree"]["wall_median_ms"],
        "scan_wall_median_ms": x["scan"]["wall_median_ms"],
        "tree_slowdown": x["tree"]["wall_median_ms"] / x["scan"]["wall_median_ms"],
        "tree_wall_p95_ms": x["tree"]["wall_p95_ms"],
        "scan_wall_p95_ms": x["scan"]["wall_p95_ms"],
        "selected_leaf_slots": x["selected_leaf_slots"],
        "result_count_mean": sum(x["result_counts"]) / len(x["result_counts"]),
        "oracle_set_mismatches": x["oracle_set_mismatches"],
    }


def fmt(v, digits=3):
    return f"{v:.{digits}f}"


def main():
    words = load(WORDS)
    protein = load(PROTEIN)
    native = load(NATIVE)
    update = load(UPDATE)
    tide = load(TIDE)

    assert words["center_radius"] == 4 and words["radii"] == [3, 4, 5]
    assert protein["center_radius"] == 23 and protein["radii"] == [22, 23, 24]
    assert words["total_result_count_mismatches"] == 0
    assert protein["total_result_count_mismatches"] == 0
    assert native["total_oracle_set_mismatches"] == 0
    assert update["total_flag_mismatches"] == 0

    word_cases = [flat_case(words, r, b) for r in words["radii"] for b in (1, 32)]
    word_center = {x["batch"]: x for x in word_cases if x["radius"] == 4}
    word_r3 = {x["batch"]: x for x in word_cases if x["radius"] == 3}
    native_cases = [native_case(native, r, b) for r in (3, 4, 5) for b in (1, 32)]
    protein_center = {b: flat_case(protein, 23, b) for b in (1, 32)}

    zero_errors = (
        words["signature_sample_mismatches"] == 0
        and words["total_result_count_mismatches"] == 0
    )
    center_both_2x = all(word_center[b]["speedup"] >= 2.0 for b in (1, 32))
    adjacent_both_1_5x = all(word_r3[b]["speedup"] >= 1.5 for b in (1, 32))
    center_both_80_reduction = all(word_center[b]["exact_call_reduction"] >= 0.8 for b in (1, 32))
    stable_r3 = all(word_r3[b]["worst_repeat_speedup_lower_bound"] >= 1.5 for b in (1, 32))
    primary_pass = all(
        [zero_errors, center_both_2x, adjacent_both_1_5x, center_both_80_reduction, stable_r3]
    )
    transfer_pass = (
        protein["total_result_count_mismatches"] == 0
        and protein_center[32]["speedup"] >= 1.5
    )
    update_pass = (
        update["total_flag_mismatches"] == 0
        and all(x["delta_self_visible"] == 16 for x in update["results"])
    )

    verdict = {
        "schema": "gts-family-gate0-final-verdict-v1",
        "date": "2026-08-25",
        "host": "CONFIGURE_ARCHIVE_HOST",
        "gpu_index": 0,
        "gpu_uuid": "GPU-CONFIGURE-ARCHIVE-DEVICE",
        "do_not_touch": "physical GPUs 1-7 and every foreign process",
        "formal_gate0": "pass" if primary_pass else "fail",
        "classification": "traversal_free_only_conditional",
        "tree_survives": False,
        "native_gts": {
            "status": "fail",
            "words_oracle_set_mismatches": native["total_oracle_set_mismatches"],
            "index_build_ms": native["index_build_ms"],
            "cases": native_cases,
            "inherited_l2_q32_tree_slowdown": tide["tide_g1"]["range900k_q32"],
        },
        "flat_primary_words": {
            "status": "pass" if primary_pass else "fail",
            "center_radius": words["center_radius"],
            "signature_build_ms": words["signature_build_ms"],
            "signature_bytes": words["signature_bytes"],
            "signature_sample_mismatches": words["signature_sample_mismatches"],
            "result_count_mismatches": words["total_result_count_mismatches"],
            "conditions": {
                "zero_errors": zero_errors,
                "center_q1_and_q32_at_least_2x": center_both_2x,
                "adjacent_q1_and_q32_at_least_1_5x": adjacent_both_1_5x,
                "center_q1_and_q32_at_least_80pct_exact_call_reduction": center_both_80_reduction,
                "nine_repeat_stability": stable_r3,
            },
            "cases": word_cases,
        },
        "flat_transfer_protein": {
            "status": "pass" if transfer_pass else "fail",
            "center_radius": protein["center_radius"],
            "signature_build_ms": protein["signature_build_ms"],
            "result_count_mismatches": protein["total_result_count_mismatches"],
            "center_cases": list(protein_center.values()),
        },
        "flat_update_words": {
            "status": "pass" if update_pass else "fail",
            "base": update["base"],
            "delta": update["delta"],
            "pivots": update["pivots"],
            "append_median_ms": update["append_median_ms"],
            "append_objects_per_second": update["append_objects_per_second"],
            "flag_mismatches": update["total_flag_mismatches"],
            "delta_self_visible": update["results"][0]["delta_self_visible"],
            "results": update["results"],
        },
        "decision": {
            "current_tide": "no_go_unchanged",
            "native_gts_tree": "no_go",
            "flat_filter_as_new_paper": "no_go_under_frozen_primary_threshold",
            "flat_filter_as_component": "retain_high_selectivity_exact_range_component",
            "research_boundary": "do_not_promote_without_a_real_application_trace_whose_operating_selectivity_matches_the_r3_window",
        },
        "artifacts": {
            "protocol": str(ROOT / "PROTOCOL.md"),
            "words": str(WORDS),
            "protein": str(PROTEIN),
            "native": str(NATIVE),
            "update": str(UPDATE),
            "inherited_l2": str(TIDE),
        },
    }
    with open(ROOT / "FINAL_VERDICT.json", "w") as f:
        json.dump(verdict, f, indent=2)
        f.write("\n")

    lines = []
    lines += [
        "# GTS-family existential Gate 0 verdict",
        "",
        "Date: 2026-08-25  ",
        "Execution host: `CONFIGURE_ARCHIVE_HOST` only  ",
        "GPU: physical GPU 0, `GPU-CONFIGURE-ARCHIVE-DEVICE`  ",
        "Do-not-touch boundary: physical GPUs 1--7 and every foreign process",
        "",
        "## Executive verdict",
        "",
        "**Formal Gate 0: FAIL under the thresholds frozen in `PROTOCOL.md`.**",
        "",
        "The evidence is not `metric_index_no_go` in every regime. It is more precise:",
        "",
        "- **native GTS tree: NO-GO** on both cheap L2 and expensive edit distance;",
        "- **flat exact lower-bound filtering: a real but selectivity-bounded lead**;",
        "- **new paper promotion: NO-GO at this gate**, because the primary central operating point misses the frozen q1 and exact-call-reduction thresholds;",
        "- keep the flat path only as a component until a real application trace independently establishes that its operating region matches the narrow-radius window.",
        "",
        "All newly reported runs had zero oracle mismatches.",
        "",
        "## 1. Native GTS tree: FAIL",
        "",
        f"The isolated native tree built 611,756 Words in {fmt(native['index_build_ms'])} ms and passed the exact result-set oracle, but lost to its same-process exact scan in every case:",
        "",
        "| Radius | Batch | Native tree | Exact scan | Tree slowdown |",
        "|---:|---:|---:|---:|---:|",
    ]
    for x in native_cases:
        lines.append(
            f"| {x['radius']} | {x['batch']} | {fmt(x['tree_wall_median_ms'])} ms | "
            f"{fmt(x['scan_wall_median_ms'])} ms | **{fmt(x['tree_slowdown'], 2)}x** |"
        )
    lines += [
        "",
        "This agrees with the inherited SIFT-L2 result: at the measured k=10 range, native GTS was already 6.72--20.38x slower than scan. Expensive distance does not rescue the current tree execution path.",
        "",
        "## 2. Traversal-free flat bound filter: conditional window",
        "",
        f"Words used a {words['signature_bytes'] / 1024**2:.2f} MiB, 64-pivot signature matrix built in {fmt(words['signature_build_ms'])} ms. Query-to-pivot distances are included. Results below are wall medians over nine timed repetitions:",
        "",
        "| Radius | Batch | Exact scan | Flat filter+verify | Speedup | Exact-call reduction | Mean results/query |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in word_cases:
        lines.append(
            f"| {x['radius']} | {x['batch']} | {fmt(x['scan_wall_median_ms'])} ms | "
            f"{fmt(x['flat_wall_median_ms'])} ms | **{fmt(x['speedup'], 2)}x** | "
            f"{100*x['exact_call_reduction']:.2f}% | {fmt(x['result_count_mean'], 1)} |"
        )
    lines += [
        "",
        "At `r=3`, the path is convincingly useful: 3.20x q1 and 3.62x q32, with 98.55% and 87.46% fewer exact distance calls. At the frozen central radius `r=4`, q32 reaches 2.09x but q1 reaches only 1.63x; q32 exact-call reduction is only 63.99%. Therefore the formal primary threshold fails. At `r=5`, the q1 gain falls to 1.17x.",
        "",
        "## 3. Protein transfer",
        "",
        f"Protein (52,799 sequences, exact edit distance) had zero mismatches. At center radius 23, the 64-pivot path achieved {fmt(protein_center[1]['speedup'], 2)}x q1 and {fmt(protein_center[32]['speedup'], 2)}x q32. The frozen transfer requirement is q32 >=1.5x, so transfer support passes, but the q1 boundary remains.",
        "",
        "## 4. Append and immediate visibility",
        "",
        f"With Base=550,000 and Delta={update['delta']:,}, preallocated 64-pivot signatures appended the Delta (including host-to-device row copy and signature computation) in a median {fmt(update['append_median_ms'])} ms ({update['append_objects_per_second']/1e6:.2f} M objects/s). All 16 Delta-held queries found their own newly appended row at radius zero, and the full filter/scan flag matrices had zero mismatches.",
        "",
        "| Radius | Exact scan q32 | Flat q32 | Speedup | Flag mismatches |",
        "|---:|---:|---:|---:|---:|",
    ]
    for x in update["results"]:
        lines.append(
            f"| {x['radius']} | {fmt(x['scan_median_ms'])} ms | {fmt(x['filter_median_ms'])} ms | "
            f"{fmt(x['speedup'], 2)}x | {x['flag_mismatches']} |"
        )
    lines += [
        "",
        "The update number assumes preallocated signature capacity; allocation and capacity growth are not measured. It is evidence for append-friendly mechanics, not a complete dynamic system result.",
        "",
        "## Decision",
        "",
        "1. Do not revive TIDE around the GTS tree.",
        "2. Do not promote the flat pivot table as a new paper: the primitive is prior art and the frozen central threshold failed.",
        "3. Preserve the flat filter as a reusable exact high-selectivity component and the Gate harness as infrastructure.",
        "4. Further work is justified only if an independently chosen real application trace has a stable operating selectivity matching the `r=3` window. The application must be chosen before additional optimization; otherwise this line stops here.",
        "",
        "## Boundaries",
        "",
        "- Queries are deterministic in-dataset IDs, not a production trace.",
        "- The new exact scan uses a packed, two-row Levenshtein kernel with zero compiler spills, but it is not claimed to be the globally fastest possible GPU edit-distance implementation.",
        "- Native GTS and the flat probe use different internal layouts; each speedup is therefore computed only against its own same-process exact scan.",
        "- The flat probe answers exact range queries with a supplied radius. It is not yet an exact kNN implementation.",
        "- Pivot tables/LAESA are prior art; the measurements establish headroom, not novelty.",
        "",
        "## Artifacts",
        "",
        f"- Protocol: `{ROOT / 'PROTOCOL.md'}`",
        f"- Machine verdict: `{ROOT / 'FINAL_VERDICT.json'}`",
        f"- Words raw result: `{WORDS}`",
        f"- Protein raw result: `{PROTEIN}`",
        f"- Native GTS raw result: `{NATIVE}`",
        f"- Update raw result: `{UPDATE}`",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")

    files = [p for p in ROOT.rglob("*") if p.is_file()]
    files = [p for p in files if p.name != "ARTIFACT_MANIFEST.sha256"]
    with open(ROOT / "ARTIFACT_MANIFEST.sha256", "w") as f:
        for path in sorted(files):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            f.write(f"{digest}  {path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()


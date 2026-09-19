#!/usr/bin/env python3
"""Create a bounded nonsecret synthetic E1 fixture with exclusive writes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

SCOPE = "BOUNDED_STANDALONE_256x16_GTSQ_FIXTURE_ONLY_NOT_LEGACY_OR_NATIVE_GTS"
TRIAL_ID = "8a7b6c5d4e3f2910fedcba9876543210"
PROTOCOL_MANIFEST = "fda2465377cd945807f924b9236b91fddcf7ff00f0c2732d7ec12720a36989b5"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as f:
        f.write(canonical(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("xb") as f:
        for row in rows:
            f.write(canonical(row))


def write_text(path: Path, data: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)


def vector(*values: int) -> List[int]:
    assert len(values) <= 16
    return list(values) + [0] * (16 - len(values))


def object_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # 252 initial base rows + 4 arrival-only rows = exactly 256 objects, 16D.
    for i in range(252):
        if i == 0:
            v = vector(*([100] * 16))
        elif i == 1:
            v = vector(10)
        else:
            v = vector(100 + i, (17 * i) % 97, (31 * i) % 89, (7 * i) % 83)
        rows.append({"record_type": "object", "stable_id": f"base-{i:03d}",
                     "vector": v, "initial_membership": "base"})
    rows.extend([
        {"record_type": "object", "stable_id": "arr-direct-near", "vector": vector(0), "initial_membership": "arrival"},
        {"record_type": "object", "stable_id": "arr-direct-far", "vector": vector(*([1000] * 16)), "initial_membership": "arrival"},
        {"record_type": "object", "stable_id": "arr-fallback", "vector": vector(1), "initial_membership": "arrival"},
        {"record_type": "object", "stable_id": "arr-reject", "vector": vector(2), "initial_membership": "arrival"},
    ])
    assert len(rows) == 256
    return rows


def catalog_rows() -> List[Dict[str, Any]]:
    rows = object_rows()
    rows.extend([
        {"record_type": "query", "query_id": "q-witness", "vector": vector(0), "partition": "heldout"},
        {"record_type": "query", "query_id": "q-overlay", "vector": vector(0), "partition": "heldout"},
        {"record_type": "query", "query_id": "q-post-rebuild", "vector": vector(0), "partition": "heldout"},
    ])
    return rows


def trace_rows(catalog_sha: str) -> List[Dict[str, Any]]:
    header = {
        "record_type": "trace_header", "schema": "tide.lifecycle-trace.v1",
        "trace_id": TRIAL_ID, "metric_contract": "integer_l2_squared_v1", "k": 2,
        "tie_order": ["distance_key", "stable_id"], "sealed": True,
        "required_cases": ["direct_admission", "sidecar_capacity_fallback", "certificate_rejection", "overlay_delete", "base_delete_rebuild_barrier", "first_query_after_rebuild", "nontrivial_receipt_witness"],
        "catalog_sha256": catalog_sha,
        "trace_generation_commitment": "public-synthetic-fixture-v1-no-private-control-material",
    }
    return [header,
        {"seq": 1, "op": "insert", "stable_id": "arr-direct-near", "expected_insert_outcome": "direct", "required_cases": ["direct_admission"]},
        {"seq": 2, "op": "insert", "stable_id": "arr-direct-far", "expected_insert_outcome": "direct", "required_cases": ["direct_admission"]},
        {"seq": 3, "op": "query", "query_id": "q-witness", "required_cases": ["nontrivial_receipt_witness"], "witness_id": "witness-prune-subset"},
        {"seq": 4, "op": "insert", "stable_id": "arr-fallback", "expected_insert_outcome": "capacity_fallback", "required_cases": ["sidecar_capacity_fallback"]},
        {"seq": 5, "op": "insert", "stable_id": "arr-reject", "expected_insert_outcome": "certificate_reject", "required_cases": ["certificate_rejection"]},
        {"seq": 6, "op": "query", "query_id": "q-overlay", "required_cases": []},
        {"seq": 7, "op": "delete", "stable_id": "arr-fallback", "required_cases": ["overlay_delete"]},
        {"seq": 8, "op": "delete", "stable_id": "base-000", "required_cases": ["base_delete_rebuild_barrier"]},
        {"seq": 9, "op": "query", "query_id": "q-post-rebuild", "required_cases": ["first_query_after_rebuild"]},
    ]


def source_snippets(root: Path) -> Dict[str, str]:
    source = root / "source"
    source.mkdir(mode=0o700)
    self_data = Path(__file__).read_text(encoding="utf-8")
    write_text(source / "synthetic_e1_fixture_builder.py", self_data)
    write_text(source / "base_oracle_identity.py", "# Synthetic fixture identity only; no legacy/native GTS claim.\nALGORITHM='independent_base_integer_l2_reference_v1'\n")
    write_text(source / "full_oracle_identity.py", "# Full oracle is frozen protocol oracle_replayer.py; this identity binds fixture wiring.\nALGORITHM='independent_full_live_integer_l2_reference_v1'\n")
    write_text(source / "mirror_oracle_identity.py", "# Synthetic traversal mirror identity only; no legacy/native GTS claim.\nALGORITHM='independent_receipt_mirror_fixture_v1'\n")
    return {p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()}


def prepare(root: Path) -> None:
    # Atomic directory creation is the one-shot root reservation.
    os.mkdir(root, 0o700)
    if root.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("root mode is not 0700")
    for name in ("input", "output", "oracles", "logs", "source"):
        # source is populated below; all child directories are private.
        if name != "source":
            (root / name).mkdir(mode=0o700)
    write_text(root / "SCOPE.md", "# Scope\n\n" + SCOPE + "\n\nThis evidence is synthetic protocol wiring only. It is not a GTS result, GPU result, FVEC result, performance result, or full paper Gate 1.\n")
    source_hashes = source_snippets(root)
    catalog = root / "input" / "raw_catalog.jsonl"
    write_jsonl(catalog, catalog_rows())
    trace = root / "input" / "sealed_trace.jsonl"
    write_jsonl(trace, trace_rows(sha(catalog)))
    stable_map = {"schema": "gtsq.synthetic-stable-map.v1", "scope": SCOPE,
                  "stable_to_local": [{"stable_id": row["stable_id"], "local_id": i}
                                      for i, row in enumerate(object_rows())]}
    write_json(root / "input" / "stable_id_map.json", stable_map)
    capacity = {"schema": "gtsq.synthetic-capacity-config.v1", "scope": SCOPE,
                "sidecar_cap_L": 1, "common_overlay_budget_C_ov": 4,
                "fault_required_capacity": 1, "fault_injected_capacity": 0,
                "gpu": "NOT_USED", "fvec": "NOT_USED"}
    write_json(root / "input" / "capacity_config.json", capacity)
    envelopes = {"schema": "gtsq.synthetic-physical-envelope-inventory.v1", "scope": SCOPE,
                 "physical_children": [{"node": "synthetic-root", "child": "synthetic-leaf-a", "upper": "10"},
                                       {"node": "synthetic-root", "child": "synthetic-leaf-b", "upper": "1000"}]}
    write_json(root / "input" / "physical_envelope_inventory.json", envelopes)
    source_manifest = {"schema": "gtsq.synthetic-source-manifest.v1", "scope": SCOPE,
                       "builder_files": source_hashes,
                       "protocol_package_manifest_sha256": PROTOCOL_MANIFEST,
                       "qualification_fixture_contract": "SOURCE_ONLY_PLAN_REFERENCED_NOT_EXECUTED"}
    write_json(root / "input" / "source_manifest.json", source_manifest)
    source_hash = source_hashes["synthetic_e1_fixture_builder.py"]
    base_hash = source_hashes["base_oracle_identity.py"]
    full_hash = source_hashes["full_oracle_identity.py"]
    mirror_hash = source_hashes["mirror_oracle_identity.py"]
    for filename, content in (("base_oracle_pin.json", {"id":"synthetic-base-oracle-v1","implementation_sha256":base_hash,"scope":SCOPE}),
                              ("full_oracle_pin.json", {"id":"synthetic-full-oracle-v1","implementation_sha256":full_hash,"scope":SCOPE}),
                              ("mirror_oracle_pin.json", {"id":"synthetic-mirror-oracle-v1","implementation_sha256":mirror_hash,"scope":SCOPE})):
        write_json(root / "oracles" / filename, content)
    artifacts = {"tree_checksum": "synthetic-tree-epoch0",
                 "receipt_checksum": "synthetic-receipt-epoch0",
                 "physical_envelope_inventory_sha256": sha(root / "input" / "physical_envelope_inventory.json"),
                 "base_oracle_impl_sha256": base_hash,
                 "full_oracle_impl_sha256": full_hash,
                 "traversal_mirror_impl_sha256": mirror_hash}
    gate1 = {"schema": "tide.gate1-qualification.v1", "status": "QUALIFIED",
             "qualification_scope": SCOPE,
             "nonclaim": "QUALIFIED labels only this bounded standalone synthetic 256x16 GTSQ fixture; it does not qualify legacy/native GTS or complete full-paper Gate 1.",
             "host_source_sha256": source_hash, "host_binary_sha256": source_hash,
             "adapter_source_sha256": source_hash, "metric_contract": "integer_l2_squared_v1",
             "stable_id_contract": {"persistent_across_rebuild": True}, "artifacts": artifacts,
             "native_properties": {k: True for k in ("deterministic_ties","finite_physical_child_upper_envelopes","pruning_and_expansion_closure","conservative_cutoffs","ordered_leaf_receipt","qualified_rebuild_publication")},
             "wrapper_obligations": {k: True for k in ("exact_single_representation","receipt_indexed_sidecar_scan","full_live_delta_scan","stable_id_dedup","unified_tie_order","fail_stop_on_uncertainty")}}
    write_json(root / "input" / "gate1_qualification.json", gate1)
    contract = {"schema": "gtsq.synthetic-fixture-artifact-contract.v1", "scope": SCOPE,
      "source": {"git_commit":"SYNTHETIC_STANDALONE_NO_GIT","git_tree":"SYNTHETIC_STANDALONE_NO_GIT"},
      "source_manifest_sha256": sha(root / "input" / "source_manifest.json"),
      "state_fingerprint_sha256": sha(catalog), "stable_id_map_sha256": sha(root / "input" / "stable_id_map.json"),
      "capacity_config_sha256": sha(root / "input" / "capacity_config.json"),
      "adapter": {"header_sha256": source_hash,"source_sha256": source_hash},
      "artifacts": {"tree_checksum64": hashlib.sha256(b"synthetic-tree-epoch0").hexdigest()[:16],
                    "expanded_leaf_receipt_checksum64": hashlib.sha256(b"synthetic-receipt-epoch0").hexdigest()[:16],
                    "canonical_receipt_sha256": hashlib.sha256(b"synthetic-receipt-fixture").hexdigest()},
      "oracles": {"base":{"id":"synthetic-base-oracle-v1","implementation_sha256":base_hash},"full":{"id":"synthetic-full-oracle-v1","implementation_sha256":full_hash},"mirror":{"id":"synthetic-mirror-oracle-v1","implementation_sha256":mirror_hash}},
      "nonclaim": "Contract-shaped synthetic fixture record only; no legacy/native GTS source, capture, CUDA, or FVEC was executed."}
    write_json(root / "input" / "fixture_artifact_contract.json", contract)
    write_json(root / "PREPARE_STATUS.json", {"status":"PREPARED_EXCLUSIVE_SYNTHETIC_INPUTS_ONLY","scope":SCOPE,"trial_id":TRIAL_ID,"objects":256,"dimensions":16,"no_gpu":True,"no_fvec":True})


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def distance(catalog: Mapping[str, Any], sid: str, qid: str) -> str:
    query = catalog["queries"][qid]
    vec = catalog["objects"][sid]
    return str(sum((a-b)*(a-b) for a,b in zip(query,vec)))


def events(root: Path) -> None:
    trace = load_jsonl(root / "input" / "sealed_trace.jsonl")[1:]
    replay_rows = load_jsonl(root / "output" / "oracle_replay.jsonl")
    if len(trace) != len(replay_rows): raise RuntimeError("oracle replay length mismatch")
    catalog_rows_loaded = load_jsonl(root / "input" / "raw_catalog.jsonl")
    catalog = {"objects": {x["stable_id"]:x["vector"] for x in catalog_rows_loaded if x["record_type"]=="object"},
               "queries": {x["query_id"]:x["vector"] for x in catalog_rows_loaded if x["record_type"]=="query"}}
    gate1 = json.loads((root / "input" / "gate1_qualification.json").read_text(encoding="utf-8"))
    hs, hb = gate1["host_source_sha256"], gate1["host_binary_sha256"]
    tree = "synthetic-tree-epoch0"; sidecars: Dict[str,str] = {}; delta=set(); out=[]
    for op, oracle in zip(trace, replay_rows):
        seq=op["seq"]; base={"schema":"gtsq.synthetic-runner-event.v1","scope":SCOPE,"seq":seq,"op":op["op"],"host_source_sha256":hs,"host_binary_sha256":hb,"pre_active_set_digest":oracle["pre_active_set_digest"],"post_active_set_digest":oracle["post_active_set_digest"]}
        if op["op"]=="insert":
            sid=op["stable_id"]
            if op["expected_insert_outcome"]=="direct":
                leaf="synthetic-leaf-a" if sid=="arr-direct-near" else "synthetic-leaf-b"; sidecars[sid]=leaf
                base.update({"routing":"sidecar","sidecar_leaf_id":leaf,"route_reason":"strict_certificate_and_capacity"})
            elif op["expected_insert_outcome"]=="capacity_fallback":
                delta.add(sid); base.update({"routing":"delta","route_reason":"sidecar_capacity_fallback"})
            else:
                delta.add(sid); base.update({"routing":"delta","route_reason":"certificate_rejection"})
        elif op["op"]=="delete":
            sid=op["stable_id"]
            if sid=="base-000":
                tree="synthetic-tree-epoch1"; sidecars.clear(); delta.clear()
                base.update({"routing":"base_delete_rebuild_barrier","rebuild":{"status":"QUALIFIED_PUBLISHED","new_tree_checksum":tree,"stable_to_local_map_sha256":sha(root / "input" / "stable_id_map.json"),"pre_state_sha256":"synthetic-pre-state","post_state_sha256":"synthetic-post-state"}})
            else:
                sidecars.pop(sid,None); delta.discard(sid); base.update({"routing":"overlay_delete"})
        else:
            receipt=["synthetic-leaf-a"] if op["query_id"] in {"q-witness","q-overlay"} else ["synthetic-base-leaf"]
            scanned_side=sorted(sid for sid,leaf in sidecars.items() if leaf in receipt); scanned_delta=sorted(delta)
            contrib=[]
            for sid in scanned_side: contrib.append({"stable_id":sid,"source":"sidecar","distance_key":distance(catalog,sid,op["query_id"])})
            for sid in scanned_delta: contrib.append({"stable_id":sid,"source":"delta","distance_key":distance(catalog,sid,op["query_id"])})
            base.update({"receipt_leaf_ids":receipt,"scanned_sidecar_ids":scanned_side,"scanned_delta_ids":scanned_delta,"scan_multiplicity":{sid:1 for sid in scanned_side+scanned_delta},"overlay_contributions":contrib,"final_answer":oracle["answer"]})
            if op["query_id"]=="q-witness": base["nontrivial_receipt_witness"]={"actual_prune":True,"receipt_is_strict_leaf_subset":True,"visible_direct_stable_id":"arr-direct-near","unreceived_direct_leaf_id":"synthetic-leaf-b","unreceived_direct_objects_outside_oracle_top_k":True}
        base["tree_checksum"]=tree; out.append(base)
    write_jsonl(root / "output" / "runner_events.jsonl", out)
    fault=[]
    for stage in ("traversal","receipt","receipt_transfer","sidecar_enumeration","delta_enumeration","candidate_transfer","distance_key","final_merge"):
        fault.append({"schema":"gtsq.synthetic-fault-record.v1","scope":SCOPE,"stage":stage,"required_cap":1,"injected_cap":0,"status":"NO_ANSWER","final_answer":None,"overlay_reads":0,"merged":False})
    write_jsonl(root / "output" / "faults.jsonl", fault)
    cap=[]
    byseq={x["seq"]:x for x in replay_rows}
    for seq,sid,leaf in ((1,"arr-direct-near","synthetic-leaf-a"),(2,"arr-direct-far","synthetic-leaf-b")):
        r=byseq[seq]; cap.append({"schema":"gtsq.synthetic-replay-capsule.v1","scope":SCOPE,"seq":seq,"kind":"direct_admission","stable_id":sid,"certificate_path":["synthetic-root",leaf],"pre_active_set_digest":r["pre_active_set_digest"],"post_active_set_digest":r["post_active_set_digest"],"tree_checksum":"synthetic-tree-epoch0"})
    r=byseq[8]; cap.append({"schema":"gtsq.synthetic-replay-capsule.v1","scope":SCOPE,"seq":8,"kind":"rebuild","pre_active_set_digest":r["pre_active_set_digest"],"post_active_set_digest":r["post_active_set_digest"],"pre_tree_checksum":"synthetic-tree-epoch0","post_tree_checksum":"synthetic-tree-epoch1","stable_to_local_map_sha256":sha(root / "input" / "stable_id_map.json"),"raw_operation_suffix_sha256":sha(root / "input" / "sealed_trace.jsonl")})
    write_jsonl(root / "output" / "replay_capsules.jsonl", cap)
    metadata={"schema":"gtsq.synthetic-run-metadata.v1","status":"COMPLETED","scope":SCOPE,"trial_id":TRIAL_ID,"host_source_sha256":hs,"host_binary_sha256":hb,"metric_contract":"integer_l2_squared_v1","objects":256,"dimensions":16,"gpu":"NOT_USED","fvec":"NOT_USED","command_line":"explicit synthetic fixture builder and frozen protocol validators only","protocol_package_manifest_sha256":PROTOCOL_MANIFEST,"no_performance_claim":True}
    write_json(root / "output" / "run_metadata.json", metadata)
    bundle={"schema":"tide.e1-bundle.v1","template_only":False,"status":"COMPLETED","trace_id":TRIAL_ID,"scope":SCOPE,"inputs":{"catalog":"input/raw_catalog.jsonl","trace":"input/sealed_trace.jsonl","gate1_qualification":"input/gate1_qualification.json"},"outputs":{"runner_events":"output/runner_events.jsonl","oracle_replay":"output/oracle_replay.jsonl","faults":"output/faults.jsonl","replay_capsules":"output/replay_capsules.jsonl","run_metadata":"output/run_metadata.json"}}
    write_json(root / "E1_BUNDLE.json", bundle)
    write_json(root / "EVENTS_STATUS.json", {"status":"SYNTHETIC_EVENTS_READY_FOR_E1_VALIDATION","scope":SCOPE,"trial_id":TRIAL_ID})


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True,type=Path); ap.add_argument("--phase",required=True,choices=("prepare","events")); args=ap.parse_args()
    if not args.root.is_absolute(): raise SystemExit("root must be absolute")
    if args.phase == "prepare": prepare(args.root)
    else: events(args.root)
    print(json.dumps({"status":"OK","phase":args.phase,"root":str(args.root),"scope":SCOPE,"trial_id":TRIAL_ID},sort_keys=True,separators=(",",":")))
    return 0
if __name__=="__main__": sys.exit(main())

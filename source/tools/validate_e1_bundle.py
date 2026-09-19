#!/usr/bin/env python3
"""Fail-closed validator for a completed (not template) TIDE E1 bundle."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set
from tide_protocol_lib import (FAULT_STAGES, ProtocolError, active_set_digest,
    ensure_no_private_fields, load_and_validate_trace, load_json, load_jsonl,
    print_result, require, require_hex32, require_identifier, require_list,
    require_mapping, require_nonnegative_int, require_relative_path,
    require_sha256, require_string, validate_distance_pairs)

NATIVE = ("deterministic_ties", "finite_physical_child_upper_envelopes",
          "pruning_and_expansion_closure", "conservative_cutoffs",
          "ordered_leaf_receipt", "qualified_rebuild_publication")
WRAPPER = ("exact_single_representation", "receipt_indexed_sidecar_scan",
           "full_live_delta_scan", "stable_id_dedup", "unified_tie_order",
           "fail_stop_on_uncertainty")

def rel(root: Path, value: Any, where: str) -> Path:
    path = root / require_relative_path(value, where)
    require(path.is_file() and not path.is_symlink(), f"{where}: missing/unsafe regular file")
    return path

def gate1(report: Mapping[str, Any]) -> Mapping[str, Any]:
    ensure_no_private_fields(report, "gate1")
    require(report.get("schema") == "tide.gate1-qualification.v1", "gate1 schema")
    require(report.get("status") == "QUALIFIED", "gate1 status is not QUALIFIED")
    require(report.get("metric_contract") == "integer_l2_squared_v1", "gate1 metric contract")
    require(report.get("stable_id_contract", {}).get("persistent_across_rebuild") is True,
            "gate1 lacks persistent stable IDs")
    for key in ("host_source_sha256", "host_binary_sha256", "adapter_source_sha256"):
        require_sha256(report.get(key), f"gate1.{key}")
    artifacts = require_mapping(report.get("artifacts"), "gate1.artifacts")
    for key in ("tree_checksum", "receipt_checksum"):
        require_string(artifacts.get(key), f"gate1.artifacts.{key}")
    impl = []
    for key in ("physical_envelope_inventory_sha256", "base_oracle_impl_sha256",
                "full_oracle_impl_sha256", "traversal_mirror_impl_sha256"):
        value = require_sha256(artifacts.get(key), f"gate1.artifacts.{key}")
        if key.endswith("impl_sha256"): impl.append(value)
    require(len(set(impl)) == 3, "gate1 oracle implementation hashes collide")
    for section, keys in (("native_properties", NATIVE), ("wrapper_obligations", WRAPPER)):
        values = require_mapping(report.get(section), f"gate1.{section}")
        for key in keys: require(values.get(key) is True, f"gate1.{section}.{key} is not true")
    return report

def ids(value: Any, where: str) -> List[str]:
    out = [require_identifier(x, f"{where}[{i}]") for i,x in enumerate(require_list(value, where))]
    require(len(out) == len(set(out)), f"{where}: duplicate ID")
    require(out == sorted(out), f"{where}: IDs must be lexically sorted")
    return out

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--bundle",required=True,type=Path); args=ap.parse_args()
    try:
        desc=load_json(args.bundle); ensure_no_private_fields(desc,"bundle")
        require(desc.get("schema")=="tide.e1-bundle.v1", "bundle schema")
        require(desc.get("template_only") is not True, "template cannot be validated as E1 evidence")
        require(desc.get("status")=="COMPLETED", "bundle status must be COMPLETED")
        root=args.bundle.parent; inputs=require_mapping(desc.get("inputs"),"bundle.inputs"); outputs=require_mapping(desc.get("outputs"),"bundle.outputs")
        catalog_path=rel(root, inputs.get("catalog"),"inputs.catalog"); trace_path=rel(root,inputs.get("trace"),"inputs.trace")
        report=gate1(load_json(rel(root,inputs.get("gate1_qualification"),"inputs.gate1_qualification")))
        catalog, trace=load_and_validate_trace(trace_path,catalog_path)
        require(desc.get("trace_id")==trace["trace_id"],"bundle trace_id mismatch")
        events=load_jsonl(rel(root,outputs.get("runner_events"),"outputs.runner_events")); replay=load_jsonl(rel(root,outputs.get("oracle_replay"),"outputs.oracle_replay"))
        faults=load_jsonl(rel(root,outputs.get("faults"),"outputs.faults")); capsules=load_jsonl(rel(root,outputs.get("replay_capsules"),"outputs.replay_capsules"))
        metadata=load_json(rel(root,outputs.get("run_metadata"),"outputs.run_metadata")); ensure_no_private_fields(metadata,"metadata")
        require(metadata.get("status")=="COMPLETED", "run metadata status")
        for key in ("host_source_sha256","host_binary_sha256"):
            require(metadata.get(key)==report.get(key), f"metadata {key} mismatch")
        require(len(events)==len(trace["operations"])==len(replay), "event/replay/trace length mismatch")
        base={sid for sid,row in catalog["objects"].items() if row["initial_membership"]=="base"}; live=set(base); sidecars: Dict[str,str]={}; delta:Set[str]=set(); current_tree=report["artifacts"]["tree_checksum"]
        capsule_needed=[]; seen_first_rebuild_queries=set()
        for op,event,oracle in zip(trace["operations"],events,replay):
            seq=op["seq"]; require(event.get("seq")==seq and oracle.get("seq")==seq,"seq mismatch")
            require(event.get("op")==op["op"] and oracle.get("op")==op["op"],"op mismatch")
            ensure_no_private_fields(event,f"event[{seq}]")
            require(event.get("host_source_sha256")==report["host_source_sha256"],f"event {seq} source pin")
            require(event.get("host_binary_sha256")==report["host_binary_sha256"],f"event {seq} binary pin")
            require(event.get("pre_active_set_digest")==oracle.get("pre_active_set_digest")==active_set_digest(live),f"event {seq} pre digest")
            rebuild=False
            if op["op"]=="insert":
                sid=op["stable_id"]; routing=require_string(event.get("routing"),f"event {seq}.routing")
                expected=op["expected_insert_outcome"]
                require((expected=="direct" and routing=="sidecar") or (expected!="direct" and routing=="delta"),f"event {seq} routing inconsistent with sealed case")
                live.add(sid)
                if routing=="sidecar":
                    leaf=require_identifier(event.get("sidecar_leaf_id"),f"event {seq}.sidecar_leaf_id"); sidecars[sid]=leaf; capsule_needed.append((seq,"direct_admission"))
                else: delta.add(sid)
                rebuild=bool(op.get("post_op_rebuild",False))
            elif op["op"]=="delete":
                sid=op["stable_id"]; is_base=sid in base; live.remove(sid)
                if is_base:
                    rebuild=True; capsule_needed.append((seq,"rebuild"))
                else:
                    require(event.get("routing")=="overlay_delete",f"event {seq}: overlay delete routing")
                    sidecars.pop(sid,None); delta.discard(sid)
            else:
                receipt=ids(event.get("receipt_leaf_ids"),f"event {seq}.receipt_leaf_ids")
                expected_side=sorted(sid for sid,leaf in sidecars.items() if leaf in set(receipt)); expected_delta=sorted(delta)
                require(ids(event.get("scanned_sidecar_ids"),f"event {seq}.scanned_sidecar_ids")==expected_side,f"event {seq}: sidecar scan incomplete/extra")
                require(ids(event.get("scanned_delta_ids"),f"event {seq}.scanned_delta_ids")==expected_delta,f"event {seq}: delta scan incomplete/extra")
                mult=require_mapping(event.get("scan_multiplicity"),f"event {seq}.scan_multiplicity")
                for sid in expected_side+expected_delta: require(mult.get(sid)==1,f"event {seq}: multiplicity for {sid}")
                contrib=require_list(event.get("overlay_contributions"),f"event {seq}.overlay_contributions")
                got=[]
                for i,row in enumerate(contrib):
                    row=require_mapping(row,f"event {seq}.overlay_contributions[{i}]"); got.append(require_identifier(row.get("stable_id"),f"contrib {i}")); require(row.get("source") in {"sidecar","delta"},f"contrib {i} source"); require_string(row.get("distance_key"),f"contrib {i}.distance_key")
                require(sorted(got)==sorted(expected_side+expected_delta) and len(got)==len(set(got)),f"event {seq}: overlay contribution multiplicity")
                expected_answer=validate_distance_pairs(oracle.get("answer"),f"oracle {seq}.answer")
                actual_answer=validate_distance_pairs(event.get("final_answer"),f"event {seq}.final_answer")
                require(actual_answer==expected_answer,f"event {seq}: full-live oracle mismatch")
                if "first_query_after_rebuild" in op["required_cases"]: seen_first_rebuild_queries.add(seq)
            if rebuild:
                rb=require_mapping(event.get("rebuild"),f"event {seq}.rebuild"); require(rb.get("status")=="QUALIFIED_PUBLISHED",f"event {seq}: rebuild not published")
                current_tree=require_string(rb.get("new_tree_checksum"),f"event {seq}.new_tree_checksum"); sidecars.clear(); delta.clear(); base=set(live)
            require(event.get("tree_checksum")==current_tree,f"event {seq}: immutable tree checksum mismatch")
            require(event.get("post_active_set_digest")==oracle.get("post_active_set_digest")==active_set_digest(live),f"event {seq} post digest")
        require(seen_first_rebuild_queries, "no first post-rebuild query observed")
        observed={(row.get("seq"),row.get("kind")) for row in capsules}
        for item in capsule_needed: require(item in observed,f"missing replay capsule {item}")
        seen_faults=set()
        for row in faults:
            stage=require_string(row.get("stage"),"fault.stage"); require(stage in FAULT_STAGES,"fault unknown stage"); seen_faults.add(stage)
            need=require_nonnegative_int(row.get("required_cap"),"fault.required_cap",positive=True); injected=require_nonnegative_int(row.get("injected_cap"),"fault.injected_cap")
            require(injected<need and row.get("status")=="NO_ANSWER", "fault did not fail-stop")
            require(row.get("final_answer") in (None,[]),"fault leaked answer")
            if stage in {"traversal","receipt","receipt_transfer"}: require(row.get("overlay_reads",0)==0 and row.get("merged") is False,"early fault touched overlay/merge")
        require(seen_faults==FAULT_STAGES,"missing required fault stages")
        print_result({"status":"PASS_E1_EVIDENCE_STRUCTURE_AND_ORACLE_CROSSCHECK","trace_id":trace["trace_id"],"operations":len(events),"nonclaim":"Schema/pin cross-check only; review source and execution provenance separately."})
        return 0
    except ProtocolError as error:
        print_result({"status":"FAIL_CLOSED","error":str(error)}); return 2
if __name__=="__main__": sys.exit(main())

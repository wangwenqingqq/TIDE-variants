#!/usr/bin/env python3
"""Validate a pre-execution matched E2/E3 campaign plan; it never runs a job."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from tide_protocol_lib import (ProtocolError, ensure_no_private_fields, load_json, print_result,
    require, require_hex32, require_list, require_mapping, require_nonnegative_int, require_sha256, require_string)

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--plan",required=True,type=Path); args=ap.parse_args()
    try:
        plan=load_json(args.plan); ensure_no_private_fields(plan,"plan")
        require(plan.get("schema")=="tide.e2e3-campaign-plan.v1","plan schema")
        require(plan.get("template_only") is not True,"template is not executable plan")
        require_hex32(plan.get("campaign_id"),"campaign_id")
        require(plan.get("metric_contract")=="integer_l2_squared_v1","metric contract")
        require_nonnegative_int(plan.get("k"),"k",positive=True)
        e2=require_mapping(plan.get("primary_e2"),"primary_e2")
        require(e2.get("variants")==["certified_sidecar","exact_delta"],"E2 variants/order")
        for key in ("trace_id",): require_hex32(e2.get(key),f"e2.{key}")
        for key in ("base_state_sha256","catalog_sha256","full_oracle_impl_sha256"): require_sha256(e2.get(key),f"e2.{key}")
        require(e2.get("tie_order")==["distance_key","stable_id"],"E2 tie order")
        require(e2.get("base_delete_barrier")=="synchronous","E2 base delete barrier")
        accounting=require_mapping(e2.get("overlay_accounting"),"E2 overlay_accounting")
        require(accounting.get("expression")=="live_sidecars + live_delta","E2 accounting expression")
        require(accounting.get("threshold_name")=="C_ov","E2 threshold name")
        require_nonnegative_int(accounting.get("C_ov"),"E2 C_ov",positive=True)
        require_nonnegative_int(e2.get("sidecar_cap_L"),"E2 L",positive=True)
        require("delta_only_B" not in e2,"E2 primary cannot carry delta_only_B")
        reps=require_list(e2.get("repetitions"),"E2 repetitions"); require(len(reps)>=5,"E2 requires >=5 repetitions")
        starts=[]
        for i,row in enumerate(reps):
            row=require_mapping(row,f"E2 repetitions[{i}]"); require(row.get("replication")==i+1,f"E2 replication index {i}")
            order=require_list(row.get("order"),f"E2 repetitions[{i}].order"); require(set(order)=={"certified_sidecar","exact_delta"} and len(order)==2,f"E2 order {i}"); starts.append(order[0])
            for run_id in require_list(row.get("run_ids"),f"E2 repetitions[{i}].run_ids"): require_hex32(run_id,f"E2 run_id")
        require(abs(starts.count("certified_sidecar")-starts.count("exact_delta"))<=1,"E2 start order must be balanced")
        e3=require_list(plan.get("e3_scale_points"),"e3_scale_points"); counts=[]
        for i,row in enumerate(e3):
            row=require_mapping(row,f"E3[{i}]"); n=require_nonnegative_int(row.get("objects"),f"E3[{i}].objects",positive=True); counts.append(n)
            require_string(row.get("trace_id"),f"E3[{i}].trace_id"); require(0<=row.get("insert_ratio",-1)<=1 and 0<=row.get("delete_ratio",-1)<=1,f"E3[{i}] ratios")
            require_nonnegative_int(row.get("query_count"),f"E3[{i}].query_count",positive=True); require(row.get("oracle_coverage")=="every_query","E3 oracle coverage")
            require_list(row.get("turnover_snapshots"),f"E3[{i}].turnover_snapshots")
        require(any(n>=100_000 for n in counts) and any(n>=1_000_000 for n in counts),"E3 needs >=100K and >=1M points")
        print_result({"status":"PASS_PLAN_ONLY","campaign_id":plan["campaign_id"],"e2_repetitions":len(reps),"e3_points":len(e3),"nonclaim":"No timing/data/GPU execution occurred."}); return 0
    except ProtocolError as error: print_result({"status":"FAIL_CLOSED","error":str(error)}); return 2
if __name__=="__main__": sys.exit(main())

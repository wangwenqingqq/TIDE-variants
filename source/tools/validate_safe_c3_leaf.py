#!/usr/bin/env python3
import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

CANDIDATE = 4102
ROUTE = [0, 5, 60, 604]
SIZE_NODES = {0, 5, 60, 604}
MAX_NODE = 604
K_EPS = 1e-5

def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")

def no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key: {key}")
        out[key] = value
    return out

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_dupes, parse_constant=reject_constant)

def fail(message):
    raise AssertionError(message)

def f32_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", int(bits)))[0]

def load_bundle(bundle):
    raw = (Path(bundle) / "trace.e1gtrc").read_bytes()
    if len(raw) < 48:
        fail("truncated trace header")
    magic, version, dim, base_n, reservoir_n, pool_n, query_n, k, radius, events = struct.unpack(
        "<8s7IfQ", raw[:48])
    if magic != b"E1GTRC01" or version != 1:
        fail("trace ABI")
    if base_n != 4096 or pool_n != 6144 or reservoir_n != pool_n - base_n:
        fail("fixture cardinality")
    pool_bytes = (Path(bundle) / "pool.i16").read_bytes()
    expected = pool_n * dim * 2
    if len(pool_bytes) != expected:
        fail("pool byte length")
    pool = struct.unpack("<" + "h" * (pool_n * dim), pool_bytes)
    return {"dim": dim, "base_n": base_n, "pool_n": pool_n, "k": k, "pool": pool}

def leaves_by_lid(snapshot):
    return sorted((node for node in snapshot["nodes"]
                   if node["empty"] == 0 and node["is_leaf"] == 1),
                  key=lambda node: (node["lid"], node["node_id"]))

def leaf_end(snapshot, leaf):
    leaves = leaves_by_lid(snapshot)
    for i, item in enumerate(leaves):
        if item["node_id"] == leaf["node_id"]:
            return leaves[i + 1]["lid"] if i + 1 < len(leaves) else snapshot["id_list_capacity"]
    fail("leaf absent")

def reconstruct(snapshot, pool_n):
    nodes = snapshot["nodes"]
    order = snapshot["tree_order"]
    ids = snapshot["id_slots"]
    by_id = {node["node_id"]: node for node in nodes}
    if len(by_id) != len(nodes) or set(by_id) != set(range(len(nodes))):
        fail("node IDs")
    memo = {}
    visiting = set()
    def visit(nid):
        if nid in memo:
            return memo[nid]
        if nid in visiting:
            fail("cycle")
        node = by_id[nid]
        if node["empty"]:
            memo[nid] = []
            return memo[nid]
        visiting.add(nid)
        if node["is_leaf"]:
            end = leaf_end(snapshot, node)
            if node["lid"] < 0 or node["size"] < 0 or node["lid"] + node["size"] > end:
                fail(f"invalid leaf span {nid}")
            values = ids[node["lid"]:node["lid"] + node["size"]]
            if any(value < 0 or value >= pool_n for value in values):
                fail(f"invalid logical id in leaf {nid}")
        else:
            values = []
            children = 0
            for slot in range(order):
                cid = nid * order + slot + 1
                if cid >= len(nodes) or by_id[cid]["empty"]:
                    continue
                children += 1
                values.extend(visit(cid))
            if children == 0:
                fail(f"internal without child {nid}")
        if len(values) != node["size"]:
            fail(f"subtree count mismatch {nid}")
        visiting.remove(nid)
        memo[nid] = values
        return values
    root = visit(0)
    return by_id, memo, root

def distance(pool, dim, a, b):
    offa = a * dim
    offb = b * dim
    return math.sqrt(sum((int(pool[offa+i]) - int(pool[offb+i])) ** 2 for i in range(dim)))

def validate_snapshot(snapshot, fixture, is_final):
    if snapshot["schema"] != "safe-c3-full-d2h-snapshot-v1":
        fail("snapshot schema")
    if snapshot["data_info"] != [fixture["dim"], fixture["base_n"], 2]:
        fail("data_info altered")
    if snapshot["tree_order"] != 10 or snapshot["max_size"] != 20:
        fail("tree constants")
    if len(snapshot["id_slots"]) != snapshot["id_list_capacity"]:
        fail("id capacity")
    by_id, members, root = reconstruct(snapshot, fixture["pool_n"])
    if len(root) != fixture["base_n"] + (1 if is_final else 0):
        fail("root member count")
    counts = Counter(root)
    for sid in range(fixture["base_n"]):
        if counts[sid] != 1:
            fail(f"base multiplicity {sid}")
    for sid in range(fixture["base_n"], fixture["pool_n"]):
        expect = 1 if is_final and sid == CANDIDATE else 0
        if counts[sid] != expect:
            fail(f"inactive/candidate multiplicity {sid}")
    # Validate all stored max envelopes (with a small cross-language float tolerance)
    pool, dim = fixture["pool"], fixture["dim"]
    for nid, node in by_id.items():
        if nid == 0 or node["empty"]:
            continue
        pivot = node["pid"]
        values = members[nid]
        actual = max((distance(pool, dim, sid, pivot) for sid in values), default=0.0)
        stored = f32_from_bits(node["max_dis_bits"])
        if stored + max(0.35, actual * 1e-6) < actual:
            fail(f"max coverage {nid}")
    # Validate all successor fences; the updated one must be strict.
    order = snapshot["tree_order"]
    for parent, node in by_id.items():
        if node["empty"]:
            continue
        for slot in range(order - 1):
            left_id = parent * order + slot + 1
            right_id = left_id + 1
            if left_id not in by_id or right_id not in by_id:
                continue
            left, right = by_id[left_id], by_id[right_id]
            if left["empty"] or right["empty"]:
                continue
            margin = f32_from_bits(right["min_dis_bits"]) - f32_from_bits(left["max_dis_bits"])
            if margin < -K_EPS:
                fail(f"successor fence {left_id}")
            if is_final and left_id == MAX_NODE and not margin > K_EPS:
                fail("updated strict fence")
    return by_id, members

def exact_topk(fixture):
    pool, dim = fixture["pool"], fixture["dim"]
    values = []
    qoff = CANDIDATE * dim
    for sid in list(range(fixture["base_n"])) + [CANDIDATE]:
        off = sid * dim
        sq = sum((int(pool[off+i]) - int(pool[qoff+i])) ** 2 for i in range(dim))
        values.append((sq, sid))
    values.sort()
    k = fixture["k"]
    if values[k-1][0] == values[k][0]:
        fail("boundary tie unexpectedly present")
    return values[:k]

def parse_jsonl(path):
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            fail(f"blank JSONL line {line_no}")
        records.append(json.loads(line, object_pairs_hook=no_dupes, parse_constant=reject_constant))
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    fixture = load_bundle(args.bundle)
    initial = load_json(args.initial)
    final = load_json(args.final)
    by_initial, _ = validate_snapshot(initial, fixture, False)
    by_final, final_members = validate_snapshot(final, fixture, True)

    changed_slots = [i for i, (a, b) in enumerate(zip(initial["id_slots"], final["id_slots"])) if a != b]
    if changed_slots != [33564] or initial["id_slots"][33564] != -1 or final["id_slots"][33564] != CANDIDATE:
        fail("raw id whitelist")
    changed_sizes = {nid for nid in by_initial if by_initial[nid]["size"] != by_final[nid]["size"]}
    if changed_sizes != SIZE_NODES or any(by_final[nid]["size"] != by_initial[nid]["size"] + 1 for nid in SIZE_NODES):
        fail("size whitelist")
    changed_max = {nid for nid in by_initial
                   if by_initial[nid]["max_dis_bits"] != by_final[nid]["max_dis_bits"]}
    if changed_max != {MAX_NODE}:
        fail("max whitelist")
    if by_initial[MAX_NODE]["max_dis_bits"] != 1193758447 or by_final[MAX_NODE]["max_dis_bits"] != 1193909109:
        fail("max bit contract")
    immutable_fields = ("empty", "pid", "min_dis_bits", "lid", "is_leaf")
    for nid in by_initial:
        if any(by_initial[nid][key] != by_final[nid][key] for key in immutable_fields):
            fail(f"immutable field changed {nid}")

    records = parse_jsonl(args.engine)
    if [record.get("record") for record in records] != [
        "meta", "plan", "postcommit_d2h_audit", "native_knn_4102", "exact_full_live_range_fallback"]:
        fail("engine record order")
    meta, plan, audit, knn, range_record = records
    if meta["candidate_id"] != CANDIDATE or plan["route_nodes"] != ROUTE or plan["write_slot"] != 33564:
        fail("engine plan binding")
    if plan["expected_sizes_before"] != [4096, 409, 49, 4] or plan["expected_sizes_after"] != [4097, 410, 50, 5]:
        fail("engine size plan")
    if plan["max_update_nodes"] != [604]:
        fail("engine max plan")
    if audit["changed_id_slots"] != 1 or audit["changed_size_nodes"] != 4 or audit["changed_max_nodes"] != 1:
        fail("engine D2H counts")
    if not all(audit[key] for key in [
        "data_info_unchanged", "data_rows_unchanged", "topology_unchanged",
        "membership_ok", "cardinality_ok", "coverage_and_fences_ok"]):
        fail("engine D2H booleans")

    exact = exact_topk(fixture)
    expect_ids = [sid for _, sid in exact]
    expect_sq = [sq for sq, _ in exact]
    if knn["gts_ids"] != expect_ids or knn["independent_exact_ids"] != expect_ids:
        fail("native KNN ID/exact mismatch")
    if knn["independent_exact_squared_distances"] != expect_sq or knn["k_boundary_tie"]:
        fail("native KNN exact record")
    if CANDIDATE not in knn["gts_ids"] or knn["gts_distances"][knn["gts_ids"].index(CANDIDATE)] != 0:
        fail("candidate not visible in native KNN")
    if range_record["native_gts_range_executed"] or range_record["native_range_correctness_claim"]:
        fail("range scope overclaimed")
    if range_record["exact_ids"] != [CANDIDATE] or range_record["exact_squared_distances"] != [0]:
        fail("full-live exact range fallback")

    summary = load_json(args.summary)
    if summary["status"] != "PASS" or summary["candidate_id"] != CANDIDATE or summary["route_nodes"] != ROUTE:
        fail("summary")
    if summary["native_range_correctness_claim"]:
        fail("summary range overclaim")
    print(json.dumps({
        "schema": "safe-c3-independent-validator-v1",
        "status": "PASS",
        "candidate_id": CANDIDATE,
        "route": ROUTE,
        "changed_id_slot": 33564,
        "updated_max_node": MAX_NODE,
        "native_knn_exact_topk_ids": expect_ids,
        "native_range_claim": False,
    }, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)


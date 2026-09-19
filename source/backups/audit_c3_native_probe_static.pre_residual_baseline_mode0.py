#!/workspace/legacy_workspace/GTS/bench_env/bin/python
"""CPU-only source/contract audit for C3 native direct-insert probe (no runner execution)."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.MULTILINE) is not None


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True,type=Path)
    p.add_argument('--runner',required=True,type=Path)
    p.add_argument('--validator',required=True,type=Path)
    p.add_argument('--tree-header',required=True,type=Path)
    p.add_argument('--search-header',required=True,type=Path)
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); errors=[]
    protocol=json.loads(a.protocol.read_text()); runner=a.runner.read_text(); validator=a.validator.read_text(); tree=a.tree_header.read_text(); search=a.search_header.read_text()
    def check(cond: bool, tag: str, **extra):
        if not cond: errors.append({'check':tag,**extra})
    check(protocol.get('schema')=='c3-native-certified-direct-insert-protocol-v2-search-upper','protocol_schema')
    check(protocol.get('status')=='PRE_REGISTERED_AFTER_STATIC_SEARCH_UPPER_AUDIT_BEFORE_GPU_EXECUTION','protocol_preregistration_status')
    check(not has(runner,r'^\s*#\s*include\s+[<"]incremental_insert\.cuh[>"]'),'no_incremental_insert_include')
    check(not has(runner,r'^\s*#\s*include\s+[<"]update\.cuh[>"]'),'no_update_include')
    check(not has(runner,r'\bincrementalInsert\s*\('),'no_legacy_incremental_insert_call')
    check(not has(runner,r'\bfindTargetLeaf\s*\('),'no_legacy_target_leaf_call')
    check(has(runner,r'void\s+native_append\s*\('),'native_append_function')
    check(has(runner,r'cudaMemcpy\(runtime\.id_list\s*\+\s*slot'),'native_real_id_list_write')
    check(has(runner,r'cudaMemcpy\(runtime\.node_list\s*\+\s*leaf_id'),'native_real_leaf_size_write')
    check(has(runner,r'leaf\.size\s*>=\s*MAX_SIZE'),'static_leaf_scan_capacity_gate')
    check(has(runner,r'void\s+searchIndexKnnV2|searchIndexKnnV2\(runtime\.data_d'),'real_static_topk_api')
    check(has(runner,r'verify_search_upper_invariant'),'runner_search_upper_audit')
    check(has(runner,r'max_upper\s*<=\s*next_min'),'runner_max_le_next_min_check')
    check(has(runner,r'actual_search_upper\s*=\s*frozen\.nodes\[next\]\.min_dis'),'runner_actual_next_sibling_upper')
    check(has(search,r'node_list\[nid\s*\+\s*1\]\.min_dis'),'archived_search_next_sibling_branch_present')
    check(has(search,r'nid\s*%\s*TREE_ORDER\s*!=\s*0'),'archived_search_nonlast_condition_present')
    check(has(tree,r'#define\s+LEAF_PAD_SLOTS\s+64'),'archived_leaf_padding_64_present')
    check(has(tree,r'__managed__\s+int\s+MAX_SIZE\s*=\s*20'),'archived_static_max_size_20_present')
    check(has(validator,r'def\s+search_upper_invariant\s*\('),'validator_search_upper_audit')
    check(has(validator,r'def\s+exact_topk\s*\('),'validator_independent_exact_oracle')
    check(has(validator,r'pool\[ids\]\.astype\(np\.int64'),'validator_int64_distance')
    check(a.binary.is_file() and a.binary.stat().st_size>0,'compiled_binary_exists')
    result={'schema':'c3-native-direct-insert-static-audit-v1','status':'PASS' if not errors else 'FAIL','gpu_used':False,
            'scope':'source/build contract audit only; no CUDA runner execution','error_count':len(errors),'errors':errors,
            'inputs':{str(x.resolve()):sha(x.resolve()) for x in (a.protocol,a.runner,a.validator,a.tree_header,a.search_header,a.binary)}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'errors':len(errors),'out':str(a.out.resolve())},sort_keys=True))
    return 0 if not errors else 2
if __name__=='__main__': raise SystemExit(main())

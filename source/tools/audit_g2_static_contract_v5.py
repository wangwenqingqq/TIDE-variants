#!/usr/bin/env python3
"""CPU-only static audit for the isolated v5 G2 rebuild witness."""
from __future__ import annotations
import argparse, hashlib, json, re
from artifact_pinning_v5 import CRITICAL_TOOL_RELS, source_include_closure
from pathlib import Path

ROOT = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v5_g2_artifact_pinned')
V3 = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v3_capacity_witness')
V2 = Path('/workspace/experiments/tide_safe_c1_20260728/safe_c1_dynamic_gts_v2_search_native')

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); root=a.root.resolve(); errors=[]; checks={}
    if root != ROOT.resolve(): errors.append('root_must_equal_canonical_v5_root')
    src=root/'src/safe_c1_dynamic_gts.cu'
    text=src.read_text() if src.is_file() else ''
    def require(name:str, needle:str) -> None:
        ok=needle in text; checks[name]=ok
        if not ok: errors.append('missing:'+name)
    def forbid(name:str, needle:str) -> None:
        ok=needle not in text; checks[name]=ok
        if not ok: errors.append('forbidden:'+name)
    require('explicit_gts_scalar_alias','using GtsScalar = short;')
    require('gts_scalar_float_static_assert','std::is_same<GtsScalar, float>::value')
    require('gts_scalar_ieee_float32_static_assert','std::numeric_limits<float>::is_iec559')
    require('gts_scalar_allocation','pool_values * sizeof(GtsScalar)')
    require('gts_scalar_device_copy','copy.size() * sizeof(GtsScalar)')
    require('gts_scalar_encoded_host_hash','immutable_pool_encoded.size() * sizeof(GtsScalar)')
    require('seeded_init_kernel','initIndexDataSeededStableIds')
    require('seed_assignment_before_pivot','id_list[idx] = seeded_stable_ids[idx]')
    require('seeded_constructor','indexConstruSeededStableIds')
    require('metadata_only_release','release_tree_metadata_only')
    require('immediate_rebuild_barrier','base deletion requires immediate REBUILD')
    require('state_after_rebuild','state.after_rebuild(live)')
    require('e0_freeze','epoch0.capture(runtime, initial_ids)')
    require('e1_freeze','epoch1.capture(runtime, live)')
    require('exact_leaf_set','epoch leaf stable-ID set is not exactly')
    require('pivot_subset','epoch pivot stable ID is absent from active seed set')
    require('full_pool_hash','metadata-only rebuild mutated immutable full pool')
    require('static_e1_probe','validate_real_static_probe(runtime, query_d, live')
    require('dynamic_full_oracle','exact_active_oracle')
    require('no_performance_claim','not_established')
    forbid('legacy_incremental_include','#include "incremental_insert.cuh"')
    forbid('legacy_update_include','#include "update.cuh"')
    # In comments both banned names are intentionally mentioned; verify no legacy function call.
    legacy_call=bool(re.search(r'(?<!SeededStableIds)\bindexConstru\s*\(', text))
    checks['no_archive_indexConstru_call']=not legacy_call
    if legacy_call: errors.append('forbidden:archive_indexConstru_call')
    for name in ('updateIndexRnn(', 'incrementalInsert(', 'deleteIncrementalInsert(', 'findTargetLeaf('):
        present=name in text; checks['no_'+name[:-1]]=not present
        if present: errors.append('forbidden:'+name)
    seed=text.find('id_list[idx] = seeded_stable_ids[idx]')
    pivot=text.find('getPivotDis<<<', seed)
    hash_start=text.find('auto device_pool_hash')
    hash_end=text.find('if (device_pool_hash()', hash_start)
    hash_body=text[hash_start:hash_end] if hash_start >= 0 and hash_end > hash_start else ''
    checks['device_hash_no_float_byte_count']='sizeof(float)' not in hash_body and 'std::vector<float> copy' not in hash_body
    if not checks['device_hash_no_float_byte_count']: errors.append('device_hash_encoding_mismatch')
    checks['seed_precedes_first_pivot']=seed>=0 and pivot>seed
    if not checks['seed_precedes_first_pivot']: errors.append('seed_not_before_pivot')
    # Metadata-only release body must not release the persistent pool / data_info.
    m=re.search(r'void release_tree_metadata_only\([^)]*\)\s*\{(.*?)\n\}',text,re.S)
    body=m.group(1) if m else ''
    checks['metadata_release_preserves_data_d']='cudaFree(runtime.data_d)' not in body and 'cudaFree(runtime.data_info)' not in body
    if not checks['metadata_release_preserves_data_d']: errors.append('metadata_release_frees_persistent_pool')
    # Verify v2/v3 frozen artifacts are still byte-identical to their original copy boundary.
    before=root/'provenance/v3_copied_code_sha256_before_g2.txt'
    baseline={}
    if before.is_file():
        for line in before.read_text().splitlines():
            cols=line.split(maxsplit=1)
            if len(cols)==2: baseline[cols[1]]=cols[0]
    for rel in ('include/safe_c1_search_native_routing_certificate.hpp','include/safe_c1_tie_free_witness.hpp','include/safe_search_v2.cuh'):
        p=V3/rel; ok=rel in baseline and p.is_file() and sha(p)==baseline[rel]
        checks['v3_frozen_'+rel.replace('/','_')]=ok
        if not ok: errors.append('v3_freeze_hash_mismatch:'+rel)
    # v3 source baseline is checked against v3, not v5 (which is intentionally changed).
    p=V3/'src/safe_c1_dynamic_gts.cu'; ok='src/safe_c1_dynamic_gts.cu' in baseline and p.is_file() and sha(p)==baseline['src/safe_c1_dynamic_gts.cu']
    checks['v3_frozen_source']=ok
    if not ok: errors.append('v3_freeze_hash_mismatch:source')
    v2sel=V2/'bundles/g1a_v2_identity_oracle_selection_20260728T110122Z/candidate_selection_v2.json'
    v3res=V3/'bundles/g1b_capacity_witness_20260728T114307Z_cpu/g1b_capacity_witness_v3.json'
    checks['external_artifacts_present']=v2sel.is_file() and v3res.is_file()
    if not checks['external_artifacts_present']: errors.append('external_artifact_missing')
    # Artifact-pinning design is statically auditable before a future guard can
    # reach nvidia-smi.  This audit itself does not query any GPU.
    pin_tool = root/'tools/artifact_pinning_v5.py'
    guard = root/'tools/run_g2_rebuild_witness_guarded_v5.sh'
    finalizer = root/'tools/finalize_g2_rebuild_witness_v5.py'
    checks['artifact_pin_tool_exists'] = pin_tool.is_file()
    if not checks['artifact_pin_tool_exists']: errors.append('artifact_pin_tool_missing')
    checks['critical_tool_set_present'] = all((root/rel).is_file() for rel in CRITICAL_TOOL_RELS)
    if not checks['critical_tool_set_present']: errors.append('artifact_pin_critical_tool_missing')
    guard_text = guard.read_text(encoding='utf-8') if guard.is_file() else ''
    finalizer_text = finalizer.read_text(encoding='utf-8') if finalizer.is_file() else ''
    pin_marker = guard_text.find('"$PYTHON" "$PIN_TOOL" verify')
    nvidia_marker = guard_text.find('command -v nvidia-smi')
    checks['artifact_pin_before_nvidia_smi'] = pin_marker >= 0 and nvidia_marker >= 0 and pin_marker < nvidia_marker
    if not checks['artifact_pin_before_nvidia_smi']: errors.append('artifact_pin_not_fail_closed_before_nvidia_smi')
    checks['guard_passes_pin_to_finalizer'] = '--artifact-pin "$PIN"' in guard_text and '--pin-verification "$PIN_VERIFICATION"' in guard_text
    if not checks['guard_passes_pin_to_finalizer']: errors.append('guard_missing_finalizer_pin_args')
    checks['finalizer_recomputes_pin'] = 'verify_execution_artifact_pin' in finalizer_text and 'artifact_pin_mismatch' in finalizer_text
    if not checks['finalizer_recomputes_pin']: errors.append('finalizer_missing_artifact_pin_revalidation')
    try:
        closure = source_include_closure()
        checks['source_include_closure_cpu_resolves'] = bool(closure.get('files_sha256')) and closure.get('source') == 'src/safe_c1_dynamic_gts.cu'
        checks['source_include_closure_local_only'] = all(key.startswith(('src/','include/')) for key in closure.get('files_sha256', {}))
        if not checks['source_include_closure_cpu_resolves']: errors.append('source_include_closure_unresolved')
        if not checks['source_include_closure_local_only']: errors.append('source_include_closure_not_local_only')
    except Exception as exc:
        checks['source_include_closure_cpu_resolves'] = False
        errors.append('source_include_closure_exception:'+str(exc))
    report={'schema':'safe-c1-g2-v5-static-audit','status':'PASS_CPU_ONLY_G2_STATIC_AUDIT' if not errors else 'FAIL_CPU_ONLY_G2_STATIC_AUDIT','gpu_used':False,'cuda_binary_executed':False,'root':str(root),'source_sha256':sha(src) if src.is_file() else None,'checks':checks,'errors':errors,'scope':'static source/freeze/artifact-pinning audit only; no CUDA binary execution, nvidia-smi, or performance claim'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(report['status']); return 0 if not errors else 2
if __name__=='__main__': raise SystemExit(main())

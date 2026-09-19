#!/usr/bin/env python3
"""Finalize and hash the CPU-only Safe-C2 v3 exact-oracle outputs."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, pathlib, platform, subprocess, sys

V2_TEST_SHA='50ccb28263bf23e499b9c50e4d3e9800fca3949aa5b5ea8a454237d5987701ce'
EXPECTED_QUERY_SHA='318e5085dfb4f50831d8f6620cffc736da9c11fb44454b507d2f5e7019d2e11e'
EXPECTED_BASE_SHA='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
MIN_COUNTS={'calibration':1800,'validation':1800,'sealed_test':5400}

def sha(p: pathlib.Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def ids(p):
 vals=[int(x) for x in p.read_text(encoding='ascii').split()]
 if vals!=sorted(vals) or len(vals)!=len(set(vals)): raise SystemExit(f'non-monotonic/duplicate IDs: {p}')
 return vals

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-root',type=pathlib.Path,required=True); ap.add_argument('--oracle-dir',type=pathlib.Path,required=True); ap.add_argument('--source',type=pathlib.Path,required=True); ap.add_argument('--binary',type=pathlib.Path,required=True); ap.add_argument('--command-file',type=pathlib.Path,required=True); args=ap.parse_args()
 ir=args.input_root.resolve(); od=args.oracle_dir.resolve()
 if not (ir/'selection_manifest_pre_gt.json').is_file(): raise SystemExit('missing frozen pre-GT selection manifest')
 pre=json.loads((ir/'selection_manifest_pre_gt.json').read_text())
 if pre.get('schema')!='gts-v3-compact-learn-selection-pre-gt-v1' or pre.get('status')!='INPUT_SELECTION_FROZEN_GT_AND_TIE_ADMISSION_PENDING': raise SystemExit('wrong pre-GT manifest')
 query=ir/'sift_learn_clean10k.fvecs'; base=pathlib.Path(pre['source']['base_fvecs_path']); gt=od/'learn_clean10k_gt_fp32_top100.ivecs'
 for p in [query,base,gt,od/'top101_int64_fp32_audit.bin',od/'tie_admission.tsv',od/'oracle_summary.json',args.source,args.binary,args.command_file]:
  if not p.is_file() or p.is_symlink(): raise SystemExit(f'missing/symlink artifact: {p}')
 if sha(query)!=EXPECTED_QUERY_SHA: raise SystemExit('compact query hash mismatch')
 if sha(base)!=EXPECTED_BASE_SHA: raise SystemExit('base hash mismatch')
 if gt.stat().st_size != 10000*(4+100*4): raise SystemExit(f'GT ivecs size mismatch {gt.stat().st_size}')
 header=(od/'tie_admission.tsv').read_text(encoding='ascii').splitlines()
 if len(header)!=10001 or not header[0].startswith('local_id\t'): raise SystemExit('tie admission length/header mismatch')
 seen={}
 for row in header[1:]:
  x=row.split('\t')
  if len(x)!=6: raise SystemExit('malformed admission row')
  q,*flags=map(int,x)
  if q in seen or q<0 or q>=10000 or any(v not in (0,1) for v in flags): raise SystemExit('invalid admission row')
  seen[q]=flags[-1]
 if set(seen)!=set(range(10000)): raise SystemExit('admission does not cover 10k local IDs')
 stages={}
 expected_ranges={'calibration':range(0,2000),'validation':range(2000,4000),'sealed_test':range(4000,10000)}
 for stage,rng in expected_ranges.items():
  p=od/(('sealed_test.ids' if stage=='sealed_test' else stage+'.ids'))
  xs=ids(p)
  if not set(xs).issubset(set(rng)): raise SystemExit(f'{stage} contains ID outside frozen partition')
  if any(seen[x]!=1 for x in xs): raise SystemExit(f'{stage} includes noneligible ID')
  if len(xs)!=sum(seen[x] for x in rng): raise SystemExit(f'{stage} drops/changes eligible ID')
  if len(xs)<MIN_COUNTS[stage]: raise SystemExit(f'{stage} eligible count {len(xs)} < predeclared minimum {MIN_COUNTS[stage]}')
  stages[stage]={'ids_path':str(p),'ids_sha256':sha(p),'eligible_count':len(xs),'unfiltered_partition':[rng.start,rng.stop],'predeclared_minimum':MIN_COUNTS[stage]}
 if set(ids(od/'calibration.ids'))&set(ids(od/'validation.ids')) or set(ids(od/'calibration.ids'))&set(ids(od/'sealed_test.ids')) or set(ids(od/'validation.ids'))&set(ids(od/'sealed_test.ids')): raise SystemExit('stage overlap')
 summary=json.loads((od/'oracle_summary.json').read_text())
 if summary.get('schema')!='gts-v3-cpu-exact-fp32-int64-oracle-v1': raise SystemExit('wrong oracle summary')
 if summary.get('eligible_calibration')!=stages['calibration']['eligible_count'] or summary.get('eligible_validation')!=stages['validation']['eligible_count'] or summary.get('eligible_sealed_test')!=stages['sealed_test']['eligible_count']: raise SystemExit('summary eligibility mismatch')
 compiler=subprocess.run(['/usr/bin/g++','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
 manifest={
  'schema':'gts-v3-compact-learn-workload-v1',
  'status':'READY_FOR_C2_V3_CALIBRATION_VALIDATION_AND_ONE_SEALED_TEST',
  'created_utc':dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z'),
  'sealed_v2_test_forbidden':True,
  'v2_sealed_test_ids_sha256_forbidden':V2_TEST_SHA,
  'discipline':'The compact split was frozen before GT. The CPU oracle may inspect sealed-test vectors only for deterministic ground-truth/tie admission; no GTS traversal, gamma selection, recall, timing, test trace, or test-based design decision has occurred.',
  'base_fvecs_path':str(base),'base_fvecs_sha256':sha(base),
  'query_fvecs_path':str(query),'query_fvecs_sha256':sha(query),
  'groundtruth_ivecs_path':str(gt),'groundtruth_ivecs_sha256':sha(gt),
  'local_to_original_mapping_path':str(ir/'local_to_original_learn_id.tsv'),'local_to_original_mapping_sha256':sha(ir/'local_to_original_learn_id.tsv'),
  'selection_manifest_pre_gt_path':str(ir/'selection_manifest_pre_gt.json'),'selection_manifest_pre_gt_sha256':sha(ir/'selection_manifest_pre_gt.json'),
  'stages':stages,
  'calibration_ids_path':stages['calibration']['ids_path'],'calibration_ids_sha256':stages['calibration']['ids_sha256'],
  'validation_ids_path':stages['validation']['ids_path'],'validation_ids_sha256':stages['validation']['ids_sha256'],
  'test_ids_path':stages['sealed_test']['ids_path'],'test_ids_sha256':stages['sealed_test']['ids_sha256'],
  'cpu_oracle':{
   'source_path':str(args.source),'source_sha256':sha(args.source),'binary_path':str(args.binary),'binary_sha256':sha(args.binary),
   'compiler':compiler,'compile_flags':['-O3','-march=native','-fopenmp','-ffp-contract=off','-fno-fast-math','-std=c++20'],
   'command_file_path':str(args.command_file),'command_file_sha256':sha(args.command_file),
   'oracle_dir':str(od),'oracle_summary_path':str(od/'oracle_summary.json'),'oracle_summary_sha256':sha(od/'oracle_summary.json'),
   'top101_audit_path':str(od/'top101_int64_fp32_audit.bin'),'top101_audit_sha256':sha(od/'top101_int64_fp32_audit.bin'),
   'tie_admission_path':str(od/'tie_admission.tsv'),'tie_admission_sha256':sha(od/'tie_admission.tsv'),
   'threads_requested':32,'nice':10,
  },
 }
 out=ir/'workload_manifest.json'
 if out.exists() or out.is_symlink(): raise SystemExit('refusing overwrite final workload manifest')
 out.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
 hashes={p.name:sha(p) for p in [query,ir/'local_to_original_learn_id.tsv',ir/'selection_manifest_pre_gt.json',gt,od/'top101_int64_fp32_audit.bin',od/'tie_admission.tsv',od/'oracle_summary.json',od/'calibration.ids',od/'validation.ids',od/'sealed_test.ids',out]}
 hpath=ir/'final_workload_artifacts.sha256'
 if hpath.exists() or hpath.is_symlink(): raise SystemExit('refusing overwrite final hash manifest')
 hpath.write_text(''.join(f'{v}  {k}\n' for k,v in sorted(hashes.items())),encoding='ascii')
 print(json.dumps({'status':'PASS','workload_manifest':str(out),'workload_manifest_sha256':sha(out),'eligible':{k:v['eligible_count'] for k,v in stages.items()}},sort_keys=True))
if __name__=='__main__': main()

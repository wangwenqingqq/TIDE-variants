#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs');queries=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs');gt=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs')
contract=root/'provenance/sift_integer_disk_contract_v1.json';audit=root/'tools/audit_diskguard_v2_sources.py';card=root/'provenance/diskguard_v2_rebuild.json';top=root/'bin/static_aabb_topk_canary_v2_diskguard';perf=root/'bin/static_aabb_perf_probe_v2_diskguard'
for p in (base,queries,gt,contract,audit,card,top,perf):
 if not p.is_file() or p.is_symlink():raise SystemExit('invalid '+str(p))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
c=json.loads(contract.read_text())
if c.get('status')!='PASS_CPU_ONLY' or c['base']['integer_range_violations'] or c['query']['integer_range_violations']:raise SystemExit('integer contract not valid')
b=json.loads(card.read_text())
if b.get('status')!='COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION' or b['topk_binary']['sha256']!=sha(top) or b['perf_binary']['sha256']!=sha(perf):raise SystemExit('build binding mismatch')
a=json.loads(subprocess.check_output([str(audit)],text=True))
if a.get('status')!='PASS' or a.get('sift_integer_contract_sha256')!=sha(contract):raise SystemExit('source/contract audit mismatch')
print(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2-preflight','status':'PASS_CPU_ONLY','scope':'SIFT-integer disk-guard v2 only; no old C2 validation/sealed access','base_sha256':sha(base),'query_sha256':sha(queries),'groundtruth_sha256':sha(gt),'integer_contract_sha256':sha(contract),'audit_sha256':sha(audit),'topk_binary_sha256':sha(top),'perf_binary_sha256':sha(perf),'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False},sort_keys=True))

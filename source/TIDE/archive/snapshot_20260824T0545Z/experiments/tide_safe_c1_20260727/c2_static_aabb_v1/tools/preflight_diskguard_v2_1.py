#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
r=pathlib.Path('/workspace/experiments/tide_safe_c1_20260727/c2_static_aabb_v1')
base=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs');q=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_query.fvecs');gt=pathlib.Path('/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_groundtruth.ivecs')
contract=r/'provenance/sift_integer_disk_contract_v1.json';audit=r/'tools/audit_diskguard_v2_1_sources.py';card=r/'provenance/diskguard_v2_1_build.json';top=r/'bin/static_aabb_topk_canary_v2_1_diskguard';perf=r/'bin/static_aabb_perf_probe_v2_1_diskguard'
for p in (base,q,gt,contract,audit,card,top,perf):
 if not p.is_file() or p.is_symlink():raise SystemExit('invalid '+str(p))
def h(p):
 x=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):x.update(z)
 return x.hexdigest()
c=json.loads(contract.read_text());b=json.loads(card.read_text());a=json.loads(subprocess.check_output([str(audit)],text=True))
if c['status']!='PASS_CPU_ONLY' or c['base']['integer_range_violations'] or c['query']['integer_range_violations']:raise SystemExit('contract')
if b['status']!='COMPLETE_BUILD_ONLY_NO_BINARY_EXECUTION' or b['topk_binary']['sha256']!=h(top) or b['perf_binary']['sha256']!=h(perf):raise SystemExit('build')
if a['status']!='PASS' or a['contract_sha256']!=h(contract):raise SystemExit('audit')
print(json.dumps({'schema':'safe-c2-static-aabb-diskguard-v2.1-preflight','status':'PASS_CPU_ONLY','scope':'SIFT-integer disk-guard v2.1; no old C2 validation/sealed access','base_sha256':h(base),'query_sha256':h(q),'groundtruth_sha256':h(gt),'contract_sha256':h(contract),'audit_sha256':h(audit),'topk_binary_sha256':h(top),'perf_binary_sha256':h(perf),'gpu_binary_executed':False,'nvidia_smi_called':False,'formal_claim_eligible':False},sort_keys=True))

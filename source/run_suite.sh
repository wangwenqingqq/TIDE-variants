#!/usr/bin/env bash
# E1-GR suite: exactly 9 locked traces x {Safe-C1, buffer-only} = 18 runs.
# Each child run independently guards physical GPU0; GPU7 is never selectable.
set -Eeuo pipefail
ROOT="/workspace/experiments/tide_safe_c1_20260727"
SELF="$ROOT/safe_c1_cuda_reference"
BATCH="$ROOT/runs/e1g0_cuda_traces_v1_20260727"
PY="/workspace/legacy_workspace/GTS/bench_env/bin/python"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%Z)"
OUT="$ROOT/runs/e1gr_cuda_safe_c1_reference_suite_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/suite.log"
RECORDS="$OUT/run_records.tsv"
START="$(date -Is)"
{
  echo "experiment=E1-GR"
  echo "scope=CUDA Safe-C1 reference correctness only; not dynamic GTS integration or latency"
  echo "host=$(hostname)"
  echo "a800_hosts_used=[]"
  echo "forbidden_gpu=7"
  echo "runner=$SELF/run_one.sh"
  echo "batch=$BATCH"
  echo "started_at=$START"
} | tee "$LOG"
mapfile -t BUNDLES < <(find "$BATCH" -mindepth 1 -maxdepth 1 -type d | sort)
[[ ${#BUNDLES[@]} -eq 9 ]] || { echo "expected 9 bundles, found ${#BUNDLES[@]}" | tee -a "$LOG"; exit 2; }
printf 'bundle\tmode\trun_dir\n' > "$RECORDS"
for bundle in "${BUNDLES[@]}"; do
  for mode in safe buffer; do
    echo "START bundle=$(basename "$bundle") mode=$mode" | tee -a "$LOG"
    run_dir="$($SELF/run_one.sh --bundle "$bundle" --mode "$mode" --runs "$ROOT/runs")"
    printf '%s\t%s\t%s\n' "$bundle" "$mode" "$run_dir" >> "$RECORDS"
    echo "PASS bundle=$(basename "$bundle") mode=$mode run_dir=$run_dir" | tee -a "$LOG"
  done
done
"$PY" - "$OUT/suite_summary.json" "$OUT" "$START" "$RECORDS" "$SELF/run_suite.sh" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
out, suite_dir, started, records_path, script = sys.argv[1:]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
rows=[]
for line in open(records_path):
    if line.startswith('bundle\t'): continue
    b,m,r=line.rstrip('\n').split('\t')
    card=json.load(open(os.path.join(r,'run_card.json')))
    verify=json.load(open(os.path.join(r,'independent_verify.json')))
    rows.append({'bundle':b,'mode':m,'run_dir':r,'status':card.get('status'),'verify_pass':verify.get('pass'),
                 'trace_counts':card.get('contract',{}).get('trace_counts',{}),
                 'placement':json.load(open(os.path.join(r,'engine_summary.json'))).get('placement',{})})
passed=sum(x['status']=='PASS_REFERENCE_CORRECTNESS' and x['verify_pass'] for x in rows)
obj={'experiment_id':'E1-GR','status':'PASS_REFERENCE_CORRECTNESS' if passed==len(rows) else 'INCOMPLETE_OR_FAILED',
     'scope':'CUDA Safe-C1 reference correctness only; not GTS dynamic integration and not a latency result',
     'host':os.uname().nodename,'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
     'a800_hosts_used':[],'forbidden_gpu':[7],'physical_gpu_used':[0],
     'expected_runs':18,'completed_runs':len(rows),'independent_verifier_passes':passed,
     'suite_script':script,'suite_script_sha256':sha(script),'run_records':records_path,'runs':rows}
open(out,'w').write(json.dumps(obj,indent=2,sort_keys=True)+'\n')
PY
echo "finished_at=$(date -Is)" | tee -a "$LOG"
printf '%s\n' "$OUT"

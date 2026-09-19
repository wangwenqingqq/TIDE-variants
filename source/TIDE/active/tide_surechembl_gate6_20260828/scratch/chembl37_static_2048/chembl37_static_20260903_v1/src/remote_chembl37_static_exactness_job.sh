#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BASE WORK GPU GPU_UUID" >&2
  exit 64
fi

BASE=$1
WORK=$2
GPU=$3
GPU_UUID=$4
ROOT=$BASE/root
CUDA=/usr/local/cuda-13.1
QUERY_INPUT=$BASE/src/gate6_query_campaign.cu
ORACLE_INPUT=$BASE/src/gate6_exact_tuple_oracle_width.cpp
QUERY_SOURCE=$WORK/src/gate6_query_campaign.cu
ORACLE_SOURCE=$WORK/src/gate6_exact_tuple_oracle_width.cpp
QUERY_BIN=$WORK/build/gate6_query_campaign
ORACLE_BIN=$WORK/build/gate6_exact_tuple_oracle_width
KEEPER=/workspace/TIDE/active/tide_surechembl_gate6_20260828/build/gate6_query_campaign

EXPECTED_QUERY_SOURCE=0d0a00450020f0067a61b8d8fd2d57a463480bac547b64d75ba93dc93b41f154
EXPECTED_ORACLE_SOURCE=eea7c2b3d6033660be4c8c74eeb16d37ea8f6a1fc4560e01f1af4904fd3fd1d2
EXPECTED_PREPARATION=c7434ef50db743318de31c531710f4150e87ce827adb4f145aa757d660a08ec2
EXPECTED_PREPARATION_AUDIT=11ebc7e0093b2e161dbd7968f2238c5f54eafeba5fedc78ec094c61bb0ec3d95

if [[ -e $WORK ]]; then
  echo "refusing to reuse work path: $WORK" >&2
  exit 65
fi
mkdir -p "$WORK"/{audit,build,fixture,oracle,results,src}
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/audit/build_start_utc.txt"

check_hash() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ $actual != "$expected" ]]; then
    echo "hash mismatch: $path: $actual != $expected" >&2
    exit 66
  fi
}
check_hash "$EXPECTED_QUERY_SOURCE" "$QUERY_INPUT"
check_hash "$EXPECTED_ORACLE_SOURCE" "$ORACLE_INPUT"
check_hash "$EXPECTED_PREPARATION" "$ROOT/PREPARATION_MANIFEST.json"
check_hash "$EXPECTED_PREPARATION_AUDIT" "$ROOT/PREPARATION_AUDIT.json"
cp "$QUERY_INPUT" "$QUERY_SOURCE"
cp "$ORACLE_INPUT" "$ORACLE_SOURCE"
check_hash "$EXPECTED_QUERY_SOURCE" "$QUERY_SOURCE"
check_hash "$EXPECTED_ORACLE_SOURCE" "$ORACLE_SOURCE"
sha256sum "$QUERY_SOURCE" "$ORACLE_SOURCE" "$ROOT/PREPARATION_MANIFEST.json" \
  "$ROOT/PREPARATION_AUDIT.json" > "$WORK/audit/input_hashes.txt"

"$CUDA/bin/nvcc" -std=c++17 -O3 -lineinfo -arch=sm_120a \
  -Xcompiler=-pthread,-Wall,-Wextra \
  "$QUERY_SOURCE" -o "$QUERY_BIN" \
  > "$WORK/audit/nvcc.stdout.txt" 2> "$WORK/audit/nvcc.stderr.txt"
g++ -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic \
  "$ORACLE_SOURCE" -o "$ORACLE_BIN" \
  > "$WORK/audit/gxx.stdout.txt" 2> "$WORK/audit/gxx.stderr.txt"
if grep -Eqi 'warning:|error:' "$WORK/audit/nvcc.stderr.txt" \
    "$WORK/audit/gxx.stderr.txt"; then
  echo "warning/error found in build logs" >&2
  exit 67
fi

for label in candidate keeper; do
  if [[ $label == candidate ]]; then binary=$QUERY_BIN; else binary=$KEEPER; fi
  test -x "$binary"
  "$CUDA/bin/cuobjdump" --dump-sass "$binary" \
    > "$WORK/audit/$label.full.sass" \
    2> "$WORK/audit/$label.sass.stderr.txt"
  "$CUDA/bin/cuobjdump" --dump-resource-usage "$binary" \
    > "$WORK/audit/$label.resources.txt" \
    2> "$WORK/audit/$label.resources.stderr.txt"
done
python3 - "$WORK/audit" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
root=Path(sys.argv[1])
def body(label):
    selected=False; instructions=[]
    for raw in (root/f'{label}.full.sass').read_text(errors='replace').splitlines():
        function=re.match(r'\s*Function\s*:\s*(\S+)',raw)
        if function:
            selected='exact_runs_kernelILi32' in function.group(1); continue
        if not selected: continue
        address=re.match(r'\s*/\*\s*[0-9a-fA-F]+\s*\*/\s*(.*)',raw)
        if not address: continue
        instruction=re.sub(r'\s*/\*\s*(?:0x)?[0-9a-fA-F]{8,}\s*\*/\s*$','',address.group(1)).strip()
        instruction=re.sub(r'\s+',' ',instruction)
        if instruction.endswith(';'): instructions.append(instruction)
    text='\n'.join(instructions)+'\n'
    (root/f'{label}.words32.instruction_body.sass').write_text(text)
    if not instructions: raise SystemExit(f'WORDS32 body absent: {label}')
    return len(instructions),hashlib.sha256(text.encode()).hexdigest()
def resources(label):
    lines=(root/f'{label}.resources.txt').read_text().splitlines()
    for i,line in enumerate(lines):
        if 'exact_runs_kernelILi32' not in line: continue
        for candidate in lines[i+1:i+5]:
            m=re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+) CONSTANT\[0\]:(\d+)',candidate)
            if m: return tuple(map(int,m.groups()))
    raise SystemExit(f'WORDS32 resources absent: {label}')
cb=body('candidate'); kb=body('keeper'); cr=resources('candidate'); kr=resources('keeper')
result={'candidate_body_lines':cb[0],'candidate_body_sha256':cb[1],
        'keeper_body_lines':kb[0],'keeper_body_sha256':kb[1],
        'candidate_resources':cr,'keeper_resources':kr,
        'body_equal':cb==kb,'resources_equal':cr==kr}
result['pass']=result['body_equal'] and result['resources_equal']
(root/'WORDS32_SASS_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['pass']: raise SystemExit(1)
PY
sha256sum "$QUERY_BIN" "$ORACLE_BIN" \
  "$WORK/audit/candidate.words32.instruction_body.sass" \
  > "$WORK/audit/binary_and_body_hashes.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/audit/build_end_utc.txt"

# Build a source-derived sanitizer fixture without changing the formal data.
python3 - "$ROOT" "$WORK/fixture/root" <<'PY'
import json,numpy as np,sys
from pathlib import Path
source=Path(sys.argv[1]); root=Path(sys.argv[2]); rows=2_897_819; words=32
fp=np.memmap(source/'data/gate0_prepared/union_fp_u64x32.bin',mode='r',dtype='<u8',shape=(rows,words))
ids=np.memmap(source/'data/gate0_prepared/union_id_i64.bin',mode='r',dtype='<i8',shape=(rows,))
pc=np.memmap(source/'data/gate0_prepared/union_popcnt_u16.bin',mode='r',dtype='<u2',shape=(rows,))
chosen=np.linspace(0,rows-1,4096,dtype=np.int64); base=chosen[:3584]; delta=chosen[3584:]
gate0=root/'data/gate0_prepared'; stage=root/'data/stage_a'; gate0.mkdir(parents=True); stage.mkdir(parents=True)
fp[chosen].astype('<u8').tofile(gate0/'union_fp_u64x32.bin'); ids[chosen].astype('<i8').tofile(gate0/'union_id_i64.bin'); pc[chosen].astype('<u2').tofile(gate0/'union_popcnt_u16.bin')
fp[base].astype('<u8').tofile(gate0/'base_fp_u64x32.bin'); ids[base].astype('<i8').tofile(gate0/'base_id_i64.bin'); pc[base].astype('<u2').tofile(gate0/'base_popcnt_u16.bin')
packed=np.empty((len(delta),34),dtype='<u8'); packed[:,0]=ids[delta].astype('<u8'); packed[:,1:33]=fp[delta]; packed[:,-1]=pc[delta]; packed.tofile(stage/'delta_u64x34.bin'); packed[:32].tofile(stage/'queries_u64x34.bin')
(root/'FIXTURE.json').write_text(json.dumps({'union_rows':4096,'base_rows':3584,'delta_rows':512,'query_rows':32,'selection':'4096 evenly spaced rows from sorted formal union'},indent=2)+'\n')
PY

exec 7>/tmp/tide_surechembl_cpu_oracle.lock
flock -n 7 || { echo "CPU oracle lock busy" >&2; exit 75; }
OMP_NUM_THREADS=32 nice -n 10 "$ORACLE_BIN" "$WORK/fixture/root" \
  "$WORK/oracle/fixture.csv" 32 32 \
  > "$WORK/oracle/fixture.stdout.txt" 2> "$WORK/oracle/fixture.stderr.txt"
OMP_NUM_THREADS=32 nice -n 10 "$ORACLE_BIN" "$ROOT" \
  "$WORK/oracle/formal.csv" 512 32 \
  > "$WORK/oracle/formal.stdout.txt" 2> "$WORK/oracle/formal.stderr.txt"
flock -u 7

exec 9>/tmp/tide_surechembl_gpu0123.lock
flock -n 9 || { echo "global GPU lock busy" >&2; exit 75; }
exec 8>/tmp/tide_surechembl_gate6_gpu2.lock
flock -n 8 || { echo "Gate-6 GPU2 lock busy" >&2; exit 75; }
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/audit/gpu_phase_start_utc.txt"
hostname > "$WORK/audit/hostname.txt"
printf 'pid=%s global_lock=%s gate6_lock=%s\n' $$ \
  /tmp/tide_surechembl_gpu0123.lock /tmp/tide_surechembl_gate6_gpu2.lock \
  > "$WORK/audit/lock_owner.txt"
nvidia-smi -L > "$WORK/audit/nvidia_smi_L.txt"
"$CUDA/bin/nvcc" --version > "$WORK/audit/nvcc_version.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits > "$WORK/audit/compute_apps_before.csv" || true
if grep -Fq "$GPU_UUID" "$WORK/audit/compute_apps_before.csv"; then
  echo "existing process on target GPU $GPU_UUID" >&2
  exit 75
fi
nvidia-smi -i "$GPU" --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$WORK/audit/gpu_before.csv"
if ! grep -Fq "$GPU_UUID" "$WORK/audit/gpu_before.csv"; then
  echo "physical GPU UUID mismatch" >&2
  exit 68
fi

run_candidate() {
  local root=$1
  local output=$2
  local query_limit=$3
  nice -n 5 "$QUERY_BIN" --root "$root" --mode correctness \
    --output "$output" --gpu "$GPU" --words 32 --query-limit "$query_limit"
}
run_candidate "$WORK/fixture/root" "$WORK/results/fixture.json" 32 \
  > "$WORK/results/fixture.stdout.txt" 2> "$WORK/results/fixture.stderr.txt"
"$CUDA/bin/compute-sanitizer" --tool memcheck --leak-check full \
  "$QUERY_BIN" --root "$WORK/fixture/root" --mode correctness \
  --output "$WORK/results/memcheck.json" --gpu "$GPU" --words 32 --query-limit 1 \
  > "$WORK/results/memcheck.stdout.txt" 2> "$WORK/results/memcheck.stderr.txt"
"$CUDA/bin/compute-sanitizer" --tool synccheck \
  "$QUERY_BIN" --root "$WORK/fixture/root" --mode correctness \
  --output "$WORK/results/synccheck.json" --gpu "$GPU" --words 32 --query-limit 1 \
  > "$WORK/results/synccheck.stdout.txt" 2> "$WORK/results/synccheck.stderr.txt"
grep -Fq 'ERROR SUMMARY: 0 errors' \
  "$WORK/results/memcheck.stdout.txt" "$WORK/results/memcheck.stderr.txt"
grep -Fq 'ERROR SUMMARY: 0 errors' \
  "$WORK/results/synccheck.stdout.txt" "$WORK/results/synccheck.stderr.txt"

run_candidate "$ROOT" "$WORK/results/smoke32.json" 32 \
  > "$WORK/results/smoke32.stdout.txt" 2> "$WORK/results/smoke32.stderr.txt"
run_candidate "$ROOT" "$WORK/results/formal512.json" 512 \
  > "$WORK/results/formal512.stdout.txt" 2> "$WORK/results/formal512.stderr.txt"

python3 - "$WORK" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1])
def load_oracle(path):
    with path.open(newline='') as f:
        return {(int(r['query']),int(r['threshold_num']),int(r['threshold_den'])):r for r in csv.DictReader(f)}
def audit(candidate_path,oracle_path,queries,label):
    oracle=load_oracle(oracle_path); rows=list(csv.DictReader(candidate_path.open()))
    expected=queries*2*9
    if len(rows)!=expected: raise SystemExit(f'{label}: candidate row count {len(rows)} != {expected}')
    errors=[]; overflow=0; internal=0
    for row in rows:
        key=(int(row['query']),int(row['threshold_num']),int(row['threshold_den']))
        expected_row=oracle[key]
        overflow+=int(row['overflow']); internal+=int(row['match'])!=1
        if int(row['hits'])!=int(expected_row['hits']) or int(row['hash'])!=int(expected_row['result_hash']):
            errors.append({'key':key,'variant':row['variant']})
        if not row['variant'].startswith('unbounded') and int(row['candidates'])!=int(expected_row['candidate_rows']):
            errors.append({'key':key,'variant':row['variant'],'kind':'candidate'})
    return {'label':label,'rows':len(rows),'expected_rows':expected,'external_mismatches':errors[:20],
            'external_mismatch_count':len(errors),'overflow_rows':overflow,'internal_mismatch_rows':internal,
            'pass':not errors and not overflow and not internal}
fixture=audit(root/'results/fixture.json.samples.csv',root/'oracle/fixture.csv',32,'fixture')
smoke=audit(root/'results/smoke32.json.samples.csv',root/'oracle/formal.csv',32,'smoke32')
formal=audit(root/'results/formal512.json.samples.csv',root/'oracle/formal.csv',512,'formal512')
mem=''.join((root/f'results/memcheck.{stream}.txt').read_text() for stream in ('stdout','stderr'))
sync=''.join((root/f'results/synccheck.{stream}.txt').read_text() for stream in ('stdout','stderr'))
result={'experiment_id':'tide_20260903_chembl37_static_single_source_2048_exactness',
        'fixture':fixture,'smoke32':smoke,'formal512':formal,
        'sanitizers':{'memcheck_zero_errors':'ERROR SUMMARY: 0 errors' in mem,
                      'synccheck_zero_errors':'ERROR SUMMARY: 0 errors' in sync},
        'sass':json.loads((root/'audit/WORDS32_SASS_AUDIT.json').read_text())}
result['pass']=all(x['pass'] for x in [fixture,smoke,formal]) and all(result['sanitizers'].values()) and result['sass']['pass']
(root/'FORMAL_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['pass']: raise SystemExit(1)
PY

nvidia-smi -i "$GPU" --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$WORK/audit/gpu_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits > "$WORK/audit/compute_apps_after.csv" || true
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK/audit/gpu_phase_end_utc.txt"
sha256sum "$WORK"/FORMAL_SUMMARY.json "$WORK"/oracle/* \
  "$WORK"/results/*.json "$WORK"/results/*.csv \
  "$WORK"/audit/WORDS32_SASS_AUDIT.json \
  > "$WORK/audit/result_hashes.txt"

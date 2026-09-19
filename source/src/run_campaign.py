#!/usr/bin/env python3
"""Run P1 only with explicit allocation; default is CPU/read-only preflight."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from prepare import digest

BACKENDS=['tide_batch','nv_row','tide_bounded']
PEERS=['nv_row','tide_bounded']


def plan():
    result=[]
    for peer in PEERS:
        for pair in range(1,7):
            order=[peer,'tide_batch'] if pair%2 else ['tide_batch',peer]
            for slot,backend in enumerate(order):
                result.append({'comparison':peer,'pair':pair,'slot':slot,'backend':backend,
                               'reverse':pair%2==0,'warmup':16})
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['preflight','validate','formal'],default='preflight')
    ap.add_argument('--admit-gpu3',action='store_true'); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    planned=plan()
    if args.phase=='preflight':
        for data in ['real','fixture']:
            m=json.loads((root/f'data/{data}/ORACLE_MANIFEST.json').read_text())
            if digest(root/f'data/{data}/oracle.bin')!=m['oracle_sha256']: raise ValueError('oracle drift')
        files=[root/'build/bridges_v2/libp1_tide.so',root/'build/bridges_v2/libp1_gpusim.so']
        print(json.dumps({'state':'CPU preflight only; no GPU context',
                          'formal_processes':len(planned),'plan':planned,
                          'libraries':{str(p):digest(p) for p in files}},indent=2))
        return
    if not args.admit_gpu3: raise RuntimeError('GPU 3 user authorization is required')
    os.environ['PYTHONPATH']='/workspace/TIDE/active/tide_surechembl_gate6_20260828/scratch/nvmolkit060_query/nvmolkit060_20260904_v1/site'
    stage=root/'raw'/args.phase
    if stage.exists(): raise FileExistsError(f'retain old stage; do not replace: {stage}')
    if args.phase=='formal':
        gates=json.loads((root/'raw/validate/GATES.json').read_text())
        if not gates['complete']: raise RuntimeError('validation gates incomplete')
        for path,sha in gates['pinned_files'].items():
            if digest(root/path)!=sha: raise RuntimeError('validation artifact drift: '+path)
    stage.mkdir(parents=True)
    (stage/'PLAN.json').write_text(json.dumps(planned,indent=2)+'\n')
    records=[]
    def invoke(label,backend,data,mode,reverse=False,sanitizer=None):
        target=stage/(label+'.jsonl')
        child=[sys.executable,str(root/'src/run_backend.py'),'--backend',backend,'--data',str(root/'data'/data),
               '--mode',mode,'--output',str(target)]
        if reverse: child+=['--reverse']
        if sanitizer:
            child=['/usr/local/bin/compute-sanitizer','--tool',sanitizer,'--error-exitcode','87']+child
        child=['taskset','-c','127']+child
        command=[sys.executable,str(root/'src/guarded_run.py'),'--admit-gpu3',
                 '--record',str(stage/(label+'.guard.json')),'--']+child
        with (stage/(label+'.log')).open('x') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record={'label':label,'backend':backend,'data':data,'mode':mode,
                'exit_code':completed.returncode,'sanitizer':sanitizer}
        records.append(record); (stage/'PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n')
        if completed.returncode: raise RuntimeError('retained failure: '+label)
        if sanitizer:
            log=(stage/(label+'.log')).read_text()
            if not re.search(r'ERROR SUMMARY: 0 errors',log): raise RuntimeError('sanitizer zero-error proof missing')
        print(label,'passed',flush=True)
    if args.phase=='validate':
        for backend in BACKENDS:
            invoke(backend+'_fixture',backend,'fixture','check')
            invoke(backend+'_real',backend,'real','check')
            for sanitizer in ['memcheck','synccheck']:
                invoke(backend+'_'+sanitizer,backend,'fixture','check',sanitizer=sanitizer)
            invoke(backend+'_churn',backend,'fixture','churn')
        # Pin the bridge, keeper, driver, input, and all completed gate evidence.
        pins={}
        candidates=list((root/'src').glob('*'))+list((root/'keeper').glob('*'))
        candidates+=list((root/'build/bridges_v2').glob('*.so'))
        candidates+=list((root/'data').rglob('*'))+list(stage.glob('*'))
        for p in candidates:
            if p.is_file() and '__pycache__' not in p.parts:
                pins[str(p.relative_to(root))]=digest(p)
        (stage/'GATES.json').write_text(json.dumps({'complete':True,'invocations':records,'pinned_files':pins},indent=2)+'\n')
    else:
        for item in planned:
            label=f"{item['comparison']}_pair{item['pair']}_slot{item['slot']}_{item['backend']}"
            invoke(label,item['backend'],'real','timing',reverse=item['reverse'])
        (stage/'COMPLETE.json').write_text(json.dumps({'complete':True,'processes':len(records)},indent=2)+'\n')


if __name__=='__main__': main()

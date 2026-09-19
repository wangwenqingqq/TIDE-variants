#!/usr/bin/env python3
"""Fail closed on missing GPU admission, occupation, locks, or child failure."""
import argparse
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

UUID='GPU-CONFIGURE-ARCHIVE-DEVICE'


def command(args):
    return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()


def snapshot():
    gpu=command(['nvidia-smi','-i','3','--query-gpu=uuid,name,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,power.limit,temperature.gpu,power.draw','--format=csv,noheader'])
    apps=command(['nvidia-smi','-i','3','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    if not gpu.startswith(UUID+','): raise RuntimeError('GPU 3 identity changed')
    return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':gpu,'apps':apps}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--admit-gpu3',action='store_true')
    ap.add_argument('--record',type=Path,required=True); ap.add_argument('child',nargs=argparse.REMAINDER)
    args=ap.parse_args()
    if not args.admit_gpu3: raise RuntimeError('explicit user-approved GPU 3 admission required')
    child=args.child[1:] if args.child[:1]==['--'] else args.child
    if not child: raise ValueError('missing child command')
    if args.record.exists(): raise FileExistsError(args.record)
    args.record.parent.mkdir(parents=True,exist_ok=True)
    locks=[]
    for path in ['/tmp/tide_surechembl_gpu0123.lock','/tmp/tide_surechembl_gpusim_gpu3.lock']:
        handle=open(path,'a'); fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(handle)
    before=snapshot()
    if before['apps']: raise RuntimeError('pre-existing GPU 3 process: '+before['apps'])
    # Occupancy is insufficient: refuse visible persistent launchers for this GPU.
    launcher=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try: line=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (PermissionError,FileNotFoundError,ProcessLookupError): continue
        if re.search(r'(?:--physical-gpu|--gpu-index|--gpu)\s+3(?:\s|$)',line) and 'guarded_run.py' not in line:
            launcher.append({'pid':int(p.name),'command':line[:500]})
    if launcher: raise RuntimeError('pending GPU 3 launcher: '+json.dumps(launcher))
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='3',P1_GUARDED_UUID=UUID,
                                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    rec={'before':before,'guard_pid':os.getpid(),'command':child,'admission':'user-confirmed GPU3 only','status':'running'}
    start=time.monotonic(); proc=subprocess.Popen(child,env=env,start_new_session=True)
    rec['child_pid']=proc.pid; args.record.write_text(json.dumps(rec,indent=2)+'\n')
    samples=[]; outsiders=[]
    while True:
        try:
            code=proc.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            sample=snapshot(); samples.append(sample)
            for line in sample['apps'].splitlines():
                pid=int(line.split(',')[0])
                try: own=os.getsid(pid)==proc.pid
                except ProcessLookupError: continue  # Exited during the read-only snapshot.
                if not own: outsiders.append({'pid':pid,'utc':sample['utc']})
    after=snapshot()
    rec.update(status='complete' if code==0 else 'failed',exit_code=code,after=after,elapsed_seconds=time.monotonic()-start)
    rec.update(sampled_device_state=samples,foreign_gpu_processes=outsiders,
               resource_scope='2-second device samples, not an exact peak-memory measurement')
    args.record.write_text(json.dumps(rec,indent=2)+'\n')
    if outsiders: raise RuntimeError('GPU interference observed; retain and reject timing')
    if after['apps']: raise RuntimeError('post-run GPU occupation; timing requires review')
    return code


if __name__=='__main__': sys.exit(main())

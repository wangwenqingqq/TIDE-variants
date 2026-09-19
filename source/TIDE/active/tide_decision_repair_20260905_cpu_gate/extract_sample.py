"""CPU-only, bounded-chunk extraction of the preregistered temporal hash sample."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np

DATES = ['2026-06-01','2026-06-15','2026-07-01','2026-07-17',
         '2026-08-04','2026-08-18','2026-08-25']
K, SEED, CHUNK = 16384, 2026090503, 1 << 20

def splitmix64(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

def file_info(p):
    s = p.stat()
    return {'path': str(p), 'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    latest=a.source/DATES[-1]
    ids=np.memmap(latest/'id_i64.bin',dtype='<i8',mode='r')
    keep_ids=np.empty(0,dtype=np.int64); keep_scores=np.empty(0,dtype=np.uint64)
    for start in range(0,len(ids),CHUNK):
        part=np.asarray(ids[start:start+CHUNK]); scores=splitmix64(part.astype(np.uint64)^np.uint64(SEED))
        ci=np.concatenate([keep_ids,part]); cs=np.concatenate([keep_scores,scores])
        if len(cs)>K:
            boundary=np.partition(cs,K-1)[K-1]; ix=np.flatnonzero(cs<=boundary)
            ix=ix[np.lexsort((ci[ix],cs[ix]))[:K]]
        else: ix=np.lexsort((ci,cs))
        keep_ids,keep_scores=ci[ix],cs[ix]
    selected=np.sort(keep_ids); assert len(np.unique(selected))==K
    presence=np.zeros((len(DATES),K),dtype=np.uint8)
    fingerprints=np.zeros((len(DATES),K,4),dtype=np.uint64)
    counts=np.zeros((len(DATES),K),dtype=np.uint16)
    records=[]
    for si,date in enumerate(DATES):
        paths={name:a.source/date/name for name in ['id_i64.bin','fp_u64x4.bin','popcnt_u16.bin']}
        before={name:file_info(p) for name,p in paths.items()}
        src=np.memmap(paths['id_i64.bin'],dtype='<i8',mode='r')
        fp=np.memmap(paths['fp_u64x4.bin'],dtype='<u8',mode='r',shape=(len(src),4))
        pc=np.memmap(paths['popcnt_u16.bin'],dtype='<u2',mode='r')
        digest=hashlib.sha256()
        for start in range(0,len(src),CHUNK):
            part=np.asarray(src[start:start+CHUNK]); digest.update(part.tobytes())
            pos=np.searchsorted(selected,part); valid=pos<K
            ix=np.flatnonzero(valid); ix=ix[selected[pos[ix]]==part[ix]]; dest=pos[ix]
            assert len(np.unique(dest))==len(dest) and not presence[si,dest].any()
            presence[si,dest]=1; fingerprints[si,dest]=fp[start+ix]; counts[si,dest]=pc[start+ix]
        assert before=={name:file_info(p) for name,p in paths.items()}, 'source changed during read'
        records.append({'date':date,'rows':len(src),'sample_rows':int(presence[si].sum()),
                        'files':before,'full_id_file_sha256':digest.hexdigest()})
        print(json.dumps(records[-1]),flush=True)
    assert presence[-1].all()
    assert np.all(presence[1:] >= presence[:-1]), 'nonmonotone membership: frozen task invalid'
    for si in range(len(DATES)):
        active=presence[si].astype(bool)
        assert np.array_equal(fingerprints[si,active],fingerprints[-1,active])
        assert np.array_equal(counts[si,active],counts[-1,active])
    pop=np.array([sum(int(w).bit_count() for w in row) for row in fingerprints[-1]],dtype=np.uint16)
    assert np.array_equal(pop,counts[-1])
    birth=presence.argmax(axis=0); order=np.lexsort((selected,birth))
    np.savez(a.output/'sample.npz',ids=selected[order],fp=fingerprints[-1,order],
             popcounts=pop[order],birth=birth[order],presence=presence[:,order])
    fingerprints[-1,order].astype('<u8').tofile(a.output/'fp_u64x4.bin')
    meta={'contract':'P3 fixed latest-snapshot minhash sample; all natural temporal deltas',
          'k':K,'seed':SEED,'chunk':CHUNK,'snapshot_dates':DATES,'snapshots':records,
          'sample_prefix_counts':presence.sum(axis=1).tolist(),
          'sample_delta_counts':np.diff(presence.sum(axis=1).astype(np.int64)).tolist(),
          'validation':'All sampled IDs unique; sampled historical fingerprints and popcounts byte-equal; monotone membership; recomputed popcounts equal.',
          'source_hash_scope':'Full ID files hashed; fingerprint and popcount files checked only at sampled positions, sizes and mtimes. No claim of full fingerprint-file hashing.',
          'host':platform.node(),'python':sys.version,'numpy':np.__version__,
          'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()}}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('VALIDATED',meta['sample_prefix_counts'],meta['sample_delta_counts'],flush=True)
if __name__=='__main__': main()

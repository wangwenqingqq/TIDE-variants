#!/usr/bin/env python3
from pathlib import Path
import csv,json,glob
E=Path('/workspace/RT-TIDE/experiments/large_batch_20260825')
def latest(pat): return Path(sorted(glob.glob(str(E/pat)))[-1])
rtp=latest('crossover_p1_*.jsonl'); cbp=latest('cublas_matrix_*.jsonl'); pp=latest('p_sweep_n1m_*.jsonl'); sp=latest('selectivity_rt_sweep_*.jsonl')
rt={(x['n'],x['q']):x for x in map(json.loads,rtp.read_text().splitlines())}
cb={(x['n'],x['q']):x for x in map(json.loads,cbp.read_text().splitlines())}
ps=list(map(json.loads,pp.read_text().splitlines()))
# Best measured P for 1M q32/64/128.
bestp={}
for x in ps:
 k=(x['n'],x['q'])
 if k not in bestp or x['result']['rt_plus_rerank_mean_ms']<bestp[k]['result']['rt_plus_rerank_mean_ms']: bestp[k]=x
rows=[]
for k in sorted(rt):
 x=rt[k]; r=x['result']; c=cb[k]['result']
 rtmean=r['rt_plus_rerank_mean_ms']; rtp50=r['rt_plus_rerank_p50_ms']; rtp95=r['rt_plus_rerank_p95_ms']; p=1
 if k in bestp:
  z=bestp[k]; rtmean=z['result']['rt_plus_rerank_mean_ms']; rtp50=z['result']['rt_plus_rerank_p50_ms']; rtp95=z['result']['rt_plus_rerank_p95_ms']; p=z['p']
 baselines={'direct':r['full_scan_mean_ms'],'tiled':r['tiled_scan_mean_ms'],'cublas':c['mean_ms']}
 kind=min(baselines,key=baselines.get); best=baselines[kind]
 rows.append({'n':k[0],'q':k[1],'candidate_ratio':r['candidate_ratio'],'rt_p':p,'rt_mean_ms':rtmean,'rt_p50_ms':rtp50,'rt_p95_ms':rtp95,
              'direct_ms':baselines['direct'],'tiled_ms':baselines['tiled'],'cublas_ms':baselines['cublas'],'cublas_p95_ms':c['p95_ms'],
              'best_scan':kind,'best_scan_ms':best,'rt_over_best_scan':rtmean/best,'best_scan_speedup_over_rt':rtmean/best,
              'threshold_count_rt':r['full_true_within_tau'],'threshold_count_cublas':c['true_within_tau'],
              'counts_equal':r['threshold_counts_equal'] and r['tiled_threshold_counts_equal'] and r['full_true_within_tau']==c['true_within_tau']})
with (E/'merged_crossover.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sel=list(map(json.loads,sp.read_text().splitlines()))
# Break-even interpolation between alpha=.25 and .1 for each q.
break_even={}
for q in [32,128]:
 a=next(x for x in sel if x['q']==q and abs(x['alpha']-.25)<1e-6)['result']
 b=next(x for x in sel if x['q']==q and abs(x['alpha']-.1)<1e-6)['result']
 target=cb[(1000000,q)]['result']['mean_ms']
 x1,y1=b['candidate_ratio'],b['rt_plus_rerank_mean_ms'];x2,y2=a['candidate_ratio'],a['rt_plus_rerank_mean_ms']
 cross=x1+(target-y1)*(x2-x1)/(y2-y1)
 break_even[str(q)]={'lower_ratio':x1,'lower_rt_ms':y1,'upper_ratio':x2,'upper_rt_ms':y2,'cublas_ms':target,'interpolated_candidate_ratio':cross}
summary={'sources':{'rt_matrix':str(rtp),'cublas_matrix':str(cbp),'p_sweep':str(pp),'selectivity_sweep':str(sp)},
         'all_35_threshold_counts_equal':all(x['counts_equal'] for x in rows),
         'all_large_batch_q_ge_32_rt_loses_to_best_scan':all(x['rt_over_best_scan']>1 for x in rows if x['q']>=32),
         'rows':rows,'best_parallelism_n1m':{str(k[1]):{'p':v['p'],'rt_ms':v['result']['rt_plus_rerank_mean_ms']} for k,v in bestp.items()},
         'selectivity_break_even':break_even,
         'topk_validation':json.loads((E/'cublas_n1m_q128_topk_validation_stable.json').read_text()),
         'decision':'NO-GO: large N and large q alone do not justify RT; exact PCA2 candidate ratio ~40% is far above the 4-6% empirical break-even against cuBLAS.'}
(E/'large_batch_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('sources',summary['sources'])
print('all counts equal',summary['all_35_threshold_counts_equal'])
print('N,Q,RT(ms),best-scan(ms),kind,RT/best,cand%,P')
for x in rows:
 if x['q']>=32: print('{},{},{:.4f},{:.4f},{},{:.2f}x,{:.2f}%,{}'.format(x['n'],x['q'],x['rt_mean_ms'],x['best_scan_ms'],x['best_scan'],x['rt_over_best_scan'],100*x['candidate_ratio'],x['rt_p']))
print('break-even',json.dumps(break_even,indent=2))

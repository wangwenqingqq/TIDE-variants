from pathlib import Path
p=Path('/workspace/RT-TIDE/experiments/bottleneck_20260824/fixed_probe_src/rt_tide_fused_probe.cu')
s=p.read_text()
if 'float p50;' not in s:
    s=s.replace('struct Stats { double mean; float minv; float p95; };','struct Stats { double mean; float minv; float p50; float p95; };',1)
    old='''  s.minv = v.front();
  s.p95 = v[static_cast<size_t>(std::ceil(0.95 * v.size())) - 1];
'''
    new='''  s.minv = v.front();
  s.p50 = v[static_cast<size_t>(std::ceil(0.50 * v.size())) - 1];
  s.p95 = v[static_cast<size_t>(std::ceil(0.95 * v.size())) - 1];
'''
    assert old in s
    s=s.replace(old,new,1)
    old='''      << ",\\"full_scan_mean_ms\\":"<<fs.mean<<",\\"full_scan_min_ms\\":"<<fs.minv
      << ",\\"full_scan_p95_ms\\":"<<fs.p95
'''
    new='''      << ",\\"full_scan_mean_ms\\":"<<fs.mean<<",\\"full_scan_min_ms\\":"<<fs.minv
      << ",\\"full_scan_p50_ms\\":"<<fs.p50<<",\\"full_scan_p95_ms\\":"<<fs.p95
'''
    assert old in s
    s=s.replace(old,new,1)
    old='''      << ",\\"tiled_scan_min_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.minv)
      << ",\\"tiled_scan_p95_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.p95)
'''
    new='''      << ",\\"tiled_scan_min_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.minv)
      << ",\\"tiled_scan_p50_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.p50)
      << ",\\"tiled_scan_p95_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.p95)
'''
    assert old in s
    s=s.replace(old,new,1)
    old='''      << ",\\"rt_plus_rerank_mean_ms\\":"<<rs.mean<<",\\"rt_plus_rerank_min_ms\\":"<<rs.minv
      << ",\\"rt_plus_rerank_p95_ms\\":"<<rs.p95
'''
    new='''      << ",\\"rt_plus_rerank_mean_ms\\":"<<rs.mean<<",\\"rt_plus_rerank_min_ms\\":"<<rs.minv
      << ",\\"rt_plus_rerank_p50_ms\\":"<<rs.p50<<",\\"rt_plus_rerank_p95_ms\\":"<<rs.p95
'''
    assert old in s
    s=s.replace(old,new,1)
    p.write_text(s)
print(p)

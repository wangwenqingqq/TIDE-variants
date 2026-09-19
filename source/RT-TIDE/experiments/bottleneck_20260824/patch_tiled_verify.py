from pathlib import Path
p=Path('/workspace/RT-TIDE/experiments/bottleneck_20260824/fixed_probe_src/rt_tide_fused_probe.cu')
s=p.read_text()
if 'tiled_counts_equal' not in s:
    old='''    auto fused_times = measure(stream.cuda_stream(), 3, repeats, fused_launch);
    full_launch();
    fused_launch();
    stream.Sync();

    std::vector<uint32_t> hc_full(nq), hc_cand(nq);
'''
    new='''    auto fused_times = measure(stream.cuda_stream(), 3, repeats, fused_launch);
    std::vector<uint32_t> hc_tiled(nq);
    if (!tiled_times.empty()) {
      tiled_launch();
      stream.Sync();
      check(cudaMemcpy(hc_tiled.data(),dfull_counts,nq*sizeof(uint32_t),cudaMemcpyDeviceToHost),"counts tiled");
    }
    full_launch();
    fused_launch();
    stream.Sync();

    std::vector<uint32_t> hc_full(nq), hc_cand(nq);
'''
    assert old in s
    s=s.replace(old,new,1)
    old='''    const bool counts_equal = hc_full == hc_cand;
'''
    new='''    const bool counts_equal = hc_full == hc_cand;
    const bool tiled_counts_equal = tiled_times.empty() || hc_full == hc_tiled;
'''
    assert old in s
    s=s.replace(old,new,1)
    old='''      << ",\\"threshold_counts_equal\\":"<<(counts_equal?"true":"false")
'''
    new=old+'''      << ",\\"tiled_threshold_counts_equal\\":"<<(tiled_counts_equal?"true":"false")
'''
    assert old in s
    s=s.replace(old,new,1)
    old='''    return counts_equal ? 0 : 3;
'''
    new='''    return (counts_equal && tiled_counts_equal) ? 0 : 3;
'''
    assert old in s
    s=s.replace(old,new,1)
    p.write_text(s)
print(p)

from pathlib import Path
p=Path('/workspace/RT-TIDE/experiments/bottleneck_20260824/fixed_probe_src/rt_tide_fused_probe.cu')
s=p.read_text()
if 'tiled_full_distance_kernel' in s:
    print('already patched')
else:
    needle='__global__ void queue_distance_kernel(const float* data, const float* queries,\n'
    kernel=r'''// Query-tiled exact scan: load each database vector once per block, then
// reuse it across all queries in the batch. Experiment-only fairness baseline.
__global__ void tiled_full_distance_kernel(const float* data, const float* queries,
                                           const float* tau2, float* out,
                                           uint32_t* counts, size_t n, int nq,
                                           int dim) {
  const size_t gi = static_cast<size_t>(blockIdx.x);
  if (gi >= n) return;
  extern __shared__ float sx[];
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    sx[d] = data[gi * static_cast<size_t>(dim) + d];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  for (int qi = warp; qi < nq; qi += nwarps) {
    float acc = 0.0f;
    for (int d = lane; d < dim; d += 32) {
      const float diff = sx[d] - queries[static_cast<size_t>(qi) * dim + d];
      acc = fmaf(diff, diff, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
      out[static_cast<size_t>(qi) * n + gi] = acc;
      if (acc <= tau2[qi]) atomicAdd(counts + qi, 1u);
    }
  }
}

'''
    assert needle in s
    s=s.replace(needle,kernel+needle,1)

    needle='    auto fused_launch = [&] {\n'
    block=r'''    auto tiled_launch = [&] {
      check(cudaMemsetAsync(dfull_counts, 0, nq*sizeof(uint32_t), stream.cuda_stream()), "clear tiled");
      tiled_full_distance_kernel<<<static_cast<unsigned int>(n),threads,
                                   static_cast<size_t>(dim)*sizeof(float),
                                   stream.cuda_stream()>>>(
          ddata,dqueries,dtau,dfull,dfull_counts,n,nq,dim);
    };
'''
    assert needle in s
    s=s.replace(needle,block+needle,1)

    old='    auto full_times = measure(stream.cuda_stream(), 3, repeats, full_launch);\n    auto fused_times = measure(stream.cuda_stream(), 3, repeats, fused_launch);\n'
    new='    auto full_times = measure(stream.cuda_stream(), 3, repeats, full_launch);\n    std::vector<float> tiled_times;\n    if (std::getenv("RTSPATIAL_MEASURE_TILED"))\n      tiled_times = measure(stream.cuda_stream(), 3, repeats, tiled_launch);\n    auto fused_times = measure(stream.cuda_stream(), 3, repeats, fused_launch);\n'
    assert old in s
    s=s.replace(old,new,1)

    old='    const auto fs=stats(full_times), rs=stats(fused_times);\n    std::cout << std::setprecision(10)\n'
    new='    const auto fs=stats(full_times), rs=stats(fused_times);\n    Stats ts{};\n    if (!tiled_times.empty()) ts=stats(tiled_times);\n    std::cout << std::setprecision(10)\n'
    assert old in s
    s=s.replace(old,new,1)

    old='      << ",\\"full_scan_p95_ms\\":"<<fs.p95\n'
    new=old + '      << ",\\"tiled_scan_mean_ms\\":"<<(tiled_times.empty() ? -1.0 : ts.mean)\n      << ",\\"tiled_scan_min_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.minv)\n      << ",\\"tiled_scan_p95_ms\\":"<<(tiled_times.empty() ? -1.0f : ts.p95)\n'
    assert old in s, repr(old)
    s=s.replace(old,new,1)
    p.write_text(s)
    print('patched',p)

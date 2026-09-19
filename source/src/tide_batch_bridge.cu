// Ordinary queued host batch. The included device kernel is byte-identical.
#define main p2_unused_main
#include "../keeper/tide_query_keeper.cu"
#undef main

namespace p2 {
constexpr unsigned MAX_Q=64;
thread_local std::string error;
struct Batch {
  MappedFile fp,ids,pc;
  HostDBView<4> host;
  Snapshot<4> snapshot;
  cudaStream_t stream=nullptr;
  cudaEvent_t start=nullptr,stop=nullptr;
  DeviceQuery<4>* dq=nullptr; DeviceQuery<4>* hq=nullptr;
  DeviceSlice* ds=nullptr; DeviceSlice* hs=nullptr;
  DeviceHit* dh=nullptr; DeviceHit* hh=nullptr;
  std::uint32_t* dm=nullptr; std::uint32_t* hm=nullptr;
  volatile std::uint64_t hash_sink=0;
  explicit Batch(const std::string& root):fp(root+"/fp_u64x4.bin"),ids(root+"/id_i64.bin"),
    pc(root+"/popcnt_u16.bin"),host(make_host_view<4>(fp,ids,pc)) {
    snapshot.runs.push_back({std::make_shared<DeviceRun<4>>(0,host),host});
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMalloc(&dq,MAX_Q*sizeof(*dq))); CUDA_CHECK(cudaMallocHost(&hq,MAX_Q*sizeof(*hq)));
    CUDA_CHECK(cudaMalloc(&ds,MAX_Q*sizeof(*ds))); CUDA_CHECK(cudaMallocHost(&hs,MAX_Q*sizeof(*hs)));
    CUDA_CHECK(cudaMalloc(&dh,MAX_Q*std::size_t(OUTPUT_CAPACITY)*sizeof(*dh)));
    CUDA_CHECK(cudaMallocHost(&hh,MAX_Q*std::size_t(OUTPUT_CAPACITY)*sizeof(*hh)));
    CUDA_CHECK(cudaMalloc(&dm,2*MAX_Q*sizeof(*dm))); CUDA_CHECK(cudaMallocHost(&hm,2*MAX_Q*sizeof(*hm)));
  }
  ~Batch() {
    cudaSetDevice(0); if(stream) cudaStreamSynchronize(stream);
    if(dq)cudaFree(dq);if(ds)cudaFree(ds);if(dh)cudaFree(dh);if(dm)cudaFree(dm);
    if(hq)cudaFreeHost(hq);if(hs)cudaFreeHost(hs);if(hh)cudaFreeHost(hh);if(hm)cudaFreeHost(hm);
    if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);if(stream)cudaStreamDestroy(stream);
  }
  void run(const std::uint64_t* packed,unsigned nq,unsigned num,unsigned den,
           std::uint64_t* out,std::uint64_t* counts) {
    if(nq!=1 && nq!=8 && nq!=64)throw std::runtime_error("unregistered batch");
    if(!num || num>den || den>65535)throw std::runtime_error("invalid threshold");
    CUDA_CHECK(cudaSetDevice(0)); std::array<unsigned,MAX_Q> blocks{};
    for(unsigned i=0;i<nq;++i) {
      const auto* p=packed+6*i; Query<4> q;q.id=p[0];q.pc=p[5];
      if(!q.pc)throw std::runtime_error("zero query outside contract");
      for(int w=0;w<4;++w)q.fp[w]=p[1+w];
      std::uint64_t candidate_rows=0;auto s=make_slices<4>(snapshot,q,num,den,true,&candidate_rows);
      hq[i]={};for(int w=0;w<4;++w)hq[i].fp[w]=q.fp[w];
      hq[i].popcount=q.pc;hq[i].threshold_num=num;hq[i].threshold_den=den;
      hs[i]={};counts[i]=0;
      if(s.empty())continue;
      if(s.size()!=1)throw std::runtime_error("one compacted run required");
      hs[i]=s[0]; blocks[i]=(s[0].rows+BLOCK-1)/BLOCK;
    }
    CUDA_CHECK(cudaMemcpyAsync(dq,hq,nq*sizeof(*dq),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(ds,hs,nq*sizeof(*ds),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemsetAsync(dm,0,2*MAX_Q*sizeof(*dm),stream));
    CUDA_CHECK(cudaEventRecord(start,stream));
    for(unsigned i=0;i<nq;++i)if(blocks[i]) {
      exact_runs_kernel<4><<<blocks[i],BLOCK,0,stream>>>(ds+i,1,dq+i,
        dh+i*std::size_t(OUTPUT_CAPACITY),OUTPUT_CAPACITY,dm+i,dm+MAX_Q+i);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop,stream));
    CUDA_CHECK(cudaMemcpyAsync(hm,dm,2*MAX_Q*sizeof(*dm),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); bool any=false;
    for(unsigned i=0;i<nq;++i) {
      if(hm[MAX_Q+i] || hm[i]>OUTPUT_CAPACITY)throw std::runtime_error("output overflow");
      if(hm[i]) {
        any=true; auto offset=i*std::size_t(OUTPUT_CAPACITY);
        CUDA_CHECK(cudaMemcpyAsync(hh+offset,dh+offset,hm[i]*sizeof(*hh),cudaMemcpyDeviceToHost,stream));
      }
    }
    if(any)CUDA_CHECK(cudaStreamSynchronize(stream));
    for(unsigned i=0;i<nq;++i) {
      std::vector<HostHit> hits;hits.reserve(hm[i]);auto offset=i*std::size_t(OUTPUT_CAPACITY);
      for(unsigned j=0;j<hm[i];++j) {const auto& h=hh[offset+j];hits.push_back({h.id,h.intersection,h.union_count});}
      sort_hits(hits);hash_sink=hash_hits(hits);
      for(const auto& h:hits)if(h.id!=packed[6*i])out[offset+counts[i]++]=h.id;
    }
  }
};
}
extern "C" const char* p2_error(){return p2::error.c_str();}
extern "C" void* p2_create(const char* root){try{return new p2::Batch(root);}catch(const std::exception& e){p2::error=e.what();return nullptr;}}
extern "C" void p2_destroy(void* h){delete static_cast<p2::Batch*>(h);}
extern "C" int p2_run(void* h,const std::uint64_t* p,unsigned n,unsigned num,unsigned den,std::uint64_t* out,std::uint64_t* counts){
  try{static_cast<p2::Batch*>(h)->run(p,n,num,den,out,counts);return 0;}
  catch(const std::exception& e){p2::error=e.what();return -1;}
}

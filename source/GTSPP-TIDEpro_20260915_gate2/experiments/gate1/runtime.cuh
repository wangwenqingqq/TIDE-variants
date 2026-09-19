// Gate 1 owns all GPU storage through one bounded arena. No Gate 0 timings reused.
#include "../../vendor/tide/frozen_query_runtime.cuh"
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <sys/resource.h>

namespace g1 {
using DB = OwnedHostDB<4>;
using Q = Query<4>;
using Clock = std::chrono::steady_clock;
const auto origin = Clock::now();
inline double now() { return std::chrono::duration<double, std::milli>(Clock::now()-origin).count(); }
inline void require(bool ok, const std::string &why) { if (!ok) throw std::runtime_error(why); }
constexpr std::size_t MiB = 1ull<<20;
constexpr int MAXQ = 64, MAXDESC = MAXQ*65;
inline std::size_t align256(std::size_t n) { return (n+255)&~std::size_t(255); }

struct Slice { DeviceSlice s; unsigned q; };
__global__ void batch_kernel(const Slice *slices, int ns, const DeviceQuery<4> *qs,
    DeviceHit *hits, unsigned capacity, unsigned *counts) {
  if (!ns) return; // also used for module prewarming without dereferencing pointers
  const std::uint64_t b = blockIdx.x;
  int lo=0, hi=ns;
  while(lo+1<hi) { int m=(lo+hi)/2; if(slices[m].s.first_block<=b) lo=m; else hi=m; }
  const Slice d=slices[lo];
  auto row=(b-d.s.first_block)*blockDim.x+threadIdx.x;
  if(row>=d.s.rows) return;
  const auto *q=qs+d.q;
  unsigned in=intersection_count<4>(q,d.s.fp+row*4);
  unsigned un=q->popcount+d.s.pc[row]-in;
  if(in*q->threshold_den < un*q->threshold_num) return;
  unsigned perq=atomicAdd(counts+2+d.q,1u);
  unsigned pos=atomicAdd(counts,1u);
  if(perq>=OUTPUT_CAPACITY || pos>=capacity) { atomicExch(counts+1,1u); return; }
  hits[pos]={d.s.ids[row],std::uint16_t(in),std::uint16_t(un),d.q};
}

struct Arena {
  unsigned char *base=nullptr;
  std::size_t capacity=0, cap=0, baseline=0, used=0, peak=0, observed_peak=0;
  std::map<std::size_t,std::size_t> free;
  std::mutex mutex;
  std::ofstream events;
  unsigned refused=0;
  Arena(std::size_t total_cap,const std::string &path):cap(total_cap),events(path) {
    require(bool(events),"cannot open arena ledger");
    events<<"time_ms,event,offset,bytes,live_bytes\n";
    CUDA_CHECK(cudaSetDevice(0));
    cudaFuncAttributes attr{}; CUDA_CHECK(cudaFuncGetAttributes(&attr,batch_kernel));
    batch_kernel<<<1,1>>>(nullptr,0,nullptr,nullptr,0,nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::size_t avail,total; CUDA_CHECK(cudaMemGetInfo(&avail,&total));
    baseline=total-avail;
    require(cap>baseline+128*MiB,"context leaves insufficient arena under registered total cap");
    capacity=((cap-baseline-64*MiB)/MiB)*MiB;
    CUDA_CHECK(cudaMalloc(&base,capacity));
    free[0]=capacity; observe();
  }
  ~Arena() { if(used) std::cerr<<"ERROR: arena leaked logical bytes="<<used<<'\n'; if(base) cudaFree(base); }
  std::size_t observe() {
    std::size_t avail,total; CUDA_CHECK(cudaMemGetInfo(&avail,&total));
    std::lock_guard<std::mutex> lock(mutex);
    observed_peak=std::max(observed_peak,total-avail);
    require(total-avail<=cap,"observed whole-card VRAM exceeds registered cap");
    return total-avail;
  }
  std::size_t alloc(std::size_t n) {
    n=align256(n); require(n>0,"zero arena allocation");
    std::lock_guard<std::mutex> lock(mutex);
    auto best=free.end();
    for(auto it=free.begin();it!=free.end();++it)
      if(it->second>=n && (best==free.end() || it->second<best->second)) best=it;
    if(best==free.end()) { refused++; throw std::bad_alloc(); }
    auto off=best->first, remaining=best->second-n; free.erase(best);
    if(remaining) free[off+n]=remaining;
    used+=n; peak=std::max(peak,used);
    events<<now()<<",alloc,"<<off<<','<<n<<','<<used<<'\n';
    return off;
  }
  void release(std::size_t off,std::size_t n) {
    n=align256(n);
    std::lock_guard<std::mutex> lock(mutex);
    used-=n; events<<now()<<",release,"<<off<<','<<n<<','<<used<<'\n';
    auto next=free.lower_bound(off);
    if(next!=free.end() && off+n==next->first) { n+=next->second; free.erase(next); }
    next=free.lower_bound(off);
    if(next!=free.begin()) { auto prev=std::prev(next); if(prev->first+prev->second==off) {
      off=prev->first; n+=prev->second; free.erase(prev);
    }}
    free.emplace(off,n);
  }
  std::size_t live() { std::lock_guard<std::mutex> lock(mutex); return used; }
  bool fits(std::size_t n) {
    std::lock_guard<std::mutex> lock(mutex);
    for(auto p:free) if(p.second>=align256(n)) return true;
    std::size_t largest=0;for(auto p:free) largest=std::max(largest,p.second);
    events<<now()<<",refuse,"<<largest<<','<<align256(n)<<','<<used<<'\n';
    refused++; return false;
  }
};
struct Buffer {
  Arena *arena=nullptr; std::size_t off=0,bytes=0;
  Buffer()=default;
  Buffer(Arena &a,std::size_t n):arena(&a),off(a.alloc(n)),bytes(align256(n)) {}
  ~Buffer() { if(arena) arena->release(off,bytes); }
  Buffer(const Buffer&)=delete;
  void *ptr() const { return arena->base+off; }
};
struct Release { std::atomic<double> at{0}; };
std::atomic<unsigned long> next_uid{1};
struct Run {
  std::shared_ptr<DB> host;
  HostDBView<4> metadata; // computed once, never scan pc in a request
  std::unique_ptr<Buffer> storage;
  std::shared_ptr<Release> released=std::make_shared<Release>();
  std::uint64_t uid=next_uid++;
  std::size_t bytes=0;
  std::uint64_t *fp=nullptr,*ids=nullptr; std::uint16_t *pc=nullptr;
  double metadata_ms=0,stage_ms=0,h2d_ms=0;
  static std::size_t footprint(std::size_t rows) {
    // Ordinary common size class, fixed 10.5M-row capacity, not a policy-specific optimization.
    // Equal large-block extents prevent the 9.46M base leaving a hole too small for a 10.28M merged run.
    constexpr std::size_t LARGE_ROWS=10500000;
    require(rows<=LARGE_ROWS,"registered run row capacity exceeded");
    return align256(rows*42 >= 256*MiB ? LARGE_ROWS*42 : rows*42);
  }
  Run(Arena &a,std::shared_ptr<DB> h,cudaStream_t stream,void *stage,int fault=0):host(std::move(h)) {
    if(fault==1) throw std::runtime_error("injected before allocation");
    double t=now(); metadata=host->view(); metadata_ms=now()-t;
    bytes=footprint(host->ids.size()); storage=std::make_unique<Buffer>(a,bytes);
    if(fault==2) throw std::runtime_error("injected after arena allocation");
    auto n=host->ids.size(); fp=static_cast<std::uint64_t*>(storage->ptr());
    ids=fp+4*n; pc=reinterpret_cast<std::uint16_t*>(ids+n);
    // Bounded pinned chunks avoid cudaMalloc / hidden pageable transfer staging during overlap.
    require(stage!=nullptr,"writer pinned staging must be preallocated");
    try {
      auto copy=[&](void *dst,const void *src,std::size_t size) {
        for(std::size_t o=0;o<size;o+=8*MiB) {
          auto len=std::min(8*MiB,size-o); double b=now();
          std::copy_n(static_cast<const char*>(src)+o,len,static_cast<char*>(stage)); stage_ms+=now()-b;
          b=now(); CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst)+o,stage,len,cudaMemcpyHostToDevice,stream));
          CUDA_CHECK(cudaStreamSynchronize(stream)); h2d_ms+=now()-b;
        }
      };
      copy(fp,host->fp.data(),n*32);
      if(fault==3) throw std::runtime_error("injected after partial H2D");
      copy(ids,host->ids.data(),n*8); copy(pc,host->pc.data(),n*2);
      if(fault==4) throw std::runtime_error("injected after full H2D");
      a.observe();
    } catch(...) { cudaStreamSynchronize(stream); throw; }
  }
  ~Run() { storage.reset(); released->at.store(now()); }
};
struct EpochInfo { std::atomic<double> last_reader_done{0}, owner_released{0}; std::atomic<int> readers{0}; };
struct Epoch {
  int number=0; std::vector<std::shared_ptr<Run>> runs;
  std::shared_ptr<EpochInfo> info=std::make_shared<EpochInfo>();
  ~Epoch() { runs.clear(); info->owner_released.store(now()); }
  std::size_t rows() const { std::size_t n=0; for(auto &r:runs) n+=r->host->ids.size(); return n; }
};
using EP=std::shared_ptr<Epoch>;
// One reader in this harness. Registration and publication share a short CPU lock,
// making the UID ownership classification an instantaneous, auditable snapshot.
std::mutex publication_mutex;
std::weak_ptr<Epoch> registered_reader;
struct Gate {
  std::mutex mutex; std::condition_variable cv; bool release=false;
  void open() { std::lock_guard<std::mutex> lock(mutex); release=true; cv.notify_all(); }
  static void CUDART_CB callback(void *p) {
    auto &g=*static_cast<Gate*>(p); std::unique_lock<std::mutex> lock(g.mutex);
    // Only CPU publication unlocks this; no dependency on any pending CUDA work.
    g.cv.wait(lock,[&]{return g.release;});
  }
};
struct Result {
  int epoch=0,batch=0,first=0,threshold=0,runs=0;
  std::size_t rows=0,candidates=0,blocks=0,descriptors=0,output_bytes=0;
  double begin=0,end=0,kernel_ms=0,device_complete=0; bool overflow=false;
  std::vector<std::vector<HostHit>> hits;
};
struct Runtime {
  cudaStream_t stream=nullptr; cudaEvent_t start=nullptr,stop=nullptr;
  Buffer workspace;
  DeviceQuery<4> *dq,*hq=nullptr; Slice *ds,*hs=nullptr;
  DeviceHit *dh,*hh=nullptr; unsigned *dc,*hc=nullptr;
  EP acquired; Gate *gate=nullptr; Result pending;
  static std::size_t bytes() { return align256(MAXQ*sizeof(DeviceQuery<4>))+align256(MAXDESC*sizeof(Slice))+
    align256(MAXQ*OUTPUT_CAPACITY*sizeof(DeviceHit))+align256((MAXQ+2)*sizeof(unsigned)); }
  Runtime(Arena &a):workspace(a,bytes()) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    auto p=static_cast<char*>(workspace.ptr()); dq=reinterpret_cast<DeviceQuery<4>*>(p);
    p+=align256(MAXQ*sizeof(DeviceQuery<4>)); ds=reinterpret_cast<Slice*>(p);
    p+=align256(MAXDESC*sizeof(Slice)); dh=reinterpret_cast<DeviceHit*>(p);
    p+=align256(MAXQ*OUTPUT_CAPACITY*sizeof(DeviceHit)); dc=reinterpret_cast<unsigned*>(p);
    CUDA_CHECK(cudaMallocHost(&hq,MAXQ*sizeof(DeviceQuery<4>)));
    CUDA_CHECK(cudaMallocHost(&hs,MAXDESC*sizeof(Slice)));
    CUDA_CHECK(cudaMallocHost(&hh,MAXQ*OUTPUT_CAPACITY*sizeof(DeviceHit)));
    CUDA_CHECK(cudaMallocHost(&hc,(MAXQ+2)*sizeof(unsigned)));
  }
  void cancel() {
    if(gate) {gate->open(); gate=nullptr;}
    if(stream) CUDA_CHECK(cudaStreamSynchronize(stream));
    if(acquired) {
      {std::lock_guard<std::mutex> lock(publication_mutex);
        acquired->info->last_reader_done.store(now()); acquired->info->readers--; registered_reader.reset();}
      acquired.reset();
    }
  }
  ~Runtime() {
    try { cancel(); } catch(...) { std::terminate(); }
    if(hq) cudaFreeHost(hq); if(hs) cudaFreeHost(hs); if(hh) cudaFreeHost(hh); if(hc) cudaFreeHost(hc);
    if(start) cudaEventDestroy(start); if(stop) cudaEventDestroy(stop); if(stream) cudaStreamDestroy(stream);
  }
  void begin(EP *slot,const QueryStore<4> &queries,int first,int qn,int threshold,Gate *g=nullptr,
      unsigned capacity_override=0) {
    require(!acquired,"runtime allows only one in-flight batch");
    pending=Result{}; pending.begin=now();
    {std::lock_guard<std::mutex> lock(publication_mutex); acquired=std::atomic_load(slot);
      require(bool(acquired),"no published epoch"); acquired->info->readers++; registered_reader=acquired;}
    require(qn>=1 && qn<=64,"bad batch size");
    pending.epoch=acquired->number; pending.rows=acquired->rows(); pending.runs=acquired->runs.size();
    pending.batch=qn; pending.first=first; pending.threshold=threshold;
    std::size_t nd=0,blocks=0;
    for(int j=0;j<qn;++j) {
      const auto &q=queries[(first+j)%queries.size()]; require(q.pc>0,"empty query unsupported");
      std::copy(q.fp.begin(),q.fp.end(),hq[j].fp); hq[j].popcount=q.pc;
      hq[j].threshold_num=threshold; hq[j].threshold_den=100; hq[j].reserved=0;
      for(const auto &r:acquired->runs) {
        auto range=bounds<4>(r->metadata,q,threshold,100); auto n=range.second-range.first; if(!n) continue;
        require(nd<MAXDESC,"descriptor capacity exceeded");
        hs[nd++]={{r->fp+range.first*4,r->ids+range.first,r->pc+range.first,n,blocks},unsigned(j)};
        blocks+=(n+BLOCK-1)/BLOCK; pending.candidates+=n;
      }
    }
    require(blocks<=std::numeric_limits<int>::max(),"grid exceeds limit");
    pending.blocks=blocks; pending.descriptors=nd;
    CUDA_CHECK(cudaMemcpyAsync(dq,hq,qn*sizeof(*hq),cudaMemcpyHostToDevice,stream));
    if(nd) CUDA_CHECK(cudaMemcpyAsync(ds,hs,nd*sizeof(*hs),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemsetAsync(dc,0,(MAXQ+2)*sizeof(unsigned),stream));
    gate=g; if(gate) CUDA_CHECK(cudaLaunchHostFunc(stream,Gate::callback,gate));
    CUDA_CHECK(cudaEventRecord(start,stream));
    if(blocks) batch_kernel<<<unsigned(blocks),BLOCK,0,stream>>>(ds,nd,dq,dh,
      capacity_override?capacity_override:qn*OUTPUT_CAPACITY,dc);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop,stream));
    CUDA_CHECK(cudaMemcpyAsync(hc,dc,(MAXQ+2)*sizeof(unsigned),cudaMemcpyDeviceToHost,stream));
  }
  Result finish() {
    require(bool(acquired),"no pending request");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    pending.overflow=hc[1]!=0; pending.hits.resize(pending.batch);
    if(!pending.overflow && hc[0]) {
      require(hc[0]<=MAXQ*OUTPUT_CAPACITY,"invalid append counter");
      CUDA_CHECK(cudaMemcpyAsync(hh,dh,hc[0]*sizeof(DeviceHit),cudaMemcpyDeviceToHost,stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    pending.device_complete=now(); acquired->info->last_reader_done.store(pending.device_complete);
    if(!pending.overflow) {
      pending.output_bytes=hc[0]*sizeof(DeviceHit);
      for(unsigned i=0;i<hc[0];++i) { auto h=hh[i]; require(h.reserved<unsigned(pending.batch),"bad query ID");
        pending.hits[h.reserved].push_back({h.id,h.intersection,h.union_count}); }
      for(auto &h:pending.hits) {sort_hits(h); (void)hash_hits(h);}
      for(int j=0;j<pending.batch;++j) require(pending.hits[j].size()==hc[j+2],"batch counter mismatch");
    }
    // Releasing the acquired owner is part of measured request completion, including last-owner destruction.
    {std::lock_guard<std::mutex> lock(publication_mutex);acquired->info->readers--;registered_reader.reset();}
    acquired.reset(); pending.end=now(); float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    pending.kernel_ms=ms; gate=nullptr; return std::move(pending);
  }
  Result run(EP *slot,const QueryStore<4> &qs,int first,int qn,int t) {
    try {begin(slot,qs,first,qn,t); return finish();} catch(...) {cancel(); throw;}
  }
};
} // namespace g1

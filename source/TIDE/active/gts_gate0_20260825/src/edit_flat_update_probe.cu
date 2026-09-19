#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  std::ostringstream os; os << #x << ": " << cudaGetErrorString(e); \
  throw std::runtime_error(os.str()); } } while(0)

namespace {
constexpr int MAXL = 40;
constexpr int P = 64;
constexpr int Q = 32;
constexpr int THREADS = 128;

struct Data {
  size_t n = 0;
  std::vector<char> bytes;
  std::vector<uint8_t> lens;
};

Data load_words(const char *path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("open words");
  int max_len=0, metric=0; size_t declared=0;
  in >> max_len >> declared >> metric;
  std::string line; std::getline(in,line);
  if (max_len>MAXL || metric!=6) throw std::runtime_error("bad header");
  Data d; d.bytes.reserve(declared*MAXL); d.lens.reserve(declared);
  while(std::getline(in,line)) {
    if(!line.empty() && line.back()=='\r') line.pop_back();
    if(line.size()>MAXL) throw std::runtime_error("long word");
    size_t at=d.bytes.size(); d.bytes.resize(at+MAXL,0);
    std::copy(line.begin(),line.end(),d.bytes.begin()+at);
    d.lens.push_back(static_cast<uint8_t>(line.size()));
  }
  d.n=d.lens.size();
  if(d.n!=declared) throw std::runtime_error("row mismatch");
  return d;
}

uint64_t splitmix(uint64_t &x) {
  uint64_t z=(x+=0x9e3779b97f4a7c15ULL);
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
  z=(z^(z>>27))*0x94d049bb133111ebULL;
  return z^(z>>31);
}

std::vector<int> ids(size_t lo,size_t hi,int count,uint64_t seed,
                     std::unordered_set<int>& used) {
  std::vector<int> v; uint64_t s=seed;
  while(static_cast<int>(v.size())<count) {
    int x=static_cast<int>(lo+splitmix(s)%(hi-lo));
    if(used.insert(x).second) v.push_back(x);
  }
  return v;
}

__device__ __forceinline__ uint8_t lev(const char *a,int n,const char *b,int m) {
  if(m>n) { const char*t=a;a=b;b=t; int z=n;n=m;m=z; }
  uint16_t x[MAXL+1],y[MAXL+1],*prev=x,*cur=y;
  for(int j=0;j<=m;++j) prev[j]=j;
  for(int i=1;i<=n;++i) {
    cur[0]=i;
    for(int j=1;j<=m;++j) {
      uint16_t u=prev[j]+1,v=cur[j-1]+1,w=prev[j-1]+(a[i-1]!=b[j-1]);
      uint16_t z=u<v?u:v; cur[j]=z<w?z:w;
    }
    uint16_t*t=prev;prev=cur;cur=t;
  }
  return static_cast<uint8_t>(prev[m]);
}

__global__ void sig_kernel(const char *data,const uint8_t*lens,size_t stride,
                           size_t begin,size_t count,const int*pivots,
                           uint8_t*sig) {
  size_t local=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  int p=blockIdx.y; if(local>=count||p>=P) return;
  size_t obj=begin+local; int pid=pivots[p];
  sig[static_cast<size_t>(p)*stride+obj]=lev(data+obj*MAXL,lens[obj],
      data+static_cast<size_t>(pid)*MAXL,lens[pid]);
}

__global__ void qp_kernel(const char*data,const uint8_t*lens,const int*qids,
                          const int*pivots,uint8_t*qp) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=Q*P) return;
  int q=i/P,p=i%P,qid=qids[q],pid=pivots[p];
  qp[i]=lev(data+static_cast<size_t>(qid)*MAXL,lens[qid],
            data+static_cast<size_t>(pid)*MAXL,lens[pid]);
}

__device__ __forceinline__ bool pass_bound(const uint8_t*sig,const uint8_t*qp,
                                            size_t n,size_t obj,int q,int r) {
  for(int p=0;p<P;++p) {
    int a=sig[static_cast<size_t>(p)*n+obj],b=qp[q*P+p];
    int z=a>b?a-b:b-a; if(z>r) return false;
  }
  return true;
}

__global__ void scan_flags(const char*data,const uint8_t*lens,size_t n,
                           const int*qids,int r,uint8_t*out) {
  size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  size_t total=n*Q; if(i>=total) return;
  int q=i/n; size_t obj=i-static_cast<size_t>(q)*n; int qid=qids[q];
  out[i]=lev(data+obj*MAXL,lens[obj],data+static_cast<size_t>(qid)*MAXL,
             lens[qid])<=r;
}

__global__ void filter_flags(const char*data,const uint8_t*lens,size_t n,
                             const int*qids,const uint8_t*sig,const uint8_t*qp,
                             int r,uint8_t*out) {
  size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  size_t total=n*Q; if(i>=total) return;
  int q=i/n; size_t obj=i-static_cast<size_t>(q)*n;
  if(!pass_bound(sig,qp,n,obj,q,r)) { out[i]=0; return; }
  int qid=qids[q];
  out[i]=lev(data+obj*MAXL,lens[obj],data+static_cast<size_t>(qid)*MAXL,
             lens[qid])<=r;
}

double pct(std::vector<double> v,double p) {
  std::sort(v.begin(),v.end()); size_t i=std::ceil(p*v.size())-1;
  return v[std::min(i,v.size()-1)];
}

template<class F> std::vector<double> bench(F f,int warm,int reps) {
  for(int i=0;i<warm;++i) { f(); CUDA_CHECK(cudaDeviceSynchronize()); }
  cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  std::vector<double> v;
  for(int i=0;i<reps;++i) { CUDA_CHECK(cudaEventRecord(a)); f();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); v.push_back(ms); }
  cudaEventDestroy(a);cudaEventDestroy(b); return v;
}

template<class T> void arr(std::ostream&o,const std::vector<T>&v) {
  o<<'['; for(size_t i=0;i<v.size();++i){if(i)o<<',';o<<v[i];} o<<']';
}
}

int main(int argc,char**argv) {
  try {
    if(argc!=5) throw std::runtime_error("usage: probe WORDS BASE_N OUT REPEATS");
    const char*path=argv[1]; size_t base=std::strtoull(argv[2],nullptr,10);
    const char*out_path=argv[3]; int reps=std::atoi(argv[4]);
    Data h=load_words(path); if(base==0||base>=h.n||reps<1) throw std::runtime_error("bad args");
    size_t delta=h.n-base;
    CUDA_CHECK(cudaSetDevice(0));
    char*d_data=nullptr;uint8_t*d_lens=nullptr,*d_sig=nullptr,*d_qp=nullptr,*d_scan=nullptr,*d_filter=nullptr;
    int*d_pivots=nullptr,*d_qids=nullptr;
    CUDA_CHECK(cudaMalloc(&d_data,h.bytes.size())); CUDA_CHECK(cudaMalloc(&d_lens,h.lens.size()));
    CUDA_CHECK(cudaMalloc(&d_sig,h.n*P)); CUDA_CHECK(cudaMalloc(&d_qp,Q*P));
    CUDA_CHECK(cudaMalloc(&d_scan,h.n*Q)); CUDA_CHECK(cudaMalloc(&d_filter,h.n*Q));
    CUDA_CHECK(cudaMalloc(&d_pivots,P*sizeof(int))); CUDA_CHECK(cudaMalloc(&d_qids,Q*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data,h.bytes.data(),base*MAXL,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lens,h.lens.data(),base,cudaMemcpyHostToDevice));
    std::unordered_set<int> used;
    auto piv=ids(0,base,P,20260825^0x11111111ULL,used);
    auto q_delta=ids(base,h.n,Q/2,20260825^0x55555555ULL,used);
    auto q_base=ids(0,base,Q/2,20260825^0x66666666ULL,used);
    std::vector<int> qids=q_delta; qids.insert(qids.end(),q_base.begin(),q_base.end());
    CUDA_CHECK(cudaMemcpy(d_pivots,piv.data(),P*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qids,qids.data(),Q*sizeof(int),cudaMemcpyHostToDevice));
    dim3 gb((base+THREADS-1)/THREADS,P);
    auto base_times=bench([&]{sig_kernel<<<gb,THREADS>>>(d_data,d_lens,h.n,0,base,d_pivots,d_sig);},1,reps);
    dim3 gd((delta+THREADS-1)/THREADS,P);
    auto append_times=bench([&]{
      CUDA_CHECK(cudaMemcpyAsync(d_data+base*MAXL,h.bytes.data()+base*MAXL,delta*MAXL,cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpyAsync(d_lens+base,h.lens.data()+base,delta,cudaMemcpyHostToDevice));
      sig_kernel<<<gd,THREADS>>>(d_data,d_lens,h.n,base,delta,d_pivots,d_sig);
    },1,reps);
    size_t total=h.n*Q;
    std::vector<int> radii={0,3,4};
    struct R {int r;std::vector<double>s,f;int mism=0;int visible=0;}; std::vector<R> rows;
    for(int r:radii) {
      auto st=bench([&]{scan_flags<<<(total+THREADS-1)/THREADS,THREADS>>>(d_data,d_lens,h.n,d_qids,r,d_scan);},3,reps);
      auto ft=bench([&]{qp_kernel<<<(Q*P+THREADS-1)/THREADS,THREADS>>>(d_data,d_lens,d_qids,d_pivots,d_qp);
        filter_flags<<<(total+THREADS-1)/THREADS,THREADS>>>(d_data,d_lens,h.n,d_qids,d_sig,d_qp,r,d_filter);},3,reps);
      qp_kernel<<<(Q*P+THREADS-1)/THREADS,THREADS>>>(d_data,d_lens,d_qids,d_pivots,d_qp);
      filter_flags<<<(total+THREADS-1)/THREADS,THREADS>>>(d_data,d_lens,h.n,d_qids,d_sig,d_qp,r,d_filter);
      scan_flags<<<(total+THREADS-1)/THREADS,THREADS>>>(d_data,d_lens,h.n,d_qids,r,d_scan);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint8_t>a(total),b(total); CUDA_CHECK(cudaMemcpy(a.data(),d_scan,total,cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(b.data(),d_filter,total,cudaMemcpyDeviceToHost));
      int mism=0; for(size_t i=0;i<total;++i)mism+=(a[i]!=b[i]);
      int visible=0; for(int q=0;q<Q/2;++q)visible+=b[static_cast<size_t>(q)*h.n+qids[q]]!=0;
      rows.push_back({r,std::move(st),std::move(ft),mism,visible});
    }
    int all_mism=0;for(auto&r:rows)all_mism+=r.mism;
    std::ofstream o(out_path);o<<std::setprecision(10);
    o<<"{\"schema\":\"gts-gate0-flat-update-v1\",\"n\":"<<h.n<<",\"base\":"<<base
     <<",\"delta\":"<<delta<<",\"pivots\":"<<P<<",\"query_ids\":";arr(o,qids);
    o<<",\"base_build_ms\":";arr(o,base_times);o<<",\"base_build_median_ms\":"<<pct(base_times,.5)
     <<",\"append_ms\":";arr(o,append_times);o<<",\"append_median_ms\":"<<pct(append_times,.5)
     <<",\"append_objects_per_second\":"<<(1000.0*delta/pct(append_times,.5))
     <<",\"total_flag_mismatches\":"<<all_mism<<",\"results\":[";
    for(size_t i=0;i<rows.size();++i){if(i)o<<',';auto&r=rows[i];o<<"{\"radius\":"<<r.r<<",\"scan_ms\":";arr(o,r.s);
      o<<",\"filter_ms\":";arr(o,r.f);o<<",\"scan_median_ms\":"<<pct(r.s,.5)
       <<",\"filter_median_ms\":"<<pct(r.f,.5)<<",\"speedup\":"<<pct(r.s,.5)/pct(r.f,.5)
       <<",\"flag_mismatches\":"<<r.mism<<",\"delta_self_visible\":"<<r.visible<<"}";}
    o<<"]}\n";o.close();
    std::cerr<<"base="<<base<<" delta="<<delta<<" append_ms="<<pct(append_times,.5)
             <<" mismatches="<<all_mism<<" delta_self_visible="<<rows[0].visible<<"/"<<Q/2<<"\n";
    return all_mism==0&&rows[0].visible==Q/2?0:3;
  } catch(const std::exception&e){std::cerr<<"fatal: "<<e.what()<<"\n";return 1;}
}


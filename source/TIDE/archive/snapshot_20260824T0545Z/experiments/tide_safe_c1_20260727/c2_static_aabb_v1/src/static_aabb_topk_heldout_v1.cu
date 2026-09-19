/*
 * Safe-C2 static-AABB held-out correctness v1.
 * Pre-registered standard SIFT1M input only. Static raw-float32 L2 tree only.
 * It compares radial traversal and full-coordinate AABB traversal against an
 * independent exact integer-squared CPU oracle. It records no timing and
 * supports no calibration, update, insertion, or rebuild.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define RP_DEFINE_CONSTANTS
#include "tree.cuh"
#include "search_static_aabb_v2_diskguard.cuh"

namespace {
constexpr int kDim=128;
constexpr int kBaseN=1000000;
constexpr int kK=10;
constexpr int kOrder=10;
constexpr double kEnvelopeRel=1.0e-4;
constexpr double kEnvelopeAbs=1.0e-4;

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("STATIC_AABB_HELDOUT_CORRECTNESS_V1: "+m); }
void check_cuda(cudaError_t s,const char* e,const char* f,int l) {
 if(s==cudaSuccess) return;
 std::ostringstream o; o<<e<<" failed at "<<f<<':'<<l<<": "<<cudaGetErrorString(s); fail(o.str());
}
#define SA_CUDA(x) ::check_cuda((x),#x,__FILE__,__LINE__)

struct Args {
 std::string base,queries,ids,outdir;
};
Args parse_args(int argc,char** argv) {
 Args a;
 for(int i=1;i<argc;++i) {
  const std::string k(argv[i]);
  auto val=[&](){if(i+1>=argc) fail("missing value for "+k);return std::string(argv[++i]);};
  if(k=="--base") a.base=val();
  else if(k=="--queries") a.queries=val();
  else if(k=="--ids") a.ids=val();
  else if(k=="--outdir") a.outdir=val();
  else fail("unknown argument "+k);
 }
 if(a.base.empty()||a.queries.empty()||a.ids.empty()||a.outdir.empty()) fail("required argument missing");
 return a;
}
std::vector<float> read_base(const std::string& p) {
 std::ifstream in(p,std::ios::binary); if(!in) fail("cannot open base");
 std::vector<float> x((size_t)kBaseN*kDim);
 for(int i=0;i<kBaseN;++i) {
  int d=0;in.read((char*)&d,4);if(!in||d!=kDim)fail("bad base fvec header");
  in.read((char*)x.data()+(size_t)i*kDim*sizeof(float),kDim*sizeof(float));if(!in)fail("short base fvec");
  for(int j=0;j<kDim;++j) {
   const float v=x[(size_t)i*kDim+j];
   if(!std::isfinite(v)||v<0.0f||v>255.0f||v!=truncf(v)) fail("base violates SIFT integer-float [0,255] contract");
  }
 }
 char c=0;if(in.read(&c,1))fail("unexpected extra base record");
 return x;
}
std::vector<int> read_ids(const std::string& p) {
 std::ifstream in(p);if(!in)fail("cannot open IDs");
 std::vector<int> v;int x=0;while(in>>x)v.push_back(x);
 if((int)v.size()!=64)fail("held-out correctness requires exactly 64 pre-registered IDs");
 if(!std::is_sorted(v.begin(),v.end())||std::adjacent_find(v.begin(),v.end())!=v.end())fail("IDs must be sorted/distinct");
 for(int id:v)if(id<0||id>=10000)fail("query ID outside standard SIFT range");
 return v;
}
std::vector<float> read_query_rows(const std::string& p,const std::vector<int>& ids) {
 std::ifstream in(p,std::ios::binary);if(!in)fail("cannot open queries");
 const std::streamoff bytes=4+(std::streamoff)kDim*4;
 std::vector<float> q((size_t)ids.size()*kDim);
 for(size_t i=0;i<ids.size();++i) {
  in.seekg((std::streamoff)ids[i]*bytes);int d=0;in.read((char*)&d,4);
  if(!in||d!=kDim)fail("bad query fvec header");
  in.read((char*)q.data()+i*kDim*sizeof(float),kDim*sizeof(float));if(!in)fail("short query fvec");
  for(int j=0;j<kDim;++j) {
   const float v=q[i*kDim+j];
   if(!std::isfinite(v)||v<0.0f||v>255.0f||v!=truncf(v)) fail("query violates SIFT integer-float [0,255] contract");
  }
 }
 return q;
}
std::vector<int> all_dimensions() {
 std::vector<int> out(kDim);
 for(int d=0; d<kDim; ++d) out[d]=d;
 return out;
}
float raw_l2_float(const float* a,const float* b) {
 volatile float s=0.0f;
 for(int d=0;d<kDim;++d){const float z=a[d]-b[d];const float t=z*z;s=s+t;}
 return sqrtf((float)s);
}
double raw_l2_double(const float* a,const float* b) {
 long double s=0;for(int d=0;d<kDim;++d){const long double z=(long double)a[d]-b[d];s+=z*z;}return (double)sqrt(s);
}
float outward_lower(double x) {
 if(!std::isfinite(x)||x<0)fail("invalid radial lower");
 const double y=std::max(0.0,x*(1.0-kEnvelopeRel)-kEnvelopeAbs);
 if(y==0)return 0.0f;
 const float r=nextafterf((float)y,-std::numeric_limits<float>::infinity());
 if((double)r>x)fail("unsafe radial lower rounding");return r;
}
float outward_upper(double x) {
 if(!std::isfinite(x)||x<0)fail("invalid radial upper");
 const float r=nextafterf((float)(x*(1.0+kEnvelopeRel)+kEnvelopeAbs),std::numeric_limits<float>::infinity());
 if((double)r<x)fail("unsafe radial upper rounding");return r;
}
struct Runtime {
 int* data_info=nullptr;float* data_d=nullptr;int* id_list=nullptr;TN* node_list=nullptr;int* max_node_num=nullptr;int* empty_list=nullptr;int height=0;
};
void release_runtime(Runtime& r) {
 if(r.id_list){cudaFree(r.id_list);r.id_list=nullptr;}if(r.node_list){cudaFree(r.node_list);r.node_list=nullptr;}
 if(r.max_node_num){cudaFree(r.max_node_num);r.max_node_num=nullptr;}if(r.empty_list){cudaFree(r.empty_list);r.empty_list=nullptr;}
 if(r.data_d){cudaFree(r.data_d);r.data_d=nullptr;}if(r.data_info){cudaFree(r.data_info);r.data_info=nullptr;}
 if(max_dis_d){cudaFree(max_dis_d);max_dis_d=nullptr;}dis_list=nullptr;split_list=nullptr;split_num=nullptr;pid_list=nullptr;
}
Runtime build_tree(const std::vector<float>& base) {
 Runtime r;SA_CUDA(cudaMallocManaged(&r.data_info,3*sizeof(int)));r.data_info[0]=kDim;r.data_info[1]=kBaseN;r.data_info[2]=2;
 SA_CUDA(cudaMalloc(&r.data_d,base.size()*sizeof(float)));SA_CUDA(cudaMemcpy(r.data_d,base.data(),base.size()*sizeof(float),cudaMemcpyHostToDevice));
 DIS_CODE=100;INFI_DIS=10000;indexConstru(r.data_d,nullptr,nullptr,r.data_info,r.id_list,r.node_list,r.max_node_num,r.height,r.empty_list);
 SA_CUDA(cudaDeviceSynchronize());SA_CUDA(cudaGetLastError());
 if(!r.id_list||!r.node_list||!r.max_node_num||!r.empty_list||r.height<=0||!max_dis_d)fail("tree build invalid");
 dis_list=nullptr;split_list=nullptr;split_num=nullptr;pid_list=nullptr;return r;
}
struct HostTree {
 std::vector<TN> nodes;std::vector<int> empty;std::vector<float> maxd;std::vector<int> ids;int count=0;int leaves=0;
};
HostTree capture_tree(const Runtime& r) {
 HostTree h;h.count=r.max_node_num[0];if(h.count<=1)fail("node count");
 h.nodes.resize(h.count);h.empty.resize(h.count);h.maxd.resize(h.count);
 SA_CUDA(cudaMemcpy(h.nodes.data(),r.node_list,h.nodes.size()*sizeof(TN),cudaMemcpyDeviceToHost));
 SA_CUDA(cudaMemcpy(h.empty.data(),r.empty_list,h.empty.size()*sizeof(int),cudaMemcpyDeviceToHost));
 SA_CUDA(cudaMemcpy(h.maxd.data(),max_dis_d,h.maxd.size()*sizeof(float),cudaMemcpyDeviceToHost));
 for(int n=1;n<h.count;++n)if(h.empty[n]==0&&h.nodes[n].is_leaf==1)++h.leaves;
 if(h.leaves<=0)fail("no leaves");
 h.ids.resize((size_t)kBaseN+(size_t)h.leaves*LEAF_PAD_SLOTS);
 SA_CUDA(cudaMemcpy(h.ids.data(),r.id_list,h.ids.size()*sizeof(int),cudaMemcpyDeviceToHost));
 return h;
}
void repair_radial_intervals(Runtime& r,const std::vector<float>& base) {
 HostTree h=capture_tree(r);std::vector<double> lo(h.count,std::numeric_limits<double>::infinity()),hi(h.count,0.0);std::vector<int> seen(h.count,0);
 for(int leaf=1;leaf<h.count;++leaf) {
  if(h.empty[leaf]!=0||h.nodes[leaf].is_leaf!=1)continue;const TN z=h.nodes[leaf];
  if(z.size<=0||z.lid<0||(size_t)z.lid+(size_t)z.size>h.ids.size())fail("bad padded leaf");
  for(int slot=0;slot<z.size;++slot){const int id=h.ids[(size_t)z.lid+slot];if(id<0||id>=kBaseN)fail("bad payload ID");int nid=leaf;
   while(nid!=0){if(h.empty[nid]!=0||h.nodes[nid].pid<0||h.nodes[nid].pid>=kBaseN)fail("bad ancestor");
    const double d=raw_l2_double(base.data()+(size_t)id*kDim,base.data()+(size_t)h.nodes[nid].pid*kDim);
    lo[nid]=std::min(lo[nid],d);hi[nid]=std::max(hi[nid],d);++seen[nid];nid=(nid-1)/kOrder;}
  }
 }
 for(int nid=1;nid<h.count;++nid)if(h.empty[nid]==0){
  if(h.nodes[nid].size<=0||seen[nid]!=h.nodes[nid].size||!std::isfinite(lo[nid])||!std::isfinite(hi[nid]))fail("radial membership reconstruction");
  h.nodes[nid].min_dis=outward_lower(lo[nid]);h.maxd[nid]=outward_upper(hi[nid]);
  if(h.nodes[nid].min_dis>h.maxd[nid])fail("invalid repaired interval");
 }
 SA_CUDA(cudaMemcpy(r.node_list,h.nodes.data(),h.nodes.size()*sizeof(TN),cudaMemcpyHostToDevice));
 SA_CUDA(cudaMemcpy(max_dis_d,h.maxd.data(),h.maxd.size()*sizeof(float),cudaMemcpyHostToDevice));
 SA_CUDA(cudaDeviceSynchronize());HostTree v=capture_tree(r);
 if(std::memcmp(v.nodes.data(),h.nodes.data(),h.nodes.size()*sizeof(TN))||std::memcmp(v.maxd.data(),h.maxd.data(),h.maxd.size()*sizeof(float)))fail("radial repair D2H mismatch");
}
struct Boxes {
 std::vector<int> dims;std::vector<float> lo,hi;std::vector<int> active;long long membership=0,cover_checks=0,cover_bad=0;
};
Boxes build_boxes(const HostTree& h,const std::vector<float>& base,std::vector<int> dims) {
 Boxes b;b.dims=std::move(dims);const int m=(int)b.dims.size();const size_t bytes=(size_t)h.count*m;
 b.lo.assign(bytes,std::numeric_limits<float>::infinity());b.hi.assign(bytes,-std::numeric_limits<float>::infinity());std::vector<int> seen(h.count,0);
 for(int leaf=1;leaf<h.count;++leaf) {
  if(h.empty[leaf]!=0||h.nodes[leaf].is_leaf!=1)continue;const TN z=h.nodes[leaf];
  for(int slot=0;slot<z.size;++slot){const int id=h.ids[(size_t)z.lid+slot];const float* x=base.data()+(size_t)id*kDim;int nid=leaf;
   while(nid!=0){++seen[nid];++b.membership;const size_t o=(size_t)nid*m;for(int j=0;j<m;++j){const float v=x[b.dims[j]];b.lo[o+j]=std::min(b.lo[o+j],v);b.hi[o+j]=std::max(b.hi[o+j],v);}nid=(nid-1)/kOrder;}
  }
 }
 for(int nid=1;nid<h.count;++nid)if(h.empty[nid]==0){if(seen[nid]!=h.nodes[nid].size||h.nodes[nid].size<=0)fail("AABB membership reconstruction");const size_t o=(size_t)nid*m;
  for(int j=0;j<m;++j){if(!std::isfinite(b.lo[o+j])||!std::isfinite(b.hi[o+j])||b.lo[o+j]>b.hi[o+j])fail("AABB invalid");b.lo[o+j]=nextafterf(b.lo[o+j],-std::numeric_limits<float>::infinity());b.hi[o+j]=nextafterf(b.hi[o+j],std::numeric_limits<float>::infinity());}b.active.push_back(nid);}
 for(int leaf=1;leaf<h.count;++leaf) {
  if(h.empty[leaf]!=0||h.nodes[leaf].is_leaf!=1)continue;const TN z=h.nodes[leaf];
  for(int slot=0;slot<z.size;++slot){const int id=h.ids[(size_t)z.lid+slot];const float* x=base.data()+(size_t)id*kDim;int nid=leaf;
   while(nid!=0){const size_t o=(size_t)nid*m;for(int j=0;j<m;++j){++b.cover_checks;const float v=x[b.dims[j]];if(!(b.lo[o+j]<=v&&v<=b.hi[o+j]))++b.cover_bad;}nid=(nid-1)/kOrder;}
  }
 }
 if(b.cover_bad)fail("AABB cover violation");return b;
}
struct DeviceBoxes {float* lo=nullptr;float* hi=nullptr;int* dims=nullptr;};
void release_boxes(DeviceBoxes& d){if(d.dims){cudaFree(d.dims);d.dims=nullptr;}if(d.hi){cudaFree(d.hi);d.hi=nullptr;}if(d.lo){cudaFree(d.lo);d.lo=nullptr;}}
DeviceBoxes upload_boxes(const Boxes& b) {
 DeviceBoxes d;const size_t bb=b.lo.size()*sizeof(float),db=b.dims.size()*sizeof(int);
 SA_CUDA(cudaMalloc(&d.lo,bb));SA_CUDA(cudaMalloc(&d.hi,bb));SA_CUDA(cudaMalloc(&d.dims,db));
 SA_CUDA(cudaMemcpy(d.lo,b.lo.data(),bb,cudaMemcpyHostToDevice));SA_CUDA(cudaMemcpy(d.hi,b.hi.data(),bb,cudaMemcpyHostToDevice));SA_CUDA(cudaMemcpy(d.dims,b.dims.data(),db,cudaMemcpyHostToDevice));return d;
}
struct Snapshot {
 HostTree tree;std::vector<float> lo,hi;std::vector<int> dims;
};
Snapshot capture_snapshot(const Runtime& r,const DeviceBoxes& d,const Boxes& b) {
 Snapshot s;s.tree=capture_tree(r);s.lo.resize(b.lo.size());s.hi.resize(b.hi.size());s.dims.resize(b.dims.size());
 SA_CUDA(cudaMemcpy(s.lo.data(),d.lo,s.lo.size()*sizeof(float),cudaMemcpyDeviceToHost));SA_CUDA(cudaMemcpy(s.hi.data(),d.hi,s.hi.size()*sizeof(float),cudaMemcpyDeviceToHost));SA_CUDA(cudaMemcpy(s.dims.data(),d.dims,s.dims.size()*sizeof(int),cudaMemcpyDeviceToHost));return s;
}
void same_snapshot(const Snapshot& a,const Snapshot& b) {
 if(a.tree.count!=b.tree.count||a.tree.ids.size()!=b.tree.ids.size()||a.lo.size()!=b.lo.size()||a.dims.size()!=b.dims.size())fail("snapshot size changed");
 if(std::memcmp(a.tree.nodes.data(),b.tree.nodes.data(),a.tree.nodes.size()*sizeof(TN))||std::memcmp(a.tree.empty.data(),b.tree.empty.data(),a.tree.empty.size()*sizeof(int))||
    std::memcmp(a.tree.maxd.data(),b.tree.maxd.data(),a.tree.maxd.size()*sizeof(float))||std::memcmp(a.tree.ids.data(),b.tree.ids.data(),a.tree.ids.size()*sizeof(int))||
    std::memcmp(a.lo.data(),b.lo.data(),a.lo.size()*sizeof(float))||std::memcmp(a.hi.data(),b.hi.data(),a.hi.size()*sizeof(float))||std::memcmp(a.dims.data(),b.dims.data(),a.dims.size()*sizeof(int)))fail("static snapshot byte mutation");
}
void write_bin(const std::string& p,const void* x,size_t n){std::ofstream f(p,std::ios::binary);if(!f)fail("write "+p);f.write((const char*)x,(std::streamsize)n);f.close();if(!f)fail("close "+p);}
void write_snapshot(const std::string& prefix,const Snapshot& s) {
 write_bin(prefix+"_nodes.bin",s.tree.nodes.data(),s.tree.nodes.size()*sizeof(TN));write_bin(prefix+"_empty.bin",s.tree.empty.data(),s.tree.empty.size()*sizeof(int));
 write_bin(prefix+"_maxd.bin",s.tree.maxd.data(),s.tree.maxd.size()*sizeof(float));write_bin(prefix+"_ids.bin",s.tree.ids.data(),s.tree.ids.size()*sizeof(int));
 write_bin(prefix+"_aabb_lo.bin",s.lo.data(),s.lo.size()*sizeof(float));write_bin(prefix+"_aabb_hi.bin",s.hi.data(),s.hi.size()*sizeof(float));write_bin(prefix+"_aabb_dims.bin",s.dims.data(),s.dims.size()*sizeof(int));
}
// Every SIFT coordinate was runtime-validated as an integer float in [0,255].
uint64_t raw_l2_sift_integer_sq(const float* a,const float* b) {
 uint64_t sum=0; for(int d=0;d<kDim;++d) { const int z=(int)a[d]-(int)b[d]; sum+=(uint64_t)(z*z); } return sum;
}
struct Neighbor{uint64_t dsq;int id;};
struct Worse{bool operator()(const Neighbor&a,const Neighbor&b)const{return a.dsq<b.dsq||(a.dsq==b.dsq&&a.id<b.id);}};
std::vector<std::array<int,kK>> exact_oracle(const std::vector<float>& base,const std::vector<float>& q) {
 const int nq=(int)(q.size()/kDim);std::vector<std::array<int,kK>> out(nq);
 #pragma omp parallel for schedule(static)
 for(int qi=0;qi<nq;++qi){std::priority_queue<Neighbor,std::vector<Neighbor>,Worse> h;const float* z=q.data()+(size_t)qi*kDim;
  for(int id=0;id<kBaseN;++id){Neighbor n{raw_l2_sift_integer_sq(z,base.data()+(size_t)id*kDim),id};if((int)h.size()<kK)h.push(n);else {const Neighbor w=h.top();if(n.dsq<w.dsq||(n.dsq==w.dsq&&n.id<w.id)){h.pop();h.push(n);}}}
  std::vector<Neighbor> x;while(!h.empty()){x.push_back(h.top());h.pop();}std::sort(x.begin(),x.end(),[](const Neighbor&a,const Neighbor&b){return a.dsq<b.dsq||(a.dsq==b.dsq&&a.id<b.id);});
  for(int j=0;j<kK;++j)out[qi][j]=x[j].id;
 }return out;
}
struct Eval{int invalid=0,duplicate=0,set_mismatch=0,distance_mismatch=0;};
Eval evaluate(const std::vector<float>& base,const std::vector<float>& q,const std::vector<std::array<int,kK>>& oracle,const std::vector<int>& ids,const std::vector<float>& dis) {
 Eval e;const int nq=(int)oracle.size();
 for(int qi=0;qi<nq;++qi){std::array<int,kK> got{};for(int j=0;j<kK;++j){const int id=ids[(size_t)qi*kK+j];got[j]=id;if(id<0||id>=kBaseN)++e.invalid;}
  auto sorted=got;std::sort(sorted.begin(),sorted.end());for(int j=1;j<kK;++j)if(sorted[j]==sorted[j-1])++e.duplicate;
  auto want=oracle[qi];std::sort(want.begin(),want.end());if(sorted!=want)++e.set_mismatch;
  const float* z=q.data()+(size_t)qi*kDim;for(int j=0;j<kK;++j){const int id=got[j];if(id<0||id>=kBaseN)continue;const float ref=raw_l2_float(z,base.data()+(size_t)id*kDim);const float gotd=dis[(size_t)qi*kK+j];if(!std::isfinite(gotd)||std::fabs(ref-gotd)>2.0e-3f)++e.distance_mismatch;}
 }return e;
}
void release_res(){if(res_dis){SA_CUDA(cudaFree(res_dis));res_dis=nullptr;}}
void run_baseline(const Runtime&r,float*q_d,int*out_d,int nq){while(!st.empty())st.pop();update_disk=false;searchIndexKnnV2(r.data_d,r.node_list,r.id_list,r.max_node_num,q_d,out_d,nq,kK,r.height,r.data_info,r.empty_list,nullptr,nullptr);}
void run_aabb(const Runtime&r,float*q_d,int*out_d,int nq,StaticProjectedAabbDeviceView v){searchIndexKnnStaticAabbV2DiskGuard(r.data_d,r.node_list,r.id_list,r.max_node_num,q_d,out_d,nq,kK,r.height,r.data_info,r.empty_list,nullptr,nullptr,v);}
void copy_outputs(int* d,int nq,std::vector<int>& ids,std::vector<float>& dis) {
 ids.resize((size_t)nq*kK);dis.resize((size_t)nq*kK);SA_CUDA(cudaMemcpy(ids.data(),d,ids.size()*sizeof(int),cudaMemcpyDeviceToHost));if(!res_dis)fail("missing res_dis");SA_CUDA(cudaMemcpy(dis.data(),res_dis,dis.size()*sizeof(float),cudaMemcpyDeviceToHost));
}
void write_result(const std::string& out,const Boxes& b,const HostTree& tree,const Eval& base,const Eval& cand,bool same,const std::vector<int>& qids) {
 std::ofstream f(out+"/result.json");if(!f)fail("result");
 auto emit=[&](const Eval&e){f<<"{\"invalid\":"<<e.invalid<<",\"duplicate\":"<<e.duplicate<<",\"oracle_set_mismatch\":"<<e.set_mismatch<<",\"distance_mismatch\":"<<e.distance_mismatch<<"}";};
 const bool ok=base.invalid==0&&base.duplicate==0&&base.set_mismatch==0&&base.distance_mismatch==0&&cand.invalid==0&&cand.duplicate==0&&cand.set_mismatch==0&&cand.distance_mismatch==0&&same&&b.cover_bad==0;
 f<<"{\"schema\":\"safe-c2-static-aabb-heldout-correctness-v1\",\"status\":\""<<(ok?"PASS_STATIC_AABB_HELDOUT_CORRECTNESS_V1":"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_V1")<<"\",\"scope\":\"pre-registered standard-SIFT held-out static-tree top-K correctness; raw-float32 L2 only; no timing, calibration, update, insertion, rebuild, validation, sealed, or general-metric claim\",\"gpu_executed\":true,\"formal_claim_eligible\":false,";
 f<<"\"projection\":{\"policy\":\"all raw SIFT coordinates in ascending order; no learned/query/GT/calibration selection\",\"dimensions\":"<<b.dims.size()<<",\"ids\":[";for(size_t i=0;i<b.dims.size();++i){if(i)f<<',';f<<b.dims[i];}f<<"]},";
 f<<"\"tree\":{\"node_count\":"<<tree.count<<",\"active_nonroot_nodes\":"<<b.active.size()<<"},\"host_cover\":{\"membership_observations\":"<<b.membership<<",\"coordinate_checks\":"<<b.cover_checks<<",\"violations\":"<<b.cover_bad<<"},";
 f<<"\"queries\":{\"count\":"<<qids.size()<<",\"ids\":[";for(size_t i=0;i<qids.size();++i){if(i)f<<',';f<<qids[i];}f<<"]},\"baseline\":";emit(base);f<<",\"static_aabb\":";emit(cand);f<<",\"baseline_and_static_aabb_id_sets_equal\":"<<(same?"true":"false")<<",\"snapshot_pre_post_byte_equal\":true}";f.close();if(!f)fail("result close");
}
} // namespace
int main(int argc,char**argv) {
 Runtime r;DeviceBoxes db;float* q_d=nullptr;int* out_d=nullptr;
 try {
  const Args a=parse_args(argc,argv);const auto base=read_base(a.base);const auto qids=read_ids(a.ids);const auto q=read_query_rows(a.queries,qids);const auto dims=all_dimensions();
  const auto oracle=exact_oracle(base,q);
  r=build_tree(base);repair_radial_intervals(r,base);const HostTree tree=capture_tree(r);const Boxes boxes=build_boxes(tree,base,dims);db=upload_boxes(boxes);
  const Snapshot frozen=capture_snapshot(r,db,boxes);write_snapshot(a.outdir+"/snapshot_before",frozen);
  SA_CUDA(cudaMalloc(&q_d,q.size()*sizeof(float)));SA_CUDA(cudaMemcpy(q_d,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice));SA_CUDA(cudaMalloc(&out_d,(size_t)qids.size()*kK*sizeof(int)));
  same_snapshot(frozen,capture_snapshot(r,db,boxes));run_baseline(r,q_d,out_d,(int)qids.size());std::vector<int> bid,cid;std::vector<float> bdis,cdis;copy_outputs(out_d,(int)qids.size(),bid,bdis);const Eval be=evaluate(base,q,oracle,bid,bdis);release_res();same_snapshot(frozen,capture_snapshot(r,db,boxes));
  StaticProjectedAabbDeviceView view{db.lo,db.hi,db.dims,(int)boxes.dims.size(),kDim};run_aabb(r,q_d,out_d,(int)qids.size(),view);copy_outputs(out_d,(int)qids.size(),cid,cdis);const Eval ce=evaluate(base,q,oracle,cid,cdis);release_res();const Snapshot final=capture_snapshot(r,db,boxes);same_snapshot(frozen,final);write_snapshot(a.outdir+"/snapshot_after",final);
  bool same=true;for(size_t qi=0;qi<qids.size();++qi){std::array<int,kK>x{},y{};for(int j=0;j<kK;++j){x[j]=bid[qi*kK+j];y[j]=cid[qi*kK+j];}std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());if(x!=y)same=false;}
  write_result(a.outdir,boxes,tree,be,ce,same,qids);
  if(out_d){SA_CUDA(cudaFree(out_d));out_d=nullptr;}if(q_d){SA_CUDA(cudaFree(q_d));q_d=nullptr;}release_boxes(db);release_runtime(r);return 0;
 } catch(const std::exception&e) {
  std::cerr<<"FAIL_STATIC_AABB_HELDOUT_CORRECTNESS_V1: "<<e.what()<<std::endl;if(out_d)cudaFree(out_d);if(q_d)cudaFree(q_d);release_res();release_boxes(db);release_runtime(r);return 2;
 }
}

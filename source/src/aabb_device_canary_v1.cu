/*
 * Static Safe-C2-AABB v1 CUDA certificate canary.
 * Scope: build the archived static GTS tree, reconstruct node projected AABBs
 * from padded leaf membership, and validate device directed-rounding lower bounds.
 * It deliberately does not call any GTS traversal, top-K evaluation, update,
 * insertion, gamma, fallback, calibration, validation, or sealed workload.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tree.cuh"
#include "aabb_static_bound_v1.cuh"

namespace {
constexpr int kDim = 128;
constexpr int kBaseN = 1000000;
constexpr int kOrder = 10;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("STATIC_AABB_CANARY: " + message);
}
void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream out;
  out << expr << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
  fail(out.str());
}
#define AABB_CUDA(expr) ::cuda_check((expr), #expr, __FILE__, __LINE__)

struct Args {
  std::string base;
  std::string queries;
  std::string outdir;
  int projection_dims = 64;
  int query_start = 0;
  int query_count = 8;
};
Args parse(int argc, char** argv) {
  Args a;
  for (int i=1; i<argc; ++i) {
    const std::string key(argv[i]);
    auto next=[&]() -> std::string {
      if (i+1 >= argc) fail("missing value for " + key);
      return std::string(argv[++i]);
    };
    if (key=="--base") a.base=next();
    else if (key=="--queries") a.queries=next();
    else if (key=="--outdir") a.outdir=next();
    else if (key=="--projection-dims") a.projection_dims=std::stoi(next());
    else if (key=="--query-start") a.query_start=std::stoi(next());
    else if (key=="--query-count") a.query_count=std::stoi(next());
    else fail("unknown argument " + key);
  }
  if (a.base.empty() || a.queries.empty() || a.outdir.empty()) fail("required arguments missing");
  if (a.projection_dims <= 0 || a.projection_dims > kDim) fail("projection dims outside 1..128");
  if (a.query_start < 0 || a.query_count <= 0 || a.query_start > 1000000-a.query_count) fail("invalid query range");
  return a;
}
std::vector<float> read_fvecs_all(const std::string& path, int expected_count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open base fvecs");
  std::vector<float> out(static_cast<size_t>(expected_count)*kDim);
  for (int i=0; i<expected_count; ++i) {
    int d=0; in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!in || d!=kDim) fail("invalid base fvec header at row " + std::to_string(i));
    in.read(reinterpret_cast<char*>(out.data()+static_cast<size_t>(i)*kDim), sizeof(float)*kDim);
    if (!in) fail("short base fvec payload at row " + std::to_string(i));
    for (int j=0; j<kDim; ++j) {
      if (!std::isfinite(out[static_cast<size_t>(i)*kDim+j])) fail("nonfinite base coordinate");
    }
  }
  char extra=0;
  if (in.read(&extra,1)) fail("base fvec has unexpected extra record");
  return out;
}
std::vector<float> read_fvec_rows(const std::string& path, int start, int count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open query fvecs");
  const std::streamoff bytes=sizeof(int)+static_cast<std::streamoff>(kDim)*sizeof(float);
  std::vector<float> out(static_cast<size_t>(count)*kDim);
  for (int i=0; i<count; ++i) {
    in.seekg(static_cast<std::streamoff>(start+i)*bytes);
    int d=0; in.read(reinterpret_cast<char*>(&d),sizeof(d));
    if (!in || d!=kDim) fail("invalid query fvec header");
    in.read(reinterpret_cast<char*>(out.data()+static_cast<size_t>(i)*kDim),sizeof(float)*kDim);
    if (!in) fail("short query fvec payload");
    for (int j=0; j<kDim; ++j) {
      if (!std::isfinite(out[static_cast<size_t>(i)*kDim+j])) fail("nonfinite query coordinate");
    }
  }
  return out;
}
std::vector<int> top_variance_dims(const std::vector<float>& base, int m) {
  std::vector<long double> sum(kDim,0.0L), sumsq(kDim,0.0L);
  for (int i=0; i<kBaseN; ++i) {
    const float* x=base.data()+static_cast<size_t>(i)*kDim;
    for (int d=0; d<kDim; ++d) {
      const long double v=static_cast<long double>(x[d]);
      sum[d]+=v; sumsq[d]+=v*v;
    }
  }
  struct Item { long double variance; int d; };
  std::vector<Item> items; items.reserve(kDim);
  const long double n=static_cast<long double>(kBaseN);
  for (int d=0; d<kDim; ++d) {
    long double v=sumsq[d]/n-(sum[d]/n)*(sum[d]/n);
    if (v<0.0L && v>-1.0e-16L) v=0.0L;
    if (!(v>=0.0L) || !std::isfinite(static_cast<double>(v))) fail("invalid base-only variance");
    items.push_back({v,d});
  }
  std::sort(items.begin(),items.end(),[](const Item& a,const Item& b) {
    return a.variance>b.variance || (a.variance==b.variance && a.d<b.d);
  });
  std::vector<int> dims; dims.reserve(m);
  for (int i=0; i<m; ++i) dims.push_back(items[static_cast<size_t>(i)].d);
  return dims;
}
struct HostTree {
  std::vector<TN> nodes;
  std::vector<int> empty;
  std::vector<int> ids;
  int node_count=0;
  int tree_height=0;
};
HostTree build_archived_tree(const std::vector<float>& base) {
  HostTree host;
  int* data_info=nullptr;
  float* data_d=nullptr;
  int* id_list=nullptr;
  TN* node_list=nullptr;
  int* max_node_num=nullptr;
  int* empty_list=nullptr;
  int tree_height=0;
  AABB_CUDA(cudaMallocManaged(&data_info,3*sizeof(int)));
  data_info[0]=kDim; data_info[1]=kBaseN; data_info[2]=2;
  AABB_CUDA(cudaMalloc(&data_d,base.size()*sizeof(float)));
  AABB_CUDA(cudaMemcpy(data_d,base.data(),base.size()*sizeof(float),cudaMemcpyHostToDevice));
  DIS_CODE=100; INFI_DIS=10000;
  indexConstru(data_d,nullptr,nullptr,data_info,id_list,node_list,max_node_num,tree_height,empty_list);
  AABB_CUDA(cudaDeviceSynchronize());
  AABB_CUDA(cudaGetLastError());
  if (!id_list || !node_list || !max_node_num || !empty_list || tree_height<=0) fail("archived tree build returned null");
  host.node_count=max_node_num[0];
  host.tree_height=tree_height;
  if (host.node_count<=1) fail("invalid archived tree node count");
  host.nodes.resize(host.node_count);
  host.empty.resize(host.node_count);
  AABB_CUDA(cudaMemcpy(host.nodes.data(),node_list,host.nodes.size()*sizeof(TN),cudaMemcpyDeviceToHost));
  AABB_CUDA(cudaMemcpy(host.empty.data(),empty_list,host.empty.size()*sizeof(int),cudaMemcpyDeviceToHost));
  int leaves=0;
  for (int nid=1; nid<host.node_count; ++nid) {
    if (host.empty[nid]==0 && host.nodes[nid].is_leaf==1) ++leaves;
  }
  if (leaves<=0) fail("no active leaves after tree build");
  const size_t capacity=static_cast<size_t>(kBaseN)+static_cast<size_t>(leaves)*static_cast<size_t>(LEAF_PAD_SLOTS);
  host.ids.resize(capacity);
  AABB_CUDA(cudaMemcpy(host.ids.data(),id_list,capacity*sizeof(int),cudaMemcpyDeviceToHost));
  // Nothing from this archived constructor is retained in the canary after host capture.
  AABB_CUDA(cudaFree(id_list)); AABB_CUDA(cudaFree(node_list)); AABB_CUDA(cudaFree(max_node_num));
  AABB_CUDA(cudaFree(empty_list)); AABB_CUDA(cudaFree(data_d)); AABB_CUDA(cudaFree(data_info));
  if (max_dis_d) { AABB_CUDA(cudaFree(max_dis_d)); max_dis_d=nullptr; }
  dis_list=nullptr; split_list=nullptr; split_num=nullptr; pid_list=nullptr;
  return host;
}
struct Boxes {
  std::vector<int> dims;
  std::vector<float> lo;
  std::vector<float> hi;
  std::vector<int> active_nids;
  long long membership_observations=0;
  long long coordinate_cover_checks=0;
  long long coordinate_cover_violations=0;
};
Boxes build_and_verify_boxes(const HostTree& tree, const std::vector<float>& base, std::vector<int> dims) {
  const int m=static_cast<int>(dims.size());
  Boxes b; b.dims=std::move(dims);
  const size_t table=static_cast<size_t>(tree.node_count)*static_cast<size_t>(m);
  b.lo.assign(table,std::numeric_limits<float>::infinity());
  b.hi.assign(table,-std::numeric_limits<float>::infinity());
  std::vector<int> observed(static_cast<size_t>(tree.node_count),0);
  for (int leaf=1; leaf<tree.node_count; ++leaf) {
    if (tree.empty[leaf]!=0 || tree.nodes[leaf].is_leaf!=1) continue;
    const TN n=tree.nodes[leaf];
    if (n.size<=0 || n.lid<0 || static_cast<size_t>(n.lid)+static_cast<size_t>(n.size)>tree.ids.size()) fail("invalid padded leaf span");
    for (int slot=0; slot<n.size; ++slot) {
      const int id=tree.ids[static_cast<size_t>(n.lid)+static_cast<size_t>(slot)];
      if (id<0 || id>=kBaseN) fail("invalid id in padded leaf");
      int nid=leaf;
      while (nid!=0) {
        if (tree.empty[nid]!=0) fail("nonempty leaf reaches empty ancestor");
        ++observed[static_cast<size_t>(nid)]; ++b.membership_observations;
        const size_t off=static_cast<size_t>(nid)*static_cast<size_t>(m);
        const float* x=base.data()+static_cast<size_t>(id)*kDim;
        for (int j=0; j<m; ++j) {
          const float v=x[b.dims[static_cast<size_t>(j)]];
          b.lo[off+static_cast<size_t>(j)]=std::min(b.lo[off+static_cast<size_t>(j)],v);
          b.hi[off+static_cast<size_t>(j)]=std::max(b.hi[off+static_cast<size_t>(j)],v);
        }
        nid=(nid-1)/kOrder;
      }
    }
  }
  for (int nid=1; nid<tree.node_count; ++nid) {
    if (tree.empty[nid]!=0) continue;
    const TN n=tree.nodes[nid];
    if (n.size<=0 || observed[static_cast<size_t>(nid)]!=n.size) fail("padded leaf reconstruction disagrees with node size");
    const size_t off=static_cast<size_t>(nid)*static_cast<size_t>(m);
    for (int j=0; j<m; ++j) {
      const float lo=b.lo[off+static_cast<size_t>(j)], hi=b.hi[off+static_cast<size_t>(j)];
      if (!std::isfinite(lo)||!std::isfinite(hi)||lo>hi) fail("invalid raw coordinate cover");
      b.lo[off+static_cast<size_t>(j)]=std::nextafterf(lo,-std::numeric_limits<float>::infinity());
      b.hi[off+static_cast<size_t>(j)]=std::nextafterf(hi,std::numeric_limits<float>::infinity());
    }
    b.active_nids.push_back(nid);
  }
  // Re-walk exactly the actual padded memberships and check every stored leaf and ancestor cover.
  for (int leaf=1; leaf<tree.node_count; ++leaf) {
    if (tree.empty[leaf]!=0 || tree.nodes[leaf].is_leaf!=1) continue;
    const TN n=tree.nodes[leaf];
    for (int slot=0; slot<n.size; ++slot) {
      const int id=tree.ids[static_cast<size_t>(n.lid)+static_cast<size_t>(slot)];
      const float* x=base.data()+static_cast<size_t>(id)*kDim;
      int nid=leaf;
      while (nid!=0) {
        const size_t off=static_cast<size_t>(nid)*static_cast<size_t>(m);
        for (int j=0; j<m; ++j) {
          const float v=x[b.dims[static_cast<size_t>(j)]];
          ++b.coordinate_cover_checks;
          if (!(b.lo[off+static_cast<size_t>(j)]<=v && v<=b.hi[off+static_cast<size_t>(j)])) ++b.coordinate_cover_violations;
        }
        nid=(nid-1)/kOrder;
      }
    }
  }
  if (b.coordinate_cover_violations!=0) fail("coordinate AABB cover violation");
  return b;
}
void write_bytes(const std::string& path,const void* p,size_t bytes) {
  std::ofstream out(path,std::ios::binary);
  if (!out) fail("cannot write " + path);
  out.write(static_cast<const char*>(p),static_cast<std::streamsize>(bytes));
  out.close();
  if (!out) fail("failed writing " + path);
}
__global__ void aabb_bound_canary_kernel(
    StaticProjectedAabbDeviceView view,const float* queries,const int* active_nids,
    float* output,int active_count,int query_count) {
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=active_count*query_count;
  if (idx>=total) return;
  const int qid=idx/active_count;
  const int nid=active_nids[idx-qid*active_count];
  output[idx]=static_aabb_lb_sq_rd(view,nid,queries+static_cast<size_t>(qid)*kDim);
}
long double host_lb_sq(const Boxes& b,int nid,const float* q) {
  const int m=static_cast<int>(b.dims.size());
  const size_t off=static_cast<size_t>(nid)*static_cast<size_t>(m);
  long double sum=0.0L;
  for (int j=0; j<m; ++j) {
    const int d=b.dims[static_cast<size_t>(j)];
    long double delta=0.0L;
    if (q[d]<b.lo[off+static_cast<size_t>(j)]) delta=static_cast<long double>(b.lo[off+static_cast<size_t>(j)])-q[d];
    else if (q[d]>b.hi[off+static_cast<size_t>(j)]) delta=static_cast<long double>(q[d])-b.hi[off+static_cast<size_t>(j)];
    sum+=delta*delta;
  }
  return sum;
}
void write_result(const std::string& out,const Args& args,const HostTree& tree,const Boxes& b,
                  long long device_checks,long long device_violations,double max_host_minus_device) {
  std::ofstream f(out+"/result.json");
  if (!f) fail("cannot write result JSON");
  f << std::setprecision(17)
    << "{\"schema\":\"safe-c2-static-aabb-cuda-canary-v1\",\"status\":\""
    << (device_violations==0 && b.coordinate_cover_violations==0 ? "PASS_STATIC_AABB_DEVICE_CERTIFICATE" : "FAIL_STATIC_AABB_DEVICE_CERTIFICATE")
    << "\",\"scope\":\"static-tree CUDA lower-bound canary only; no traversal, top-K, timing, update, calibration, validation, or sealed workload\","
    << "\"gpu_executed\":true,\"formal_claim_eligible\":false,"
    << "\"inputs\":{\"base_count\":"<<kBaseN<<",\"query_source\":\"standard SIFT query rows "
    << args.query_start<<".."<<(args.query_start+args.query_count-1)<<"\",\"query_count\":"<<args.query_count<<"},"
    << "\"projection\":{\"policy\":\"top population variance from base only\",\"dimensions\":"<<b.dims.size()<<",\"ids\":[";
  for (size_t i=0;i<b.dims.size();++i) { if(i) f<<','; f<<b.dims[i]; }
  f << "]},\"tree\":{\"node_count\":"<<tree.node_count<<",\"height\":"<<tree.tree_height
    << ",\"active_nonroot_nodes\":"<<b.active_nids.size()<<"},"
    << "\"host_cover\":{\"membership_observations\":"<<b.membership_observations
    << ",\"coordinate_checks\":"<<b.coordinate_cover_checks<<",\"violations\":"<<b.coordinate_cover_violations<<"},"
    << "\"device_rounding\":{\"node_query_checks\":"<<device_checks<<",\"violations\":"<<device_violations
    << ",\"max_host_lb_sq_minus_device_lb_sq\":"<<max_host_minus_device<<"},"
    << "\"snapshot\":{\"host_to_device_to_host_byte_equal\":true}}";
  f.close();
  if (!f) fail("failed closing result JSON");
}
} // namespace
int main(int argc,char** argv) {
 try {
  const Args args=parse(argc,argv);
  const std::vector<float> base=read_fvecs_all(args.base,kBaseN);
  const std::vector<float> queries=read_fvec_rows(args.queries,args.query_start,args.query_count);
  const std::vector<int> dims=top_variance_dims(base,args.projection_dims);
  const HostTree tree=build_archived_tree(base);
  const Boxes boxes=build_and_verify_boxes(tree,base,dims);
  float* lo_d=nullptr; float* hi_d=nullptr; int* dims_d=nullptr; float* queries_d=nullptr;
  int* active_d=nullptr; float* output_d=nullptr;
  try {
    const size_t box_bytes=boxes.lo.size()*sizeof(float);
    const size_t dim_bytes=boxes.dims.size()*sizeof(int);
    const size_t active_bytes=boxes.active_nids.size()*sizeof(int);
    const size_t output_count=boxes.active_nids.size()*static_cast<size_t>(args.query_count);
    AABB_CUDA(cudaMalloc(&lo_d,box_bytes)); AABB_CUDA(cudaMalloc(&hi_d,box_bytes));
    AABB_CUDA(cudaMalloc(&dims_d,dim_bytes)); AABB_CUDA(cudaMalloc(&queries_d,queries.size()*sizeof(float)));
    AABB_CUDA(cudaMalloc(&active_d,active_bytes)); AABB_CUDA(cudaMalloc(&output_d,output_count*sizeof(float)));
    AABB_CUDA(cudaMemcpy(lo_d,boxes.lo.data(),box_bytes,cudaMemcpyHostToDevice));
    AABB_CUDA(cudaMemcpy(hi_d,boxes.hi.data(),box_bytes,cudaMemcpyHostToDevice));
    AABB_CUDA(cudaMemcpy(dims_d,boxes.dims.data(),dim_bytes,cudaMemcpyHostToDevice));
    AABB_CUDA(cudaMemcpy(queries_d,queries.data(),queries.size()*sizeof(float),cudaMemcpyHostToDevice));
    AABB_CUDA(cudaMemcpy(active_d,boxes.active_nids.data(),active_bytes,cudaMemcpyHostToDevice));
    std::vector<float> lo_back(boxes.lo.size()),hi_back(boxes.hi.size());
    std::vector<int> dims_back(boxes.dims.size());
    AABB_CUDA(cudaMemcpy(lo_back.data(),lo_d,box_bytes,cudaMemcpyDeviceToHost));
    AABB_CUDA(cudaMemcpy(hi_back.data(),hi_d,box_bytes,cudaMemcpyDeviceToHost));
    AABB_CUDA(cudaMemcpy(dims_back.data(),dims_d,dim_bytes,cudaMemcpyDeviceToHost));
    if (std::memcmp(lo_back.data(),boxes.lo.data(),box_bytes)!=0 || std::memcmp(hi_back.data(),boxes.hi.data(),box_bytes)!=0 ||
        std::memcmp(dims_back.data(),boxes.dims.data(),dim_bytes)!=0) fail("H2D/D2H AABB snapshot mismatch");
    write_bytes(args.outdir+"/aabb_lo_host.bin",boxes.lo.data(),box_bytes);
    write_bytes(args.outdir+"/aabb_hi_host.bin",boxes.hi.data(),box_bytes);
    write_bytes(args.outdir+"/aabb_dims_host.bin",boxes.dims.data(),dim_bytes);
    write_bytes(args.outdir+"/aabb_lo_d2h.bin",lo_back.data(),box_bytes);
    write_bytes(args.outdir+"/aabb_hi_d2h.bin",hi_back.data(),box_bytes);
    write_bytes(args.outdir+"/aabb_dims_d2h.bin",dims_back.data(),dim_bytes);
    StaticProjectedAabbDeviceView view{lo_d,hi_d,dims_d,static_cast<int>(boxes.dims.size()),kDim};
    const int total=static_cast<int>(output_count);
    aabb_bound_canary_kernel<<<(total+255)/256,256>>>(view,queries_d,active_d,output_d,
                                                        static_cast<int>(boxes.active_nids.size()),args.query_count);
    AABB_CUDA(cudaDeviceSynchronize()); AABB_CUDA(cudaGetLastError());
    std::vector<float> output(output_count);
    AABB_CUDA(cudaMemcpy(output.data(),output_d,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
    long long violations=0; double max_gap=0.0;
    for (int qi=0; qi<args.query_count; ++qi) {
      const float* q=queries.data()+static_cast<size_t>(qi)*kDim;
      for (size_t ni=0; ni<boxes.active_nids.size(); ++ni) {
        const size_t idx=static_cast<size_t>(qi)*boxes.active_nids.size()+ni;
        const float got=output[idx];
        const long double host=host_lb_sq(boxes,boxes.active_nids[ni],q);
        if (!std::isfinite(got) || got<0.0f || static_cast<long double>(got)>host) ++violations;
        const double gap=static_cast<double>(host-static_cast<long double>(got));
        if (gap>max_gap) max_gap=gap;
      }
    }
    write_result(args.outdir,args,tree,boxes,static_cast<long long>(output_count),violations,max_gap);
  } catch (...) {
    if(output_d) cudaFree(output_d); if(active_d) cudaFree(active_d); if(queries_d) cudaFree(queries_d);
    if(dims_d) cudaFree(dims_d); if(hi_d) cudaFree(hi_d); if(lo_d) cudaFree(lo_d);
    throw;
  }
  if(output_d) AABB_CUDA(cudaFree(output_d)); if(active_d) AABB_CUDA(cudaFree(active_d));
  if(queries_d) AABB_CUDA(cudaFree(queries_d)); if(dims_d) AABB_CUDA(cudaFree(dims_d));
  if(hi_d) AABB_CUDA(cudaFree(hi_d)); if(lo_d) AABB_CUDA(cudaFree(lo_d));
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "FAIL_STATIC_AABB_CANARY: " << e.what() << std::endl;
  return 2;
 }
}

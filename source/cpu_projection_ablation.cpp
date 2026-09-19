/*
 * Exploratory CPU-only feasibility probe.
 *
 * This v4 ablates query-independent coordinate projections for an exact
 * radial-plus-AABB lower bound on a fixed radial tree. Projection dimensions
 * are selected solely from the fixed base sample's coordinate variances before
 * any query file is read. It is NOT a GTS benchmark and NOT formal evidence.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int kDim=128, kBaseN=1'000'000, kTreeOrder=10, kLeaf=20;
constexpr int kSampleN=16'384, kProbeQ=256, kCheckQ=8, kK=10;
const char* kBasePath="/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs";
const char* kQueryPath="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/fresh_selection_pre_gt_v1/sift_learn_fresh12524.fvecs";
const char* kCalibrationIds="/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4/inputs/final_workload_v2/calibration.ids";

struct Node {
  std::vector<int> ids;
  std::vector<int> children;
  int pivot=-1;
  double lo=0.0, hi=0.0;
  std::array<float,kDim> coord_lo{};
  std::array<float,kDim> coord_hi{};
  bool leaf=false;
};
struct Cand {
  double d;
  int id;
  bool operator==(const Cand& other) const { return d==other.d && id==other.id; }
};
struct Better {
  bool operator()(const Cand& a,const Cand& b) const {
    return a.d<b.d || (a.d==b.d && a.id<b.id);
  }
};
using Heap=std::priority_queue<Cand,std::vector<Cand>,Better>;
struct Stats { long long nodes=0, pruned=0, leaves=0, points=0; };
struct ModeResult {
  int m=0;
  std::vector<int> dims;
  Stats st;
  int certificate_checks=0;
  int certificate_violations=0;
  int output_mismatches=0;
};
enum class BoundMode { kImmediateRadial, kMultiLandmarkRadial, kRadialPlusProjectedAabb };

std::vector<Node> nodes;
const std::vector<float>* base_ptr=nullptr;

std::vector<int> read_ids() {
  std::ifstream in(kCalibrationIds);
  if(!in) throw std::runtime_error("open calibration IDs");
  std::vector<int> out; int x;
  while(in>>x && static_cast<int>(out.size())<kProbeQ) out.push_back(x);
  if(static_cast<int>(out.size())!=kProbeQ) throw std::runtime_error("short calibration ID list");
  std::sort(out.begin(),out.end());
  if(std::adjacent_find(out.begin(),out.end())!=out.end()) throw std::runtime_error("duplicate calibration IDs");
  return out;
}
std::vector<float> read_rows(const char* path,const std::vector<int>& rows,int total) {
  std::ifstream in(path,std::ios::binary);
  if(!in) throw std::runtime_error(std::string("open ")+path);
  const std::streamoff row_bytes=4+std::streamoff(kDim)*4;
  std::vector<float> out(static_cast<size_t>(rows.size())*kDim);
  for(size_t j=0;j<rows.size();++j) {
    if(rows[j]<0 || rows[j]>=total) throw std::runtime_error("row range");
    in.seekg(static_cast<std::streamoff>(rows[j])*row_bytes);
    int d=0; in.read(reinterpret_cast<char*>(&d),4);
    if(!in || d!=kDim) throw std::runtime_error("fvec header");
    in.read(reinterpret_cast<char*>(out.data()+j*kDim),kDim*4);
    if(!in) throw std::runtime_error("fvec payload");
  }
  return out;
}
double l2(const float* a,const float* b) {
  double sum=0.0;
  for(int d=0;d<kDim;++d) {
    const double delta=static_cast<double>(a[d])-static_cast<double>(b[d]);
    sum+=delta*delta;
  }
  return std::sqrt(sum);
}
double interval_lb(double q_to_p,double lo,double hi) {
  return std::max(0.0,std::max(lo-q_to_p,q_to_p-hi));
}
bool better(const Cand& a,const Cand& b) { return Better{}(a,b); }
std::vector<Cand> sorted_heap(Heap h) {
  std::vector<Cand> out;
  while(!h.empty()) { out.push_back(h.top()); h.pop(); }
  std::sort(out.begin(),out.end(),better);
  return out;
}
void add(Heap& h,Cand c) {
  if(static_cast<int>(h.size())<kK) h.push(c);
  else if(better(c,h.top())) { h.pop(); h.push(c); }
}
double tau(const Heap& h) {
  return static_cast<int>(h.size())==kK ? h.top().d : std::numeric_limits<double>::infinity();
}

void split_node(int nid) {
  if(static_cast<int>(nodes[nid].ids.size())<=kLeaf) { nodes[nid].leaf=true; return; }
  const int pivot=nodes[nid].ids[nodes[nid].ids.size()/2];
  std::vector<std::pair<double,int>> radial;
  radial.reserve(nodes[nid].ids.size());
  const float* p=base_ptr->data()+static_cast<size_t>(pivot)*kDim;
  for(int id:nodes[nid].ids) radial.emplace_back(l2(base_ptr->data()+static_cast<size_t>(id)*kDim,p),id);
  std::sort(radial.begin(),radial.end(),[](const auto& a,const auto& b) {
    return a.first<b.first || (a.first==b.first && a.second<b.second);
  });
  const int n=static_cast<int>(radial.size());
  const int chunk=(n+kTreeOrder-1)/kTreeOrder;
  for(int start=0;start<n;start+=chunk) {
    const int end=std::min(n,start+chunk);
    Node c; c.pivot=pivot; c.lo=radial[start].first; c.hi=radial[end-1].first;
    c.ids.reserve(end-start);
    for(int i=start;i<end;++i) c.ids.push_back(radial[i].second);
    const int cid=static_cast<int>(nodes.size());
    nodes.push_back(std::move(c));
    nodes[nid].children.push_back(cid);
  }
  const auto kids=nodes[nid].children;
  for(int cid:kids) split_node(cid);
}
void build_boxes(int nid) {
  Node& n=nodes[nid];
  for(int d=0;d<kDim;++d) {
    float lo=std::numeric_limits<float>::infinity();
    float hi=-std::numeric_limits<float>::infinity();
    for(int id:n.ids) {
      const float value=base_ptr->at(static_cast<size_t>(id)*kDim+d);
      lo=std::min(lo,value);
      hi=std::max(hi,value);
    }
    if(!std::isfinite(lo) || !std::isfinite(hi) || lo>hi) throw std::runtime_error("invalid AABB");
    n.coord_lo[d]=lo;
    n.coord_hi[d]=hi;
  }
  const auto kids=n.children;
  for(int cid:kids) build_boxes(cid);
}

/*
 * Uses only a subset of nonnegative squared coordinate terms. Therefore this is
 * an L2 lower bound for every vector inside the stored coordinate box.
 */
double projected_aabb_lb(const Node& n,const float* q,const std::vector<int>& dims) {
  double sq=0.0;
  for(int d:dims) {
    double delta=0.0;
    if(q[d]<n.coord_lo[d]) delta=static_cast<double>(n.coord_lo[d])-q[d];
    else if(q[d]>n.coord_hi[d]) delta=static_cast<double>(q[d])-n.coord_hi[d];
    sq+=delta*delta;
  }
  return std::sqrt(sq);
}
double node_lb(int nid,const float* q,double inherited,BoundMode mode,const std::vector<int>& dims) {
  const Node& n=nodes[nid];
  const double own=n.pivot<0 ? 0.0 :
    interval_lb(l2(q,base_ptr->data()+static_cast<size_t>(n.pivot)*kDim),n.lo,n.hi);
  if(mode==BoundMode::kImmediateRadial) return own;
  if(mode==BoundMode::kMultiLandmarkRadial) return std::max(inherited,own);
  return std::max(std::max(inherited,own),projected_aabb_lb(n,q,dims));
}
void search_tree(int nid,const float* q,BoundMode mode,double inherited,
                 const std::vector<int>& dims,Heap& h,Stats& st) {
  const double lb=node_lb(nid,q,inherited,mode,dims);
  /* Strict pruning preserves canonical (distance,id) ties conservatively. */
  if(static_cast<int>(h.size())==kK && lb>tau(h)) { ++st.pruned; return; }
  ++st.nodes;
  const Node& n=nodes[nid];
  if(n.leaf) {
    ++st.leaves;
    for(int id:n.ids) {
      ++st.points;
      add(h,{l2(q,base_ptr->data()+static_cast<size_t>(id)*kDim),id});
    }
    return;
  }
  std::vector<std::pair<double,int>> kids;
  kids.reserve(n.children.size());
  for(int cid:n.children) kids.emplace_back(node_lb(cid,q,lb,mode,dims),cid);
  std::sort(kids.begin(),kids.end(),[](const auto& a,const auto& b) {
    return a.first<b.first || (a.first==b.first && a.second<b.second);
  });
  for(const auto& child:kids) search_tree(child.second,q,mode,lb,dims,h,st);
}
Heap exact_knn(const float* q) {
  Heap h;
  for(int id=0;id<kSampleN;++id)
    add(h,{l2(q,base_ptr->data()+static_cast<size_t>(id)*kDim),id});
  return h;
}
void check_certificate(int nid,const float* q,double inherited,const std::vector<int>& dims,
                       int& checks,int& violations) {
  const double lb=node_lb(nid,q,inherited,BoundMode::kRadialPlusProjectedAabb,dims);
  double actual=std::numeric_limits<double>::infinity();
  for(int id:nodes[nid].ids)
    actual=std::min(actual,l2(q,base_ptr->data()+static_cast<size_t>(id)*kDim));
  ++checks;
  if(lb>actual+1e-9) ++violations;
  for(int cid:nodes[nid].children) check_certificate(cid,q,lb,dims,checks,violations);
}
int max_depth(int nid) {
  int depth=0;
  for(int child:nodes[nid].children) depth=std::max(depth,1+max_depth(child));
  return depth;
}

/*
 * This computation consumes only the already-loaded static base sample. It is
 * deliberately executed before read_ids()/query-file access.
 */
std::vector<int> rank_dims_by_base_variance(const std::vector<float>& base,
                                            std::array<double,kDim>& variances) {
  std::array<double,kDim> sums{};
  std::array<double,kDim> sums_sq{};
  for(int i=0;i<kSampleN;++i) {
    const float* x=base.data()+static_cast<size_t>(i)*kDim;
    for(int d=0;d<kDim;++d) {
      const double v=static_cast<double>(x[d]);
      sums[d]+=v;
      sums_sq[d]+=v*v;
    }
  }
  for(int d=0;d<kDim;++d) {
    const double mean=sums[d]/static_cast<double>(kSampleN);
    variances[d]=sums_sq[d]/static_cast<double>(kSampleN)-mean*mean;
    if(variances[d]<0.0 && variances[d]>-1e-12) variances[d]=0.0;
    if(variances[d]<0.0 || !std::isfinite(variances[d]))
      throw std::runtime_error("invalid static base variance");
  }
  std::vector<int> ranked(kDim);
  std::iota(ranked.begin(),ranked.end(),0);
  std::sort(ranked.begin(),ranked.end(),[&](int a,int b) {
    return variances[a]>variances[b] || (variances[a]==variances[b] && a<b);
  });
  return ranked;
}
std::vector<int> projection_dims(int m,const std::vector<int>& ranked) {
  if(m<=0 || m>kDim) throw std::runtime_error("projection m");
  if(m==kDim) {
    /* Natural order makes m=128 arithmetic-identical to v3 full-AABB. */
    std::vector<int> all(kDim);
    std::iota(all.begin(),all.end(),0);
    return all;
  }
  return std::vector<int>(ranked.begin(),ranked.begin()+m);
}
void emit_int_array(std::ostream& out,const std::vector<int>& xs) {
  out<<"[";
  for(size_t i=0;i<xs.size();++i) {
    if(i) out<<",";
    out<<xs[i];
  }
  out<<"]";
}
void emit_double_array(std::ostream& out,const std::array<double,kDim>& xs,
                       const std::vector<int>& dims) {
  out<<"[";
  for(size_t i=0;i<dims.size();++i) {
    if(i) out<<",";
    out<<xs[dims[i]];
  }
  out<<"]";
}
} // namespace

int main() {
  try {
    std::vector<int> base_rows;
    base_rows.reserve(kSampleN);
    for(int i=0;i<kSampleN;++i) base_rows.push_back(i*61);
    auto base=read_rows(kBasePath,base_rows,kBaseN);
    base_ptr=&base;

    std::array<double,kDim> variances{};
    const auto ranked_dims=rank_dims_by_base_variance(base,variances);
    const std::vector<int> ms={8,16,32,64,128};
    std::vector<ModeResult> modes;
    modes.reserve(ms.size());
    for(int m:ms) {
      ModeResult r;
      r.m=m;
      r.dims=projection_dims(m,ranked_dims);
      modes.push_back(std::move(r));
    }

    /* Query IDs/files are read only after projection selection is complete. */
    const auto qids=read_ids();
    const auto queries=read_rows(kQueryPath,qids,12'524);

    nodes.reserve(2048);
    Node root; root.ids.resize(kSampleN);
    for(int i=0;i<kSampleN;++i) root.ids[i]=i;
    nodes.push_back(std::move(root));
    split_node(0);
    build_boxes(0);

    Stats immediate_stats;
    Stats multi_stats;
    int immediate_mismatches=0;
    int multi_mismatches=0;
    const std::vector<int> no_dims;
    for(int qi=0;qi<kProbeQ;++qi) {
      const float* q=queries.data()+static_cast<size_t>(qi)*kDim;
      const Heap exact=exact_knn(q);
      Heap immediate;
      Heap multi;
      Stats si,sm;
      search_tree(0,q,BoundMode::kImmediateRadial,0.0,no_dims,immediate,si);
      search_tree(0,q,BoundMode::kMultiLandmarkRadial,0.0,no_dims,multi,sm);
      if(sorted_heap(exact)!=sorted_heap(immediate)) ++immediate_mismatches;
      if(sorted_heap(exact)!=sorted_heap(multi)) ++multi_mismatches;
      immediate_stats.nodes+=si.nodes; immediate_stats.pruned+=si.pruned;
      immediate_stats.leaves+=si.leaves; immediate_stats.points+=si.points;
      multi_stats.nodes+=sm.nodes; multi_stats.pruned+=sm.pruned;
      multi_stats.leaves+=sm.leaves; multi_stats.points+=sm.points;
      for(auto& mode:modes) {
        Heap h;
        Stats st;
        search_tree(0,q,BoundMode::kRadialPlusProjectedAabb,0.0,mode.dims,h,st);
        if(sorted_heap(exact)!=sorted_heap(h)) ++mode.output_mismatches;
        mode.st.nodes+=st.nodes; mode.st.pruned+=st.pruned;
        mode.st.leaves+=st.leaves; mode.st.points+=st.points;
      }
    }
    for(auto& mode:modes) {
      for(int qi=0;qi<kCheckQ;++qi) {
        check_certificate(0,queries.data()+static_cast<size_t>(qi)*kDim,0.0,mode.dims,
                          mode.certificate_checks,mode.certificate_violations);
      }
    }

    bool pass=(immediate_mismatches==0 && multi_mismatches==0);
    for(const auto& mode:modes)
      pass=pass && mode.output_mismatches==0 && mode.certificate_violations==0;

    std::cout<<std::setprecision(17)
      <<"{\"schema\":\"c2-projection-aabb-cpu-feasibility-probe-v1\","
      <<"\"status\":\""<<(pass ? "PASS_CPU_CERTIFICATE_AND_EXACTNESS" :
                                   "FAIL_CPU_CERTIFICATE_OR_EXACTNESS")<<"\","
      <<"\"scope\":\"nonformal CPU-only exploratory radial-tree probe; not GTS performance evidence and not a heldout result\","
      <<"\"gpu_executed\":false,"
      <<"\"validation_or_sealed_inputs_accessed\":false,"
      <<"\"base_sample\":{\"count\":"<<kSampleN
      <<",\"selector\":\"base_id=i*61 for i in [0,16383]\"},"
      <<"\"query_sample\":{\"count\":"<<kProbeQ
      <<",\"source\":\"existing C2 calibration IDs only; no validation/sealed IDs\"},"
      <<"\"tree\":{\"order\":"<<kTreeOrder<<",\"leaf_cap\":"<<kLeaf
      <<",\"nodes\":"<<nodes.size()<<",\"depth\":"<<max_depth(0)<<"},"
      <<"\"projection_selection\":{\"query_independent\":true,"
      <<"\"policy\":\"m<128: top-m coordinate population variance on fixed base sample, ties by ascending coordinate index; m=128: all coordinates in ascending index order\","
      <<"\"ranked_dims_by_base_variance\":";
    emit_int_array(std::cout,ranked_dims);
    std::cout<<",\"ranked_variances\":";
    emit_double_array(std::cout,variances,ranked_dims);
    std::cout<<"},"
      <<"\"exactness_baselines\":{\"canonical_topk\":\"(Euclidean-L2,id), strict-prune-only\","
      <<"\"immediate_radial_output_mismatches_vs_bruteforce\":"<<immediate_mismatches
      <<",\"multilandmark_radial_output_mismatches_vs_bruteforce\":"<<multi_mismatches<<"},"
      <<"\"work_baselines\":{\"immediate\":{\"point_distances\":"<<immediate_stats.points
      <<",\"nodes\":"<<immediate_stats.nodes<<",\"pruned_nodes\":"<<immediate_stats.pruned<<"},"
      <<"\"multilandmark\":{\"point_distances\":"<<multi_stats.points
      <<",\"nodes\":"<<multi_stats.nodes<<",\"pruned_nodes\":"<<multi_stats.pruned
      <<",\"point_distance_reduction_vs_immediate_fraction\":"
      <<(immediate_stats.points ? 1.0-static_cast<double>(multi_stats.points)/static_cast<double>(immediate_stats.points) : 0.0)<<"}},"
      <<"\"modes\":[";
    for(size_t i=0;i<modes.size();++i) {
      const auto& mode=modes[i];
      if(i) std::cout<<",";
      const double reduction=immediate_stats.points ?
        1.0-static_cast<double>(mode.st.points)/static_cast<double>(immediate_stats.points) : 0.0;
      std::cout<<"{\"m\":"<<mode.m<<",\"projection_dims\":";
      emit_int_array(std::cout,mode.dims);
      std::cout<<",\"certificate\":{\"checked_node_query_pairs\":"<<mode.certificate_checks
        <<",\"violations\":"<<mode.certificate_violations<<"},"
        <<"\"exactness\":{\"queries\":"<<kProbeQ
        <<",\"output_mismatches_vs_bruteforce\":"<<mode.output_mismatches<<"},"
        <<"\"work\":{\"point_distances\":"<<mode.st.points
        <<",\"nodes\":"<<mode.st.nodes<<",\"pruned_nodes\":"<<mode.st.pruned
        <<",\"point_distance_reduction_vs_immediate_fraction\":"<<reduction<<"}}";
    }
    std::cout<<"],\"formal_claim_eligible\":false}"<<std::endl;
  } catch(const std::exception& e) {
    std::cerr<<"FAIL_CPU_PROJECTION_ABLATION: "<<e.what()<<std::endl;
    return 2;
  }
}

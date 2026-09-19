/*
 * Exploratory CPU-only feasibility probe.
 * It builds a deterministic radial 10-ary tree over a fixed SIFT1M base sample.
 * The multi-landmark mode uses the max of all ancestor pivot interval lower bounds.
 * It is NOT a GTS benchmark and NOT formal evidence.\n * v1 adds a CPU-only directed-rounding arithmetic audit; it does not execute CUDA.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
struct Cand { double d; int id; bool operator==(const Cand& other) const { return d==other.d && id==other.id; } };
struct Better {
  bool operator()(const Cand& a, const Cand& b) const {
    return a.d < b.d || (a.d == b.d && a.id < b.id);
  }
};
using Heap=std::priority_queue<Cand,std::vector<Cand>,Better>;
struct Stats { long long nodes=0, pruned=0, leaves=0, points=0; };
enum class BoundMode { kImmediateRadial, kMultiLandmarkRadial, kRadialPlusAabb };

std::vector<int> read_ids() {
  std::ifstream in(kCalibrationIds);
  if (!in) throw std::runtime_error("open calibration IDs");
  std::vector<int> out; int x;
  while (in>>x && (int)out.size()<kProbeQ) out.push_back(x);
  if ((int)out.size()!=kProbeQ) throw std::runtime_error("short calibration ID list");
  std::sort(out.begin(),out.end());
  if (std::adjacent_find(out.begin(),out.end())!=out.end()) throw std::runtime_error("duplicate calibration IDs");
  return out;
}
std::vector<float> read_rows(const char* path,const std::vector<int>& rows,int total) {
  std::ifstream in(path,std::ios::binary);
  if (!in) throw std::runtime_error(std::string("open ")+path);
  const std::streamoff rb=4+std::streamoff(kDim)*4;
  std::vector<float> out((size_t)rows.size()*kDim);
  for(size_t j=0;j<rows.size();++j) {
    if(rows[j]<0 || rows[j]>=total) throw std::runtime_error("row range");
    in.seekg((std::streamoff)rows[j]*rb);
    int d=0; in.read(reinterpret_cast<char*>(&d),4);
    if(!in || d!=kDim) throw std::runtime_error("fvec header");
    in.read(reinterpret_cast<char*>(out.data()+j*kDim),kDim*4);
    if(!in) throw std::runtime_error("fvec payload");
  }
  return out;
}
double l2(const float* a,const float* b) {
  double s=0;
  for(int d=0;d<kDim;++d){double z=(double)a[d]-(double)b[d];s+=z*z;}
  return std::sqrt(s);
}
double interval_lb(double q_to_p,double lo,double hi) {
  return std::max(0.0,std::max(lo-q_to_p,q_to_p-hi));
}
bool better(const Cand& a,const Cand& b) { return Better{}(a,b); }
std::vector<Cand> sorted_heap(Heap h) {
  std::vector<Cand> out;
  while(!h.empty()){out.push_back(h.top());h.pop();}
  std::sort(out.begin(),out.end(),better);
  return out;
}
void add(Heap& h,Cand c) {
  if((int)h.size()<kK) h.push(c);
  else if(better(c,h.top())){h.pop();h.push(c);}
}
double tau(const Heap& h) { return (int)h.size()==kK ? h.top().d : std::numeric_limits<double>::infinity(); }

std::vector<Node> nodes;
const std::vector<float>* base_ptr=nullptr;
void split_node(int nid) {
  if((int)nodes[nid].ids.size()<=kLeaf){nodes[nid].leaf=true;return;}
  const int pivot=nodes[nid].ids[nodes[nid].ids.size()/2];
  std::vector<std::pair<double,int>> radial;
  radial.reserve(nodes[nid].ids.size());
  const float* p=base_ptr->data()+(size_t)pivot*kDim;
  for(int id:nodes[nid].ids) radial.emplace_back(l2(base_ptr->data()+(size_t)id*kDim,p),id);
  std::sort(radial.begin(),radial.end(),[](const auto& a,const auto& b){
    return a.first<b.first || (a.first==b.first && a.second<b.second);
  });
  const int n=(int)radial.size();
  const int chunk=(n+kTreeOrder-1)/kTreeOrder;
  for(int start=0;start<n;start+=chunk) {
    const int end=std::min(n,start+chunk);
    Node c; c.pivot=pivot; c.lo=radial[start].first; c.hi=radial[end-1].first;
    c.ids.reserve(end-start);
    for(int i=start;i<end;++i)c.ids.push_back(radial[i].second);
    const int cid=(int)nodes.size(); nodes.push_back(std::move(c));
    nodes[nid].children.push_back(cid);
  }
  const auto kids=nodes[nid].children;
  for(int cid:kids) split_node(cid);
}
void build_boxes(int nid) {
  Node& n=nodes[nid];
  for (int d=0; d<kDim; ++d) {
    float lo=std::numeric_limits<float>::infinity();
    float hi=-std::numeric_limits<float>::infinity();
    for (int id:n.ids) {
      const float value=base_ptr->at((size_t)id*kDim+d);
      lo=std::min(lo,value); hi=std::max(hi,value);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo>hi) throw std::runtime_error("invalid AABB");
    n.coord_lo[d]=lo; n.coord_hi[d]=hi;
  }
  const auto kids=n.children;
  for (int cid:kids) build_boxes(cid);
}
double aabb_lb(const Node& n,const float* q) {
  double sq=0.0;
  for (int d=0;d<kDim;++d) {
    double delta=0.0;
    if (q[d]<n.coord_lo[d]) delta=(double)n.coord_lo[d]-q[d];
    else if (q[d]>n.coord_hi[d]) delta=(double)q[d]-n.coord_hi[d];
    sq+=delta*delta;
  }
  return std::sqrt(sq);
}

// CPU model for the CUDA directed-rounding contract proposed for Safe-C2-AABB.
// Each primitive moves a default IEEE-754 round-to-nearest result one representable
// float toward -infinity.  This is conservative (possibly looser) and models
// __fsub_rd/__fmul_rd/__fadd_rd for the finite nonnegative inputs checked below.
float rd_down(float x) {
  return std::nextafterf(x, -std::numeric_limits<float>::infinity());
}
float nonnegative_rd_sub(float a, float b) {
  const float value = rd_down(a-b);
  return value > 0.0f ? value : 0.0f;
}
float nonnegative_rd_mul(float a, float b) {
  const float value = rd_down(a*b);
  return value > 0.0f ? value : 0.0f;
}
float nonnegative_rd_add(float a, float b) {
  const float value = rd_down(a+b);
  return value > 0.0f ? value : 0.0f;
}
float aabb_lb_sq_rn_float(const Node& n, const float* q) {
  float sq=0.0f;
  for (int d=0; d<kDim; ++d) {
    float delta=0.0f;
    if (q[d] < n.coord_lo[d]) delta=n.coord_lo[d]-q[d];
    else if (q[d] > n.coord_hi[d]) delta=q[d]-n.coord_hi[d];
    sq += delta*delta;
  }
  return sq;
}
float aabb_lb_sq_rd_float(const Node& n, const float* q) {
  float sq=0.0f;
  for (int d=0; d<kDim; ++d) {
    float delta=0.0f;
    if (q[d] < n.coord_lo[d]) delta=nonnegative_rd_sub(n.coord_lo[d],q[d]);
    else if (q[d] > n.coord_hi[d]) delta=nonnegative_rd_sub(q[d],n.coord_hi[d]);
    const float term=nonnegative_rd_mul(delta,delta);
    sq=nonnegative_rd_add(sq,term);
  }
  return sq;
}
struct RoundingAudit {
  long long node_query_pairs=0;
  long long point_domination_checks=0;
  long long rn_overestimation_violations=0;
  long long rd_overestimation_violations=0;
  long long rd_exceeds_rn_violations=0;
  double max_rn_minus_rd=0.0;
};
double exact_sq_to_point(const float* q, int id) {
  double sq=0.0;
  const float* x=base_ptr->data()+(size_t)id*kDim;
  for (int d=0; d<kDim; ++d) {
    const double delta=(double)q[d]-(double)x[d];
    sq+=delta*delta;
  }
  return sq;
}
void check_rounding_contract(int nid, const float* q, RoundingAudit& audit) {
  const Node& n=nodes[nid];
  const float rn=aabb_lb_sq_rn_float(n,q);
  const float rd=aabb_lb_sq_rd_float(n,q);
  ++audit.node_query_pairs;
  if (rd > rn) ++audit.rd_exceeds_rn_violations;
  audit.max_rn_minus_rd=std::max(audit.max_rn_minus_rd,(double)rn-(double)rd);
  for (int id:n.ids) {
    const double actual=exact_sq_to_point(q,id);
    ++audit.point_domination_checks;
    if ((double)rn > actual) ++audit.rn_overestimation_violations;
    if ((double)rd > actual) ++audit.rd_overestimation_violations;
  }
  for (int cid:n.children) check_rounding_contract(cid,q,audit);
}
double node_lb(int nid,const float* q,double inherited,BoundMode mode) {
  const Node& n=nodes[nid];
  const double own=n.pivot<0 ? 0.0 : interval_lb(l2(q,base_ptr->data()+(size_t)n.pivot*kDim),n.lo,n.hi);
  if (mode==BoundMode::kImmediateRadial) return own;
  if (mode==BoundMode::kMultiLandmarkRadial) return std::max(inherited,own);
  return std::max(std::max(inherited,own),aabb_lb(n,q));
}
void search_tree(int nid,const float* q,BoundMode mode,double inherited,Heap& h,Stats& st) {
  const double lb=node_lb(nid,q,inherited,mode);
  if((int)h.size()==kK && lb>tau(h)){++st.pruned;return;}
  ++st.nodes;
  const Node& n=nodes[nid];
  if(n.leaf) {
    ++st.leaves;
    for(int id:n.ids) {++st.points; add(h,{l2(q,base_ptr->data()+(size_t)id*kDim),id});}
    return;
  }
  std::vector<std::pair<double,int>> kids;
  kids.reserve(n.children.size());
  for(int cid:n.children) kids.emplace_back(node_lb(cid,q,lb,mode),cid);
  std::sort(kids.begin(),kids.end(),[](const auto& a,const auto& b){return a.first<b.first || (a.first==b.first&&a.second<b.second);});
  for(const auto& x:kids) search_tree(x.second,q,mode,lb,h,st);
}
Heap exact_knn(const float* q) {
  Heap h;
  for(int id=0;id<kSampleN;++id)add(h,{l2(q,base_ptr->data()+(size_t)id*kDim),id});
  return h;
}
void check_certificate(int nid,const float* q,double inherited,BoundMode mode,int& checks,int& violations) {
  const double lb=node_lb(nid,q,inherited,mode);
  double actual=std::numeric_limits<double>::infinity();
  for(int id:nodes[nid].ids)actual=std::min(actual,l2(q,base_ptr->data()+(size_t)id*kDim));
  ++checks;
  if(lb>actual+1e-9)++violations;
  for(int cid:nodes[nid].children)check_certificate(cid,q,lb,mode,checks,violations);
}
int max_depth(int nid) {
  int d=0;for(int c:nodes[nid].children)d=std::max(d,1+max_depth(c));return d;
}
}
int main() {
 try {
  std::vector<int> base_rows; base_rows.reserve(kSampleN);
  for(int i=0;i<kSampleN;++i)base_rows.push_back(i*61);
  auto base=read_rows(kBasePath,base_rows,kBaseN); base_ptr=&base;
  auto qids=read_ids(); auto queries=read_rows(kQueryPath,qids,12'524);
  nodes.reserve(2048); Node root; root.ids.resize(kSampleN);
  for(int i=0;i<kSampleN;++i) { root.ids[i]=i; }
  nodes.push_back(std::move(root));
  split_node(0);
  build_boxes(0);
  long long immediate_points=0,multi_points=0,aabb_points=0,immediate_nodes=0,multi_nodes=0,aabb_nodes=0,immediate_pruned=0,multi_pruned=0,aabb_pruned=0;
  int output_mismatches=0;
  for(int qi=0;qi<kProbeQ;++qi) {
    const float* q=queries.data()+(size_t)qi*kDim;
    Heap exact=exact_knn(q), a,b,c; Stats sa,sb,sc;
    search_tree(0,q,BoundMode::kImmediateRadial,0.0,a,sa);
    search_tree(0,q,BoundMode::kMultiLandmarkRadial,0.0,b,sb);
    search_tree(0,q,BoundMode::kRadialPlusAabb,0.0,c,sc);
    if(sorted_heap(exact)!=sorted_heap(a) || sorted_heap(exact)!=sorted_heap(b) || sorted_heap(exact)!=sorted_heap(c))++output_mismatches;
    immediate_points+=sa.points;multi_points+=sb.points;aabb_points+=sc.points;
    immediate_nodes+=sa.nodes;multi_nodes+=sb.nodes;aabb_nodes+=sc.nodes;
    immediate_pruned+=sa.pruned;multi_pruned+=sb.pruned;aabb_pruned+=sc.pruned;
  }
  int checks=0,violations=0;
  for(int qi=0;qi<kCheckQ;++qi)check_certificate(0,queries.data()+(size_t)qi*kDim,0.0,BoundMode::kRadialPlusAabb,checks,violations);
  RoundingAudit rounding;
  for(int qi=0;qi<kCheckQ;++qi)check_rounding_contract(0,queries.data()+(size_t)qi*kDim,rounding);
  const double multi_reduction=immediate_points?1.0-(double)multi_points/(double)immediate_points:0.0;
  const double aabb_reduction=immediate_points?1.0-(double)aabb_points/(double)immediate_points:0.0;
  std::cout<<std::setprecision(12)
    <<"{\"schema\":\"c2-multilandmark-cpu-feasibility-probe-v1\",\"status\":\""
    <<(output_mismatches==0&&violations==0&&rounding.rd_overestimation_violations==0&&rounding.rd_exceeds_rn_violations==0?"PASS_CPU_CERTIFICATE_EXACTNESS_AND_ROUNDING_AUDIT":"FAIL_CPU_CERTIFICATE_EXACTNESS_OR_ROUNDING_AUDIT")
    <<"\",\"scope\":\"nonformal CPU-only exploratory radial-tree probe; not GTS performance evidence and not a heldout result\","
    <<"\"base_sample\":{\"count\":"<<kSampleN<<",\"selector\":\"base_id=i*61 for i in [0,16383]\"},"
    <<"\"query_sample\":{\"count\":"<<kProbeQ<<",\"source\":\"existing C2 calibration IDs only; no validation/sealed IDs\"},"
    <<"\"tree\":{\"order\":"<<kTreeOrder<<",\"leaf_cap\":"<<kLeaf<<",\"nodes\":"<<nodes.size()<<",\"depth\":"<<max_depth(0)<<"},"
    <<"\"certificate\":{\"checked_node_query_pairs\":"<<checks<<",\"violations\":"<<violations<<"},"
    <<"\"rounding_audit\":{\"scope\":\"CPU model of directed-down float AABB squared lower bound; not a CUDA execution\","
    <<"\"node_query_pairs\":"<<rounding.node_query_pairs<<",\"point_domination_checks\":"<<rounding.point_domination_checks
    <<",\"rn_overestimation_violations\":"<<rounding.rn_overestimation_violations
    <<",\"rd_overestimation_violations\":"<<rounding.rd_overestimation_violations
    <<",\"rd_exceeds_rn_violations\":"<<rounding.rd_exceeds_rn_violations
    <<",\"max_rn_minus_rd\":"<<rounding.max_rn_minus_rd<<"},"
    <<"\"exactness\":{\"queries\":"<<kProbeQ<<",\"canonical_topk\":\"(Euclidean-L2,id), strict-prune-only\","
    <<"\"output_mismatches_vs_bruteforce\":"<<output_mismatches<<"},"
    <<"\"work\":{\"immediate\":{\"point_distances\":"<<immediate_points<<",\"nodes\":"<<immediate_nodes<<",\"pruned_nodes\":"<<immediate_pruned<<"},"
    <<"\"multilandmark\":{\"point_distances\":"<<multi_points<<",\"nodes\":"<<multi_nodes<<",\"pruned_nodes\":"<<multi_pruned
    <<",\"point_distance_reduction_fraction\":"<<multi_reduction<<"},"
    <<"\"radial_plus_full_aabb\":{\"point_distances\":"<<aabb_points<<",\"nodes\":"<<aabb_nodes<<",\"pruned_nodes\":"<<aabb_pruned
    <<",\"point_distance_reduction_fraction\":"<<aabb_reduction<<"}},"
    <<"\"formal_claim_eligible\":false,\"gpu_executed\":false}"<<std::endl;
 } catch(const std::exception& e) {std::cerr<<"FAIL_CPU_MULTILANDMARK_PROBE: "<<e.what()<<std::endl;return 2;}
}

// Reuse the measured Gate 1 translation unit without editing its bytes. Its old
// entry point is unreachable here; kernel, arena, oracle and lifecycle helpers
// are literally the same implementation. Gate 2 adds host scheduling only.
#define main gate1_unused_entry
#include "../gate1/replay.cu"
#undef main
#include <cmath>

namespace g2 {
using namespace g1;
struct Writer : g1::Writer {
  using g1::Writer::Writer;
  EP build(const EP &old,std::shared_ptr<DB> delta,const std::string &policy,int epoch,Maintenance &m) {
    // Same ordinary upload/merge implementation, with only k/factor configurable.
    m.begin=now();if(!m.arrival)m.arrival=m.begin;m.epoch=epoch;m.policy=policy;
    auto next=std::make_shared<Epoch>();next->number=epoch;if(old)next->runs=old->runs;
    auto upload=[&](std::shared_ptr<DB> h) {
      auto r=std::make_shared<Run>(arena,h,stream,stage);
      m.uploaded+=h->ids.size()*42;m.metadata_ms+=r->metadata_ms;m.stage_ms+=r->stage_ms;m.h2d_ms+=r->h2d_ms;
      return r;
    };
    if(!arena.fits(Run::footprint(delta->ids.size()))) {m.status="rejected_delta_budget";m.ready=now();return old;}
    next->runs.push_back(upload(delta));
    auto compact=[&](std::size_t first) {
      std::size_t n=0;for(std::size_t i=first;i<next->runs.size();++i)n+=next->runs[i]->host->ids.size();
      if(!arena.fits(Run::footprint(n))) {m.deferred++;m.status="published_merge_deferred";return false;}
      double t=now();auto host=merge(next->runs,first);m.build_ms+=now()-t;
      auto combined=upload(host);m.merges++;m.merge_rw+=n*84;
      next->runs.erase(next->runs.begin()+first,next->runs.end());next->runs.push_back(combined);return true;
    };
    int period=policy.rfind("periodic",0)==0?std::stoi(policy.substr(8)):0;
    int factor=policy.rfind("tier",0)==0?std::stoi(policy.substr(4)):0;
    if(epoch>0&&(policy=="compact"||(period&&epoch%period==0)))compact(0);
    else if(factor)while(next->runs.size()>=2) {
      auto n=next->runs.size();if(next->runs[n-2]->host->ids.size()>std::size_t(factor)*next->runs[n-1]->host->ids.size())break;
      if(!compact(n-2))break;
    }
    m.ready=now();return next;
  }
};

std::shared_ptr<DB> final_host(const History &h) {
  auto out=std::make_shared<DB>();std::vector<HostDBView<4>> views;std::size_t n=0;
  for(auto &r:h){views.push_back(r->view());n+=r->ids.size();}
  out->ids.resize(n);out->pc.resize(n);out->fp.resize(4*n);std::size_t pos=0;
  for(int pc=0;pc<=256;++pc)for(std::size_t i=0;i<h.size();++i) {
    auto b=views[i].cumulative[pc],len=views[i].cumulative[pc+1]-b;
    std::copy_n(h[i]->ids.data()+b,len,out->ids.data()+pos);
    std::copy_n(h[i]->pc.data()+b,len,out->pc.data()+pos);
    std::copy_n(h[i]->fp.data()+4*b,4*len,out->fp.data()+4*pos);pos+=len;
  }
  require(pos==n,"final static row conservation failed");return out;
}
void until(double ms) {
  std::this_thread::sleep_until(origin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(ms)));
}
struct Answer { Result result;double returned=0;int maint_backlog=0; };

void run(const std::map<std::string,std::string> &args) {
  auto root=args.at("--root"),prefix=args.at("--output"),mode=args.at("--mode"),policy=args.at("--policy");
  require(mode=="growth"||mode=="shadow"||mode=="static_base"||mode=="static_final"||mode=="calibrate","bad mode");
  require(policy=="all_delta"||policy=="compact"||policy=="periodic2"||policy=="periodic3"||policy=="periodic6"||policy=="tier2"||policy=="tier4","bad policy");
  int cap=std::stoi(args.at("--cap-mib")),rotation=std::stoi(args.at("--rotation"));
  double interval=std::stod(args.at("--interval-ms")),window=std::stod(args.at("--window-ms"));
  require((cap==1280||cap==2048)&&rotation>=0&&rotation<4&&interval>=.1&&window==8000,"unregistered configuration");
  require(!std::filesystem::exists(prefix+".arena.csv"),"refusing existing evidence");
  auto h=load(root);MappedFile qfile(root+"/queries.bin");QueryStore<4> qs(qfile);require(qs.size()==64,"64 queries required");
  Oracles oracle(root);Arena arena(cap*MiB,prefix+".arena.csv");
  std::size_t total=0;double start=0,finish=0;bool coverage=true;
  {
    Runtime rt(arena);Writer writer(arena);Logs logs(prefix);
    EP visible;Maintenance initial;
    bool final=mode=="static_final"||mode=="calibrate",dynamic=mode=="growth"||mode=="shadow";
    auto base=final?final_host(h):h[0];
    writer.publish(&visible,writer.build(nullptr,base,"all_delta",final?6:0,initial),initial);
    base.reset();
    for(int i=0;i<32;++i)oracle.check(rt.run(&visible,qs,(i*8)%64,8,80));
    EP shadow=mode=="shadow"?visible:nullptr;
    auto retained_base=mode=="shadow"?visible->runs[0]->released:std::shared_ptr<Release>{};
    std::vector<Maintenance> ledger;ledger.reserve(6);std::atomic<int> done{0};
    std::atomic<bool> failed{false},stop{false};std::exception_ptr writer_failure;
    total=mode=="calibrate"?512:std::size_t(std::ceil(window/interval));
    require(total<=80000,"host response count bound exceeded");
    std::vector<Answer> answers;answers.reserve(total);
    start=now()+100;
    std::thread worker;
    if(dynamic)worker=std::thread([&] {
      try {
        CUDA_CHECK(cudaSetDevice(0));EP *slot=mode=="growth"?&visible:&shadow;
        for(int e=1;e<=6;++e) {
          Maintenance m;m.arrival=start+750+(e-1)*1000;m.mode=mode;m.rotation=rotation;m.batch=8;
          until(m.arrival);if(stop.load())break;
          auto old=std::atomic_load(slot);auto next=writer.build(old,h[e],policy,e,m);old.reset();
          require(next&&next->number==e,"delta rejected; cannot silently lose maintenance");
          writer.publish(slot,next,m);next.reset();ledger.push_back(m);done.store(e);
        }
      }catch(...){writer_failure=std::current_exception();failed.store(true);}
    });
    try {
      until(start);
      for(std::size_t i=0;i<total;++i) {
        if(mode!="calibrate")until(start+i*interval);
        require(!failed.load(),"writer failed; attempt invalid");
        require(now()<start+window*2,"fixed drain deadline exceeded; attempt invalid");
        int due=dynamic?std::clamp(int(std::floor((now()-start-750)/1000))+1,0,6):0;
        int backlog=std::max(0,due-done.load());
        auto result=rt.run(&visible,qs,(i*8)%64,8,80);double returned=now();
        answers.push_back({std::move(result),returned,backlog});
      }
      finish=now();if(mode!="calibrate")until(start+window);
      if(worker.joinable())worker.join();
      if(writer_failure)std::rethrow_exception(writer_failure);
    }catch(...) {stop.store(true);if(worker.joinable())worker.join();throw;}
    // Determine real lifecycle coverage before releasing the final visible owners.
    std::ofstream life(prefix+".coverage.csv");life<<std::setprecision(12);
    life<<"epoch,arrival_ms,published_ms,owner_released_ms,exclusive_reclaimed_ms,retained_base_exemption,within_window\n";
    if(dynamic)require(ledger.size()==6,"missing maintenance events");
    for(auto &m:ledger) {
      bool exemption=mode=="shadow"&&m.epoch==1;
      double owner=m.retired?m.retired->owner_released.load():m.writer_done;
      double reclaimed=m.writer_done;
      for(auto &r:m.exclusive_releases)if(r!=retained_base) {
        auto at=r->at.load();if(at==0)reclaimed=start+window+1;else reclaimed=std::max(reclaimed,at);
      }
      bool ok=m.published<start+window&&(exemption||(owner>0&&owner<start+window))&&reclaimed<start+window;
      coverage&=ok;
      life<<m.epoch<<','<<m.arrival<<','<<m.published<<','<<owner<<','<<reclaimed<<','<<exemption<<','<<ok<<'\n';
    }
    std::array<int,7> seen{};
    std::ofstream arrivals(prefix+".arrivals.csv");arrivals<<std::setprecision(12);
    arrivals<<"index,arrival_ms,dispatch_ms,returned_ms,queue_ms,response_ms,full_service_ms,waiting_batches,maintenance_backlog,within_window\n";
    for(std::size_t i=0;i<answers.size();++i) {
      auto &a=answers[i];auto &r=a.result;oracle.check(r);
      double arrival=mode=="calibrate"?r.begin:start+i*interval;
      bool inside=a.returned<=start+window;if(inside)seen[r.epoch]++;
      int due=mode=="calibrate"?int(i+1):std::min<int>(total,std::max(0,int(std::floor((r.begin-start)/interval))+1));
      int waiting=std::max(0,due-int(i)-1);
      arrivals<<i<<','<<arrival<<','<<r.begin<<','<<a.returned<<','<<r.begin-arrival<<','<<a.returned-arrival<<','
        <<a.returned-r.begin<<','<<waiting<<','<<a.maint_backlog<<','<<inside<<'\n';
      logs.request(r,policy,mode,rotation,"measured",r.begin-arrival,a.maint_backlog);
    }
    if(mode=="growth")for(auto n:seen)coverage&=n>0;
    visible.reset();shadow.reset();for(auto &m:ledger)logs.maintenance(m);
    arena.observe();
    std::ofstream meta(prefix+".case.json");meta<<std::setprecision(12);
    meta<<"{\"mode\":\""<<mode<<"\",\"policy\":\""<<policy<<"\",\"rotation\":"<<rotation
      <<",\"cap_mib\":"<<cap<<",\"interval_ms\":"<<interval<<",\"window_ms\":"<<window<<",\"start_ms\":"<<start
      <<",\"drain_finished_ms\":"<<finish<<",\"scheduled_batches\":"<<total<<",\"completed_batches\":"<<answers.size()
      <<",\"dropped_batches\":0,\"coverage_pass\":"<<(coverage?"true":"false")<<",\"cpu_complete_match\":true,\"seen_epochs\":[";
    for(int e=0;e<7;++e)meta<<(e?",":"")<<seen[e];meta<<"]}\n";
  }
  require(arena.live()==0,"nonzero final arena balance");arena.observe();struct rusage rss{};getrusage(RUSAGE_SELF,&rss);
  std::ofstream memory(prefix+".memory.json");
  memory<<"{\"total_cap_bytes\":"<<arena.cap<<",\"baseline_observed_bytes\":"<<arena.baseline<<",\"fixed_arena_bytes\":"<<arena.capacity
    <<",\"peak_logical_arena_bytes\":"<<arena.peak<<",\"observed_whole_card_peak_bytes\":"<<arena.observed_peak
    <<",\"budget_refusals\":"<<arena.refused<<",\"final_live_arena_bytes\":0,\"host_maxrss_kib\":"<<rss.ru_maxrss<<"}\n";
  std::cout<<"PASS batches="<<total<<" coverage="<<coverage<<" observed_peak="<<arena.observed_peak<<std::endl;
}
} // namespace g2

int main(int argc,char **argv) {
  try {
    std::map<std::string,std::string> args;g1::require(argc%2==1,"expected key/value arguments");
    for(int i=1;i<argc;i+=2)g1::require(args.emplace(argv[i],argv[i+1]).second,"duplicate argument");
    if(args.at("--mode")=="oracle") {
      auto root=args.at("--root");auto h=g1::load(root);MappedFile f(root+"/queries.bin");QueryStore<4> qs(f);
      g1::oracle(h,qs,root);return 0;
    }
    if(args.at("--mode")=="guards") {
      // Gate 1's entry point still validates its own stricter argument surface.
      return gate1_unused_entry(argc,argv);
    }
    g2::run(args);return 0;
  }catch(const std::exception &e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 2;}
}

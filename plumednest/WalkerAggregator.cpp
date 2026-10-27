/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   WalkerAggregator implementation
   (c) 2025
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "WalkerAggregator.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/ActionSet.h"
#include "core/ActionWithValue.h"
#include "tools/Tools.h"
#include "tools/Communicator.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <time.h>

namespace PLMD {
namespace bias {

PLUMED_REGISTER_ACTION(WalkerAggregator,"WALKER_AGGREGATOR")

void WalkerAggregator::registerKeywords(Keywords& keys) {
  function::Function::registerKeywords(keys);
  // Advertise usage of ARG provided by ActionWithArguments
  keys.use("ARG");
  keys.add("compulsory","WALKERS_N","number of walkers");
  keys.add("compulsory","WALKERS_ID","walker id (0-based)");
  keys.add("optional","WALKERS_DIR","shared directory for value files (default current)");
  keys.add("optional","FILE_PREFIX","prefix for value files (default label)");
  keys.add("optional","METHOD","aggregation method: MEAN (default) or COLLECT (expose all walker values as components)");
  keys.add("optional","WALKERS_RSTRIDE","write/read stride to synchronize walker values (default 1)");
  keys.add("optional","STARTUP_WAIT","wait this many seconds once at the start to allow replicas to start (default 0)");
  keys.add("optional","WVALUES_RETENTION","APPEND (default) to append values, or LATEST to keep only the most recent value per stride");
  keys.add("optional","HISTORY_NT","keep a rolling window of the last NT stride-aligned updates per walker and expose them in COLLECT mode (default 0)");
  keys.addFlag("WALKERS_MPI",false,"use MPI multi-sim communicator to exchange walker values (avoids file I/O)");
  // Flags must be added with default 'false' per Keywords::addFlag contract; we invert logic in parseFlag
  keys.addFlag("WAIT_ALL",false,"wait until all walkers have produced at least one value before aggregating");
  useCustomisableComponents(keys); // allow *.component syntax
}

WalkerAggregator::WalkerAggregator(const ActionOptions& ao)
  : Action(ao), function::Function(ao), ncomp_(0), mw_n_(1), mw_id_(0), method_("MEAN"), wait_all_(true), ready_(false) {

  parse("WALKERS_N",mw_n_);
  parse("WALKERS_ID",mw_id_);
  if(mw_n_<=0) error("WALKERS_N must be >0");
  if(mw_id_<0 || mw_id_>=mw_n_) error("WALKERS_ID must be in [0, WAlKERS_N-1]");
  parse("WALKERS_DIR",mw_dir_);
  parse("FILE_PREFIX",file_prefix_);
  if(file_prefix_.empty()) file_prefix_=getLabel();
  parse("METHOD",method_);
  // Uppercase METHOD manually
  for(char &ch: method_) ch=std::toupper(static_cast<unsigned char>(ch));
  if(method_!="MEAN" && method_!="COLLECT") error("METHOD must be MEAN or COLLECT");
  parse("WVALUES_RETENTION", retention_mode_);
  if(retention_mode_.empty()) retention_mode_ = "APPEND";
  for(char &ch: retention_mode_) ch=std::toupper(static_cast<unsigned char>(ch));
  if(retention_mode_!="APPEND" && retention_mode_!="LATEST") error("WVALUES_RETENTION must be APPEND or LATEST");
  parseFlag("WAIT_ALL",wait_all_);
  parseFlag("WALKERS_MPI",walkers_mpi_);
  parse("WALKERS_RSTRIDE", rstride_);
  if(rstride_<=0) rstride_=1;
  parse("STARTUP_WAIT", startup_wait_sec_);
  if(startup_wait_sec_ < 0.0) startup_wait_sec_ = 0.0;
  parse("HISTORY_NT", history_nt_);
  if(history_nt_ < 0) history_nt_ = 0;

  // Arguments were already parsed by ActionWithArguments (Function base). If not, parse here.
  if(getNumberOfArguments()==0){
    std::vector<Value*> arg;
    parseArgumentList("ARG", arg);
    if(arg.empty()) error("No ARG provided");
    requestArguments(arg);
  }
  ncomp_=getNumberOfArguments();
  if(ncomp_==0) error("No ARG provided");
  if(method_=="COLLECT") {
    setupCollectComponents();
  } else {
    for(unsigned i=0;i<ncomp_;++i){
      std::string cname=getPntrToArgument(i)->getName();
      addComponentWithDerivatives(cname);
      getPntrToComponent(i)->setNotPeriodic();
    }
  }
  last_values_.assign(mw_n_, std::vector<double>(ncomp_,0.0));
  have_value_.assign(mw_n_, false);
  last_step_seen_.assign(mw_n_, -1);
  last_hist_step_.assign(mw_n_, -1);
  if(history_nt_>0){
    history_.assign(mw_n_, std::deque<std::vector<double>>() );
  }
  if(!walkers_mpi_){
    ifiles_.resize(mw_n_);
    ifilenames_.resize(mw_n_);
    for(int i=0;i<mw_n_;++i){
      if(i==mw_id_) continue;
      std::string fname=file_prefix_+".WVALUES."+std::to_string(i);
      if(!mw_dir_.empty()) fname=mw_dir_+"/"+fname;
      ifilenames_[i]=fname;
      ifiles_[i].reset(new IFile());
      ifiles_[i]->link(*this);
    }
  }
  log.printf("  WalkerAggregator using %d walkers (id %d) method=%s components=%u (comm=%s)\n",mw_n_,mw_id_,method_.c_str(),ncomp_, walkers_mpi_?"MPI":"FILES");
}

void WalkerAggregator::setupCollectComponents() {
  // In COLLECT mode, expose either current values per walker, or if HISTORY_NT>0, expose k=0..NT history slices per walker.
  // Naming:
  //  - single-component ARG: w<id> (no history) or w<id>.<k>
  //  - multi-component ARG: w<id>.<comp> (no history) or w<id>.<comp>.<k>
  if(history_nt_==0){
    for(int w=0; w<mw_n_; ++w) {
      for(unsigned c=0; c<ncomp_; ++c) {
        std::string cname = "w" + std::to_string(w);
        if(ncomp_>1) cname += "." + getPntrToArgument(c)->getName();
        addComponentWithDerivatives(cname);
        getPntrToComponent(w*ncomp_+c)->setNotPeriodic();
      }
    }
  } else {
    for(int w=0; w<mw_n_; ++w) {
      for(unsigned c=0; c<ncomp_; ++c) {
        for(int k=0; k<=history_nt_; ++k){
          std::string cname = "w" + std::to_string(w);
          if(ncomp_>1) cname += "." + getPntrToArgument(c)->getName();
          cname += "." + std::to_string(k);
          addComponentWithDerivatives(cname);
          // Flat index: (w * ncomp_ + c) * (history_nt_+1) + k
          getPntrToComponent( (w*ncomp_ + c)*(history_nt_+1) + k )->setNotPeriodic();
        }
      }
    }
  }
}

void WalkerAggregator::openOutputIfNeeded(){
  if(value_ofile_.isOpen()) return;
  std::string fname=file_prefix_+".WVALUES."+std::to_string(mw_id_);
  if(!mw_dir_.empty()) fname=mw_dir_+"/"+fname;
  if(retention_mode_=="APPEND"){
    value_ofile_.link(*this);
    value_ofile_.enforceBackup();
    value_ofile_.open(fname);
    value_ofile_.addConstantField("step");
    value_ofile_.addConstantField("time");
    for(unsigned i=0;i<ncomp_;++i){
      value_ofile_.addConstantField(getPntrToArgument(i)->getName());
    }
    // Set precision for doubles
    value_ofile_.fmtField("%.14f");
  }
}

void WalkerAggregator::writeLocalValues(const std::vector<double>& basevals){
  // Only write on stride-aligned steps to synchronize across walkers
  if( (getStep() % rstride_) != 0 ) {
    // still update local cache for use in MEAN/COLLECT
    std::vector<double> sanitized(ncomp_);
    for(unsigned i=0;i<ncomp_;++i){
      double v = basevals[i];
      if(!(std::isfinite(v))) v = 0.0;
      sanitized[i]=v;
    }
    last_values_[mw_id_]=sanitized;
    have_value_[mw_id_]=true;
    last_step_seen_[mw_id_] = static_cast<long long>(getStep());
    return;
  }
  // sanitize time and values to avoid writing NaNs that downstream readers may reject
  double t = getTime();
  if(!(std::isfinite(t))) t = 0.0;
  std::vector<double> sanitized(ncomp_);
  for(unsigned i=0;i<ncomp_;++i){
    double v = basevals[i];
    if(!(std::isfinite(v))) v = 0.0;
    sanitized[i]=v;
  }
  if(retention_mode_=="APPEND"){
    openOutputIfNeeded();
    value_ofile_.printField("step", static_cast<int>(getStep()));
    value_ofile_.printField("time", t);
    for(unsigned i=0;i<ncomp_;++i){
      value_ofile_.printField(getPntrToArgument(i)->getName(), sanitized[i]);
    }
    value_ofile_.printField();
    value_ofile_.flush(); // small file; cost negligible
  } else { // LATEST: truncate and write a single complete block atomically
    std::string fname=file_prefix_+".WVALUES."+std::to_string(mw_id_);
    if(!mw_dir_.empty()) fname=mw_dir_+"/"+fname;
    writeLatestBlockAtomic(fname, static_cast<long long>(getStep()), t, sanitized);
  }
  last_values_[mw_id_]=sanitized;
  have_value_[mw_id_]=true;
  last_step_seen_[mw_id_] = static_cast<long long>(getStep());
  // Only push to history deque on stride-aligned steps
  if(history_nt_>0 && (getStep() % rstride_) == 0){
    updateHistoryIfAdvanced(mw_id_);
  }
}

void WalkerAggregator::writeLatestBlockAtomic(const std::string& fname, long long step, double time, const std::vector<double>& vals){
  // Write to a temporary file and atomically rename to target to avoid readers seeing partial blocks
  std::string tmp = fname + ".tmp";
  {
    std::ofstream os(tmp.c_str(), std::ios::trunc | std::ios::out);
    if(!os.good()) return; // silently ignore; reader will use last value
    // FIELDS header
    os << "#! FIELDS step time";
    for(unsigned i=0;i<ncomp_;++i){ os << " " << getPntrToArgument(i)->getName(); }
    os << "\n";
    // complete block
    os << "#! SET step " << step << "\n";
    os.setf(std::ios::fixed); os.precision(14);
    os << "#! SET time " << time << "\n";
    for(unsigned i=0;i<ncomp_;++i){
      os << "#! SET " << getPntrToArgument(i)->getName() << " " << vals[i] << "\n";
    }
    os.flush();
  }
  // atomic rename
  std::rename(tmp.c_str(), fname.c_str());
}

void WalkerAggregator::readOthers(){
  // Read per-walker files that are written with the PLUMED replica suffix (.<walker_id>)
  // and extract the most recent '#! SET <name> <value>' entries for our component names.
  for(int i=0;i<mw_n_;++i){
    if(i==mw_id_) continue;
    // Candidate filenames to try
    std::string base = ifilenames_[i];
    std::vector<std::string> candidates;
    candidates.push_back(base + "." + std::to_string(i)); // preferred: writer's suffix
    candidates.push_back(base); // fallback: no suffix
    bool opened=false; std::ifstream is;
    for(const auto& cand : candidates){
      is.close();
      is.clear();
      is.open(cand.c_str());
      if(is.good()){ opened=true; break; }
    }
    if(!opened) continue; // file not yet present
    // Robust block parser: commit only complete blocks to avoid partial reads during writer flush
    std::vector<double> block_vals(ncomp_, 0.0);
    std::vector<char> block_has(ncomp_, 0);
    long long block_step = -1;
    long long committed_step = -1;
    std::vector<double> committed_vals = last_values_[i];
    auto commit_block = [&](void){
      // Require all components present to commit (prevents trailing by one entry)
      bool complete=true; for(unsigned c=0;c<ncomp_;++c) if(!block_has[c]) { complete=false; break; }
      if(complete && block_step>=0){
        committed_vals = block_vals;
        committed_step = block_step;
      }
    };
    std::string line;
    while(std::getline(is, line)){
      if(line.rfind("#! FIELDS", 0) == 0){
        // New block starts; commit previous
        commit_block();
        // reset current block
        std::fill(block_has.begin(), block_has.end(), 0);
        block_step = -1;
        continue;
      }
      if(line.size()<3 || !(line[0]=='#' && line[1]=='!')) continue;
      std::vector<std::string> tok = Tools::getWords(line);
      if(tok.size()<3 || tok[1]!="SET") continue;
      const std::string& key = tok[2];
      if(key=="step"){
        if(tok.size()>=4){
          char* e=nullptr; long long s = std::strtoll(tok[3].c_str(), &e, 10);
          if(e!=tok[3].c_str()) block_step = s;
        }
        continue;
      }
      if(tok.size()<4) continue;
      for(unsigned c=0;c<ncomp_;++c){
        if(key==getPntrToArgument(c)->getName()){
          char* endptr=nullptr; double v=std::strtod(tok[3].c_str(), &endptr);
          if(endptr!=tok[3].c_str() && std::isfinite(v)){
            block_vals[c]=v;
            block_has[c]=1;
          }
          break;
        }
      }
    }
    // commit the last block at EOF
    commit_block();
    if(committed_step>=0){
      last_values_[i] = committed_vals;
      have_value_[i] = true;
      last_step_seen_[i] = committed_step;
      if(history_nt_>0){
        updateHistoryIfAdvanced(i);
      }
    }
  }
  // Update readiness
  if(!ready_){
    bool all=true;
    // If using stride, only consider ready if all walkers have a value for the current stride step
    long long targetStep = static_cast<long long>(getStep());
    if(rstride_>1) targetStep -= (targetStep % rstride_);
    for(int i=0;i<mw_n_;++i){
      if(!have_value_[i]) { all=false; break; }
      if(rstride_>1 && last_step_seen_[i] < targetStep) { all=false; break; }
    }
    ready_=all;
  }
}

void WalkerAggregator::calculate(){
  // One-time startup wait to mitigate initial desync across replicas
  if(!startup_wait_done_ && startup_wait_sec_ > 0.0){
    // We use nanosleep for better portability/precision
    double secs = startup_wait_sec_;
    long sec = static_cast<long>(secs);
    long nsec = static_cast<long>((secs - static_cast<double>(sec)) * 1e9);
    struct timespec req{sec, nsec<0?0:nsec};
    while(nanosleep(&req, &req)==-1) {
      // interrupted by signal, continue sleeping remaining time
      continue;
    }
    startup_wait_done_ = true;
  }
  // base values (local ARGs)
  std::vector<double> basevals(ncomp_);
  for(unsigned i=0;i<ncomp_;++i) basevals[i]=getArgument(i);
  if(walkers_mpi_){
    gatherViaMPI(basevals);
  } else {
    writeLocalValues(basevals);
    readOthers();
  }
  if(method_=="COLLECT") {
    // Expose all walker values as components: w0, w1, ...
    // If not ready, fill with local values for missing walkers. In history mode, only k=0 is meaningful before ready.
    if(history_nt_==0){
      for(int w=0; w<mw_n_; ++w) {
        for(unsigned c=0; c<ncomp_; ++c) {
          Value* comp = getPntrToComponent(w*ncomp_+c);
          comp->clearDerivatives();
          double v;
          if(w==mw_id_) {
            v = basevals[c];
            setDerivative(comp, c, 1.0); // local derivative
          } else {
            v = have_value_[w] ? last_values_[w][c] : basevals[c];
          }
          comp->set(v);
        }
      }
    } else {
      const int slot = history_nt_ + 1;
      for(int w=0; w<mw_n_; ++w) {
        for(unsigned c=0; c<ncomp_; ++c) {
          for(int k=0; k<slot; ++k){
            // flat index
            Value* comp = getPntrToComponent( (w*ncomp_ + c)*slot + k );
            comp->clearDerivatives();
            double v;
            if(k==0){
              if(w==mw_id_) {
                v = basevals[c];
                setDerivative(comp, c, 1.0);
              } else {
                v = have_value_[w] ? last_values_[w][c] : basevals[c];
              }
            } else {
              // historical slots only populated once we have pushed history; fallback to current
              if(static_cast<int>(history_[w].size())>k){
                v = history_[w][k][c];
              } else {
                v = (w==mw_id_) ? basevals[c] : (have_value_[w] ? last_values_[w][c] : basevals[c]);
              }
            }
            comp->set(v);
          }
        }
      }
    }
    return;
  }
  // MEAN mode (default):
  std::vector<double> agg(ncomp_,0.0);
  int denom=0;
  if(wait_all_){
    if(!ready_){
      // not ready: return local values; derivative 1.0 (identity) so downstream biases act on base CV only until ready
      for(unsigned i=0;i<ncomp_;++i){
        Value* comp=getPntrToComponent(i);
        comp->clearDerivatives();
        setDerivative(comp, i, 1.0); // derivative wrt own local argument only
        comp->set(basevals[i]);
      }
      return;
    }
    denom=mw_n_;
  } else {
    // include only walkers with a value; ensure at least local
    for(int i=0;i<mw_n_;++i) if(have_value_[i]) denom++;
    if(denom==0){ denom=1; }
  }
  double invdenom=1.0/static_cast<double>(denom);
  for(unsigned c=0;c<ncomp_;++c){
    double sum=0.0;
    if(wait_all_){
      for(int i=0;i<mw_n_;++i) sum+= last_values_[i][c];
    } else {
      for(int i=0;i<mw_n_;++i) if(have_value_[i]) sum+= last_values_[i][c]; else sum+= basevals[c];
    }
    agg[c]=sum*invdenom;
  }
  // Set value and derivative (mean -> derivative 1/N wrt local component, zero wrt others)
  for(unsigned c=0;c<ncomp_;++c){
    Value* comp=getPntrToComponent(c);
    // clear existing derivatives
    comp->clearDerivatives();
    // derivative index mapping: arguments are contiguous and correspond to components
    setDerivative(comp, c, invdenom);
    comp->set(agg[c]);
  }
}

void WalkerAggregator::gatherViaMPI(const std::vector<double>& basevals){
  // Update local cache regardless, for use in non-stride steps and outputs
  std::vector<double> sanitized(ncomp_);
  for(unsigned i=0;i<ncomp_;++i){
    double v = basevals[i];
    if(!(std::isfinite(v))) v=0.0;
    sanitized[i]=v;
  }
  last_values_[mw_id_] = sanitized;
  have_value_[mw_id_] = true;
  last_step_seen_[mw_id_] = static_cast<long long>(getStep());

  // Only synchronize on stride-aligned steps
  if( (getStep() % rstride_) != 0 ) {
    // readiness stays false until the first stride sync (or stays true once achieved)
    return;
  }

  // Gather all walkers' values (size = mw_n_ * ncomp_)
  std::vector<double> all_vals(mw_n_ * ncomp_, 0.0);
  // CRITICAL: multi_sim_comm only connects rank 0 of each simulation.
  // Only rank 0 should participate in the collective, then broadcast to other ranks.
  if(comm.Get_rank()==0) {
    // fill my slot
    for(unsigned c=0;c<ncomp_;++c) all_vals[mw_id_*ncomp_+c] = sanitized[c];
    // MPI allgather: gather ncomp_ values from each walker into all_vals on rank 0s
    std::vector<double> send = sanitized; // size ncomp_
    multi_sim_comm.Allgather(send, all_vals);
  }
  // Broadcast the gathered values to all ranks within each simulation
  comm.Bcast(all_vals, 0);

  // Commit into last_values_
  for(int w=0; w<mw_n_; ++w){
    for(unsigned c=0;c<ncomp_;++c){
      last_values_[w][c] = all_vals[w*ncomp_+c];
    }
    have_value_[w] = true;
    last_step_seen_[w] = static_cast<long long>(getStep());
    if(history_nt_>0){
      updateHistoryIfAdvanced(w);
    }
  }

  // All present and stride-aligned -> ready
  ready_ = true;
}

void WalkerAggregator::updateHistoryIfAdvanced(int walker_index){
  if(history_nt_<=0) return;
  const long long s = last_step_seen_[walker_index];
  if(s<0) return;
  // Only push new entry if this step is newer than the last pushed step for this walker
  if(last_hist_step_[walker_index] == s) return;
  // push_front current last_values_ snapshot
  if(static_cast<int>(history_[walker_index].size())==0 || last_hist_step_[walker_index] != s){
    history_[walker_index].push_front(last_values_[walker_index]);
    if(static_cast<int>(history_[walker_index].size()) > history_nt_+1){
      history_[walker_index].pop_back();
    }
    last_hist_step_[walker_index] = s;
  }
}

} // namespace bias
} // namespace PLMD

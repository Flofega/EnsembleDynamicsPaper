/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   WalkerAggregator: aggregate CV values across multiple walkers (filesystem)
   Copyright (c) 2025
   Part of PLUMED (see PEOPLE file)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifndef __PLUMED_bias_WalkerAggregator_h
#define __PLUMED_bias_WalkerAggregator_h

#include "Function.h"
#include "tools/OFile.h"
#include "tools/IFile.h"
#include <deque>

namespace PLMD {
namespace bias {

//+PLUMEDOC FUNCTION WALKER_AGGREGATOR
/*
Aggregate the instantaneous values of a base collective variable across multiple walkers.

This action is intended for use with multi-walker metadynamics in *aggregated* mode, where the bias is
applied to a secondary CV that is a function of the per-walker CV values (e.g. the mean). Each walker writes
its local CV value(s) to a small text file in a shared directory and reads the latest values from the other
walkers. Two operation modes are supported:
  - METHOD=MEAN (default): compute the mean across walkers for each input component and expose it as the value.
    Derivatives are scaled by 1/N so that forces from a downstream bias are distributed equally among walkers.
  - METHOD=COLLECT: expose all per-walker values as components (w0, w1, ...). This lets you build new CVs that
    operate on the set of walker values (e.g., the distance or difference between walkers) and bias those.

Keywords:
  - ARG: the underlying base CV (can be multi-component).
  - WALKERS_N: total number of walkers.
  - WALKERS_ID: id of this walker (0-based).
  - WALKERS_DIR: directory where value files are shared (defaults to current directory).
  - FILE_PREFIX: optional prefix for value files (default action label).
  - METHOD: aggregation method: MEAN (default) or COLLECT.
  - HISTORY_NT: keep a history window of the last NT stride-synchronized values per walker and expose them in
    COLLECT mode. Components are named w<id>.<k> (k=0 is the most recent, k=NT the oldest in the window). For
    multi-component ARG, names become w<id>.<comp>.<k>.
  - WAIT_ALL: (default=TRUE) if set FALSE the aggregate uses available values (missing walkers replaced by local value) until all appear.

The per-walker files are named <FILE_PREFIX>.WVALUES.<id> and contain one line per MD step:
  step time v1 [v2 ...]

Example (two walkers averaging a distance before metadynamics):
\plumedfile
d: DISTANCE ATOMS=1,2
agg: WALKER_AGGREGATOR ARG=d WALKERS_N=2 WALKERS_ID=0 WALKERS_DIR=../shared METHOD=MEAN
METAD ARG=agg.* SIGMA=0.1 HEIGHT=1.2 PACE=500 AGGREGATED
\endplumedfile

Example (two walkers, biasing the absolute difference between their per-walker distances):
\plumedfile
cv: DISTANCE ATOMS=1,2
agg: WALKER_AGGREGATOR ARG=cv WALKERS_N=2 WALKERS_ID=@replicas:{0,1} METHOD=COLLECT WAIT_ALL
absdiff: CUSTOM ARG=agg.w0,agg.w1 VAR=a,b FUNC=sqrt((a-b)*(a-b)) PERIODIC=NO
METAD ARG=absdiff SIGMA=0.1 HEIGHT=1 PACE=2 AGGREGATED
\endplumedfile

*/
//+ENDPLUMEDOC

class WalkerAggregator : public function::Function {
private:
  unsigned ncomp_;                 // number of input components (size of ARG list)
  int mw_n_;                       // total walkers
  int mw_id_;                      // this walker id
  std::string mw_dir_;             // shared directory
  std::string file_prefix_;        // file name prefix
  std::string method_;             // aggregation method (MEAN or COLLECT)
  std::string retention_mode_;     // WVALUES_RETENTION: APPEND (default) or LATEST (truncate to latest)
  bool wait_all_;                  // wait until all walkers seen
  bool walkers_mpi_ = false;       // use MPI for communication instead of files
  bool ready_;                     // have we gathered at least one value from all walkers
  int rstride_ = 1;                // synchronized stride for writing/reading (WALKERS_RSTRIDE)
  double startup_wait_sec_ = 0.0;  // one-time wait at start (STARTUP_WAIT)
  bool startup_wait_done_ = false; // guard to ensure we wait only once
  int history_nt_ = 0;             // number of past stride-aligned values to keep (0 = only current)
  OFile value_ofile_;              // writer for this walker
  std::vector<std::unique_ptr<IFile>> ifiles_; // readers for other walkers
  std::vector<std::string> ifilenames_;
  std::vector< std::vector<double> > last_values_; // last values per walker (size mw_n_ x ncomp_)
  std::vector<bool> have_value_;   // per walker flag
  std::vector<long long> last_step_seen_; // last step index seen per walker (from files; local updated on write)
  std::vector<long long> last_hist_step_; // last step we pushed into history for each walker
  std::vector< std::deque< std::vector<double> > > history_; // per walker deque of size <= history_nt_+1, element 0 = most recent

  void openOutputIfNeeded();
  void writeLocalValues(const std::vector<double>& basevals);
  void readOthers();
  void setupCollectComponents();
  void writeLatestBlockAtomic(const std::string& fname, long long step, double time, const std::vector<double>& vals);
  void gatherViaMPI(const std::vector<double>& basevals);
  void updateHistoryIfAdvanced(int walker_index);
public:
  explicit WalkerAggregator(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculate() override;
  bool isReady() const { return ready_; }
};

} // namespace bias
} // namespace PLMD

#endif

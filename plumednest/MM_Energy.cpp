/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Minimal bonded-only internal energy CV (MM_ENERGY)
   Computes harmonic bond energy for a specified atom GROUP with analytic derivatives.
   Parameter file format (1-based indices within GROUP):
     i  j  k  r0
   where k is in energy units per (length)^2 and r0 in length units of the simulation.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "tools/Pbc.h"
#include "tools/Angle.h"
#include "tools/Torsion.h"
#include "tools/OpenMP.h"
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR MM_ENERGY
/*
Compute an approximate intramolecular energy for a GROUP using simple bonded and intra-group nonbonded terms with analytic derivatives.

This action is designed to provide a biasable "protein-only" energy-like CV. It reads lightweight topology files (local to the GROUP) to derive connectivity for exclusions and optional 1–4 scaling, and it can add simple Lennard–Jones (12–6) and/or Coulomb electrostatics among atoms in the GROUP.

Mandatory keywords
------------------
- GROUP: the atom group for which the CV is computed.
- BONDS_FILE: path to a bonds parameter file. Even when only nonbonded is used, this file is parsed to infer connectivity (you may pass an empty placeholder if desired; see below).

Optional topology keywords
--------------------------
- ANGLES_FILE: path to angles parameter file.
- DIHEDRALS_FILE: path to dihedrals parameter file.

The above three files use indices local to GROUP (1-based). Only minimal parameters are required; force constants can be zero if you only want exclusions/scaling.

File formats
------------
1) Bonds (required):
\verbatim
# i j k r0
1 2 30000.0 0.153
2 3 30000.0 0.153
\endverbatim
Here k is the bond force constant (energy / length^2) and r0 the equilibrium length. Set k=0 if you only want to inform 1–2 exclusions for nonbonded.

2) Angles (optional):
\verbatim
# i j k k theta0deg
1 2 3 100.0 180.0
\endverbatim
Here k is the angle force constant and theta0deg is in degrees. Set k=0 to only inform 1–3 exclusions.

3) Dihedrals (optional):
\verbatim
# i j k l k n phi0deg
1 2 3 4 50.0 1 180.0
\endverbatim
This implements a periodic torsion of the form k*(1−cos(n*(phi−phi0))). Set k=0 if you only want to mark i–l as a 1–4 pair (for optional scaling of nonbonded).

Intra-group nonbonded options
-----------------------------
- NONBONDED_LJ: enable 12–6 Lennard–Jones among atoms in GROUP.
  - LJ_EPSILON: epsilon (energy units).
  - LJ_SIGMA: sigma (length units).
  - LJ_CUTOFF: hard cutoff (length units). No switching is applied.
  - LJ_PERATOM_FILE: optional per-atom LJ parameters. Each line is either "sigma epsilon" (sequential, one per atom in GROUP order) or "i sigma epsilon" (indexed, 1-based). If provided, pair parameters are mixed with LJ_COMB.
  - LJ_COMB: mixing rule when using per-atom LJ params. LB = Lorentz–Berthelot (sigma arithmetic, epsilon geometric). GEOM = both geometric. Default LB.

- NONBONDED_COULOMB: enable simple Coulomb electrostatics among atoms in GROUP.
  - CHARGES_FILE: charges list. Two accepted formats:
    * Sequential: one charge per line in GROUP order.
    * Indexed: lines "i q" with i 1-based in GROUP.
  - EPSILON_R: relative dielectric constant (default 1.0). The Coulomb term is scaled as k_e*q_i*q_j/(epsilon_r*r).
  - COULOMB_CUTOFF: hard cutoff (length units); 0 means no cutoff.

Exclusions and 1–4 scaling
--------------------------
- EXCLUDE_12: exclude 1–2 bonded pairs from nonbonded (derived from BONDS_FILE).
- EXCLUDE_13: exclude 1–3 angle pairs from nonbonded (derived strictly from ANGLES_FILE entries).
- SCALE_14: scaling factor for 1–4 pairs (i–l across a dihedral path). 1.0 = no scaling; 0.0 = exclude 1–4s. 1–4 pairs are taken from DIHEDRALS_FILE.

Restricting force propagation
-----------------------------
- FORCE_ATOMS: optional absolute atom list. When provided, the CV value is computed on the full GROUP but derivatives (and thus bias forces) are only propagated to the atoms listed here (those also present in GROUP). Atoms in GROUP but not in FORCE_ATOMS receive zero derivatives. This is useful to target the bias to a subset such as the protein backbone while still measuring interactions involving all GROUP atoms.

Periodicity and units
---------------------
- Uses periodic boundary conditions by default (as in DISTANCE). Use NOPBC to disable.
- All quantities use PLUMED units (set with UNITS). The Coulomb constant is internally converted so that charges are read in PLUMED charge units.

Notes
-----
- This CV is an approximation decoupled from the MD code’s full potential. It only includes the terms you enable here; e.g., if you enable NONBONDED_COULOMB it will add intra-group Coulomb using the provided charges.
- For performance, the nonbonded loops are O(N^2) within GROUP. For large groups consider reducing cutoff or splitting the group.

Examples
--------
1) Minimal bonded-only energy (bonds):
\plumedfile
prot: GROUP ATOMS=1-1000
eint: MM_ENERGY GROUP=prot BONDS_FILE=bonds.dat
PRINT ARG=eint FILE=COLVAR STRIDE=100
\endplumedfile

2) Coulomb-only protein electrostatics (protein atoms 1..N, charges in a file):
\plumedfile
UNITS LENGTH=nm TIME=ps ENERGY=kj/mol
prot: GROUP ATOMS=1-164
E: MM_ENERGY GROUP=prot \\
   BONDS_FILE=bonds_mm.dat ANGLES_FILE=angles_mm.dat DIHEDRALS_FILE=dihedrals_mm.dat \\
   NONBONDED_COULOMB CHARGES_FILE=charges_mm.dat EPSILON_R=1.0 EXCLUDE_12 EXCLUDE_13 SCALE_14=1.0 NOPBC
PRINT ARG=E STRIDE=500 FILE=COLVAR_MM_ENERGY
\endplumedfile

3) LJ + Coulomb with cutoff and 1–4 scaling:
\plumedfile
prot: GROUP ATOMS=1-164
E: MM_ENERGY GROUP=prot \\
   BONDS_FILE=bonds_mm.dat ANGLES_FILE=angles_mm.dat DIHEDRALS_FILE=dihedrals_mm.dat \\
   NONBONDED_LJ LJ_EPSILON=0.2 LJ_SIGMA=0.35 LJ_CUTOFF=1.2 \\
   NONBONDED_COULOMB CHARGES_FILE=charges_mm.dat EPSILON_R=80 COULOMB_CUTOFF=1.2 \\
   EXCLUDE_12 EXCLUDE_13 SCALE_14=0.5
PRINT ARG=E FILE=COLVAR
\endplumedfile
*/
//+ENDPLUMEDOC

class MM_Energy : public Colvar {
public:
  explicit MM_Energy(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculate() override;

private:
  struct Bond { int i, j; double k, r0; };
  struct AngleTerm { int i, j, k; double k_angle, theta0; }; // theta0 in radians
  struct DihedralTerm { int i, j, k, l; double k_tors; double phi0; int n; }; // periodic torsion: k*(1-cos(n*(phi-phi0)))
  std::vector<Bond> bonds_;
  std::vector<AngleTerm> angles_;
  std::vector<DihedralTerm> dihedrals_;
  bool pbc_ = true;

  // Nonbonded (LJ 12-6) options
  bool use_lj_ = false;
  double lj_epsilon_ = 0.0;
  double lj_sigma_   = 0.0;
  double lj_cutoff_  = 0.0;
  // Optional per-atom LJ parameters and mixing rule
  bool lj_peratom_ = false;
  std::vector<double> lj_sigma_i_;
  std::vector<double> lj_epsilon_i_;
  std::string lj_comb_ = "LB"; // LB (Lorentz-Berthelot) or GEOM
  // Nonbonded Coulomb options
  bool use_coulomb_ = false;
  double coul_epsilon_r_ = 1.0;  // relative permittivity
  double coul_cutoff_ = 0.0;     // 0 means no cutoff
  double ke_ = 0.0;              // Coulomb constant in internal PLUMED units
  std::vector<double> charges_;  // charges per atom in GROUP
  bool exclude12_ = true;
  bool exclude13_ = true;
  double scale14_ = 1.0;
  // Safety clamp for nonbonded distances (in length units). 0 disables clamping.
  double min_dist_ = 0.0;
  // Option to skip internal makeWhole() even if PBC is enabled (useful to avoid reimaging artifacts)
  bool skip_makewhole_ = false;
  // Use orthorhombic minimum-image distances for nonbonded (mirror Python logic):
  // rij = rij - L*round(rij/L) per component using box diagonal (ignores triclinic tilts).
  bool nonbonded_minimage_ortho_ = false;
  // Exclusion maps built from topology
  std::vector< std::unordered_set<int> > bonded12_; // exclude
  std::vector< std::unordered_set<int> > bonded13_; // exclude
  std::unordered_map<long,double> scale14map_;      // scaled (default factor scale14_)

  static inline long pairKey(int i,int j) { return ( (long)std::min(i,j) << 32 ) | (long)std::max(i,j); }

  // Optional: restrict derivative propagation to a subset of GROUP atoms
  bool restrict_forces_ = false;
  std::vector<char> force_mask_; // size = GROUP, 1 if derivatives apply, 0 otherwise

  // Neighbor list for nonbonded interactions (performance optimization)
  bool use_nlist_ = false;
  int nl_stride_ = 10;           // update neighbor list every N steps
  double nl_skin_ = 0.0;         // skin distance for neighbor list (cutoff + skin = actual cutoff)
  std::vector<std::pair<int,int>> nb_pairs_; // cached neighbor pairs
  std::vector<double> nb_scales_;            // cached scale factors for each pair
  long last_nl_update_ = -1;     // step of last neighbor list update
};

PLUMED_REGISTER_ACTION(MM_Energy, "MM_ENERGY")

void MM_Energy::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms","GROUP","the group of atoms for which internal bonded energy is computed");
  keys.add("compulsory","BONDS_FILE","path to a file with bond parameters (columns: i j k r0), indices 1-based within GROUP");
  keys.add("optional","ANGLES_FILE","optional file with angle parameters (columns: i j k k theta0[deg])");
  keys.add("optional","DIHEDRALS_FILE","optional file with dihedral parameters (columns: i j k l k n phi0[deg])");
  keys.addFlag("NONBONDED_LJ",false,"enable 12-6 Lennard-Jones nonbonded interactions within GROUP");
  keys.add("optional","LJ_EPSILON","global epsilon parameter for LJ (energy units)");
  keys.add("optional","LJ_SIGMA","global sigma parameter for LJ (length units)");
  keys.add("optional","LJ_CUTOFF","cutoff for LJ interactions (length units)");
  keys.add("optional","LJ_PERATOM_FILE","file with per-atom LJ parameters: either sequential lines: sigma epsilon, or indexed lines: i sigma epsilon (1-based)");
  keys.add("optional","LJ_COMB","LJ mixing rule for per-atom parameters: LB (sigma arithmetic, epsilon geometric) or GEOM (both geometric). Default LB");
  keys.addFlag("NONBONDED_COULOMB",false,"enable Coulomb electrostatics within GROUP using listed charges");
  keys.add("optional","CHARGES_FILE","file with charges (either one q per line in GROUP order, or lines: i q with 1-based indices)");
  keys.add("optional","EPSILON_R","relative dielectric constant for Coulomb (default 1.0)");
  keys.add("optional","COULOMB_CUTOFF","cutoff for Coulomb interactions (length units; 0 means no cutoff)");
  keys.addFlag("EXCLUDE_12",false,"exclude 1-2 bonded pairs from nonbonded (runtime default: on)");
  keys.addFlag("EXCLUDE_13",false,"exclude 1-3 angle pairs from nonbonded (runtime default: on)");
  keys.add("optional","SCALE_14","scale factor for 1-4 nonbonded pairs (1.0=no scaling, 0.0=exclude)");
  keys.add("atoms","FORCE_ATOMS","optional list of (absolute) atoms that should receive propagated bias; default is all atoms in GROUP");
  keys.add("optional","MIN_DIST","minimum distance clamp for nonbonded (length units); 0 disables clamping");
  keys.addFlag("SKIP_MAKEWHOLE",false,"do not call internal makeWhole(); rely on input imaging (still uses PBC in delta unless NOPBC)");
  keys.addFlag("NONBONDED_MINIMAGE_ORTHO",false,"for nonbonded terms, use orthorhombic minimum-image distances using the box diagonal lengths (x,y,z): dv -= L*round(dv/L); requires SKIP_MAKEWHOLE");
  // Neighbor list optimization for nonbonded
  keys.addFlag("NLIST",false,"use a neighbor list for nonbonded interactions (requires LJ_CUTOFF or COULOMB_CUTOFF)");
  keys.add("optional","NL_STRIDE","frequency of neighbor list updates in steps (default: 10)");
  keys.add("optional","NL_SKIN","skin distance for neighbor list; pairs within cutoff+skin are cached (default: 0.2*cutoff)");
}

MM_Energy::MM_Energy(const ActionOptions& ao) : PLUMED_COLVAR_INIT(ao) {
  // Parse group atoms
  std::vector<AtomNumber> grp;
  parseAtomList("GROUP", grp);
  if (grp.empty()) error("GROUP must list at least one atom");

  // Parse bonds/angles/dihedrals files
  std::string bonds_file;
  parse("BONDS_FILE", bonds_file);
  if (bonds_file.empty()) error("BONDS_FILE is required for MM_ENERGY");
  std::string angles_file; parse("ANGLES_FILE", angles_file);
  std::string diheds_file; parse("DIHEDRALS_FILE", diheds_file);

  bool nopbc = false; parseFlag("NOPBC", nopbc); pbc_ = !nopbc;
  // Nonbonded LJ options
  parseFlag("NONBONDED_LJ", use_lj_);
  parse("LJ_EPSILON", lj_epsilon_);
  parse("LJ_SIGMA", lj_sigma_);
  parse("LJ_CUTOFF", lj_cutoff_);
  std::string lj_peratom_file; parse("LJ_PERATOM_FILE", lj_peratom_file);
  parse("LJ_COMB", lj_comb_);
  for(char &ch: lj_comb_) ch = std::toupper(static_cast<unsigned char>(ch));
  if(lj_comb_ != "LB" && lj_comb_ != "GEOM") error("LJ_COMB must be LB or GEOM");
  // Nonbonded Coulomb options
  parseFlag("NONBONDED_COULOMB", use_coulomb_);
  parse("EPSILON_R", coul_epsilon_r_);
  parse("COULOMB_CUTOFF", coul_cutoff_);
  std::string charges_file; parse("CHARGES_FILE", charges_file);
  // Defaults: exclude 1-2 and 1-3 unless explicitly overridden
  exclude12_ = true; exclude13_ = true;
  parseFlag("EXCLUDE_12", exclude12_);
  parseFlag("EXCLUDE_13", exclude13_);
  parse("SCALE_14", scale14_);
  parse("MIN_DIST", min_dist_);
  parseFlag("SKIP_MAKEWHOLE", skip_makewhole_);
  parseFlag("NONBONDED_MINIMAGE_ORTHO", nonbonded_minimage_ortho_);
  if(nonbonded_minimage_ortho_ && !skip_makewhole_) error("NONBONDED_MINIMAGE_ORTHO requires SKIP_MAKEWHOLE");
  // Define output value and periodicity BEFORE requesting atoms
  // so derivative arrays are correctly sized.
  addValueWithDerivatives();
  setNotPeriodic();
  // Now request atoms in the provided order
  requestAtoms(grp);

  // Build force propagation mask: by default, all atoms in GROUP receive derivatives.
  // Important: parse FORCE_ATOMS BEFORE checkRead() so the keyword is consumed.
  force_mask_.assign((int)grp.size(), 1);
  std::vector<AtomNumber> force_atoms_abs;
  parseAtomList("FORCE_ATOMS", force_atoms_abs);
  if( !force_atoms_abs.empty() ) {
    // If provided, only listed absolute atoms within GROUP will receive derivatives
    restrict_forces_ = true;
    std::unordered_set<unsigned> abs_set;
    abs_set.reserve(force_atoms_abs.size());
    for(const auto& a : force_atoms_abs) abs_set.insert(a.index());
    // Map GROUP indices to absolute and mark mask accordingly
    for(int gi=0; gi<static_cast<int>(grp.size()); ++gi){
      const unsigned abs_idx = grp[gi].index();
      force_mask_[gi] = abs_set.count(abs_idx) ? 1 : 0;
    }
    int n_on = 0; for(char c: force_mask_) if(c) n_on++;
    if(n_on==0) error("FORCE_ATOMS mask selects zero atoms from GROUP");
    log.printf("  MM_ENERGY: FORCE_ATOMS enabled; propagating to %d/%d atoms in GROUP\n", n_on, (int)grp.size());
  }

  // Parse neighbor list options
  parseFlag("NLIST", use_nlist_);
  parse("NL_STRIDE", nl_stride_);
  parse("NL_SKIN", nl_skin_);
  if(use_nlist_) {
    if(nl_stride_ <= 0) nl_stride_ = 10;
    // Default skin to 20% of the larger cutoff if not specified
    double max_cut = std::max(lj_cutoff_, coul_cutoff_);
    if(max_cut <= 0.0) error("NLIST requires a positive LJ_CUTOFF or COULOMB_CUTOFF");
    if(nl_skin_ <= 0.0) nl_skin_ = 0.2 * max_cut;
    log.printf("  MM_ENERGY: using neighbor list with stride=%d skin=%.4f\n", nl_stride_, nl_skin_);
  }

  // Finalize options parsing
  checkRead();

  // Units-aware Coulomb constant (kJ/mol, nm, e): 138.935458111 scaled to internal units
  // Convert: constant / E_unit / L_unit * (Q_unit)^2
  ke_ = 138.935458111;

  // Load bonds
  std::ifstream in(bonds_file.c_str());
  if(!in) error("Cannot open BONDS_FILE: " + bonds_file);
  std::string line;
  int lineno=0; int nbonds=0;
  while(std::getline(in,line)){
    lineno++;
    // skip comments/empty
    std::string s=line; for(char& c:s){ if(c=='\t') c=' '; }
    std::istringstream iss(s);
    if(s.empty() || s[0]=='#') continue;
    // Try to read four tokens
    int i,j; double k,r0;
  if(!(iss>>i>>j>>k>>r0)) continue; // ignore malformed lines silently
  if(i<=0 || j<=0) error(std::string("BONDS_FILE indices must be 1-based within GROUP at line ") + std::to_string(lineno));
    Bond b; b.i=i-1; b.j=j-1; b.k=k; b.r0=r0; bonds_.push_back(b); nbonds++;
  }
  if(nbonds==0) {
    if(!angles_file.empty() || !diheds_file.empty() || use_lj_ || use_coulomb_) {
      log.printf("  MM_ENERGY: no bonds read from %s; proceeding because other terms are present (angles/dihedrals/LJ)\n", bonds_file.c_str());
    } else {
      error("No valid bonds read and no other terms enabled; provide at least one bonded term or enable NONBONDED_LJ");
    }
  }

  // Load angles (optional)
  if(!angles_file.empty()){
    std::ifstream ain(angles_file.c_str());
    if(!ain) error("Cannot open ANGLES_FILE: " + angles_file);
    std::string line2; int lno=0; int nang=0;
    while(std::getline(ain,line2)){
      lno++; std::string s=line2; for(char& c:s){ if(c=='\t') c=' '; }
      if(s.empty() || s[0]=='#') continue;
      std::istringstream iss(s);
  int i,j,k; double kk, t0deg; if(!(iss>>i>>j>>k>>kk>>t0deg)) continue;
  if(i<=0||j<=0||k<=0) error(std::string("ANGLES_FILE indices must be 1-based within GROUP at line ")+std::to_string(lno));
  AngleTerm at; at.i=i-1; at.j=j-1; at.k=k-1; at.k_angle=kk; at.theta0=t0deg*M_PI/180.0; angles_.push_back(at); nang++;
    }
    log.printf("  MM_ENERGY: loaded %d angles\n", (int)angles_.size());
  }

  // Load dihedrals (optional)
  if(!diheds_file.empty()){
    std::ifstream din(diheds_file.c_str());
    if(!din) error("Cannot open DIHEDRALS_FILE: " + diheds_file);
    std::string line3; int lno2=0; int ndih=0;
    while(std::getline(din,line3)){
      lno2++; std::string s=line3; for(char& c:s){ if(c=='\t') c=' '; }
      if(s.empty() || s[0]=='#') continue;
      std::istringstream iss(s);
  int i,j,k,l,n; double kk, p0deg; if(!(iss>>i>>j>>k>>l>>kk>>n>>p0deg)) continue;
      if(i<=0||j<=0||k<=0||l<=0) error(std::string("DIHEDRALS_FILE indices must be 1-based within GROUP at line ")+std::to_string(lno2));
  DihedralTerm dt; dt.i=i-1; dt.j=j-1; dt.k=k-1; dt.l=l-1; dt.k_tors=kk; dt.n=n; dt.phi0=p0deg*M_PI/180.0; dihedrals_.push_back(dt); ndih++;
    }
    log.printf("  MM_ENERGY: loaded %d dihedrals\n", (int)dihedrals_.size());
  }

  // Build LJ exclusions based on topology
  const int nat = (int)grp.size();
  bonded12_.assign(nat, {});
  bonded13_.assign(nat, {});
  for(const auto& b : bonds_) { bonded12_[b.i].insert(b.j); bonded12_[b.j].insert(b.i); }
  for(const auto& a : angles_) { bonded13_[a.i].insert(a.k); bonded13_[a.k].insert(a.i); }
  // If no angles were provided but EXCLUDE_13 is requested, infer 1-3 pairs from bonds
  if(angles_.empty() && exclude13_ && !bonds_.empty()){
    std::vector< std::unordered_set<int> > nbrs(nat);
    for(const auto& b : bonds_) { nbrs[b.i].insert(b.j); nbrs[b.j].insert(b.i); }
    for(int j=0;j<nat;++j){
      const auto& neigh = nbrs[j];
      if(neigh.size()<2) continue;
      for(auto it1=neigh.begin(); it1!=neigh.end(); ++it1){
        auto it2 = it1; ++it2;
        for(; it2!=neigh.end(); ++it2){
          int i=*it1, k=*it2; // i and k are neighbors of j
          bonded13_[i].insert(k); bonded13_[k].insert(i);
        }
      }
    }
    log.printf("  MM_ENERGY: inferred 1-3 exclusions from bonds (no ANGLES_FILE)\n");
  }
  if(!dihedrals_.empty() && scale14_!=1.0) {
    for(const auto& d : dihedrals_) {
      scale14map_[ pairKey(d.i,d.l) ] = scale14_;
    }
  }

  if(use_lj_) {
    if(!lj_peratom_file.empty()) {
      // Load per-atom LJ parameters
      lj_peratom_ = true;
      const int nat = getNumberOfAtoms();
      lj_sigma_i_.assign(nat, 0.0);
      lj_epsilon_i_.assign(nat, 0.0);
      std::vector<char> filled(nat, 0);
      std::ifstream ljf(lj_peratom_file.c_str());
      if(!ljf) error("Cannot open LJ_PERATOM_FILE: " + lj_peratom_file);
      std::string ll; int lno=0; bool mode_indexed=false; bool mode_seq=false; int seq_count=0;
      while(std::getline(ljf, ll)){
        lno++; std::string s=ll; for(char& c:s){ if(c=='\t') c=' '; }
        if(s.empty() || s[0]=='#') continue;
        std::istringstream iss(s);
        std::vector<std::string> toks; std::string t; while(iss>>t) toks.push_back(t);
        if(toks.empty()) continue;
        if(toks.size()==2){
          if(mode_indexed) error("LJ_PERATOM_FILE mixes indexed and sequential formats");
          mode_seq=true; seq_count++;
          double sig, eps; std::istringstream s0(toks[0]), s1(toks[1]);
          if(!(s0>>sig) || !(s1>>eps)) error(std::string("LJ_PERATOM_FILE invalid entry at line ")+std::to_string(lno));
          if(seq_count>nat) error("LJ_PERATOM_FILE sequential format has more lines than atoms");
          lj_sigma_i_[seq_count-1]=sig; lj_epsilon_i_[seq_count-1]=eps; filled[seq_count-1]=1;
        } else if(toks.size()==3){
          if(mode_seq) error("LJ_PERATOM_FILE mixes sequential and indexed formats");
          mode_indexed=true;
          int idx; double sig, eps; std::istringstream s0(toks[0]), s1(toks[1]), s2(toks[2]);
          if(!(s0>>idx) || !(s1>>sig) || !(s2>>eps)) error(std::string("LJ_PERATOM_FILE invalid indexed entry at line ")+std::to_string(lno));
          if(idx<=0 || idx>nat) error(std::string("LJ_PERATOM_FILE index out of range at line ")+std::to_string(lno));
          lj_sigma_i_[idx-1]=sig; lj_epsilon_i_[idx-1]=eps; filled[idx-1]=1;
        } else {
          error(std::string("LJ_PERATOM_FILE invalid format at line ")+std::to_string(lno));
        }
      }
      for(int i=0;i<nat;i++) if(!filled[i]) error("LJ_PERATOM_FILE missing entry for atom index "+std::to_string(i+1));
      if(lj_cutoff_<=0.0) error("When NONBONDED_LJ is set with LJ_PERATOM_FILE, LJ_CUTOFF must be > 0");
      log.printf("  MM_ENERGY LJ(per-atom): comb=%s cutoff=%.6g; excl12=%s excl13=%s scale14=%.3g\n",
                 lj_comb_.c_str(), lj_cutoff_, exclude12_?"on":"off", exclude13_?"on":"off", scale14_);
    } else {
      if(lj_epsilon_<=0.0 || lj_sigma_<=0.0 || lj_cutoff_<=0.0) error("When NONBONDED_LJ is set, LJ_EPSILON, LJ_SIGMA, and LJ_CUTOFF must be > 0");
      log.printf("  MM_ENERGY LJ: epsilon=%.6g sigma=%.6g cutoff=%.6g; excl12=%s excl13=%s scale14=%.3g\n",
                 lj_epsilon_, lj_sigma_, lj_cutoff_, exclude12_?"on":"off", exclude13_?"on":"off", scale14_);
    }
  }

  // Load charges for Coulomb
  if(use_coulomb_) {
    if(charges_file.empty()) error("When NONBONDED_COULOMB is set, CHARGES_FILE must be provided");
    charges_.assign(nat, 0.0);
    std::vector<bool> filled(nat,false);
    std::ifstream cinf(charges_file.c_str());
    if(!cinf) error("Cannot open CHARGES_FILE: " + charges_file);
    std::string cl; int linec=0; bool mode_indexed=false; bool mode_seq=false; std::vector<double> seq;
    while(std::getline(cinf,cl)){
      linec++; std::string s=cl; for(char& c:s){ if(c=='\t') c=' '; }
      if(s.empty() || s[0]=='#') continue;
      // split tokens
      std::istringstream iss_tok(s);
      std::vector<std::string> toks; std::string tok;
      while(iss_tok>>tok) toks.push_back(tok);
      if(toks.empty()) continue;
      if(toks.size()==1) {
        if(mode_indexed) error("CHARGES_FILE mixes indexed and sequential formats");
        mode_seq=true;
        double q1; std::istringstream ss(toks[0]); if(!(ss>>q1)) error(std::string("CHARGES_FILE invalid charge at line ")+std::to_string(linec));
        seq.push_back(q1);
      } else if(toks.size()==2) {
        if(mode_seq) error("CHARGES_FILE mixes sequential and indexed formats");
        mode_indexed=true;
        int idx; double q;
        std::istringstream s0(toks[0]); std::istringstream s1(toks[1]);
        if(!(s0>>idx) || !(s1>>q)) error(std::string("CHARGES_FILE invalid indexed entry at line ")+std::to_string(linec));
        if(idx<=0 || idx>nat) error(std::string("CHARGES_FILE index out of range at line ")+std::to_string(linec));
        charges_[idx-1]=q; filled[idx-1]=true;
      } else {
        error(std::string("CHARGES_FILE invalid format at line ")+std::to_string(linec));
      }
    }
    if(mode_indexed) {
      for(int i=0;i<nat;i++) if(!filled[i]) error("CHARGES_FILE missing charge for atom index " + std::to_string(i+1));
    } else {
      if((int)seq.size()!=nat) error("CHARGES_FILE sequential format must provide one charge per atom in GROUP order");
      for(int i=0;i<nat;i++) charges_[i]=seq[i];
    }
    if(coul_epsilon_r_<=0.0) error("EPSILON_R must be > 0");
    // Log Coulomb setup
    if(coul_cutoff_>0.0) {
      log.printf("  MM_ENERGY Coulomb: eps_r=%.6g cutoff=%.6g; excl12=%s excl13=%s scale14=%.3g\n",
                 coul_epsilon_r_, coul_cutoff_, exclude12_?"on":"off", exclude13_?"on":"off", scale14_);
    } else {
      log.printf("  MM_ENERGY Coulomb: eps_r=%.6g no cutoff; excl12=%s excl13=%s scale14=%.3g\n",
                 coul_epsilon_r_, exclude12_?"on":"off", exclude13_?"on":"off", scale14_);
    }
  }

  log.printf("  MM_ENERGY: %d atoms in GROUP, %d bonds loaded (%s)\n", (int)grp.size(), (int)bonds_.size(), pbc_?"PBC":"no PBC");
}

void MM_Energy::calculate() {
  if(pbc_ && !skip_makewhole_) makeWhole();

  double E = 0.0;
  // zero derivatives first
  clearDerivatives();

  // (No component reimaging: nonbonded minimum image can be enforced orthorhombically via NONBONDED_MINIMAGE_ORTHO)

  for(const auto& b : bonds_) {
    // positions are in requested order
    // delta(a,b) returns b - a, so rij = r_j - r_i
    const Vector rij = delta(getPosition(b.i), getPosition(b.j));
    const double r = rij.modulo();
    if(r<=0) continue; // safeguard
    const double dr = r - b.r0;
    const double dE_dr = b.k * dr;              // dE/dr for 1/2*k*(r-r0)^2 is k*(r-r0)
    const double E_bond = 0.5 * b.k * dr * dr;  // energy
    E += E_bond;
    const Vector dir = (1.0/r) * rij;           // unit vector from i to j (since rij = r_j - r_i)
    // dr/d(r_i) = -(r_j - r_i)/|r_j - r_i| = -dir
    // dr/d(r_j) = +(r_j - r_i)/|r_j - r_i| = +dir
    // dE/d(r_i) = dE_dr * dr/d(r_i) = dE_dr * (-dir)
    // dE/d(r_j) = dE_dr * dr/d(r_j) = dE_dr * (+dir)
    const Vector dE_dri = -dE_dr * dir;
    if(!restrict_forces_ || force_mask_[b.i]) setAtomsDerivatives(b.i,  dE_dri);
    if(!restrict_forces_ || force_mask_[b.j]) setAtomsDerivatives(b.j, -dE_dri);
  }
  // Angles: harmonic 0.5*k*(theta-theta0)^2, analytic derivatives using tools::Angle
  for(const auto& a : angles_) {
    // rji = pos[j] - pos[i], rjk = pos[j] - pos[k] (vectors pointing into the vertex j)
    const Vector rji = delta(getPosition(a.i), getPosition(a.j));
    const Vector rjk = delta(getPosition(a.k), getPosition(a.j));
    Vector ddij, ddik; // derivatives: ddij = d(theta)/d(rji), ddik = d(theta)/d(rjk)
    PLMD::Angle ang;
    const double theta = ang.compute(rji, rjk, ddij, ddik);
    const double dtheta = theta - a.theta0;
    const double Eang = 0.5 * a.k_angle * dtheta * dtheta;
    E += Eang;
    const double dE_dtheta = a.k_angle * dtheta;
    // Chain rule for derivatives w.r.t. atom positions:
    // rji = pos[j] - pos[i]  =>  d(rji)/d(pos_i) = -I,  d(rji)/d(pos_j) = +I
    // rjk = pos[j] - pos[k]  =>  d(rjk)/d(pos_k) = -I,  d(rjk)/d(pos_j) = +I
    // dE/d(pos_i) = dE_dtheta * ddij * (-1) = -dE_dtheta * ddij
    // dE/d(pos_k) = dE_dtheta * ddik * (-1) = -dE_dtheta * ddik
    // dE/d(pos_j) = dE_dtheta * (ddij + ddik)
    const Vector dE_dri = -dE_dtheta * ddij;
    const Vector dE_drk = -dE_dtheta * ddik;
    const Vector dE_drj = dE_dtheta * (ddij + ddik);
    if(!restrict_forces_ || force_mask_[a.i]) setAtomsDerivatives(a.i, dE_dri);
    if(!restrict_forces_ || force_mask_[a.j]) setAtomsDerivatives(a.j, dE_drj);
    if(!restrict_forces_ || force_mask_[a.k]) setAtomsDerivatives(a.k, dE_drk);
  }

  // Dihedrals: k*(1 - cos(n*(phi-phi0))) ; dE/dphi = k*n*sin(n*(phi-phi0))
  for(const auto& d : dihedrals_) {
    // Using delta(a,b) = b - a convention:
    // r12 = pos[i] - pos[j] (vector from j to i)
    // r23 = pos[j] - pos[k] (vector from k to j)
    // r34 = pos[k] - pos[l] (vector from l to k)
    const Vector r12 = delta(getPosition(d.j), getPosition(d.i));
    const Vector r23 = delta(getPosition(d.k), getPosition(d.j));
    const Vector r34 = delta(getPosition(d.l), getPosition(d.k));
    PLMD::Torsion tor;
    Vector dv1, dv2, dv3; // derivatives dphi/d(r12), dphi/d(r23), dphi/d(r34)
    const double phi = tor.compute(r12, r23, r34, dv1, dv2, dv3);
    const double x = d.n*(phi - d.phi0);
    const double Edi = d.k_tors * (1.0 - std::cos(x));
    E += Edi;
    const double dE_dphi = d.k_tors * d.n * std::sin(x);
    // Chain rule for derivatives w.r.t. atom positions:
    // r12 = pos_i - pos_j  =>  d(r12)/d(pos_i) = +I,  d(r12)/d(pos_j) = -I
    // r23 = pos_j - pos_k  =>  d(r23)/d(pos_j) = +I,  d(r23)/d(pos_k) = -I
    // r34 = pos_k - pos_l  =>  d(r34)/d(pos_k) = +I,  d(r34)/d(pos_l) = -I
    // dE/d(pos_i) = dE_dphi * dv1
    // dE/d(pos_j) = dE_dphi * (-dv1 + dv2)
    // dE/d(pos_k) = dE_dphi * (-dv2 + dv3)
    // dE/d(pos_l) = dE_dphi * (-dv3)
    const Vector dE_dri = dE_dphi * dv1;
    const Vector dE_drj = dE_dphi * (-dv1 + dv2);
    const Vector dE_drk = dE_dphi * (-dv2 + dv3);
    const Vector dE_drl = dE_dphi * (-dv3);
    if(!restrict_forces_ || force_mask_[d.i]) setAtomsDerivatives(d.i, dE_dri);
    if(!restrict_forces_ || force_mask_[d.j]) setAtomsDerivatives(d.j, dE_drj);
    if(!restrict_forces_ || force_mask_[d.k]) setAtomsDerivatives(d.k, dE_drk);
    if(!restrict_forces_ || force_mask_[d.l]) setAtomsDerivatives(d.l, dE_drl);
  }

  // Nonbonded interactions (LJ and/or Coulomb) - optimized with neighbor list and OpenMP
  if(use_lj_ || use_coulomb_) {
    const int n = getNumberOfAtoms();
    const double max_cutoff = std::max(lj_cutoff_, coul_cutoff_);
    
    // Update neighbor list if needed
    if(use_nlist_) {
      const long current_step = getStep();
      if(last_nl_update_ < 0 || (current_step - last_nl_update_) >= nl_stride_) {
        // Rebuild neighbor list
        nb_pairs_.clear();
        nb_scales_.clear();
        const double nl_cutoff_sq = (max_cutoff + nl_skin_) * (max_cutoff + nl_skin_);
        
        for(int i=0; i<n; i++) {
          for(int j=i+1; j<n; j++) {
            // Check exclusions
            if(exclude12_ && bonded12_[i].count(j)) continue;
            if(exclude13_ && bonded13_[i].count(j)) continue;
            
            // Distance check with skin
            Vector rij;
            if(nonbonded_minimage_ortho_) {
              rij = getPosition(j) - getPosition(i);
              const Tensor& B = getBox();
              const double Lx = B[0][0], Ly = B[1][1], Lz = B[2][2];
              if(Lx>0.0){ double nx = std::round(rij[0]/Lx); rij[0] -= nx*Lx; }
              if(Ly>0.0){ double ny = std::round(rij[1]/Ly); rij[1] -= ny*Ly; }
              if(Lz>0.0){ double nz = std::round(rij[2]/Lz); rij[2] -= nz*Lz; }
            } else {
              rij = delta(getPosition(i), getPosition(j));
            }
            const double r2 = rij.modulo2();
            if(r2 >= nl_cutoff_sq) continue;
            
            // Compute scale factor once
            double scale = 1.0;
            if(!scale14map_.empty()) {
              auto it = scale14map_.find(pairKey(i,j));
              if(it != scale14map_.end()) scale = it->second;
            }
            
            nb_pairs_.emplace_back(i, j);
            nb_scales_.push_back(scale);
          }
        }
        last_nl_update_ = current_step;
      }
    }

    // OpenMP parallelization setup
    const unsigned nt = OpenMP::getNumThreads();
    const unsigned npairs = use_nlist_ ? nb_pairs_.size() : (n*(n-1))/2;
    
    // Only use OpenMP if we have enough work
    const unsigned nt_use = (nt * 10 > npairs) ? 1 : nt;
    
    if(nt_use > 1) {
      // Parallel version with thread-local derivatives
      double E_nb = 0.0;
      std::vector<std::vector<Vector>> thread_derivs(nt_use, std::vector<Vector>(n, Vector(0,0,0)));
      
      #pragma omp parallel num_threads(nt_use) reduction(+:E_nb)
      {
        const int tid = OpenMP::getThreadNum();
        std::vector<Vector>& my_deriv = thread_derivs[tid];
        
        if(use_nlist_) {
          // Use cached neighbor pairs
          #pragma omp for nowait
          for(unsigned pi=0; pi<nb_pairs_.size(); pi++) {
            const int i = nb_pairs_[pi].first;
            const int j = nb_pairs_[pi].second;
            const double scale = nb_scales_[pi];
            
            // Compute distance
            Vector rij;
            if(nonbonded_minimage_ortho_) {
              rij = getPosition(j) - getPosition(i);
              const Tensor& B = getBox();
              const double Lx = B[0][0], Ly = B[1][1], Lz = B[2][2];
              if(Lx>0.0){ double nx = std::round(rij[0]/Lx); rij[0] -= nx*Lx; }
              if(Ly>0.0){ double ny = std::round(rij[1]/Ly); rij[1] -= ny*Ly; }
              if(Lz>0.0){ double nz = std::round(rij[2]/Lz); rij[2] -= nz*Lz; }
            } else {
              rij = delta(getPosition(i), getPosition(j));
            }
            double r = rij.modulo();
            if(r <= 0.0) continue;
            
            const double invr = 1.0/r;
            const Vector dir = invr * rij;
            double dVdr_total = 0.0;
            
            // LJ contribution
            if(use_lj_ && r < lj_cutoff_) {
              if(min_dist_ > 0.0 && r < min_dist_) r = min_dist_;
              double sigma_ij = lj_sigma_, eps_ij = lj_epsilon_;
              if(lj_peratom_) {
                const double si = lj_sigma_i_[i], sj = lj_sigma_i_[j];
                const double ei = lj_epsilon_i_[i], ej = lj_epsilon_i_[j];
                if(lj_comb_=="LB") { sigma_ij = 0.5*(si+sj); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
                else { sigma_ij = std::sqrt(std::max(0.0,si*sj)); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
              }
              const double sr = sigma_ij / r;
              const double sr2 = sr*sr, sr6 = sr2*sr2*sr2, sr12 = sr6*sr6;
              E_nb += scale * 4.0 * eps_ij * (sr12 - sr6);
              dVdr_total += -24.0 * eps_ij * (2.0*sr12 - sr6) / r;
            }
            
            // Coulomb contribution
            if(use_coulomb_ && (coul_cutoff_ <= 0.0 || r < coul_cutoff_)) {
              const double qq = charges_[i] * charges_[j];
              if(qq != 0.0) {
                E_nb += scale * ke_ * qq * invr / coul_epsilon_r_;
                dVdr_total += -ke_ * qq * invr * invr / coul_epsilon_r_;
              }
            }
            
            // Accumulate derivatives
            const Vector dE_dri = -scale * dVdr_total * dir;
            if(!restrict_forces_ || force_mask_[i]) my_deriv[i] += dE_dri;
            if(!restrict_forces_ || force_mask_[j]) my_deriv[j] -= dE_dri;
          }
        } else {
          // O(N^2) loop without neighbor list
          #pragma omp for nowait
          for(int i=0; i<n; i++) {
            for(int j=i+1; j<n; j++) {
              if(exclude12_ && bonded12_[i].count(j)) continue;
              if(exclude13_ && bonded13_[i].count(j)) continue;
              
              double scale = 1.0;
              if(!scale14map_.empty()) {
                auto it = scale14map_.find(pairKey(i,j));
                if(it != scale14map_.end()) scale = it->second;
              }
              
              Vector rij;
              if(nonbonded_minimage_ortho_) {
                rij = getPosition(j) - getPosition(i);
                const Tensor& B = getBox();
                const double Lx = B[0][0], Ly = B[1][1], Lz = B[2][2];
                if(Lx>0.0){ double nx = std::round(rij[0]/Lx); rij[0] -= nx*Lx; }
                if(Ly>0.0){ double ny = std::round(rij[1]/Ly); rij[1] -= ny*Ly; }
                if(Lz>0.0){ double nz = std::round(rij[2]/Lz); rij[2] -= nz*Lz; }
              } else {
                rij = delta(getPosition(i), getPosition(j));
              }
              double r = rij.modulo();
              if(r <= 0.0) continue;
              
              const double invr = 1.0/r;
              const Vector dir = invr * rij;
              double dVdr_total = 0.0;
              
              // LJ
              if(use_lj_ && r < lj_cutoff_) {
                if(min_dist_ > 0.0 && r < min_dist_) r = min_dist_;
                double sigma_ij = lj_sigma_, eps_ij = lj_epsilon_;
                if(lj_peratom_) {
                  const double si = lj_sigma_i_[i], sj = lj_sigma_i_[j];
                  const double ei = lj_epsilon_i_[i], ej = lj_epsilon_i_[j];
                  if(lj_comb_=="LB") { sigma_ij = 0.5*(si+sj); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
                  else { sigma_ij = std::sqrt(std::max(0.0,si*sj)); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
                }
                const double sr = sigma_ij / r;
                const double sr2 = sr*sr, sr6 = sr2*sr2*sr2, sr12 = sr6*sr6;
                E_nb += scale * 4.0 * eps_ij * (sr12 - sr6);
                dVdr_total += -24.0 * eps_ij * (2.0*sr12 - sr6) / r;
              }
              
              // Coulomb
              if(use_coulomb_ && (coul_cutoff_ <= 0.0 || r < coul_cutoff_)) {
                const double qq = charges_[i] * charges_[j];
                if(qq != 0.0) {
                  E_nb += scale * ke_ * qq * invr / coul_epsilon_r_;
                  dVdr_total += -ke_ * qq * invr * invr / coul_epsilon_r_;
                }
              }
              
              const Vector dE_dri = -scale * dVdr_total * dir;
              if(!restrict_forces_ || force_mask_[i]) my_deriv[i] += dE_dri;
              if(!restrict_forces_ || force_mask_[j]) my_deriv[j] -= dE_dri;
            }
          }
        }
      } // end parallel
      
      // Merge thread-local derivatives
      for(unsigned t=0; t<nt_use; t++) {
        for(int i=0; i<n; i++) {
          if(thread_derivs[t][i].modulo2() > 0.0) {
            setAtomsDerivatives(i, thread_derivs[t][i]);
          }
        }
      }
      E += E_nb;
      
    } else {
      // Serial version (fallback for small systems or single thread)
      if(use_nlist_) {
        for(unsigned pi=0; pi<nb_pairs_.size(); pi++) {
          const int i = nb_pairs_[pi].first;
          const int j = nb_pairs_[pi].second;
          const double scale = nb_scales_[pi];
          
          Vector rij;
          if(nonbonded_minimage_ortho_) {
            rij = getPosition(j) - getPosition(i);
            const Tensor& B = getBox();
            const double Lx = B[0][0], Ly = B[1][1], Lz = B[2][2];
            if(Lx>0.0){ double nx = std::round(rij[0]/Lx); rij[0] -= nx*Lx; }
            if(Ly>0.0){ double ny = std::round(rij[1]/Ly); rij[1] -= ny*Ly; }
            if(Lz>0.0){ double nz = std::round(rij[2]/Lz); rij[2] -= nz*Lz; }
          } else {
            rij = delta(getPosition(i), getPosition(j));
          }
          double r = rij.modulo();
          if(r <= 0.0) continue;
          
          const double invr = 1.0/r;
          const Vector dir = invr * rij;
          double dVdr_total = 0.0;
          
          if(use_lj_ && r < lj_cutoff_) {
            if(min_dist_ > 0.0 && r < min_dist_) r = min_dist_;
            double sigma_ij = lj_sigma_, eps_ij = lj_epsilon_;
            if(lj_peratom_) {
              const double si = lj_sigma_i_[i], sj = lj_sigma_i_[j];
              const double ei = lj_epsilon_i_[i], ej = lj_epsilon_i_[j];
              if(lj_comb_=="LB") { sigma_ij = 0.5*(si+sj); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
              else { sigma_ij = std::sqrt(std::max(0.0,si*sj)); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
            }
            const double sr = sigma_ij / r;
            const double sr2 = sr*sr, sr6 = sr2*sr2*sr2, sr12 = sr6*sr6;
            E += scale * 4.0 * eps_ij * (sr12 - sr6);
            dVdr_total += -24.0 * eps_ij * (2.0*sr12 - sr6) / r;
          }
          
          if(use_coulomb_ && (coul_cutoff_ <= 0.0 || r < coul_cutoff_)) {
            const double qq = charges_[i] * charges_[j];
            if(qq != 0.0) {
              E += scale * ke_ * qq * invr / coul_epsilon_r_;
              dVdr_total += -ke_ * qq * invr * invr / coul_epsilon_r_;
            }
          }
          
          const Vector dE_dri = -scale * dVdr_total * dir;
          if(!restrict_forces_ || force_mask_[i]) setAtomsDerivatives(i, dE_dri);
          if(!restrict_forces_ || force_mask_[j]) setAtomsDerivatives(j, -dE_dri);
        }
      } else {
        // Original O(N^2) fallback
        for(int i=0; i<n; i++) {
          for(int j=i+1; j<n; j++) {
            if(exclude12_ && bonded12_[i].count(j)) continue;
            if(exclude13_ && bonded13_[i].count(j)) continue;
            
            double scale = 1.0;
            if(!scale14map_.empty()) {
              auto it = scale14map_.find(pairKey(i,j));
              if(it != scale14map_.end()) scale = it->second;
            }
            
            Vector rij;
            if(nonbonded_minimage_ortho_) {
              rij = getPosition(j) - getPosition(i);
              const Tensor& B = getBox();
              const double Lx = B[0][0], Ly = B[1][1], Lz = B[2][2];
              if(Lx>0.0){ double nx = std::round(rij[0]/Lx); rij[0] -= nx*Lx; }
              if(Ly>0.0){ double ny = std::round(rij[1]/Ly); rij[1] -= ny*Ly; }
              if(Lz>0.0){ double nz = std::round(rij[2]/Lz); rij[2] -= nz*Lz; }
            } else {
              rij = delta(getPosition(i), getPosition(j));
            }
            double r = rij.modulo();
            if(r <= 0.0) continue;
            
            const double invr = 1.0/r;
            const Vector dir = invr * rij;
            double dVdr_total = 0.0;
            
            if(use_lj_ && r < lj_cutoff_) {
              if(min_dist_ > 0.0 && r < min_dist_) r = min_dist_;
              double sigma_ij = lj_sigma_, eps_ij = lj_epsilon_;
              if(lj_peratom_) {
                const double si = lj_sigma_i_[i], sj = lj_sigma_i_[j];
                const double ei = lj_epsilon_i_[i], ej = lj_epsilon_i_[j];
                if(lj_comb_=="LB") { sigma_ij = 0.5*(si+sj); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
                else { sigma_ij = std::sqrt(std::max(0.0,si*sj)); eps_ij = std::sqrt(std::max(0.0,ei*ej)); }
              }
              const double sr = sigma_ij / r;
              const double sr2 = sr*sr, sr6 = sr2*sr2*sr2, sr12 = sr6*sr6;
              E += scale * 4.0 * eps_ij * (sr12 - sr6);
              dVdr_total += -24.0 * eps_ij * (2.0*sr12 - sr6) / r;
            }
            
            if(use_coulomb_ && (coul_cutoff_ <= 0.0 || r < coul_cutoff_)) {
              const double qq = charges_[i] * charges_[j];
              if(qq != 0.0) {
                E += scale * ke_ * qq * invr / coul_epsilon_r_;
                dVdr_total += -ke_ * qq * invr * invr / coul_epsilon_r_;
              }
            }
            
            const Vector dE_dri = -scale * dVdr_total * dir;
            if(!restrict_forces_ || force_mask_[i]) setAtomsDerivatives(i, dE_dri);
            if(!restrict_forces_ || force_mask_[j]) setAtomsDerivatives(j, -dE_dri);
          }
        }
      }
    }
  }

  setBoxDerivativesNoPbc();
  setValue(E);
}

}
}

#ifdef MINIMIZE_CLASS

MinimizeStyle(mc,MinMC)

#else

#ifndef MIN_MC_H
#define MIN_MC_H

#include "dump_atom.h"
#include "random_park.h"
#include "min_linesearch.h"

using namespace std;

namespace LAMMPS_NS {

class MinMC: public MinLineSearch {
public:
  MinMC(class LAMMPS *);
  ~MinMC();
  int iterate(int);

private:
  int me, np;
  void read_control();
  void MC_setup();
  void MC_final();

  bigint evalf;
  int it, iter, stage;
  int max_iter;
  void MC_disp();
  void MC_swap();
  void MC_vol();

  int Metropolis(const double);

  int remapall;
  int remap(const int, const double);
  int nrigid, *rfix;

  int seed;
  RanPark *random;

  int groupbit, ngroup, n_all; // group bit & # of atoms for MC
  char *groupname;             // group name for MC
  int that, *glist;            // ID and list to choose

  int *nat_loc, *nat_all;
  double eref, delE;
  double T, kT, inv_kT;
  double *type2mass;
  double **ChemBias;
  int nMC_disp, nMC_swap, nMC_vol;
  int acc_disp, acc_swap, acc_vol;
  int att_disp, att_swap, att_vol;
  int acc_total, att_total;
  double dm_vol;

  // for output
  FILE *fp1;
  char *flog;
  int log_level;               // 1, all; 0, main
  int freq_out;                // frequency to output during MC cycle
  int freq_disp;               // frequency to output log info during MC_disp
  int freq_swap;               // frequency to output log info during MC_swap
  int freq_vol;                // frequency to output log info during MC_vol
  void print_info(const int);
};

}
#endif
#endif

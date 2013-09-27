/* -------------------------------------------------------------------------------------------------
 * Some features of this code are writtern according to
 * Norman's Code version 3.0 MinMC. The explanlation of 
 * the parameters I used here can be found in the doc of
 * his code.
 * This code don't do minimizing include extra peratom dof or 
 * extra global dof.
------------------------------------------------------------------------------------------------- */
#include "min_mc.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "timer.h"
#include "error.h"
#include "modify.h"
#include "fix_minimize.h"
#include "memory.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "compute.h"
#include "force.h"
#include "group.h"
#include "math.h"
#include "output.h"

#define MAXLINE 512
#define ZERO  1.e-10

using namespace LAMMPS_NS;

enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,ZEROQUAD};

/* -------------------------------------------------------------------------------------------------
 * Constructor of MC
------------------------------------------------------------------------------------------------- */
MinMC::MinMC(LAMMPS *lmp): MinLineSearch(lmp)
{
  // set default values
  T = 300.;
  seed = 1234;
  remapall = 1;
  log_level = neval = 0;
  freq_out = freq_disp = freq_swap = freq_vol = 1;

  fp1 = NULL;
  rfix = NULL;
  random = NULL;
  ChemBias = NULL;
  type2mass = NULL;
  groupname = flog = NULL;
  nat_loc = nat_all = NULL;

  dm_vol = 0.001;
  nrigid = 0;
  nMC_disp = nMC_swap = nMC_vol = 0;
  acc_disp = acc_swap = acc_vol = acc_total = 0;
  att_disp = att_swap = att_vol = att_total = 0;

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &np);

  if (atom->ntypes <= 1) error->all(FLERR, "ntypes should be greater than 1 for MinMC!");

  // default chemical bias is zero
  memory->create(ChemBias, atom->ntypes+1, atom->ntypes+1, "ChemBias");
  for (int i = 0; i <= atom->ntypes; ++i)
  for (int j = 0; j <= atom->ntypes; ++j) ChemBias[i][j] = 0.;

return;
}

/* -------------------------------------------------------------------------------------------------
 * deconstructor of MC
------------------------------------------------------------------------------------------------- */
MinMC::~MinMC()
{
  if (fp1) fclose(fp1);
  if (rfix) delete [] rfix;
  if (flog)   delete [] flog;
  if (glist)  delete [] glist;
  if (random) delete random;
  if (groupname) delete [] groupname;

  memory->destroy(nat_loc);
  memory->destroy(nat_all);
  memory->destroy(ChemBias);
  memory->destroy(type2mass);

return;
}
/* -------------------------------------------------------------------------------------------------
 * The main loops of MC
------------------------------------------------------------------------------------------------- */
int MinMC::iterate(int maxevent)
{
  max_iter = maxevent;

  // read in control parameters
  read_control();
  MC_setup();

  // write header of log file
  print_info(0);

  iter = it = stage = 0;
  eref = ecurrent;

  // log info for the zero step
  print_info(10);

  // main loop of MC
  for (iter = 1; iter <= max_iter; ++iter){
    // Displacement
    ++stage;
    MC_disp();

    // Swap chemical type
    ++stage;
    MC_swap();

    // Volumetric adjustment
    ++stage;
    MC_vol();

    att_total += att_swap;
    acc_total += acc_swap;

    stage = it = 0;
    if (iter==1 || iter==max_iter || (iter%freq_out)==0) print_info(10);

    // output for thermo, dump, restart files
    int ntimestep = ++update->ntimestep; ++niter;
    if (output->next == ntimestep){
      energy_force(0);
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(TIME_OUTPUT);
    }
  }

  MC_final();

return MAXITER;
}

/* -------------------------------------------------------------------------------------------------
 * read MC control parameters from file "mc.control"
------------------------------------------------------------------------------------------------- */
void MinMC::read_control()
{
  char oneline[MAXLINE], str[MAXLINE], *token1, *token2;
  FILE *fp = fopen("MC.ctrl", "r");
  if (fp == NULL){
    error->warning(FLERR, "Cannot open Min_MC control parameter file: MC.ctrl. Default parameters will be used.\n");

  } else {
    while ( 1 ) {
      fgets(oneline, MAXLINE, fp);
      if (feof(fp)) break;

      if (token1 = strchr(oneline,'#')) *token1 = '\0';

      token1 = strtok(oneline," \t\n\r\f");
      if (token1 == NULL) continue;
      token2 = strtok(NULL," \t\n\r\f");
      if (token2 == NULL){
        sprintf(str, "Insufficient parameter for %s of MC!", token1);
        error->all(FLERR, str);
      }

      if (strcmp(token1, "random_seed") == 0){
        seed = force->inumeric(FLERR, token2);
        if (seed < 1) error->all(FLERR, "MC: seed must be greater than 0");

      } else if (strcmp(token1, "temperature") == 0){
        T = force->numeric(FLERR, token2);
        if (seed < 1) error->all(FLERR, "MC: seed must be greater than 0");

      } else if (strcmp(token1, "group_name") == 0){
        if (groupname) delete [] groupname;
        groupname = new char [strlen(token2)+1];
        strcpy(groupname, token2);

      } else if (strcmp(token1, "chem_bias") == 0){
        int is = force->inumeric(FLERR, token2);
        if (is < 1 || is > atom->ntypes){
          sprintf(str, "Incorrect parameter for %s of MC!", token1);
          error->all(FLERR, str);
        }

        token2 = strtok(NULL," \t\n\r\f");
        if (token2 == NULL){
          sprintf(str, "Insufficient parameter for %s of MC!", token1);
          error->all(FLERR, str);
        }
        int it = force->inumeric(FLERR, token2);
        if (it < 1 || it > atom->ntypes){
          sprintf(str, "Incorrect parameter for %s of MC!", token1);
          error->all(FLERR, str);
        }

        token2 = strtok(NULL," \t\n\r\f");
        if (token2 == NULL){
          sprintf(str, "Insufficient parameter for %s of MC!", token1);
          error->all(FLERR, str);
        }
        ChemBias[is][it] = force->numeric(FLERR, token2);
        
      } else if (strcmp(token1, "nMC_disp") == 0){
        nMC_disp = force->inumeric(FLERR, token2);
        if (nMC_disp < 0) error->all(FLERR, "Min_MC: nMC_disp must be greater than 0.");

      } else if (strcmp(token1, "nMC_swap") == 0){
        nMC_swap = force->inumeric(FLERR, token2);
        if (nMC_swap < 0) error->all(FLERR, "Min_MC: nMC_swap must be greater than 0.");

      } else if (strcmp(token1, "nMC_vol") == 0){
        nMC_vol = force->inumeric(FLERR, token2);
        if (nMC_vol < 0) error->all(FLERR, "Min_MC: nMC_vol must be greater than 0.");

      } else if (strcmp(token1, "max_disp") == 0){
        dmax = force->numeric(FLERR, token2);
        if (dmax <= 0.) error->all(FLERR, "MC: max_disp must be greater than 0.");

      } else if (strcmp(token1, "max_dvol") == 0){
        dm_vol = force->numeric(FLERR, token2);
        if (dm_vol <= 0.) error->all(FLERR, "MC: max_disp must be greater than 0.");

      } else if (!strcmp(token1, "log_file")){
        if (flog) delete []flog;
        flog = new char [strlen(token2)+1];
        strcpy(flog, token2);

      } else if (!strcmp(token1, "log_level")){
        log_level = force->inumeric(FLERR, token2); // 1, only swap info; 3, both swap and disp; 7, all info

      } else if (!strcmp(token1, "freq_out")){
        freq_out = force->inumeric(FLERR, token2);
        if (freq_out < 1) error->all(FLERR, "MC: freq_out must be greater than 0.");

      } else if (!strcmp(token1, "freq_disp")){
        freq_disp = force->inumeric(FLERR, token2);
        if (freq_disp < 1) error->all(FLERR, "MC: freq_disp must be greater than 0.");

      } else if (!strcmp(token1, "freq_swap")){
        freq_swap = force->inumeric(FLERR, token2);
        if (freq_swap < 1) error->all(FLERR, "MC: freq_swap must be greater than 0.");

      } else if (!strcmp(token1, "freq_vol")){
        freq_vol = force->inumeric(FLERR, token2);
        if (freq_vol < 1) error->all(FLERR, "MC: freq_vol must be greater than 0.");

      } else {
        sprintf(str, "Unknown control parameter for MC: %s", token1);
        error->all(FLERR, str);
      }
    }
    fclose(fp);
  }

  // set default output file names
  if (flog == NULL){
    flog = new char [9];
    strcpy(flog, "log.mc");
  }
  for (int i = 1;   i <= atom->ntypes; ++i){ // Make sure ChemBias is symmetric
    ChemBias[i][i] = 0.;
    for (int j = i+1; j <= atom->ntypes; ++j){
      if (fabs(ChemBias[i][j]) > ZERO && fabs(ChemBias[j][i]) < ZERO) ChemBias[j][i] = -ChemBias[i][j];
      if (fabs(ChemBias[j][i]) > ZERO && fabs(ChemBias[i][j]) < ZERO) ChemBias[i][j] = -ChemBias[j][i];
    }
  }

  // default group name is all
  if (groupname == NULL){
    groupname = new char [4];
    strcpy(groupname, "all");
  }
  int igroup = group->find(groupname);

  if (igroup == -1){
    sprintf(str, "Cannot find MC group: %s", groupname);
    error->all(FLERR, str);
  }
  groupbit = group->bitmask[igroup];
  ngroup = group->count(igroup);
  if (ngroup < 1) error->all(FLERR, "No atom is found in your desired group for MC.");

  // group info for all
  int groupall = group->find("all");
  n_all = group->count(groupall);

  if (n_all != ngroup) remapall = 0;

  // open log file and output control parameter info
  if (me == 0 && strcmp(flog, "NULL") != 0){
    fp1 = fopen(flog, "w");
    if (fp1 == NULL){
      sprintf(str, "Cannot open MC log file: %s for writing", flog);
      error->one(FLERR, str);
    }

    fprintf(fp1, "\n#======================================= MC based on LAMMPS ========================================\n");
    fprintf(fp1, "# max_iter          %-18d  # %s\n", max_iter,"Max number of MC iterations.");
    fprintf(fp1, "# global control parameters\n");
    fprintf(fp1, "random_seed         %-18d  # %s\n", seed, "Seed for random generator.");
    fprintf(fp1, "temperature         %-18g  # %s\n", T, "Temperature for Metropolis algorithm, in K.");
    fprintf(fp1, "group_name          %-18s  # %s\n", groupname, "The lammps group ID of the atoms that will be displaced.");
    fprintf(fp1, "\n# Chemical bias:  src des  bias\n");
    for (int i = 1;   i <= atom->ntypes; ++i)
    for (int j = i+1; j <= atom->ntypes; ++j)
    fprintf(fp1, "chem_bias           %-3d %-3d %-10g  # %s\n", i,j,ChemBias[i][j], "Chemical bias between pairs.");
    fprintf(fp1, "\n# MC steps\n");
    fprintf(fp1, "nMC_disp            %-18d  # %s\n", nMC_disp, "# of iteraction for atomic  relax in each MC cycle.");
    fprintf(fp1, "nMC_swap            %-18d  # %s\n", nMC_swap, "# of iteraction for chemical swap in each MC cycle.");
    fprintf(fp1, "nMC_vol             %-18d  # %s\n", nMC_vol,  "# of iteraction for volume adjust in each MC cycle.");
    fprintf(fp1, "max_disp            %-18g  # %s\n", dmax, "Maximum displacement per step during relaxation.");
    fprintf(fp1, "max_dvol            %-18g  # %s\n", dm_vol, "Maximum relative change of box length.");
    fprintf(fp1, "\n# Output related parameters\n");
    fprintf(fp1, "log_file            %-18s  # %s\n", flog, "File to write MC log info; NULL to skip");
    fprintf(fp1, "log_level           %-18d  # %s\n", log_level, "Level of MC log ouput: 1, swap; 3, swap and disp; 7, all.");
    fprintf(fp1, "freq_out            %-18d  # %s\n", freq_out,  "Frequency to output MC log info.");
    fprintf(fp1, "freq_disp           %-18d  # %s\n", freq_disp, "Frequency to output log info during MC_disp.");
    fprintf(fp1, "freq_swap           %-18d  # %s\n", freq_swap, "Frequency to output log info during MC_swap.");
    fprintf(fp1, "freq_vol            %-18d  # %s\n", freq_vol,  "Frequency to output log info during MC_vol.");
    fprintf(fp1, "#====================================================================================================\n");
  }

return;
}

/* -----------------------------------------------------------------------------
 * Setup all derived parameters and other necessary quantities
 * ---------------------------------------------------------------------------*/
void MinMC::MC_setup()
{
  // some default values
  if (nMC_disp < 1) nMC_disp = ngroup;
  if (nMC_swap < 1) nMC_swap = atom->ntypes * 10;
  if (nMC_vol  < 1) nMC_vol  = 1;

  // other derived values
  kT = force->boltz * T;
  inv_kT = 1./kT;
  random = new RanPark(lmp, seed+me);

  // group list
  int nlocal = atom->nlocal;
  int *tag  = atom->tag;
  int *mask = atom->mask;
  int *llist; memory->create(llist, MAX(1, nlocal), "llist");

  int n = 0;
  for (int i = 0; i < nlocal; ++i) if (mask[i] & groupbit) llist[n++] = tag[i];

  int nsingle = n, nall;
  MPI_Allreduce(&nsingle,&nall,1,MPI_INT,MPI_SUM,world);

  if (nall != ngroup) error->all(FLERR, "# of atoms in group mismatch!");
  if (me == 0) memory->create(glist, ngroup, "glist");

  int *disp = new int [np];
  int *recv = new int [np];
  for (int i = 0; i < np; ++i) disp[i] = recv[i] = 0;
  MPI_Gather(&nsingle,1,MPI_INT,recv,1,MPI_INT,0,world);
  for (int i = 1; i < np; ++i) disp[i] = disp[i-1] + recv[i-1];

  MPI_Gatherv(llist,nsingle,MPI_INT,glist,recv,disp,MPI_INT,0,world);
  delete [] disp;
  delete [] recv;
  memory->destroy(llist);

  // get atomic mass for each type
  memory->create(nat_loc,   atom->ntypes+1, "nat_loc");
  memory->create(nat_all,   atom->ntypes+1, "nat_all");
  memory->create(type2mass, atom->ntypes+1, "MC:type2mass");

  int *type  = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;

  for (int ip = 1; ip <= atom->ntypes; ++ip) nat_loc[ip] = 0;
  for (int i = 0; i < nlocal; ++i){
    if (mask[i] & groupbit){
      int ip = type[i];
      ++nat_loc[ip];
    }
  }
  MPI_Allreduce(nat_loc, nat_all, atom->ntypes+1, MPI_INT, MPI_SUM, world);

  if (rmass){
    double *mass_one = new double[atom->ntypes+1];
    for (int ip = 1; ip <= atom->ntypes; ++ip) mass_one[ip] = 0.;
    for (int i = 0; i < nlocal; ++i){
      if (mask[i] & groupbit){
        int ip = type[i];
        mass_one[ip] += rmass[i];
      }
    }
    MPI_Allreduce(mass_one, type2mass, atom->ntypes+1, MPI_DOUBLE, MPI_SUM, world);
    for (int ip = 1; ip <= atom->ntypes; ++ip){
      if (nat_all[ip] > 0) type2mass[ip] /= double(nat_all[ip]);
      else type2mass[ip] = 1.;
    }
    delete [] mass_one;

  } else {
    for (int ip = 1; ip <= atom->ntypes; ++ip) type2mass[ip] = mass[ip];
  }

  // rigid
  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) nrigid++;
  if (nrigid) {
    rfix = new int[nrigid];
    nrigid = 0;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->rigid_flag) rfix[nrigid++] = i;
  }

return;
}

/* -----------------------------------------------------------------------------
 * MC for atomic displacements
 * ---------------------------------------------------------------------------*/
void MinMC::MC_disp()
{
  double x0_loc[3], x0_all[3];
  const double dmax2 = dmax + dmax;

  att_disp = acc_disp = 0;
  for (it = 1; it <= nMC_disp; ++it){
    // define the atom that will be displaced
    if (me == 0){
      int index = int(random->uniform()*double(ngroup))%ngroup;
      that = glist[index];
    }
    MPI_Bcast(&that, 1, MPI_INT, 0, world);

    x0_loc[0] = x0_loc[1] = x0_loc[2] = 0.;
    // find the atom and displace it
    int *tag   = atom->tag;
    double **x = atom->x;
    for (int i = 0; i < atom->nlocal; ++i){
      if (tag[i] == that){
        for (int idim = 0; idim < 3; ++idim){
          x0_loc[idim] = x[i][idim];
          x[i][idim] += dmax2 * (0.5 - random->uniform());
        }
        break;
      }
    }
    MPI_Allreduce(x0_loc, x0_all, 3, MPI_DOUBLE, MPI_SUM, world);
    
    // evaluate the new energy
    ecurrent = energy_force(0); ++neval;

    // Metropolis
    delE = ecurrent - eref;
    int acc = Metropolis(delE);
    if (acc == 0){
      // restore the atomic position
      for (int i = 0; i < atom->nlocal; ++i){
        if (tag[i] == that){
          for (int idim = 0; idim < 3; ++idim) x[i][idim] = x0_all[idim];
          break;
        }
      }

    } else {
      eref = ecurrent;
      ++acc_disp;
    }

    ++att_disp;
    if (me == 0 && (log_level&2) && ((it%freq_disp)==0 || it==nMC_disp || it==1)) print_info(1);
  }

return;
}

/* -----------------------------------------------------------------------------
 * MC for exchange
 * ---------------------------------------------------------------------------*/
void MinMC::MC_swap()
{
  att_swap = acc_swap = 0;
  for (it = 1; it <= nMC_swap; ++it){
    // define the central atom whose type will be changed
    if (me == 0){
      int index = int(random->uniform()*double(ngroup))%ngroup;
      that = glist[index];
    }
    MPI_Bcast(&that, 1, MPI_INT, 0, world);

    // change atomic type for the randomly chosen atom
    for (int i = 1; i <= atom->ntypes; ++i) nat_loc[i] = 0;
    int ip_new, ip_old;
    int phit = -1, id_hit = 0;
    int nlocal = atom->nlocal;
    int *tag   = atom->tag;
    double *rmass = atom->rmass;

    int ip_loc[2], ip_all[2];
    ip_loc[0] = ip_loc[1] = 0.;
   
    for (int i = 0; i < nlocal; ++i){
      int ip = atom->type[i];
      ++nat_loc[ip];

      if (tag[i] == that){
        ip_old = ip_new = ip;
   
        while (ip_new == ip_old) ip_new = int(random->uniform() * double(atom->ntypes))%atom->ntypes + 1;
        atom->type[i] = ip_new;
        if (rmass) rmass[i] = type2mass[ip_new];

        ++nat_loc[ip_new]; --nat_loc[ip_old];

		  ip_loc[0] = ip_old; ip_loc[1] = ip_new;

        phit = me;
        id_hit = i;
      }
    }
    MPI_Reduce(nat_loc, nat_all, atom->ntypes+1, MPI_INT, MPI_SUM, 0, world);
    MPI_Allreduce(ip_loc, ip_all, 2, MPI_INT, MPI_SUM, world);
    ip_old = ip_all[0]; ip_new = ip_all[1];

    // evaluate energy and do metropolis
    ecurrent = energy_force(0); ++neval;

    double fagu = type2mass[ip_new]/type2mass[ip_old];
    double cb   = ChemBias[ip_old][ip_new];

    delE = (ecurrent - eref) - 1.5 * kT * log(fagu) - cb;
    int acc = Metropolis(delE);

    if (acc){
      eref = ecurrent;
      ++acc_swap;

    } else {

      if (me == phit){
        atom->type[id_hit] = ip_old;
        if (rmass) rmass[id_hit] = type2mass[ip_old];
      }
      if (me == 0){ --nat_all[ip_new]; ++nat_all[ip_old];}
    }
    ++att_swap;

    if (me == 0 && (log_level&1) && ((it%freq_swap)==0 || it==nMC_swap || it==1)) print_info(2);
  }

return;
}

/* -----------------------------------------------------------------------------
 * MC for volumetric adjustment
 * ---------------------------------------------------------------------------*/
void MinMC::MC_vol()
{
  double ratio;
  const double dv = dm_vol + dm_vol;

  att_vol = acc_vol = 0;
  for (it = 1; it <= nMC_vol; ++it){
    for (int dir = 0; dir < 3; ++dir){
      if (domain->periodicity[dir] == 0) continue;

      if (me == 0) ratio = dv * (0.5-random->uniform());
      MPI_Bcast(&ratio, 1, MPI_INT, 0, world);

      remap(dir, ratio);

      ecurrent = energy_force(0); ++neval;
      delE = ecurrent - eref - 3.*double(ngroup)*kT*log(1.+ratio);
      int acc = Metropolis(delE);

      if (acc == 0){
        ratio = 1./(1. + ratio) - 1.;
        remap(dir, ratio);

      } else {
        eref = ecurrent;
        ++acc_vol;
      }
      ++att_vol;
    }

    if (me == 0 && (log_level&4) && ((it%freq_vol)==0 || it==nMC_vol || it==1)) print_info(3);
  }

return;
}

/* ---------------------------------------------------------------------------
 *  A little bit statistics on exit
 * -------------------------------------------------------------------------*/
void MinMC::MC_final()
{
  if (me == 0){
    if (fp1){
      fprintf(fp1, "\n");
      fprintf(fp1, "#=========================================================================================\n");
      fprintf(fp1, "# Total number of MC cycles  : %d\n", max_iter);
      fprintf(fp1, "# Total number atomic types  : %d\n", atom->ntypes);
      fprintf(fp1, "# Total # of accepted swaps  : %d (%4.2f%% success)\n", acc_total, double(acc_total)/double(MAX(1,att_total)));
      fprintf(fp1, "# Number of force evaluation : %d\n", neval);
      fprintf(fp1, "#=========================================================================================\n");
    }
  }

return;
}

/* -------------------------------------------------------------------------------------------------
 *  Write out related info
------------------------------------------------------------------------------------------------- */
void MinMC::print_info(const int flag)
{
  if (me != 0) return;

  if (flag == 0){
    if (fp1){
      fprintf(fp1, "# MCiter stage iter     PotentialEng   acc%%   ");
      for (int ip = 1; ip <= atom->ntypes; ++ip) fprintf(fp1, "  Type-%02d%%", ip);
      fprintf(fp1,"\n#");
      for (int i = 0; i < 48 + atom->ntypes*10; ++i) fprintf(fp1,"-");
      fprintf(fp1,"\n");
    }

  } else if (flag == 1){
    double succ = double(acc_disp)/double(MAX(1, att_disp))*100.;
    if (fp1){
      fprintf(fp1, "%9d %2d %7d %16.6f %9.5f", iter, stage, it, eref, succ);
      for (int ip = 1; ip <= atom->ntypes; ++ip) fprintf(fp1, " %9.5f", double(nat_all[ip])/double(n_all)*100.);
      fprintf(fp1, "\n");
    }

  } else if (flag == 2){
    double succ = double(acc_swap)/double(MAX(1,att_swap))*100.;
    if (fp1){
      fprintf(fp1, "%9d %2d %7d %16.6f %9.5f", iter, stage, it, eref, succ);
      for (int ip = 1; ip <= atom->ntypes; ++ip) fprintf(fp1, " %9.5f", double(nat_all[ip])/double(n_all)*100.);
      fprintf(fp1, "\n");
    }

  } else if (flag == 3){
    double succ = double(acc_vol)/double(MAX(1,att_vol))*100.;
    if (fp1){
      fprintf(fp1, "%9d %2d %7d %16.6f %9.5f", iter, stage, it, eref, succ);
      for (int ip = 1; ip <= atom->ntypes; ++ip) fprintf(fp1, " %9.5f", double(nat_all[ip])/double(n_all)*100.);
      fprintf(fp1, "\n");
    }

  } else if (flag == 10){
    double succ = double(acc_total)/double(MAX(1,att_total))*100.;
    if (fp1){
      fprintf(fp1, "%9d %2d %7d %16.6f %9.5f", iter, stage, it, eref, succ);
      for (int ip = 1; ip <= atom->ntypes; ++ip) fprintf(fp1, " %9.5f", double(nat_all[ip])/double(n_all)*100.);
      fprintf(fp1, "\n");
      fflush(fp1);
    }

  }

return;
}

/* -------------------------------------------------------------------------------------------------
 * Metropolis
 *----------------------------------------------------------------------------------------------- */
int MinMC::Metropolis(const double dE)
{
  int res = 0;
  if (me == 0){
    if (dE < 0.) res = 1;
    else {
      double rnd = random->uniform();
      if (exp(-dE * inv_kT) > rnd) res = 1;
    }
  }
  MPI_Bcast(&res, 1, MPI_INT, 0, world);

return res;
}

/* ----------------------------------------------------------------------------------------------
 * change box size
 * remap all atoms or fix group atoms depending on allremap flag
 * if rigid bodies exist, scale rigid body centers-of-mass
 *----------------------------------------------------------------------------------------------- */
void MinMC::remap(const int dir, const double ratio)
{
  double **x = atom->x;
  int *mask = atom->mask;

  // convert pertinent atoms and rigid bodies to lamda coords
  if (remapall) domain->x2lamda(atom->nlocal);
  else {
    for (int i = 0; i < atom->nlocal; i++){
      if (mask[i] & groupbit) domain->x2lamda(x[i],x[i]);
    }
  }

  if (nrigid){
    for (int i = 0; i < nrigid; i++) modify->fix[rfix[i]]->deform(0);
  }

  // reset global and local box to new size/shape
  double oldlo = domain->boxlo[dir];
  double oldhi = domain->boxhi[dir];
  double ctr = 0.5 * (oldlo + oldhi);
  domain->boxlo[dir] = (oldlo-ctr)*(1. + ratio) + ctr;
  domain->boxhi[dir] = (oldhi-ctr)*(1. + ratio) + ctr;

  domain->set_global_box();
  domain->set_local_box();

  // convert pertinent atoms and rigid bodies back to box coords
  if (remapall) domain->lamda2x(atom->nlocal);
  else {
    for (int i = 0; i < atom->nlocal; i++){
      if (mask[i] & groupbit) domain->lamda2x(x[i],x[i]);
    }
  }

  if (nrigid){
    for (int i = 0; i < nrigid; i++) modify->fix[rfix[i]]->deform(1);
  }

return;
}

/* ---------------------------------------------------------------------------------------------- */

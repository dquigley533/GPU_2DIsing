#include <stdlib.h>
#include <math.h>
#include "mt19937ar.h"   // External RNG for CPU reference implementation


// pre-compute acceptance probabilities for spin flips
void preComputeProbs_cpu(double beta, double h);

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps);

// Compute magnetisation on the cpu
void compute_magnetisation_cpu(int L, int *ising_grids, int grid_index, double *magnetisation);

// Main driver routine
float mc_driver_cpu(int L, int ngrids, int* ising_grids, double beta, double h, int* grid_fate, int tot_nsweeps, int mag_output_int, int grid_output_int, int itask, double up_thr, double dn_thr);

#include <stdlib.h>
#include <math.h>
#include "mt19937ar.h"   // External RNG for CPU reference implementation

typedef struct {
  int  L;                // Size of each L x L grid
  int  ngrids;           // Number of grids
  int* ising_grids;      // Pointer to first entry in first grid
} mc_grids_t;

typedef struct {
  int tot_nsweeps;     // Max number of MC sweeps to perform 
  int mag_output_int;  // Interval at which to calculate magnetisation
  int grid_output_int; // Interval at which to write grids to file
} mc_sampler_t;

typedef struct {
  int itask;     // What to calculate?
  double dn_thr; // Magnetisation below which lies the down macrostate
  double up_thr; // Magnetisation above which lies the up macrostate
  int ninputs;   // Number of input grids
  float* result; // Pointer to result array
} mc_function_t;



// Function typedef obeyed by functions which output grids
typedef int (*GridOutputFunc)(int L, int ngrids, int* grid_data, int isweep, float* magnetisation);

// pre-compute acceptance probabilities for spin flips
void preComputeProbs_cpu(double beta, double h);

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps);

// Compute magnetisation on the cpu
void compute_magnetisation_cpu(int L, int *ising_grids, int grid_index, float *magnetisation);

// Main driver routine
void mc_driver_cpu(mc_grids_t grids, double beta, double h, int* grid_fate, mc_sampler_t samples, mc_function_t calc, GridOutputFunc func);

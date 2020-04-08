#include <stdlib.h>
#include <math.h>
#include "mt19937ar.h"   // External RNG for CPU reference implementation

double Pacc[20];         // Cache of acceptance probabilities

// pre-compute acceptance probabilities for spin flips
void preComputeProbs_cpu(double beta, double h);

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h);


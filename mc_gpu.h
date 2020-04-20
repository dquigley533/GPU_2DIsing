#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include "gpu_tools.h"

// Cache of acceptance probabilities 
__constant__ float d_Pacc[20];   // gpu constant memory


// pre-compute acceptance probabilities for spin flips
void preComputeProbs_gpu(double beta, double h);

// MC sweep on the GPU - 3 versions
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h);
__global__ void mc_sweep_gpu_bitrep(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h);
__global__ void mc_sweep_gpu_bitmap(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h);


// Compute magnetisation on the GPU
__global__ void compute_magnetisation_gpu(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation);


// The size of any shared memory arrays needs to be known at compile time. If using shared memory as
// a cache for the Ising grids managed by a thread block we use one bit per spin. Hence can only handle
// SHGRIDSIZEx8/(LxL) threads per block. Shared memory is a limited resource, so increasing this 
// to hangle more threads per block or bigger grids will mean fewer thread blocks can run at once, which
// might lead to underutilisation of SMs.

// E.g. GTX 1660 Ti, has 32 threads per warp and enough shared memory per SM for each of SMs to hold
// 32 64x64 grids.
#define MAXL 64
#define BLOCKSIZE 48
#define SHGRIDSIZE BLOCKSIZE*MAXL*MAXL/8  // Shared memory per block in *bytes for bitrep kernel

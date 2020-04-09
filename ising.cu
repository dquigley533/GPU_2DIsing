// -*- mode: C -*-
/* ==========================================================================================
                                 GPU_2DIsing.cu

Implementation of the 2D Ising model in CUDA. Each CUDA thread simulates an independent 
instance of the 2D Ising model in parallel with an independent random number sequence. Draws
heavily from the work of Weigel et al, [J. Phys.: Conf. Ser.921 012017 (2017)] but used here
for gathering rare event statistics on nucleation during magnetisation reversal. 
 ===========================================================================================*/
// D. Quigley. Univeristy of Warwick

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>

extern "C" {
  #include "mc_cpu.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"

const bool run_gpu = true;  // Run using GPU
const bool run_cpu = true;  // Run using CPU

int main (int argc, char *argv[]) {

/*=================================
   Defaults and constants 
  =================================*/ 
  
  int L       = 64;   // Size of 2D Ising grid. LxL grid squares.
  int ngrids  = 1;    // Number of replicas of 2D grid to simulate
  int nsweeps = 100;  // Number of MC sweeps to simulate on each grid

  double beta = 1.0/1.5;  // Inverse temperature
  double h = 0.05;        // External field
 
  unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  
  int threadsPerBlock = 1;  // Number of threads/replicas to run in each threadBlock
  int blocksPerGrid   = 1;  // Total number of threadBlocks

/*=================================
   Process command line arguments 
  =================================*/ 
  if (argc != 4) {
    printf("Usage : ./ising2D nsweeps nreplicas GPUthreadsPerBlock\n");
    exit(EXIT_FAILURE);
  }

  nsweeps = atoi(argv[1]);         // Number of MC sweeps to simulate
  ngrids  = atoi(argv[2]);         // Number of replicas (grids) to simulate
  threadsPerBlock = atoi(argv[3]); // Number of threads per GPU threadblock
  
/*=================================
   Initialise simulations
  =================================*/ 
  // Host copy of Ising grid configurations
  int *ising_grids = (int *)malloc(L*L*ngrids*sizeof(int));
  
  // Initialise as spin down
  int i;
  for (i=0;i<L*L*ngrids;i++) { ising_grids[i] = -1; }

  // Initialise host RNG
  init_genrand(rngseed);

  int *d_ising_grids;                    // Pointer to device grid configurations
  curandStatePhilox4_32_10_t *d_state;   // Pointer to device RNG states

  if (run_gpu==true) {
    
    gpuInit(); // Initialise GPU device(s)

    // Device copy of Ising grid configurations
    gpuErrchk( cudaMalloc(&d_ising_grids,L*L*ngrids*sizeof(int)) );

    // Populate from host copy
    gpuErrchk( cudaMemcpy(d_ising_grids,ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyHostToDevice) );

    // Initialise GPU RNG
    gpuErrchk (cudaMalloc((void **)&d_state, ngrids*sizeof(curandStatePhilox4_32_10_t)) );
    unsigned long long gpuseed = (unsigned long long)rngseed;
    init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, d_state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Allocate threads to thread blocks
    blocksPerGrid = ngrids/threadsPerBlock;
    if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }
  
    fprintf(stderr, "threadsPerBlock = %d, blocksPerGrid = %d\n",threadsPerBlock, blocksPerGrid);

  }

/*=================================
    Run simulations
  =================================*/ 
  clock_t t1,t2;  // For measuring time taken
  int isweep;     // MC sweep loop counter

  if (run_cpu==true) {

    int igrid;   // counter for loop over replicas

    t1 = clock();

    for (isweep=0;isweep<nsweeps;isweep++){

      // MC Sweep - CPU
      for (igrid=0;igrid<ngrids;igrid++) {
        mc_sweep_cpu(L,ising_grids,igrid,1.0/1.5,0.05);
      }
    
    }

    t2 = clock();

    printf("Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

  }

  if (run_gpu==true){

    t1 = clock();

    for (isweep=0;isweep<nsweeps;isweep++){

      // MC Sweep - GPU
      mc_sweep_gpu<<<blocksPerGrid,threadsPerBlock>>>(L,d_state,ngrids,d_ising_grids,(float)beta,(float)h);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );

    }

    t2 = clock();

    printf("Time taken on GPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

  }

  return EXIT_SUCCESS;

}

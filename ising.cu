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

extern "C" {
  #include "mc_cpu.h"
}

#include "mc_gpu.h"
//#include "gpu_tools.h"

int main (int argc, char *argv[]) {


  if (argc != 4) {
    printf("Usage : ./ising2D nsweeps nreplicas GPUthreadsPerBlock\n");
    exit(EXIT_FAILURE);
  }

  gpuInit();

  // Variables to hold the dimensions of the block
  // and thread grids. 
  int blocksPerGrid,threadsPerBlock;

  int i,igrid;

  char *a = argv[2];
  int ngrids = atoi(a); // Number of grids to simulate
  int L = 64;           // Size of L x L 2D grid

  //threadsPerBlock = 1 ; // Max if using single thread with 32 bits per spin at L=64

  a = argv[3];
  threadsPerBlock = atoi(a);

  blocksPerGrid = ngrids/threadsPerBlock;
  if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }

  fprintf(stderr, "threadsPerBlock = %d, blocksPerGrid = %d\n",threadsPerBlock, blocksPerGrid);

  // Host memory
  int *ising_grids = (int *)malloc(L*L*ngrids*sizeof(int));

  // Device memory
  int *d_ising_grids;
  gpuErrchk(cudaMalloc(&d_ising_grids,L*L*ngrids*sizeof(int)));

  // Initialise as spin down
  for (i=0;i<L*L*ngrids;i++) { ising_grids[i] = -1; }

  // Copy to devicef
  gpuErrchk(cudaMemcpy(d_ising_grids,ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyHostToDevice));
  
  // RNG 
  unsigned long rngseed = 2894203475;

  // - CPU
  init_genrand(rngseed);

  // - GPU
  curandStatePhilox4_32_10_t *d_state;
  gpuErrchk (cudaMalloc((void **)&d_state, ngrids*sizeof(curandStatePhilox4_32_10_t)) );
  unsigned long long gpuseed = (unsigned long long)rngseed;
  init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, d_state);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  double beta = 1.0/1.5;
  double h = 0.05;


  


  int isweep;
  a = argv[1];
  int nsweeps = atoi(a);

  clock_t t1 = clock();

  for (isweep=0;isweep<nsweeps;isweep++){

    // MC Sweep - CPU
    for (igrid=0;igrid<ngrids;igrid++) {
      mc_sweep_cpu(L,ising_grids,igrid,1.0/1.5,0.05);
    }
    
  }

  clock_t t2 = clock();

  printf("Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

  t1 = clock();

  for (isweep=0;isweep<nsweeps;isweep++){

    // MC Sweep - GPU
    mc_sweep_gpu<<<blocksPerGrid,threadsPerBlock>>>(L,d_state,ngrids,d_ising_grids,(float)beta,(float)h);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  }

  t2 = clock();

  printf("Time taken on GPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

  return 0;

}

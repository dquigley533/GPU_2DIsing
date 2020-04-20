// -*- mode: C -*-
/* ==========================================================================================
                                 GPU_2DIsing.cu

Implementation of the 2D Ising model in CUDA. Each CUDA thread simulates an independent 
instance of the 2D Ising model in parallel with an independent random number sequence. Draws
heavily from the work of Weigel et al, [J. Phys.: Conf. Ser.921 012017 (2017)] but used here
for gathering rare event statistics on nucleation during magnetisation reversal. 
 ===========================================================================================*/
// D. Quigley. Univeristy of Warwick

// TODO
// 1. sweep counter probably needs to be a long and not an int
// 2. set magnetisation output and grid output intervals to variables
// 3. read input configuration from file
// 4. clustering using nvgraph?


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>

extern "C" {
  #include "mc_cpu.h"
  #include "io.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"

const bool run_gpu = true;     // Run using GPU
const bool run_cpu = false;    // Run using CPU

int main (int argc, char *argv[]) {

/*=================================
   Constants and variables
  =================================*/ 
  
  int L       = 64;   // Size of 2D Ising grid. LxL grid squares.
  int ngrids  = 1;    // Number of replicas of 2D grid to simulate
  int nsweeps = 100;  // Number of MC sweeps to simulate on each grid

  double beta = 1.0/1.5;  // Inverse temperature
  double h = 0.05;        // External field
 
  unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  
  int threadsPerBlock = BLOCKSIZE;  // Number of threads/replicas to run in each threadBlock
  int blocksPerGrid   = 1;          // Total number of threadBlocks

  int gpudevice = -1;               // GPU device to use

/*=================================
   Process command line arguments 
  =================================*/ 
  if (argc != 4) {
    printf("Usage : ./ising2D nsweeps nreplicas gpudevice \n");
    exit(EXIT_FAILURE);
  }

  nsweeps   = atoi(argv[1]);         // Number of MC sweeps to simulate
  ngrids    = atoi(argv[2]);         // Number of replicas (grids) to simulate
  gpudevice = atoi(argv[3]);         // Number of threads per GPU threadblock
  

/*=================================
   Delete old output 
  ================================*/
  remove("gridstates.dat");


/*=================================
   Initialise simulations
  =================================*/ 
  // Host copy of Ising grid configurations
  int *ising_grids = (int *)malloc(L*L*ngrids*sizeof(int));
  if (ising_grids==NULL){
    fprintf(stderr,"Error allocating memory for Ising grids!\n");
    exit(EXIT_FAILURE);
  }
  
  // Initialise as spin down
  int i;
  for (i=0;i<L*L*ngrids;i++) { ising_grids[i] = -1; }

  // Initialise host RNG
  init_genrand(rngseed);

  // Precompute acceptance probabilities for flip moves
  preComputeProbs_cpu(beta, h);

  int *d_ising_grids;                    // Pointer to device grid configurations
  curandState *d_state;   // Pointer to device RNG states

  if (run_gpu==true) {
    
    gpuInit(gpudevice); // Initialise GPU device(s)

    // Allocate threads to thread blocks
    blocksPerGrid = ngrids/threadsPerBlock;
    if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }

    // Device copy of Ising grid configurations
    gpuErrchk( cudaMalloc(&d_ising_grids,L*L*ngrids*sizeof(int)) );

    // Populate from host copy
    gpuErrchk( cudaMemcpy(d_ising_grids,ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyHostToDevice) );

    // Initialise GPU RNG
    gpuErrchk (cudaMalloc((void **)&d_state, ngrids*sizeof(curandState)) );
    unsigned long long gpuseed = (unsigned long long)rngseed;
    init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, ngrids, d_state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    fprintf(stderr, "threadsPerBlock = %d, blocksPerGrid = %d\n",threadsPerBlock, blocksPerGrid);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);

    // Test RNG (DEBUG)
    /*
    float   *testrnd = (float *)malloc(ngrids*sizeof(float));
    float *d_testrnd;
    gpuErrchk( cudaMalloc(&d_testrnd, ngrids*sizeof(float)) );

    int trial;
    for (trial=0;trial<10;trial++){

      populate_random<<<blocksPerGrid,threadsPerBlock>>>(ngrids, d_testrnd, d_state);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      gpuErrchk( cudaMemcpy(testrnd, d_testrnd, ngrids*sizeof(float), cudaMemcpyDeviceToHost) );

      for (i=0;i<ngrids;i++){
        printf("Random number on grid %d : %12.4f\n",i,testrnd[i]);
      }
  
  }

    free(testrnd);
    cudaFree(d_testrnd);
    exit(EXIT_SUCCESS);
    */

  }


/*=================================
    Run simulations - CPU version
  =================================*/ 

  clock_t t1,t2;  // For measuring time taken
  int isweep;     // MC sweep loop counter
  int igrid;      // counter for loop over replicas

  if (run_cpu==true) {

    // Magnetisation of each grid
    double *magnetisation = (double *)malloc(ngrids*sizeof(double));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation array!\n");
      exit(EXIT_FAILURE);
    }

    t1 = clock();

    for (isweep=0;isweep<nsweeps;isweep++){

      // Output grids to file
      if (isweep%1000==0){
        write_ising_grids(L, ngrids, ising_grids, isweep);  
      }

      // MC Sweep - CPU
      for (igrid=0;igrid<ngrids;igrid++) {
        mc_sweep_cpu(L, ising_grids, igrid, beta, h);
      }

      // Report magnetisations
      if (isweep%100==0){
        for (igrid=0;igrid<ngrids;igrid++){
          compute_magnetisation_cpu(L, ising_grids, igrid, magnetisation);
          //printf("Magnetisation of grid %d at sweep %d = %8.4f\n",igrid, isweep, magnetisation[igrid]);
        }
      } 


    }

    t2 = clock();

    printf("Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

    // Release memory
    free(magnetisation);

  }

  /*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){

    // Host copy of magnetisation
    float *magnetisation = (float *)malloc(ngrids*sizeof(float));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation host array!\n");
      exit(EXIT_FAILURE);
    }

    // Device copy of magnetisation
    float *d_magnetisation;
    gpuErrchk( cudaMalloc(&d_magnetisation,ngrids*sizeof(float)) );

    t1 = clock();

    for (isweep=0;isweep<nsweeps;isweep++){

      // Output grids to file
      if (isweep%1000==0){
        gpuErrchk( cudaMemcpy(ising_grids,d_ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyDeviceToHost) );
        write_ising_grids(L, ngrids, ising_grids, isweep);  
      }

      // MC Sweep - GPU
      mc_sweep_gpu<<<blocksPerGrid,threadsPerBlock>>>(L,d_state,ngrids,d_ising_grids,(float)beta,(float)h);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );

      if (isweep%100==0){
        compute_magnetisation_gpu<<<blocksPerGrid, threadsPerBlock>>>(L, ngrids, d_ising_grids, d_magnetisation);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaMemcpy(magnetisation,d_magnetisation,ngrids*sizeof(float),cudaMemcpyDeviceToHost) );
        //for (igrid=0;igrid<ngrids;igrid++){
        //  printf("Magnetisation of grid %d at sweep %d = %8.4f\n",igrid, isweep, magnetisation[igrid]);
        //}
      } 

    }

    t2 = clock();

    printf("Time taken on GPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

    // Free magnetisation arrays
    free(magnetisation);
    gpuErrchk( cudaFree(d_magnetisation) );

  }


/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  free(ising_grids);
  if (run_gpu==true) {
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_state) );
  }

  return EXIT_SUCCESS;

}

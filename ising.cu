// -*- mode: C -*-

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  // clock() and clock_t
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

extern "C" {
#include "mt19937ar.h"
}


double Pacc[20];
__constant__ float d_Pacc[20];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


// Initialise RNG on the GPU
__global__ void init_gpurand(unsigned long long seed, curandState *state){

  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned long long seq = (unsigned long long)idx;
  curand_init(seed, seq, 0ull, &state[idx]);

}


// sweep on the cpu
void mc_sweep(int L, int *ising_grids, int grid_index, double beta, double h) {

  // Pointer to local grid
  int *loc_grid = &ising_grids[grid_index*L*L];

  int imove, row, col;;

  for (imove=0;imove<L*L;imove++){

    // pick random spin
    row = int(L*genrand_real3());  // Cannot generate L
    col = int(L*genrand_real3()); 

    // find neighbours
    int my_idx = L*row+col;
    int up_idx = L*(row+1)%L + col;
    int dn_idx = L*(row+L-1)%L + col;
    int rt_idx = L*row + (col+1)%L;
    int lt_idx = L*row + (col+L-1)%L;

    // energy before flip
    int n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 

    // flip
    loc_grid[my_idx] = -1*loc_grid[my_idx];

    int index = 5*(loc_grid[my_idx]+1) + n_sum + 4;

    //double energy_old = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );


    // energy after flip
    //n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_new = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    //double delta_energy = energy_new - energy_old;
    //double prob = exp(-beta*delta_energy);

    double xi = genrand_real3();
    if (xi < Pacc[index] ) {
      // accept
    } else {
      loc_grid[my_idx] = -1*loc_grid[my_idx]; // reject
    }


  } // end for

}


// sweep on the gpu
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;


  // Should also use curand functions which generate 4 floats at once where possible?

  if (idx < ngrids) {

    // Should make a local copy of RNG state for current threads and copy
    // back to global memory.
    curandState localState = state[idx];

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*L*L]; // pointer to device global memory

    //const int shsize = L*L*blockDim.x;
    //__shared__ int block_grid[64*64]; // HACK FOR TESTING SHARED MEMORY SPEED (One thread per block only).
    //
    //int *loc_grid = &block_grid[L*L*threadIdx.x];
    //int k;
    //for (k=0;k<L*L;k++) {
    //  loc_grid[k] = d_ising_grids[idx*L*L+k];
    //}


    float shrink = 1.0f - FLT_EPSILON;

    // 4 random numbers
    //float4 rnd = curand_uniform4

    int imove, row, col, my_idx, up_idx, dn_idx, rt_idx, lt_idx,n_sum,index;
    float xi;
    for (imove=0;imove<L*L;imove++){
      
      row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      col = int((float)L*shrink*curand_uniform(&localState));  
      //if (row==L) row--; if (col==L) col--;
      
      // find neighbours
      my_idx = L*row+col;
      up_idx = L*(row+1)%L + col;
      dn_idx = L*(row+L-1)%L + col;
      rt_idx = L*row + (col+1)%L;
      lt_idx = L*row + (col+L-1)%L;
      
      // energy before flip
      n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
      //float energy_old = -1.0 * (float)loc_grid[my_idx] * ( (float)n_sum + h );

      index = 5*(loc_grid[my_idx]+1) + n_sum + 4;
      
      // flip
      loc_grid[my_idx] = -1*loc_grid[my_idx];
      
      // energy after flip 
      //n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx];  
      //float energy_new = -1.0 * (float)loc_grid[my_idx] * ( (float)n_sum + h ); 
      
      //float delta_energy = energy_new - energy_old;
      //float delta_energy = -2.0 * (float)loc_grid[my_idx] * ( (float)n_sum + h ); 
      
      //float prob = expf(-beta*delta_energy);
      
      xi = curand_uniform(&localState);
      if (xi < d_Pacc[index] ) {
	// accept 
      } else { 
	loc_grid[my_idx] = -1*loc_grid[my_idx]; // reject 
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    //state[idx] = localState;
    //for (k=0;k<L*L;k++) {
    //  d_ising_grids[idx*L*L+k] = loc_grid[k];
    //}


  }

}



int main (int argc, char *argv[]) {


  if (argc != 4) {
    printf("Usage : ./ising2D nsweeps nreplicas GPUthreadsPerBlock\n");
    exit(EXIT_FAILURE);
  }

 
  // Variables to hold the dimensions of the block
  // and thread grids. 
  int blocksPerGrid,threadsPerBlock;

  // cudaError_t is a type defined in cuda.h
  cudaError_t err;

  int i,igrid,count,idev;


  // Make sure we have a CUDA capable device to work with
  err = cudaGetDeviceCount(&count);
  if ( (count==0) || (err!=cudaSuccess) ) {
    printf("No CUDA supported devices are available in this system.\n");
    exit(EXIT_FAILURE);
  } else {
    printf("Found %d CUDA devices in this system\n",count);
  }


  // cudaDeviceProp is a type of struct which we will
  // populate with information about the available 
  // GPU compute devices.
  cudaDeviceProp prop;

  // Loop over the available CUDA devices
  for (idev=0;idev<count;idev++) {

    // Call another CUDA helper function to populate prop
    err = cudaGetDeviceProperties(&prop,idev);
    if ( err!=cudaSuccess ) {
      printf("Error getting device properties\n");
      exit(EXIT_FAILURE);
    }

    // Print out a member of the prop struct which tells
    // us the name of the CUDA device. Other members of this
    // struct tell us the clock speed and compute capability
    // of the device.
    printf("Device %d : %s\n",idev,prop.name);

  }

  err = cudaGetDevice(&idev);
  if ( err!=cudaSuccess ) {
    printf("Error identifying active device\n");
    exit(EXIT_FAILURE);
  }
  printf("Using device %d\n",idev);


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
  curandState *d_state;
  gpuErrchk (cudaMalloc((void **)&d_state, ngrids*sizeof(curandState)) );
  unsigned long long gpuseed = (unsigned long long)rngseed;
  init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, d_state);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  // Pre-compute acceptance probabilities for all possibilities
  // s  : nsum               : index 5*(s+1) + nsum + 4
  // +1 : -4 = -1 -1 -1 -1   : 10
  // +1 : -2 = -1 -1 -1 +1   : 12
  // +1 :  0 = -1 -1 +1 +1   : 14
  // +1 : +2 = -1 +1 +1 +1   : 16
  // +1 : +4 = +1 +1 +1 +1   : 18
  // -1 : -4 = -1 -1 -1 -1   : 0
  // -1 : -2 = -1 -1 -1 +1   : 2
  // -1 :  0 = -1 -1 +1 +1   : 4
  // -1 : +2 = -1 +1 +1 +1   : 6
  // -1 : +4 = +1 +1 +1 +1   : 8

  double beta = 1.0/1.5;
  double h = 0.05;

  float h_Pacc[20];
  
  int s, nsum, index;
  for (s=-1;s<2;s=s+2){
    for (nsum=-4;nsum<5;nsum=nsum+2){
      index = 5*(s+1) + nsum + 4;
      Pacc[index] = 2.0*(double)s*((double)nsum+h);
      Pacc[index] = exp(-beta*Pacc[index]);
      h_Pacc[index] = (float)Pacc[index]; // single precision version for GPU
    }
  }

  gpuErrchk( cudaMemcpyToSymbol(d_Pacc,h_Pacc,20) );

  int isweep;
  a = argv[1];
  int nsweeps = atoi(a);

  clock_t t1 = clock();

  for (isweep=0;isweep<nsweeps;isweep++){

    // MC Sweep - CPU
    for (igrid=0;igrid<ngrids;igrid++) {
      mc_sweep(L,ising_grids,igrid,1.0/1.5,0.05);
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

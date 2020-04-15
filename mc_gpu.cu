// -*- mode: C -*-

#include "mc_gpu.h"

#include <stdio.h>



// populate acceptance probabilities
void preComputeProbs_gpu(double beta, double h) {

    float *h_Pacc=(float *)malloc(20*sizeof(float));

    //cudaMalloc(&d_Pacc, 20*sizeof(float));

    int s, nsum, index;
    
    for (s=-1;s<2;s=s+2){
      for (nsum=-4;nsum<5;nsum=nsum+2){
        index = 5*(s+1) + nsum + 4;
        h_Pacc[index] = 2.0f*(float)s*((float)nsum+(float)h);
        h_Pacc[index] = expf(-(float)beta*h_Pacc[index]); // single precision version for GPU
        //printf(" %d ",index);
      }
    }
  
    gpuErrchk( cudaMemcpyToSymbol(d_Pacc, h_Pacc, 20*sizeof(float),0, cudaMemcpyHostToDevice ) );
    //gpuErrchk( cudaMemcpy(d_Pacc, &h_Pacc[0], 20*sizeof(float),cudaMemcpyHostToDevice ) );

    free(h_Pacc);

  }  

// sweep on the gpu
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    //curandStatePhilox4_32_10_t localState = state[idx];
    curandState localState = state[idx];

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*L*L]; // pointer to device global memory

    //const int shsize = L*L*blockDim.x;
    //__shared__ int block_grid[64*64]; // HACK FOR TESTING SHARED MEMORY SPEED (One thread per block only).
    
    //int *loc_grid = &block_grid[L*L*threadIdx.x];
    //int k;
    //for (k=0;k<L*L;k++) {
    //  loc_grid[k] = d_ising_grids[idx*L*L+k];
    //}

   

    float shrink = 1.0f - FLT_EPSILON;

    int imove, row, col, my_idx, n_sum, index, spin;

    //float4 rnd;
    //rnd = curand_uniform4(&localState);


    for (imove=0;imove<L*L;imove++){

      // 4 random numbers 
      //rnd = curand_uniform4(&localState); 


      row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      col = int((float)L*shrink*curand_uniform(&localState)); 
      
      //col = imove/L;
      //row = imove%L;
      my_idx = L*row+col;

      spin = loc_grid[my_idx];
      
      // find neighbours
      n_sum = 0;
      n_sum += loc_grid[L*((row+1)%L) + col];
      n_sum += loc_grid[L*((row+L-1)%L) + col];
      n_sum += loc_grid[L*row + (col+1)%L];
      n_sum += loc_grid[L*row + (col+L-1)%L];

      //n_sum = 4;
      index = 5*(spin+1) + n_sum + 4;

      // flip
      //spin = 0;//spin*-1;
      spin = -1*spin;


      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept
          loc_grid[my_idx] = spin;
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

    //for (k=0;k<L*L;k++) {
    //  d_ising_grids[idx*L*L+k] = loc_grid[k];
    //}



  }

  return;

}

// compute magnetisation on the gpu
__global__ void compute_magnetisation_gpu(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    int *loc_grid = &d_ising_grids[idx*L*L]; // pointer to device global memory

    float m = 0.0f;

    int i;
    for (i=0;i<L*L;i++) { m += loc_grid[i]; }
    d_magnetisation[idx] = m/(float)(L*L);

  }

  return;

}
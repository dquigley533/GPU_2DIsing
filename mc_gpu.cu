// -*- mode: C -*-

#include "mc_gpu.h"
#include <stdio.h>

__constant__ int blookup[2] = {-1, 1};

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

// sweep on the gpu - default version
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*L*L]; // pointer to device global memory 

    float shrink = 1.0f - FLT_EPSILON;

    int imove, row, col, my_idx, n_sum, index, spin;
    for (imove=0;imove<L*L;imove++){

      row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      col = int((float)L*shrink*curand_uniform(&localState)); 
      my_idx = L*row+col;

      //my_idx = int((float)L*L*shrink*curand_uniform(&localState));
      //row = my_idx/L;
      //col = my_idx%L;


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
      spin = -1*spin;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept
          loc_grid[my_idx] = spin;
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

  }

  return;

}

inline __device__ int bits_to_ints(uint8_t* grid, int L, int row, int col){
  // L must be the number of entries in each row here. Important if ever adapting to non-square grids.

  uint8_t one = 1U;

  int ibyte = (row*L+col)/8;
  int ibit  = (row*L+col)%8;
  return blookup[(grid[ibyte] >> ibit) & one]; 

}

// sweep on the gpu - packs a cache of the current grid into on-GPU shared memory
// for efficiency, and using a single bit represenation to acheive this. Mustn't
// be used if L*L*threadsPerBlock/8 > SHGRIDSIZE. 
__global__ void mc_sweep_gpu_bitrep(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // how many bytes do we need per thread to store L*L spins as single bytes
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes ++; }

    // If nbytes x threadsPerBlock is > SHGRIDSIZE we've got a problem....

    // Shared memory for storage of bits
    __shared__ uint8_t shared_grid[SHGRIDSIZE]; 

    // Pointer to part of this storage used by the current thread
    uint8_t *loc_grid = &shared_grid[nbytes*threadIdx.x];

    uint8_t one  = 1U;
    uint8_t zero = 0U;

    // zero out
    int ibyte;
    for (ibyte=0;ibyte<nbytes;ibyte++){ loc_grid[ibyte] = zero ; }

    // Fill this with the current state of the grid to be manipulated by this thread
    int ibit=0, spin;
    ibyte = 0;
    for (spin=0;spin<L*L;spin++){ 
        if ( d_ising_grids[L*L*idx + spin] == 1 ) {
          loc_grid[ibyte] |= one << ibit ;
        }
        ibit++;
        if (ibit==8) {
            ibit=0;
            ibyte++;
        }
    }
     

    float shrink = 1.0f - FLT_EPSILON;

    int imove, row, col, n_sum, index;

    for (imove=0;imove<L*L;imove++){

      row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      col = int((float)L*shrink*curand_uniform(&localState)); 
      //index = L*row+col;

      //index = int((float)L*L*shrink*curand_uniform(&localState));
      //row = index/L;
      //col = index%L;
      
      spin = bits_to_ints(loc_grid, L, row, col);
      
      // find neighbours
      n_sum = 0;
      n_sum += bits_to_ints(loc_grid, L, (row+1)%L, col); 
      n_sum += bits_to_ints(loc_grid, L, (row+L-1)%L, col);
      n_sum += bits_to_ints(loc_grid, L, row, (col+1)%L);
      n_sum += bits_to_ints(loc_grid, L, row, (col+L-1)%L); 

      //n_sum = 4;
      index = 5*(spin+1) + n_sum + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          ibyte = (row*L+col)/8;
          index = (row*L+col)%8; 
          loc_grid[ibyte] ^= one << index;
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

    for (row=0;row<L;row++){
      for (col=0;col<L;col++){
        d_ising_grids[L*L*idx + L*row + col] = bits_to_ints(loc_grid, L, row, col);
      }
    }

  }

  return;

}

// Similar to mc_sweep_gpu_bitrep, but maps each thread in a block of 8, 16 or 32 threads to a 
// fixed bit in a datatype of size 1, 2 or 4 bytes for faster addressing.
__global__ void mc_sweep_gpu_bitmap(const int L, curandState *state, const int ngrids, int *d_ising_grids, const float beta, const float h) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // Shared memory for storage of bits
#if (BLOCKSIZE==8) 
      __shared__ uint8_t shared_grid[MAXL*MAXL]; 
      uint8_t one  = 1U;
      uint8_t zero = 0U; 
#endif
#if (BLOCKSIZE==16)
      __shared__ uint16_t shared_grid[MAXL*MAXL];
      uint16_t one  = 1U;
      uint16_t zero = 0U;
#endif 
#if (BLOCKSIZE==32)
      __shared__ uint32_t shared_grid[MAXL*MAXL];
      uint32_t one  = 1U;
      uint32_t zero = 0U; 
#endif

    // initialise
    int spin;
    for (spin=0;spin<L*L;spin++) { shared_grid[spin] = zero; }

    // Fill this with the current state of the grid to be manipulated by this thread
    for (spin=0;spin<L*L;spin++){ 
        if ( d_ising_grids[L*L*idx + spin] == 1 ) {
          shared_grid[spin] |= one << threadIdx.x ;
        }
    }
     

    float shrink = 1.0f - FLT_EPSILON;

    int imove, row, col, n_sum, index, my_idx;

    for (imove=0;imove<L*L;imove++){

      row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      col = int((float)L*shrink*curand_uniform(&localState)); 
      
      my_idx = row*L + col;
      spin = blookup[(shared_grid[my_idx] >> threadIdx.x) & one];
      
      // find neighbours
      n_sum = 0;
      n_sum += blookup[(shared_grid[L*((row+1)%L) + col] >> threadIdx.x) & one];
      n_sum += blookup[(shared_grid[L*((row+L-1)%L) + col] >> threadIdx.x) & one];
      n_sum += blookup[(shared_grid[L*row + (col+1)%L] >> threadIdx.x) & one];
      n_sum += blookup[(shared_grid[L*row + (col+L-1)%L] >> threadIdx.x) & one];

      //n_sum = 4;
      index = 5*(spin+1) + n_sum + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          shared_grid[my_idx] ^= one << threadIdx.x;
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

    for (row=0;row<L;row++){
      for (col=0;col<L;col++){
        my_idx = L*row + col;
        d_ising_grids[L*L*idx+my_idx] = blookup[(shared_grid[my_idx] >> threadIdx.x) & one];
      }
    }

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
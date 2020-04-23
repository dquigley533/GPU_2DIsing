// -*- mode: C -*-

#include "mc_gpu.h"
#include <stdio.h>

__constant__ int blookup[2] = {-1, 1};

// Shared memory for storage of bits
//__shared__ uint8_t shared_grid[SHGRIDSIZE]; 
extern __shared__ uint8_t shared_grid[];

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

void preComputeNeighbours_gpu(const int L, int *d_ising_grids, int *d_neighbour_list){

  // These could probably be cached in shared memory since they are the same for all threads.

 int *h_neighbour_list = (int *)malloc(L*L*4*sizeof(int));

  int spin_index;
  for (spin_index=0;spin_index<L*L;spin_index++){

      int row = spin_index/L;
      int col = spin_index%L;

    h_neighbour_list[4*(row*L+col) + 0] = L*((row+1)%L) + col;
    h_neighbour_list[4*(row*L+col) + 1] = L*((row+L-1)%L) + col;
    h_neighbour_list[4*(row*L+col) + 2] = L*row + (col+1)%L;
    h_neighbour_list[4*(row*L+col) + 3] = L*row + (col+L-1)%L;

  }

  gpuErrchk( cudaMemcpy(d_neighbour_list, h_neighbour_list, 4*L*L*sizeof(int),cudaMemcpyHostToDevice ) );

  free(h_neighbour_list); 

  /// Also store a version in constant memory
  uint8_t *hc_next = (uint8_t *)malloc(MAXL*sizeof(uint8_t));
  uint8_t *hc_prev = (uint8_t *)malloc(MAXL*sizeof(uint8_t));

  for (spin_index=0;spin_index<L;spin_index++){

    hc_next[spin_index] = (spin_index+1)%L;
    hc_prev[spin_index] = (spin_index+L-1)%L;

  }

  gpuErrchk( cudaMemcpyToSymbol(dc_next, hc_next, MAXL*sizeof(uint8_t),0, cudaMemcpyHostToDevice ) );
  gpuErrchk( cudaMemcpyToSymbol(dc_prev, hc_prev, MAXL*sizeof(uint8_t),0, cudaMemcpyHostToDevice ) );
  
  free(hc_next); 
  free(hc_prev);

}


// sweep on the gpu - default version
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  //__shared__ int *neighbour_ptr[4*MAXL*MAXL];
  int index;
  //for (index=0;index<4*L*L;index++){
  //  neighbour_ptr[index] = d_neighbour_list[index];
  //}  


  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    int N = L*L;
    float shrink = (1.0f - FLT_EPSILON)*(float)N;

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*N]; // pointer to device global memory 


    int imove, my_idx, spin, n1, n2, n3, n4, row, col;
    float diff;
    for (imove=0;imove<N*nsweeps;imove++){

      //row = __float2int_rd(shrink*curand_uniform(&localState));  // CAN generate L without shrink
      //col = __float2int_rd(shrink*curand_uniform(&localState)); 
      //my_idx = L*row+col;

      //my_idx = imove;
      my_idx = __float2int_rd(shrink*curand_uniform(&localState));
      //my_idx = int(shrink*curand_uniform(&localState));
      //index = 4*my_idx;

      row = my_idx/L;
      col = my_idx%L;

      spin = loc_grid[my_idx];
      //spin = d_ising_grids[idx*N+my_idx];

      // find neighbours
      //n1 = *(d_neighbour_list[index+0] + &loc_grid[0]);
      //n2 = *(d_neighbour_list[index+1] + &loc_grid[0]);
      //n3 = *(d_neighbour_list[index+2] + &loc_grid[0]);
      //n4 = *(d_neighbour_list[index+3] + &loc_grid[0]);

      //n1 = loc_grid[d_neighbour_list[index + 0]];
      //n2 = loc_grid[d_neighbour_list[index + 1]];
      //n3 = loc_grid[d_neighbour_list[index + 2]];
      //n4 = loc_grid[d_neighbour_list[index + 3]];

      n1 = loc_grid[L*((row+1)%L) + col];
      n2 = loc_grid[L*((row+L-1)%L) + col];
      n3 = loc_grid[L*row + (col+1)%L];
      n4 = loc_grid[L*row + (col+L-1)%L];

      // Using this is massively slower despite being in constant memory
      //n1 = loc_grid[dc_neighbour_list[index + 0]];
      //n2 = loc_grid[dc_neighbour_list[index + 1]];
      //n3 = loc_grid[dc_neighbour_list[index + 2]];
      //n4 = loc_grid[dc_neighbour_list[index + 3]];

      //n_sum = 4;
      index = 5*(spin+1) + n1+n2+n3+n4 + 4;

      // The store back to global memory, not the branch or the RNG generation
      // seems to be the killer here.
      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept
          loc_grid[my_idx] = -1*spin;
          //d_ising_grids[idx*N+my_idx] = -1*spin;
      } //else {
          //loc_grid[my_idx] = +1*spin;
      //} 
      
      // Try avoiding the branch entirely - this seems quite slow
      //diff = curand_uniform(&localState) - d_Pacc[index] ;
      //spin = spin * lrintf(copysignf(1.0f,diff)); 
      //loc_grid[my_idx] = spin;

      // This is even slower (and has a hidden branch)
      //diff = curand_uniform(&localState) - d_Pacc[index] ;
      //spin = signbit(diff) ? -1*spin : spin ;
      //loc_grid[my_idx] = spin;
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

  }

  return;

}

inline __device__ int bits_to_ints(uint8_t* grid, int index){
  // L must be the number of entries in each row here. Important if ever adapting to non-square grids.

  uint8_t one = 1U;
  //int lookup[2] = {-1, 1};

  //nt ibyte = (index)/8;
  //int ibit  = (index)%8;

  // From CUDA-C best practices guide
  int ibyte = index >> 3;
  int ibit  = index & 7 ;

  return blookup[(grid[ibyte] >> ibit) & one]; 

}

// sweep on the gpu - packs a cache of the current grid into on-GPU shared memory
// for efficiency, and using a single bit represenation to acheive this. Mustn't
// be used if L*L*threadsPerBlock/8 > SHGRIDSIZE. 
__global__ void mc_sweep_gpu_bitrep(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // how many bytes do we need per thread to store L*L spins as single bytes
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes ++; }

    // If nbytes x threadsPerBlock is > SHGRIDSIZE we've got a problem....




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

    int imove, row, col, n_sum, index,  n1, n2, n3, n4;
    uint16_t spin_index;

    for (imove=0;imove<L*L*nsweeps;imove++){

      //row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      //col = int((float)L*shrink*curand_uniform(&localState)); 
      //index = L*row+col;

      spin_index = __float2uint_rd((float)L*L*shrink*curand_uniform(&localState));
      row = spin_index/L;
      col = spin_index%L;
      
      spin = bits_to_ints(loc_grid, spin_index);
      
      // find neighbours

      //n1 = *d_neighbour_list[index+0];
      //n2 = *d_neighbour_list[index+1];
      //n3 = *d_neighbour_list[index+2];
      //n4 = *d_neighbour_list[index+3];

      //n1 = bits_to_ints(loc_grid, d_neighbour_list[4*spin_index+0]); 
      //n2 = bits_to_ints(loc_grid, d_neighbour_list[4*spin_index+1]); 
      //n3 = bits_to_ints(loc_grid, d_neighbour_list[4*spin_index+2]); 
      //n4 = bits_to_ints(loc_grid, d_neighbour_list[4*spin_index+3]); 

      n1 = bits_to_ints(loc_grid, L*((row+1)%L) + col);
      n2 = bits_to_ints(loc_grid, L*((row+L-1)%L) + col);
      n3 = bits_to_ints(loc_grid, L*row + (col+1)%L);
      n4 = bits_to_ints(loc_grid, L*row + (col+L-1)%L);

      //n1 = bits_to_ints(loc_grid, L*dc_next[row] + col);
      //n2 = bits_to_ints(loc_grid, L*dc_prev[row] + col);
      //n3 = bits_to_ints(loc_grid, L*row + dc_next[col]);
      //n4 = bits_to_ints(loc_grid, L*row + dc_prev[col]);

      //n1 = bits_to_ints(loc_grid, dc_neighbour_list[4*spin_index+0]); 
      //n2 = bits_to_ints(loc_grid, dc_neighbour_list[4*spin_index+1]); 
      //n3 = bits_to_ints(loc_grid, dc_neighbour_list[4*spin_index+2]); 
      //n4 = bits_to_ints(loc_grid, dc_neighbour_list[4*spin_index+3]); 



      //n_sum = 4;
      index = 5*(spin+1) + n1 + n2 + n3 + n4 + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          ibyte = spin_index >> 3;
          index = spin_index & 7;

          //ibyte = spin_index / 8;
          //index = spin_index % 8; 
          loc_grid[ibyte] ^= one << index;
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

    //for (row=0;row<L;row++){
    //  for (col=0;col<L;col++){
    for (spin_index=0;spin_index<L*L;spin_index++){
      d_ising_grids[L*L*idx + spin_index] = bits_to_ints(loc_grid,spin_index);
      //}
    }

    

  }

  return;

}

// Similar to mc_sweep_gpu_bitrep, but maps each thread in a block of 32 threads to a 
// fixed bit in a datatype of size 4 bytes for faster addressing.
__global__ void mc_sweep_gpu_bitmap32(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // Shared memory for storage of bits
    uint32_t *bit_grid = (uint32_t*)&shared_grid[0];
    uint32_t one  = 1U;
    uint32_t zero = 0U;

    // initialise
    int spin;
    for (spin=0;spin<L*L;spin++) { bit_grid[spin] = zero; }

    // Fill this with the current state of the grid to be manipulated by this thread
    for (spin=0;spin<L*L;spin++){ 
        if ( d_ising_grids[L*L*idx + spin] == 1 ) {
          bit_grid[spin] |= one << threadIdx.x ;
        }
    }
     
    int N=L*L;
    float shrink = (1.0f - FLT_EPSILON)*(float)N;
    //float shrink = (1.0f - FLT_EPSILON);
    int imove, row, col, index, my_idx, n1, n2 , n3, n4;

    for (imove=0;imove<N*nsweeps;imove++){

      //row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      //col = int((float)L*shrink*curand_uniform(&localState)); 
      
      //my_idx = row*L + col;

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));
      //row = my_idx/L;
      //col = my_idx%L;
      row = my_idx >> 6;
      col = my_idx & 63;

      spin = blookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      
      // find neighbours
      n1 = blookup[(bit_grid[L*((row+1)%L) + col] >> threadIdx.x) & one];
      n2 = blookup[(bit_grid[L*((row+L-1)%L) + col] >> threadIdx.x) & one];
      n3 = blookup[(bit_grid[L*row + (col+1)%L] >> threadIdx.x) & one];
      n4 = blookup[(bit_grid[L*row + (col+L-1)%L] >> threadIdx.x) & one];

      //n1 = blookup[(bit_grid[L*((row+1) & 63) + col] >> threadIdx.x) & one];
      //n2 = blookup[(bit_grid[L*((row+L-1) & 63) + col] >> threadIdx.x) & one];
      //n3 = blookup[(bit_grid[L*row + (col+1) & 63 ] >> threadIdx.x) & one];
      //n4 = blookup[(bit_grid[L*row + (col+L-1) & 63 ] >> threadIdx.x) & one];

      //n_sum = 4;
      index = 5*(spin+1) + n1 + n2 + n3 + n4 + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          bit_grid[my_idx] ^= one << threadIdx.x;
      } 
      
      
    } //end for

    // Copy local data back to device global memory
    state[idx] = localState;

    for (row=0;row<L;row++){
      for (col=0;col<L;col++){
        my_idx = L*row + col;
        d_ising_grids[N*idx+my_idx] = blookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      }
    }

  }

  return;

}

__global__ void mc_sweep_gpu_bitmap64(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // Shared memory for storage of bits
    uint64_t *bit_grid = (uint64_t*)&shared_grid[0];
    uint64_t one  = 1U;
    uint64_t zero = 0U;

    // initialise
    int spin;
    for (spin=0;spin<L*L;spin++) { bit_grid[spin] = zero; }

    // Fill this with the current state of the grid to be manipulated by this thread
    for (spin=0;spin<L*L;spin++){ 
        if ( d_ising_grids[L*L*idx + spin] == 1 ) {
          bit_grid[spin] |= one << threadIdx.x ;
        }
    }
     
    int N=L*L;
    float shrink = (1.0f - FLT_EPSILON)*(float)N;
    //float shrink = (1.0f - FLT_EPSILON);
    int imove, row, col, n_sum, index, my_idx;

    for (imove=0;imove<L*L*nsweeps;imove++){

      //row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      //col = int((float)L*shrink*curand_uniform(&localState)); 
      
      //my_idx = row*L + col;

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));
      row = my_idx/L;
      col = my_idx%L;

      spin = blookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      
      // find neighbours
      n_sum = 0;
      n_sum += blookup[(bit_grid[L*((row+1)%L) + col] >> threadIdx.x) & one];
      n_sum += blookup[(bit_grid[L*((row+L-1)%L) + col] >> threadIdx.x) & one];
      n_sum += blookup[(bit_grid[L*row + (col+1)%L] >> threadIdx.x) & one];
      n_sum += blookup[(bit_grid[L*row + (col+L-1)%L] >> threadIdx.x) & one];

      //n_sum = 4;
      index = 5*(spin+1) + n_sum + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          bit_grid[my_idx] ^= one << threadIdx.x;
      } 
      
      
    } //end for

    // Copy local data back to device global memory
    state[idx] = localState;

    for (row=0;row<L;row++){
      for (col=0;col<L;col++){
        my_idx = L*row + col;
        d_ising_grids[L*L*idx+my_idx] = blookup[(bit_grid[my_idx] >> threadIdx.x) & one];
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
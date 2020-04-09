#include "mc_gpu.h"



// populate acceptance probabilities
void preComputeProbs_gpu(double beta, double h) {

    float h_Pacc[20];

    int s, nsum, index;
    
    for (s=-1;s<2;s=s+2){
      for (nsum=-4;nsum<5;nsum=nsum+2){
        index = 5*(s+1) + nsum + 4;
        h_Pacc[index] = 2.0*(float)s*((float)nsum+h);
        h_Pacc[index] = expf(-beta*h_Pacc[index]); // single precision version for GPU
      }
    }
  
    gpuErrchk( cudaMemcpyToSymbol(d_Pacc, h_Pacc, 20) );
    

  }  

// sweep on the gpu
__global__ void mc_sweep_gpu(const int L, curandStatePhilox4_32_10_t *state, const int ngrids, int *d_ising_grids, const float beta, const float h) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandStatePhilox4_32_10_t localState = state[idx];

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
    float4 rnd = curand_uniform4(&localState);

    int imove, row, col, my_idx, n_sum, index;
    //float xi;
    for (imove=0;imove<L*L;imove++){
      
      row = int((float)L*shrink*rnd.x);  // CAN generate L without shrink
      col = int((float)L*shrink*rnd.y);  
      
      //row = int((float)L*shrink*curand_uniform(&localState));  // CAN generate L without shrink
      //col = int((float)L*shrink*curand_uniform(&localState));  
      //if (row==L) row--; if (col==L) col--;
      
      // find neighbours
      n_sum = 0;

      my_idx = L*row+col;
      n_sum += loc_grid[L*(row+1)%L + col];
      n_sum += loc_grid[L*(row+L-1)%L + col];
      n_sum += loc_grid[L*row + (col+1)%L];
      n_sum += loc_grid[L*row + (col+L-1)%L];

      //up_idx = L*(row+1)%L + col;
      //dn_idx = L*(row+L-1)%L + col;
      //rt_idx = L*row + (col+1)%L;
      //lt_idx = L*row + (col+L-1)%L;
      
      // energy before flip
      //n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
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
      
      //xi = curand_uniform(&localState);
      if (rnd.z < d_Pacc[index] ) {
	// accept 
      } else { 
	loc_grid[my_idx] = -1*loc_grid[my_idx]; // reject 
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
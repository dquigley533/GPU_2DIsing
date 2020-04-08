// -*- mode: C -*-

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  // clock() and clock_t
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <limits.h>

extern "C" {
#include "mt19937ar.h"
}

__global__ void bitstoints(unsigned int *d_grid, int *d_grid_ints, int L, int row, int col) {

  // expect this to run as one thread within a block of 32
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  int tile = row*L+col;

  d_grid_ints[sizeof(unsigned int)*8*tile+tid] = (d_grid[tile] >> tid) & 1U ? +1:-1; // Convert to +/- 1

}

int main () {

  int L=64;
  
  int uint_size = sizeof(unsigned int);
  printf("Size of unsigned int is %d\n",uint_size);

  // Array of 64 x 64 unsigned ints
  unsigned int grid[L*L];

  // Initialise all bits as 1 (spin up)
  int i;
  for (i=0;i<L*L;i++){
    grid[i] = UINT_MAX;
  }

  // Row and column in this array to interrogate
  int row=0;
  int col=0;


  // Row and column of 2D grid, plus current thread (bit) of interest
  unsigned int tile = L*row + col;

  int myspin;  // integer representation of spin (-1/+1)
  int tid;     // thread/bit number within this integer

  printf("================================================\n");
  printf("Initial state of %d spins stored at (%d,%d)\n",sizeof(unsigned int),row,col);
  printf("================================================\n");
  for (tid=0;tid<8*sizeof(unsigned int);tid++){

    // Read the bit
    myspin = (grid[tile] >> tid) & 1U ? +1:-1; // Convert to +/- 1

    //printf("Read bit  from row %d, column %d, thread %d as %d\n",row, col, tid, mybit);
    printf("Read spin from row %d, column %d, thread %d as %d\n",row, col, tid, myspin);

  }

  printf("\n");

  // Toggle bit (XOR)
  tid = 0;  // which bit to flip
  grid[tile] ^= 1U << tid;

  printf("Flipped bit %d at row %d col %d\n",tid,row,col);

  printf("================================================\n");
  printf("Current state of %d spins stored at (%d,%d)\n",sizeof(unsigned int),row,col);
  printf("================================================\n");
  for (tid=0;tid<8*sizeof(unsigned int);tid++){

    //    mybit = (grid[tile] >> tid) & 1; // Read bit no. tid from unsigned int at row, col
    myspin = (grid[tile] >> tid) & 1 ? +1:-1; // Convert to +/- 1

    //printf("Read bit  from row %d, column %d, thread %d as %d\n",row, col, tid, mybit);
    printf("Read spin from row %d, column %d, thread %d as %d\n",row, col, tid, myspin);


  }

  printf("\n");

  printf("Copying grid of spins to GPU\n");

  // check that this way of addressing single bits works in device kernels
  unsigned int *d_grid;
  cudaMalloc(&d_grid,L*L*sizeof(unsigned int));
    
  // array of equivalent integers for checking output
  int *d_grid_ints;
  cudaMalloc(&d_grid_ints,L*L*sizeof(unsigned int)*8*sizeof(int));
  int *grid_ints = (int *)malloc(L*L*sizeof(unsigned int)*sizeof(int));
  

  // copy bit representation to the device
  cudaMemcpy(d_grid,grid,L*L*sizeof(unsigned int),cudaMemcpyHostToDevice);

  bitstoints<<<1,32>>>(d_grid, d_grid_ints, L, row, col);

  cudaMemcpy(grid_ints,d_grid_ints,L*L*sizeof(unsigned int)*8*sizeof(int),cudaMemcpyDeviceToHost);

  printf("================================================\n");
  printf("GPU converted %d spins stored at (%d,%d)\n",sizeof(unsigned int),row,col);
  printf("================================================\n");
  for (tid=0;tid<8*sizeof(unsigned int);tid++){

    printf("Read spin from row %d, column %d, thread %d as %d\n",row, col, tid, grid_ints[tid]);

  }


}

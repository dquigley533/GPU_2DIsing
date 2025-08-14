// -*- mode: C -*-

#include <stdio.h>
#include "gpu_tools.h"
#include "mc_gpu.h"

// Boilerplate error checking code borrowed from stackoverflow
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

// Initialisation 
int gpuInitDevice(int deviceIndex, int* gpu_nsms){

    int idev, count, max_shmem;    

    cudaError_t err;  // cudaError_t is a type defined in cuda.h
    
    // Make sure we have a CUDA capable device to work with
    err = cudaGetDeviceCount(&count);

    if ( (count==0) || (err!=cudaSuccess) ) {
        fprintf(stdout,"No CUDA supported devices are available in this system.\n");
        return -1;
    } else {
        fprintf(stdout,"Found %d CUDA devices in this system\n",count);
    }


    // cudaDeviceProp is a type of struct which we will
    // populate with information about the available 
    // GPU compute devices.
    cudaDeviceProp prop;

    // Loop over the available CUDA devices
    for (idev=0;idev<count;idev++) {

        // Call another CUDA helper function to populate prop
        gpuErrchk( cudaGetDeviceProperties(&prop,idev) );

        // Print out a member of the prop struct which tells
        // us the name of the CUDA device. Other members of this
        // struct tell us the clock speed and compute capability
        // of the device.
        fprintf(stdout,"Device %d : %s\n",idev,prop.name);
        fprintf(stdout,"================================\n");
        fprintf(stdout,"Number of SMs       : %d\n",prop.multiProcessorCount);
        fprintf(stdout,"Max SHMEM per block : %ld KB\n",prop.sharedMemPerBlock/1024);
        max_shmem = prop.sharedMemPerBlock;

      	int value;
      	gpuErrchk(cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, idev));
	      fprintf(stdout,"Max Opt in shared   : %d KB\n",value/1024);




	//fprintf(stderr,"Warp size           : %d\n",prop.warpSize);
        //fprintf(stderr,"Global DRAM         : %ld\n",prop.totalGlobalMem);
	fprintf(stdout,"Recommended ngrids  : %d\n", 4*prop.warpSize*prop.multiProcessorCount);
  fprintf(stdout,"\n");
  


  double dvalue = (8.0 * max_shmem) / (32);
  int maxL = (int)sqrt(dvalue);
  maxL = int(sqrt(32*(maxL*maxL/32)));

  fprintf(stdout,"Estimated largest L for shared memory code path : %d\n", maxL);
  fprintf(stdout,"(Assumes threadsPerBlock = 32)\n");
  fprintf(stdout,"\n");

    }


    // Only set device if specified 
    if (deviceIndex != -1) {
        gpuErrchk( cudaSetDevice(deviceIndex) );
    }

    gpuErrchk( cudaGetDevice(&idev ) );
    fprintf(stdout,"Using CUDA device : %d\n",idev);
    

    // Store these for use elsewhere
    //gpu_idev = idev;
    gpuErrchk( cudaGetDeviceProperties(&prop,idev) );
    *gpu_nsms = prop.multiProcessorCount;

    fflush(stdout);

    return idev;
    
}


__global__ void populate_random(int length, float *rnd_array, curandState *state){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length){

        // 4 random numbers
        float rnd = curand_uniform(&state[idx]);

        // use one of these
        rnd_array[idx] = rnd;

    }       

    return;

}



// Kernel to initialise RNG on the GPU. Used the cuRAND device API with one
// RNG sequence per CUDA thread.
__global__ void init_gpurand(unsigned long long seed, int ngrids, curandState *state){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx<ngrids){

        unsigned long long seq = (unsigned long long)idx;
        
        // Seperate subsequence for each thread
        curand_init(seed, seq, 0ull, &state[idx]);
    
        // Different seed for each thread (faster but risky)
        //curand_init(seed+23498*idx, 0ull, 0ull, &state[idx]);
    }


  }


void gpuInitGrid(int L, int ngrids, int threadsPerBlock, int* ising_grids, int** d_ising_grids, int** d_neighbour_list){

    // Allocate threads to thread blocks
    int blocksPerGrid = ngrids/threadsPerBlock;
    if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }

    // Device copy of Ising grid configurations
    gpuErrchk( cudaMalloc(d_ising_grids,L*L*ngrids*sizeof(int)) );

    // Populate from host copy
    gpuErrchk( cudaMemcpy(*d_ising_grids,ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyHostToDevice) );

    // Neighbours
    gpuErrchk (cudaMalloc((void **)d_neighbour_list, L*L*4*sizeof(int)) );
    preComputeNeighbours_gpu(L, *d_ising_grids, *d_neighbour_list);

}


void gpuInitRand(int ngrids, int threadsPerBlock, unsigned long rngseed, curandState** d_state){


    // Allocate threads to thread blocks
    int blocksPerGrid = ngrids/threadsPerBlock;
    if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }
  
    // Initialise GPU RNG
    gpuErrchk (cudaMalloc((void **)d_state, ngrids*sizeof(curandState)) );
    unsigned long long gpuseed = (unsigned long long)rngseed;
    init_gpurand<<<blocksPerGrid,threadsPerBlock>>>(gpuseed, ngrids, *d_state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Test CUDA RNG (DEBUG)
    
    /*
    float   *testrnd = (float *)malloc(ngrids*sizeof(float));
    float *d_testrnd;
    gpuErrchk( cudaMalloc(&d_testrnd, ngrids*sizeof(float)) );

    int trial;
    for (trial=0;trial<10;trial++){

      populate_random<<<blocksPerGrid,threadsPerBlock>>>(ngrids, d_testrnd, *d_state);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      gpuErrchk( cudaMemcpy(testrnd, d_testrnd, ngrids*sizeof(float), cudaMemcpyDeviceToHost) );

      int i;
      for (i=0;i<ngrids;i++){
        printf("Random number on grid %d : %12.4f\n",i,testrnd[i]);
      }
  
    }

    free(testrnd);
    cudaFree(d_testrnd);
    exit(EXIT_SUCCESS);
    */

}

int select_gpu_method(int L, int ngrids, int threadsPerBlock, int gpu_device ) {

  cudaDeviceProp prop;
  
  /* Max shared memory available to a thread block */
  gpuErrchk( cudaGetDeviceProperties(&prop,gpu_device) );
  int max_shmem = prop.sharedMemPerBlock;

  double value = (8.0 * max_shmem) / (threadsPerBlock);
  int maxL = (int)sqrt(value);
  maxL = int(sqrt(32*(maxL*maxL/32)));

  int method;
  if ( L <= maxL ) {
   //printf("Problem size fits into shared memory with multi-spin coding : using method 2.\n");
    method = 2;
  } else {
    //printf("Problem size too large for shared memory. Using (slow) global memory.\n");
    method = 0;
  }
  
  //printf("For reference, estimated largest L for method 2 : %d\n", maxL);

  fflush(stdout);
  
  return method;

}





#include <stdio.h>
#include "gpu_tools.h"

cudaError_t err;  // cudaError_t is a type defined in cuda.h

// Initialisation 
void gpuInit(int deviceIndex){

    //TODO - use error checking Macro?

    int idev, count;    

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
        printf("Error getting CUDA device properties\n");
        exit(EXIT_FAILURE);
    }

    // Print out a member of the prop struct which tells
    // us the name of the CUDA device. Other members of this
    // struct tell us the clock speed and compute capability
    // of the device.
    printf("Device %d : %s\n",idev,prop.name);

    }

    // Only set device if specified 
    if (deviceIndex != -1) {
        err = cudaSetDevice(deviceIndex);
        if (err!=cudaSuccess) {
            printf("Error setting requested CUDA device");
            exit(EXIT_FAILURE);
        }
    }

    err = cudaGetDevice(&idev);
    if ( err!=cudaSuccess ) {
        printf("Error identifying active CUDA device\n");
        exit(EXIT_FAILURE);
    }

    printf("Using CUDA device : %d\n",idev);

}



// Boilerplate error checking code borrowed from stackoverflow
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


// Kernel to initialise RNG on the GPU. Used the cuRAND device API with one
// RNG sequence per CUDA thread.
__global__ void init_gpurand(unsigned long long seed, curandStatePhilox4_32_10_t *state){

    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    unsigned long long seq = (unsigned long long)idx;
    curand_init(seed, seq, 0ull, &state[idx]);
  
  }
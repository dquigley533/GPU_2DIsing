// -*- mode: C -*-
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>

#include <numpy/arrayobject.h>
#include <Python.h>

extern "C" {
  #include "io.h"
  #include "grid.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"

bool run_gpu = true;    // Run using GPU
bool run_cpu = false;   // Run using CPU

//const int L=64;  /* Size of 2D Ising grid to simulate */


static PyObject* reset_grids_list(PyObject* self, PyObject* args) {
    PyObject *module, *new_list;

    // Import the module
    module = PyImport_ImportModule("gasp");
    if (!module) return NULL;

    // Create a new empty list
    new_list = PyList_New(0);
    if (!new_list) {
        Py_DECREF(module);
        return NULL;
    }

    // Replace the module-level attribute
    if (PyModule_AddObject(module, "grids", new_list) < 0) {
        Py_DECREF(new_list);
        Py_DECREF(module);
        return NULL;
    }

    Py_DECREF(module);
    Py_RETURN_NONE;
}


PyObject* populate_grids_list(int L, int ngrids, int* grid_data) {

  PyObject *module, *list;
  
  // Import the module to get the list
  module = PyImport_ImportModule("gasp");
  if (!module) return NULL;

  list = PyObject_GetAttrString(module, "grids");
  Py_DECREF(module);
  if (!list) return NULL;
  
  for (int g = 0; g < ngrids; ++g) {
    npy_intp dims[2] = {L, L};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (!array) {
      Py_DECREF(list);
      return NULL;
    }
    
    int* arr_data = (int*)PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data, grid_data + g * L * L, L * L * sizeof(int));
    
    if (PyList_Append(list, array) < 0) {
      Py_DECREF(array);
      Py_DECREF(list);
      return NULL;
    }
    
    Py_DECREF(array);  // PyList_Append increments ref count
  }
  
    Py_DECREF(list);
    return NULL;
    
}




int append_grids_list(int L, int ngrids, int* grid_data, int isweep, float* magnetisation) {

  PyObject *module, *list;
  
  // Import the module to get the list
  module = PyImport_ImportModule("gasp");
  if (!module) return -1;

  list = PyObject_GetAttrString(module, "grids");
  Py_DECREF(module);
  if (!list) return -1;

  // Create a new empty list
  PyObject *new_list = PyList_New(0);
  if (!new_list) {
    Py_DECREF(module);
    return -1;
  }
  
  for (int g = 0; g < ngrids; ++g) {
    npy_intp dims[2] = {L, L};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (!array) {
      Py_DECREF(list);
      Py_DECREF(new_list);
      return -1;
    }
    
    int* arr_data = (int*)PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data, grid_data + g * L * L, L * L * sizeof(int));
    
    if (PyList_Append(new_list, array) < 0) {
      Py_DECREF(array);
      Py_DECREF(list);
      Py_DECREF(new_list);
      return -1;
    }
    
    Py_DECREF(array);  // PyList_Append increments ref count
  }

  if (PyList_Append(list, new_list) < 0) {
    // Error handling
    Py_DECREF(list);
    Py_DECREF(new_list);
    return -1;
  }

  Py_DECREF(list);
  Py_DECREF(new_list);
  return 0; // Success
    
}


// Corresponds to itask = 0 in the original ising.cu
static PyObject* method_run_nucleation_swarm(PyObject* self, PyObject* args, PyObject* kwargs){

  //unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  unsigned long rngseed = (long)time(NULL);
  float result;
  
  
  /* Positional arguments to extract */
  
  int L = 64;             // Size of LxL grid
  int ngrids = 128;       // Number of grids in the swarm
  int tot_nsweeps = 100;  // Number of sweeps to run for each grid

  double beta = 0.54;     // inverse temperature
  double h = 0.07;        // magnetic field


  /* Keyword arguments to extract */

  int initial_spin = -1;       // Initial value to asign to all spins
   
  double up_threshold = -0.9*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.9*(double)initial_spin;  // Threshold mag at which assumed returned

  int mag_output_int = 100;      // Sweeps between output of magnetisation
  int grid_output_int = 1000;    // Sweeps between output of grids

  int threadsPerBlock = 32;      // Threads per block
  int gpu_method = 0;            // GPU method to use - see mc_gpu.cu

  /* list of keywords */
  static char* kwlist[] = {"initial_spin", "up_threshold", "dn_threshold", "mag_output_int",
    "grid_output_int", "threadsPerBlock", "gpu_method", NULL};


  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiidd|iddiiii", kwlist,
				   &L, &ngrids, &tot_nsweeps, &beta, &h, &initial_spin,
				   &up_threshold, &dn_threshold, &mag_output_int,
				   &grid_output_int, &threadsPerBlock, &gpu_method)) {
    return NULL;
  }

  /*=================================
   Delete old output 
  ================================*/
  remove("gridstates.bin");

  /* Reset the module level list of grids to empty */
  reset_grids_list(self, NULL);

  
/*=================================
   Write output header 
  ================================*/
  printf("# isweep    nucleated fraction\n");


/*=================================
   Initialise simulations
  =================================*/ 
  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  

  
  
  // Initialise as 100% spin down for all grids
  ising_grids = init_grids_uniform(L, ngrids, initial_spin);
  grid_fate = NULL ; // not used
    

/*=================================
    Run simulations - CPU version
  =================================*/ 

  if (run_cpu==true) {


    // Initialise host RNG
    init_genrand(rngseed);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_cpu(beta, h);

    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 0; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold;
        
    // Perform the MC simulations
    //result = mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, write_ising_grids);
    result = mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){


    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    // Initialise model grid on GPU
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    curandState *d_state;                  // Pointer to device RNG states
   
    // Iniialise RNG on GPU
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state); 
      
    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);


    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 0; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method;

    //result = mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids);
    result = mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);
    
    // Free device arrays
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_neighbour_list) );

    
 }

  //populate_grids_list(L, ngrids, ising_grids);
  
/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  free(ising_grids);

  return PyFloat_FromDouble((double)result);
    
}


static PyMethodDef GPUIsingMethods[] = {
  {"run_nucleation_swarm", (PyCFunction)method_run_nucleation_swarm, METH_VARARGS | METH_KEYWORDS, "DocString placeholder!"},
  {NULL, NULL, 0, NULL} /* DQ - not sure why we need this second member of the struct? */
};

/* Defines the Python module that our wrapped function will live inside */
static struct PyModuleDef gaspmodule = {
    PyModuleDef_HEAD_INIT,   /* Always this https://docs.python.org/3/c-api/module.html#initializing-c-modules */
    "gasp",                  /* Name of the module (same as name of wrapped function in this case */
    "Python interface to the GASP code for GPU accelerated 2D Ising Model", /* DocString for the module */
    -1,                      /* Memory required (bytes) for each instance, or -1 if multiple instances not supported */
    GPUIsingMethods,         /* Name of the PyMethodDef object to use for the list of member functions */
};

/* The init function for the module - doesn't do much other than call a function to create the module as specified in the above*/
PyMODINIT_FUNC PyInit_gasp(void) { 

  

   // Initialize NumPy API
  import_array();
  
  /* Assign created module to a variable */
  PyObject* module = PyModule_Create(&gaspmodule);
  if (!module) return NULL;  

  /* Initialise GPU device if available on import */
  if (run_gpu==true) {

    // Initialise GPU device(s)
    int igpu = gpuInitDevice(-1); 
    if (igpu==-1){
      printf("Falling back to CPU\n");
      run_cpu=true;
      run_gpu=false;
    }
    
  }

  /* Create a module level object to hold the most recent grids */
  PyObject* list = PyList_New(0);
  if (!list) {
    Py_DECREF(module);
    return NULL;
  }

  // Set the list as a module-level attribute called grids
  if (PyModule_AddObject(module, "grids", list) < 0) {
    Py_DECREF(list);
    Py_DECREF(module);
    return NULL;
  }

  

  
  return module;
    
}

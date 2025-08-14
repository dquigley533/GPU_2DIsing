// -*- mode: C -*-
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>

extern "C" {
  #include "io.h"
  #include "grid.h"
}

#include "mc_gpu.h"


bool run_gpu = true;    // Run using GPU
bool run_cpu = false;   // Run using CPU

int idev = -1; // GPU device to use
int gpu_nsms;  // Number of multiprocessors on the GPU


/* Pointer to memory in which we might store copies of all grids 
   generated to pass back to Python. 8 bit integers to save RAM */
int8_t* grid_history = NULL;
int ihist = 0;                 // Current snapshot number

static PyObject* reset_grids_list(PyObject* self, PyObject* args) {
    PyObject *module, *new_list;

    // Import the module
  module = PyImport_ImportModule("gasp");
  if (!module) {
    PyErr_SetString(PyExc_ImportError, "Could not import 'gasp' module");
    return NULL;
  }

    // Create a new empty list
    new_list = PyList_New(0);
  if (!new_list) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create new list for grids");
    return NULL;
  }

    // Replace the module-level attribute
  if (PyModule_AddObject(module, "grids", new_list) < 0) {
    Py_DECREF(new_list);
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'grids' attribute to module");
    return NULL;
  }

    Py_DECREF(module);
    Py_RETURN_NONE;
}


PyObject* populate_grids_list(int L, int ngrids, int* grid_data) {

  PyObject *module, *list;
  
  // Import the module to get the list
  module = PyImport_ImportModule("gasp");
  if (!module) {
    PyErr_SetString(PyExc_ImportError, "Could not import 'gasp' module");
    return NULL;
  }

  list = PyObject_GetAttrString(module, "grids");
  Py_DECREF(module);
  if (!list) {
    PyErr_SetString(PyExc_AttributeError, "Could not get 'grids' attribute from module");
    return NULL;
  }
  
  for (int g = 0; g < ngrids; ++g) {
    npy_intp dims[2] = {L, L};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (!array) {
      Py_DECREF(list);
      PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for grid");
      return NULL;
    }
    
    int* arr_data = (int*)PyArray_DATA((PyArrayObject*)array);
    memcpy(arr_data, grid_data + g * L * L, L * L * sizeof(int));
    
    if (PyList_Append(list, array) < 0) {
      Py_DECREF(array);
      Py_DECREF(list);
      PyErr_SetString(PyExc_RuntimeError, "Could not append array to grids list");
      return NULL;
    }
    
  Py_DECREF(array);  // PyList_Append increments ref count
  }
  
    Py_DECREF(list);
    return NULL;
    
}




int append_grids_list(int L, int ngrids, int* grid_data, int isweep, float* magnetisation) {

  npy_intp dims[2] = {L, L};

  // Create a Python list to hold the NumPy arrays
  PyObject* grid_list = PyList_New(ngrids);
  if (!grid_list) return -1;


  int snapsize = ngrids*L*L;
  
  for (int i = 0; i < ngrids; ++i) {


    // Copy current grid data into history array. Not using memcpy as there's a cast involved.
    int isite;
    for (isite=0;isite<L*L;++isite){
      grid_history[snapsize*ihist+i*L*L+isite] = (int8_t)grid_data[i*L*L+isite];
    }
    
    // Create a NumPy array from the data. This just wraps the grid_history_array
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_INT8, (int8_t*)(grid_history + ihist*snapsize + i * L * L));

    if (!array) {
      Py_DECREF(grid_list);
      return -1;
    }


    // Set the base object to None to prevent NumPy from freeing the data
    Py_INCREF(Py_None);
    PyArray_SetBaseObject((PyArrayObject*)array, Py_None);
    
    PyList_SET_ITEM(grid_list, i, array);  // Steals reference

    
  }

  // Get the module attribute "list"
  PyObject* module = PyImport_AddModule("gasp"); 
  if (!module) {
    Py_DECREF(grid_list);
    return -1;
  }
  
  PyObject* existing_list = PyObject_GetAttrString(module, "grids");
  if (!existing_list || !PyList_Check(existing_list)) {
    Py_XDECREF(existing_list);
    Py_DECREF(grid_list);
    PyErr_SetString(PyExc_RuntimeError, "Attribute 'grids' not found or not a list");
    return -1;
  }
  
  // Append the new list of arrays
  if (PyList_Append(existing_list, grid_list) < 0) {
    Py_DECREF(existing_list);
    Py_DECREF(grid_list);
    return -1;
  }
  
  Py_DECREF(existing_list);
  Py_DECREF(grid_list);

  ihist++; // Increment snapshot history counter
  
  return 0;
  
}


// Corresponds to itask = 0 in the original ising.cu
static PyObject* method_run_nucleation_swarm(PyObject* self, PyObject* args, PyObject* kwargs){

  //unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  unsigned long rngseed = (long)time(NULL);
  
  /* Positional arguments to extract */
  
  int L = 64;             // Size of LxL grid
  int ngrids = 128;       // Number of grids in the swarm
  int tot_nsweeps = 100;  // Number of sweeps to run for each grid

  double beta = 0.54;     // inverse temperature
  double h = 0.07;        // magnetic field


  /* Keyword arguments to extract */

  int initial_spin = -1;         // Majority spin in parent phase
   
  double up_threshold = -0.90*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.95*(double)initial_spin;  // Threshold mag at which assumed returned

  int mag_output_int = 100;      // Sweeps between output of magnetisation
  int grid_output_int = 1000;    // Sweeps between output of grids

  int threadsPerBlock = 32;      // Threads per block
  int gpu_method = 0;            // GPU method to use - see mc_gpu.cu

  /* list of keywords */
  static char* kwlist[] = {"L", "ngrid", "tot_nsweeps", "beta", "h",
    "initial_spin", "up_threshold", "dn_threshold", "mag_output_int",
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

  /* Allocate the C array this list of grids wraps */
  if (grid_history != NULL) { free(grid_history); }  
  int nsnaps = tot_nsweeps/grid_output_int + 1;
  grid_history = (int8_t *)malloc(nsnaps*ngrids*L*L*sizeof(int8_t));
  if (grid_history == NULL){
    PyErr_SetString(PyExc_MemoryError, "Error allocating RAM to hold grid history!");
    return NULL;   
  }
  ihist = 0;
  
  


/*=================================
   Initialise simulations
  =================================*/ 
  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  
  
  
  
  // Initialise as 100% spin down for all grids
  ising_grids = init_grids_uniform(L, ngrids, initial_spin);
  grid_fate = NULL ; // not used
  float result; // result of calculation

/*=================================
    Run simulations - CPU version
  =================================*/ 

  if (run_cpu==true) {


    fprintf(stdout, "Using CPU\n");
    
    // Initialise host RNG
    init_genrand(rngseed);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_cpu(beta, h);

    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 0; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = &result;

    /*=================================
      Write output header 
      ================================*/
#ifndef PYTHON
    fprintf(stdout, "# isweep    nucleated fraction\n");
#endif
    
    // Perform the MC simulations
    //result = mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, write_ising_grids);
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){


    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    // Initialise model grid on GPU
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    // Select gpu_method 
    //printf("Calling select_gpu_method\n");
    gpu_method = select_gpu_method(L, ngrids, threadsPerBlock, idev);
    
    curandState *d_state;                  // Pointer to device RNG states

    // Initialise RNG on GPU
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);


    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 0; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = &result;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method;

    /*=================================
      Write output header 
      ================================*/
#ifndef PYTHON
    fprintf(stdout, "# isweep    nucleated fraction\n");
#endif

    //result = mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids);
    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);
    
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

static PyObject* method_run_committor_calc(PyObject* self, PyObject* args, PyObject* kwargs){

  unsigned long rngseed = (long)time(NULL);

  int L = 64;
  int ngrids = 128;
  int tot_nsweeps = 100;
  double beta = 0.54;
  double h = 0.07;
 int initial_spin = -1;         // Majority spin in parent phase   
  double up_threshold = -0.90*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.93*(double)initial_spin;  // Threshold mag at which assumed returned
  int mag_output_int = 100;
  int grid_output_int = 1000;
  int threadsPerBlock = 32;
  int gpu_method = 0;
  const char* grid_input = "gridstates.bin";
  PyObject* grid_array_obj = NULL;

  static char* kwlist[] = {"L", "ngrid", "tot_nsweeps", "beta", "h",
    "initial_spin", "up_threshold", "dn_threshold", "mag_output_int",
    "grid_output_int", "threadsPerBlock", "gpu_method", "grid_input", "grid_array", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiidd|iddiiiisO", kwlist,
       &L, &ngrids, &tot_nsweeps, &beta, &h, &initial_spin,
       &up_threshold, &dn_threshold, &mag_output_int,
       &grid_output_int, &threadsPerBlock, &gpu_method, &grid_input, &grid_array_obj)) {
    return NULL;
  }

  int* grid_array_c = NULL;
  int grid_array_count = 1; // Default to 1 if no grid_array provided
  if (grid_array_obj && grid_array_obj != Py_None) {
    if (!PyList_Check(grid_array_obj)) {
      PyErr_SetString(PyExc_TypeError, "grid_array must be a list of 2D NumPy arrays");
      return NULL;
    }
    grid_array_count = (int)PyList_Size(grid_array_obj);
    if (grid_array_count > gpu_nsms) {
      PyErr_SetString(PyExc_ValueError, "grid_array list length exceeds gpu_nsms");
      return NULL;
    }
    if (grid_array_count == 0) {
      PyErr_SetString(PyExc_ValueError, "grid_array list is empty");
      return NULL;
    }
    if (gpu_nsms % grid_array_count != 0) {
      PyErr_SetString(PyExc_ValueError, "grid_array list length must divide gpu_nsms exactly");
      return NULL;
    }
    if (ngrids % gpu_nsms != 0) {
      PyErr_SetString(PyExc_ValueError, "ngrids must be a multiple of gpu_nsms");
      return NULL;
    }
    if (ngrids % grid_array_count != 0) {
      PyErr_SetString(PyExc_ValueError, "ngrids must be a multiple of grid_array list length");
      return NULL;
    }

    // Allocate space for all arrays
    grid_array_c = (int*)malloc(grid_array_count * L * L * sizeof(int));
    if (!grid_array_c) {
      PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for grid_array_c");
      return NULL;
    }
    for (int arr_idx = 0; arr_idx < grid_array_count; ++arr_idx) {
      PyObject* arr_obj = PyList_GetItem(grid_array_obj, arr_idx);
      if (!PyArray_Check(arr_obj)) {
        PyErr_SetString(PyExc_TypeError, "All items in grid_array must be NumPy arrays");
        free(grid_array_c);
        return NULL;
      }
      PyArrayObject* arr = (PyArrayObject*)arr_obj;
      if (PyArray_TYPE(arr) != NPY_INT8 || PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_TypeError, "Each grid_array item must be a 2D NumPy array of type int8");
        free(grid_array_c);
        return NULL;
      }
      int nrows = (int)PyArray_DIM(arr, 0);
      int ncols = (int)PyArray_DIM(arr, 1);
      if (nrows != L || ncols != L) {
        PyErr_SetString(PyExc_ValueError, "Each grid_array item must have shape LxL");
        free(grid_array_c);
        return NULL;
      }
      npy_int8* arr_data = (npy_int8*)PyArray_DATA(arr);
      for (int i = 0; i < L * L; ++i) {
        grid_array_c[arr_idx * L * L + i] = (int)arr_data[i];
      }
    }
  }

/*=================================
   Delete old output 
  ================================*/
  remove("gridstates.bin");

  /* Reset the module level list of grids to empty */
  reset_grids_list(self, NULL);

  /* Allocate the C array this list of grids wraps */
  if (grid_history != NULL) { free(grid_history); }  
  int nsnaps = tot_nsweeps/grid_output_int + 1;
  grid_history = (int8_t *)malloc(nsnaps*ngrids*L*L*sizeof(int8_t));
  if (grid_history == NULL){
    PyErr_SetString(PyExc_MemoryError, "Error allocating RAM to hold grid history!");
    return NULL;   
  }
  ihist = 0;
  
  


/*=================================
   Initialise simulations
  =================================*/ 
  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  
  
  // Initialise as 100% spin down for all grids
  //ising_grids = init_grids_uniform(L, ngrids, initial_spin);
    if (strcmp(grid_input, "gridstates.bin") == 0) {
      ising_grids = init_grids_from_file(L, ngrids); // read from gridinput.bin
    } else if ((strcmp(grid_input, "NumPy") == 0) && grid_array_obj != NULL) {
      ising_grids = init_grids_from_array(L, ngrids, grid_array_count, grid_array_c); // read from the supplied array
    } else {
      PyErr_SetString(PyExc_ValueError, "Invalid grid_input option or no grid_array supplied");
      return NULL;
    }


  grid_fate = init_fates(ngrids); // grid fates

  float *result = (float *)malloc(grid_array_count * sizeof(float)); // result of committor calcs
  if (result == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for result array");
    free(ising_grids);
    if (grid_fate) free(grid_fate);
    if (grid_array_c) free(grid_array_c);
    return NULL;
  }

/*=================================
    Run simulations - CPU version
  =================================*/ 

  if (run_cpu==true) {


#ifndef PYTHON
    fprintf(stdout,"Using CPU\n");
#endif

    // Initialise host RNG
    init_genrand(rngseed);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_cpu(beta, h);

    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 1; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = grid_array_count; calc.result=result;

    
    // Perform the MC simulations
    //result = mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, write_ising_grids);
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){


    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    // Initialise model grid on GPU
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    // Select gpu_method 
    //printf("Calling select_gpu_method\n");
    gpu_method = select_gpu_method(L, ngrids, threadsPerBlock, idev);
    
    curandState *d_state;                  // Pointer to device RNG states

    // Initialise RNG on GPU
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);



    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 1; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = grid_array_count; calc.result=result;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method; 


    //result = mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids);
    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);
    
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
  free(grid_fate);
  if (grid_array_c) free(grid_array_c);

  //return PyFloat_FromDouble((double)result);

  // Convert result array to a list of Python tuples
  // Second entry in the tuple will hold the uncertainty on the pB estimate
  PyObject* pylist = PyList_New(grid_array_count);
  if (!pylist) {
    if (result) free(result);
    return NULL;
  }
  for (int i = 0; i < grid_array_count; ++i) {
    PyObject* tuple = PyTuple_New(2);
    if (!tuple) {
      Py_DECREF(pylist);
      if (result) free(result);
      return NULL;
    }
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble((double)result[i]));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(0.0));
    PyList_SET_ITEM(pylist, i, tuple);
  }
  if (result) free(result);
  return pylist;


}



static PyMethodDef GPUIsingMethods[] = {
  {"run_committor_calc", (PyCFunction)method_run_committor_calc, METH_VARARGS | METH_KEYWORDS, "DocString placeholder for committor calc!"},
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
  if (!module) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create gasp module");
    return NULL;
  }

  /* Initialise GPU device if available on import */
  if (run_gpu==true) {

    // Initialise GPU device(s)
    idev = gpuInitDevice(-1, &gpu_nsms); 
    if (idev==-1){
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
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'grids' attribute to gasp module");
    return NULL;
  }


  return module;
    
}

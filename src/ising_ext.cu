// -*- mode: C -*-
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>

#include <Python.h>

extern "C" {
  #include "mc_cpu.h"
  #include "io.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"

const int L=64;  /* Size of 2D Ising grid to simulate */

static PyMethodDef GPUIsingMethods[] = {
  //  {"fputs", method_fputs, METH_VARARGS, "Python interface for fputs C library function"},
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

  /* Assign created module to a variable */
  PyObject* module = PyModule_Create(&gaspmodule);

  /* Add int constant by name */
  PyModule_AddIntConstant(module, "GASP_LG", L);

  gpuInit(-1); // Initialise GPU device(s)

  
  return module;
    
}

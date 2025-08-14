#include <stdio.h>
#include <stdlib.h>

/* Function to initialise initial grids */
int *init_grids_uniform(int L, int ngrids, int spin);
int *init_grids_from_file(int L, int ngrids);
int *init_grids_from_array(int L, int ngrids, int ninputs, int *array);

/* Initialise array which stores eventual fate of grids */
int *init_fates(int ngrids);

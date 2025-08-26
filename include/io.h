#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <hdf5.h>

int write_ising_grids(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size);
int create_ising_grids_hdf5(int L, int ngrids, int tot_nsweeps, double h, double beta);
int write_ising_grids_hdf5(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size);
void read_input_grid(int L, int ngrids, int *ising_grids);

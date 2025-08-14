#include "grid.h"
#include "io.h"

/* Create host grids and populate all of them with a uniform spin */
int *init_grids_uniform(int L, int ngrids, int spin){

  int i;

  // Host copy of Ising grid configurations
  int *ising_grids = (int *)malloc(L*L*ngrids*sizeof(int));
  if (ising_grids==NULL){
    fprintf(stderr,"Error allocating memory for Ising grids!\n");
    exit(EXIT_FAILURE);
  }

  // Initialise as spin   
  for (i=0;i<L*L*ngrids;i++) { ising_grids[i] = spin; }

  return ising_grids;
   
}

/* Create host grids and initialise all from gridstates.bin */
int *init_grids_from_file(int L, int ngrids) {

  // Host copy of Ising grid configurations
  int *ising_grids = (int *)malloc(L*L*ngrids*sizeof(int));
  if (ising_grids==NULL){
    fprintf(stderr,"Error allocating memory for Ising grids!\n");
    exit(EXIT_FAILURE);
  }

  // Read from file
  read_input_grid(L, ngrids, ising_grids);

  return ising_grids;
  
}

/* Create host grids and initialise all to match provided grid */
int *init_grids_from_array(int L, int ngrids, int *array) {

  // Host copy of Ising grid configurations
  int *ising_grids = (int *)malloc(L*L*ngrids*sizeof(int));
  if (ising_grids==NULL){
    fprintf(stderr,"Error allocating memory for Ising grids!\n");
    exit(EXIT_FAILURE);
  }

  int i, j;
  
  for (i=0;i<ngrids;i++) {
    for (j=0;j<L*L;j++) {
      ising_grids[i*L*L + j] = array[j];
    }
  }

  return ising_grids;
  
}

/* Create array which holds the fate of each Ising grid */
int *init_fates(int ngrids) {

  int *grid_fate = (int *)malloc(ngrids*sizeof(int));
  if (grid_fate==NULL) {
    printf("Error allocating memory for grid fates\n");
    exit(EXIT_FAILURE);
  }

  int i;
  for (i=0;i<ngrids;i++) { grid_fate[i] = -1; } // all pending

  return grid_fate;

}



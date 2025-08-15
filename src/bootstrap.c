#include <bootstrap.h>



// Compute an estimate of the error bar on the committor given the array of grid fates
double bootstrap_errbar(int ngrids, int m, int *grid_fate_host){

  int nA, nB, igrid, im, irand;
  double standard_deviation = 0.0;
  
  int *ngrids_rand_ind = (int*)malloc(ngrids*sizeof(int));
  if (ngrids_rand_ind == NULL) {
    fprintf(stderr, "Memory allocation failed for ngrids_rand_ind\n");
    return -1.0; // Indicate error
  }

  double *committor_store = (double*)malloc(m*sizeof(double));
  if (committor_store == NULL) {
    fprintf(stderr, "Memory allocation failed for committor_store\n");
    free(ngrids_rand_ind);
    return -1.0; // Indicate error
  }

  for (im=0;im<m;im++) {
    
    nA=0; nB=0;
    
    // Build a list resampling indices
    for (igrid=0;igrid<ngrids;igrid++){
      ngrids_rand_ind[igrid] = rand()%ngrids;
    }
    
    // Calculate committor for this resampling
    for (igrid=0;igrid<ngrids;igrid++){
      irand = ngrids_rand_ind[igrid];
      if ((int)grid_fate_host[irand]==0 ) {
        nA++;
      } else if ((int)grid_fate_host[irand]==1 ) {
        nB++;
      } // fate  
    } //grids

    committor_store[im] = (double)nB/(double)(nA+nB); // Store committor
  } // m
  
  // Calculate the standard deviation of the committor estimates
  double standard_deviation_numerator = 0.0, mean_committor = 0.0;            
        
  // Standard deviation calculation
  for (im=0;im<m;im++) {
      mean_committor = mean_committor + committor_store[im];
  } 
  mean_committor = mean_committor/m;
  for (im=0;im<m;im++) {
    standard_deviation_numerator = standard_deviation_numerator + (committor_store[im]-mean_committor)*(committor_store[im]-mean_committor);
  }
  standard_deviation = sqrt(standard_deviation_numerator/m);

  // Clean up
  free(ngrids_rand_ind);
  free(committor_store);

  return standard_deviation;

}
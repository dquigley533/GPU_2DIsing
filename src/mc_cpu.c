#include <stdio.h>
#include <time.h>
#include "mc_cpu.h"
#include "io.h"

double Pacc[20];         // Cache of acceptance probabilities


// populate acceptance probabilities
void preComputeProbs_cpu(double beta, double h) {

  // Pre-compute acceptance probabilities for all possibilities
  // s  : nsum               : index 5*(s+1) + nsum + 4
  // +1 : -4 = -1 -1 -1 -1   : 10
  // +1 : -2 = -1 -1 -1 +1   : 12
  // +1 :  0 = -1 -1 +1 +1   : 14
  // +1 : +2 = -1 +1 +1 +1   : 16
  // +1 : +4 = +1 +1 +1 +1   : 18
  // -1 : -4 = -1 -1 -1 -1   : 0
  // -1 : -2 = -1 -1 -1 +1   : 2
  // -1 :  0 = -1 -1 +1 +1   : 4
  // -1 : +2 = -1 +1 +1 +1   : 6
  // -1 : +4 = +1 +1 +1 +1   : 8

  int s, nsum, index;
  for (s=-1;s<2;s=s+2){
    for (nsum=-4;nsum<5;nsum=nsum+2){
      index = 5*(s+1) + nsum + 4;
      Pacc[index] = 2.0*(double)s*((double)nsum+h);
      Pacc[index] = exp(-beta*Pacc[index]);
    }
  }

}     

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps) {

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

  int imove, row, col;

  for (imove=0;imove<L*L*nsweeps;imove++){

    // pick random spin
    row = floor(L*genrand_real3());  // RNG cannot generate 1.0 so safe
    col = floor(L*genrand_real3()); 

    // find neighbours
    int my_idx = L*row+col;
    int up_idx = L*((row+L-1)%L) + col;  
    int dn_idx = L*((row+1)%L)   + col;   
    int rt_idx = L*row + (col+1)%L;
    int lt_idx = L*row + (col+L-1)%L;
    
    // energy before flip
    int n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_old = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    int index = 5*(loc_grid[my_idx]+1) + n_sum + 4;

    // flip
    loc_grid[my_idx] = -1*loc_grid[my_idx];

    // energy after flip
    //n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_new = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    //double delta_energy = energy_new - energy_old;
    //double prob = exp(-beta*delta_energy);

    double xi = genrand_real3();
    //if (xi < prob) {
    if (xi < Pacc[index] ) {
      // accept
      //fprintf(stderr,"Accepted a move\n");
    } else {
      loc_grid[my_idx] = -1*loc_grid[my_idx]; // reject
    }


  } // end for

}

void compute_magnetisation_cpu(int L, int *ising_grids, int grid_index, float *magnetisation){

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

    double m = 0.0f;

    int i;
    for (i=0;i<L*L;i++) { m += (double)loc_grid[i]; }
    magnetisation[grid_index] = (float)(m/(double)(L*L));

  return;

}

float mc_driver_cpu(mc_grids_t grids, double beta, double h, int* grid_fate, mc_sampler_t samples, mc_function_t calc, GridOutputFunc outfunc){

    clock_t t1,t2;  // For measuring time taken
    int isweep;     // MC sweep loop counter
    int igrid;      // counter for loop over replicas

    // Unpack structs
    int L = grids.L;
    int ngrids = grids.ngrids;
    int *ising_grids = grids.ising_grids;

    int tot_nsweeps = samples.tot_nsweeps;
    int mag_output_int = samples.mag_output_int;
    int grid_output_int = samples.grid_output_int;

    int itask = calc.itask;
    double dn_thr = calc.dn_thr;
    double up_thr = calc.up_thr;

    
    // How many sweeps to run in each call to mc_sweeps_cpu
    int sweeps_per_call;
    sweeps_per_call = mag_output_int < grid_output_int ? mag_output_int : grid_output_int;
    
    // Magnetisation of each grid
    float *magnetisation = (float *)malloc(ngrids*sizeof(float));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation array!\n");
      exit(EXIT_FAILURE);
    }

    // result - either fraction of nucleated trajectories (itask=0) or comittor (itask=1)
    float result;
    
    t1 = clock();  // Start timer

    isweep = 0;
    while (isweep < tot_nsweeps){

      // Output grids to file
      if (isweep%grid_output_int==0){
        outfunc(L, ngrids, ising_grids, isweep, magnetisation);  
      }

      // Report magnetisations
      if (isweep%mag_output_int==0){
        for (igrid=0;igrid<ngrids;igrid++){
          compute_magnetisation_cpu(L, ising_grids, igrid, magnetisation);
          //printf("Magnetisation of grid %d at sweep %d = %8.4f\n",igrid, isweep, magnetisation[igrid]);
        }
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<ngrids;igrid++){
            if ( magnetisation[igrid] > up_thr ) nnuc++;
          }
#ifndef PYTHON
          fprintf(stdout, "%10d  %12.6f\n",isweep, (double)nnuc/(double)ngrids);
#endif
          result = (float)((double)nnuc/(double)ngrids);
	  if (nnuc==ngrids) break; // Stop if everyone has nucleated
	  
        } else if ( itask == 1 ){

          // Statistics on fate of trajectories
          int nA=0, nB=0;
          for (igrid=0;igrid<ngrids;igrid++){
            //printf("grid_fate[%d] = %d\n",igrid, grid_fate[igrid]);
            if (grid_fate[igrid]==0 ) {
              nA++;
            } else if (grid_fate[igrid]==1 ) {
              nB++;
            } else {
              if ( magnetisation[igrid] > up_thr ){
                grid_fate[igrid] = 1;
                nB++;
              } else if (magnetisation[igrid] < dn_thr ){
                grid_fate[igrid] = 0;
                nA++;
              }
            } // fate
          } //grids

          // Monitor progress
          result = (double)nB/(double)(nA+nB);
#ifndef PYTHON
          fprintf(stdout, "\r Sweep : %10d, Reached m = %6.2f : %4d , Reached m = %6.2f : %4d , Unresolved : %4d, pB = %10.6f",
		 isweep, dn_thr, nA, up_thr, nB, ngrids-nA-nB, result);
          fflush(stdout);
#endif

          if (nA + nB == ngrids) break; // all fates resolved
        } // task
      } 

      // MC Sweep - CPU
      for (igrid=0;igrid<ngrids;igrid++) {
        mc_sweep_cpu(L, ising_grids, igrid, beta, h, sweeps_per_call);
      }
      isweep += sweeps_per_call;

    }
      
    t2 = clock();  // Stop Timer

#ifndef PYTHON
    fprintf(stdout, "\n# Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

    if (itask==1) { printf("pB estimate : %10.6f\n",result); }; 
#endif

    // Release memory
    free(magnetisation);  

    return result;
  
}

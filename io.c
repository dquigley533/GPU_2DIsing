#include "io.h"


void write_ising_grids(int L, int ngrids, int *ising_grids, int isweep){

    // Set filename
    char filename[14];
    sprintf(filename, "gridstates.dat");
    //printf("%s\n",filename);

    // open file
    FILE *ptr = fopen(filename,"ab");
    if (ptr==NULL){
        FILE *ptr = fopen(filename,"wb"); // open for write if not available for append 
        if (ptr==NULL){
            fprintf(stderr,"Error opening %s for write!\n",filename);
            exit(EXIT_FAILURE);
        }   
    }

    // file header - size of grid, number of grids and current sweep
    fwrite(&L,sizeof(int),1,ptr);
    fwrite(&ngrids,sizeof(int),1,ptr);
    fwrite(&isweep,sizeof(int),1,ptr);

    // pack everything into as few bits as possible
    int nbytes = L*L*ngrids/8;
    if ( (L*L*ngrids)%8 !=0 ) { nbytes++; }
    char *bitgrids = (char *)malloc(nbytes);
    if (bitgrids==NULL){
        fprintf(stderr,"Error allocating output buffer!");
        exit(EXIT_FAILURE);
    }

    // Set zero 
    memset(bitgrids, 0U, nbytes);
    
    uint8_t one = 1;

    int ibit=0, ibyte=0, iint;
    for (iint=0;iint<L*L*ngrids;iint++){ //loop over grid x squares per grid

        if ( ising_grids[iint] == 1 ) {
            bitgrids[ibyte] |= one << ibit;
            printf("Set bit %d of byte %d\n", ibit, ibyte);
        }

        ibit++;
        if (ibit==8) {
            ibit=0;
            ibyte++;
        }

    }

    // write to file
    fwrite(bitgrids,sizeof(char),nbytes,ptr);

    // Release memory
    free(bitgrids);

    // close file
    fclose(ptr);
  
}
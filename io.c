#include "io.h"


void read_input_grid(int L, int ngrids, int *ising_grids){

    // converts [0,1] to [-1,1]
    const int blookup[2] = {-1, 1};

    // Set filename
    char filename[14];
    sprintf(filename, "gridinput.bin");

    uint32_t one = 1U;   

    // open file
    FILE *ptr = fopen(filename, "rb");
    if (ptr==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename);
        exit(EXIT_FAILURE);
    }

    // read header specifying size of grid
    int Lcheck;
    fread(&Lcheck, sizeof(int), 1, ptr);
    if (Lcheck!=L) {
        fprintf(stderr, "Error - size of grid in input file does not match L!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer!");
        exit(EXIT_FAILURE);
    }

    // Read the grid
    fread(bitgrid, sizeof(char), nbytes, ptr);  

    // Loop over grid points
    int ibit=0, ibyte=0;
    int isite=0, igrid;

    //printf("nbytes = %d\n",nbytes);
    for (ibyte=0;ibyte<nbytes;ibyte++){
        for (ibit=0;ibit<8;ibit++){
            //printf(" %2d ",blookup[(bitgrid[ibyte] >> ibit) & one]);
            // Read into every copy of the grid
            for (igrid=0;igrid<ngrids;igrid++){
                ising_grids[L*L*igrid+isite] = blookup[(bitgrid[ibyte] >> ibit) & one];
            }
            isite++;
            //if (isite%L==0) {printf("\n");}
        }
        if (isite>L*L) break;
    }

    free(bitgrid);  // free input buffer
    fclose(ptr);    // close input file

    fprintf(stderr, "Read initial configuration of all grids from gridinput.bin\n");

}

void write_ising_grids(int L, int ngrids, int *ising_grids, int isweep){

    // Set filename
    char filename[15];
    sprintf(filename, "gridstates.bin");
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
            //printf("Set bit %d of byte %d\n", ibit, ibyte);
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

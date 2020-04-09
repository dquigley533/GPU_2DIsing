#!/usr/bin/env python

import sys
import numpy as np

igrid = 0       # Grid number to visualise
if (len(sys.argv)>1):    # Read from command line if present
    igrid = sys.argv[1]  

# Open file for binary reading
gridfile = open("gridstates.dat", "rb")   

# Loop until we break due to an error
while True:

    # Size of grid
    try:
        L = np.fromfile(gridfile, dtype=np.int32, count=1)[0]
    except: # reached EOF
        break; 

    # Number of grids
    ngrids = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

    # Current MC sweep
    isweep = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

    # Sanity check
    if (igrid >= ngrids):
        printf("Specified grid index is larger than number of grids in file!")
        exit()

    # Report header
    print("Header reports L      = ",L)
    print("               ngrids = ",ngrids)
    print("        current sweep = ",isweep)

    # Array to hold state of lattice
    grid = np.empty([L, L], dtype=np.int)    

    # Loop over grids
    smap = np.array([-1,+1])
    for jgrid in range(ngrids):
        gridbuffer = np.fromfile(gridfile, dtype=np.ubyte, count=L*L//8)
        if igrid==jgrid:
            row = 0
            col = 0
            for byte in gridbuffer:
                bits = np.unpackbits(byte)
                for bit in bits:
                    grid[row, col] = smap[bit]
                    print(row, col, grid[row, col])

                    col = col + 1
                    if col==L:
                        col = 0;
                        row = row+1;

            #print(grid)            


gridfile.close()
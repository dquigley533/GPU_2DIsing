#!/usr/bin/env python

import sys
import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

igrid = 0       # Grid number to visualise
if (len(sys.argv)>1):    # Read from command line if present
    igrid = sys.argv[1]  

# Open file for binary reading
gridfile = open("gridstates.dat", "rb")   

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
grid = [np.empty([L, L], dtype=np.int)]  

# Set display
pygame.init()
block_size = 10
size = (block_size*L, block_size*L)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Sweep : %.d" % isweep)
screen.fill(BLACK)
pygame.display.flip()
clock = pygame.time.Clock()

# Loop until we break due to an error
while True:

    # Loop over grids
    for jgrid in range(ngrids):
    
        # Read the current grid
        gridbuffer = np.fromfile(gridfile, dtype=np.ubyte, count=L*L//8)

        # If this is the grid of interest then populate 
        if igrid==jgrid:
            row = 0
            col = 0
            for byte in gridbuffer:
                bits = np.unpackbits(byte)
                for bit in bits:
                    grid[row, col] = bit
                    col = col + 1
                    if col==L:
                        col = 0;
                        row = row+1;

            for irow, row in enumerate(grid):
                for icol, col in enumerate(row):
                    pixel = model.lattice_state[irow][icol]
                    pygame.draw.rect(screen, color_map[pixel],
                        [icol*block_size, irow*block_size, block_size, block_size])

            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update and limit frame rate
            time_string = "Sweep : %.d" % isweep
            pygame.display.set_caption(time_string)
            pygame.display.flip()
            clock.tick()

            if not running:
                 break  

    # Read next header and quit if not present
    try:
        L = np.fromfile(gridfile, dtype=np.int32, count=1)[0]
    except: # reached EOF
        break; 

    # Number of grids
    ngrids = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

    # Current MC sweep
    isweep = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

gridfile.close()
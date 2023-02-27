#!/usr/bin/env python

import sys
import numpy as np
import pygame
from argparse import ArgumentParser

parser = ArgumentParser(prog='vis_gridstates',
                        description='Visulise the output from GPU_2DISING',
                        )

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

color_map = [BLACK, WHITE]

igrid = 0       # Grid number to visualise
parser.add_argument('--igrid', default=0)
parser.add_argument('-d', '--output_dir', default=".")

args = parser.parse_args()
print(args)
igrid = int(args.igrid)

# Open file for binary reading
gridfile = open(f"{args.output_dir}/gridstates.bin", "rb")   

# Size of grid
try:
    L = np.fromfile(gridfile, dtype=np.int32, count=1)[0]
except: # reached EOF
    print(f"Could not read header from {args.output_dir}/gridstates.bin") 
    exit()
    
# Number of grids
ngrids = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

# Current MC sweep
isweep = np.fromfile(gridfile, dtype=np.int32, count=1)[0]


# Sanity check
if (igrid >= ngrids):
    print("Specified grid index is larger than number of grids in file!")
    exit()

# Report header
print("Header reports L      = ",L)
print("               ngrids = ",ngrids)
print("        current sweep = ",isweep)

# Number of bytes per timeslice
bytes_per_slice = 12+ngrids*(L*L//8)

# Array to hold state of lattice
grid = np.empty([L, L], dtype=np.int32)

# Populate with first entry
gridfile.seek(bytes_per_slice*isweep + 12 + igrid*(L*L//8))
gridbuffer = np.fromfile(gridfile, dtype=np.ubyte, count=L*L//8)


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
running = True
advance = True

iframe = 0

while True:
  
                  
    # Unpack grid from gridbuffer
    irow = 0
    icol = 0
    for byte in gridbuffer:
        bits = np.unpackbits(byte, bitorder='little')
        for bit in bits:
            grid[irow][icol] = bit
            icol = icol + 1
            if icol==L:
                icol = 0;
                irow = irow+1;

    # Draw grid
    for irow, row in enumerate(grid):
        for icol, col in enumerate(row):
            pixel = grid[irow][icol]
            pygame.draw.rect(screen, color_map[pixel],
                        [icol*block_size, irow*block_size, block_size, block_size])

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                advance = not advance;
            if event.key == pygame.K_LEFT:
                iframe = max(iframe-1,0)
                advance = False
            if event.key == pygame.K_RIGHT:
                iframe += 1
                advance = False
            if event.key == pygame.K_UP:
                igrid = min(igrid+1,ngrids-1)
            if event.key == pygame.K_DOWN:
                igrid = max(igrid-1,0)
            if event.key == pygame.K_w:
                outfile = open(f"{args.output_dir}/gridinput.bin","wb")
                outfile.write(L.tobytes())
                gridbuffer.tofile(outfile)
                outfile.close()
                print(f"Grid snapshot written to {args.output_dir}/gridinput.bin")
                
                    
    # Update and limit frame rate
    time_string = "Grid : %d, Sweep : %.d" % (igrid,isweep)
    pygame.display.set_caption(time_string)
    pygame.display.flip()
    clock.tick()

    if not running:
         break

    if (advance):         
        iframe += 1
        
        
    # Read next header and quit if not present
    try:
        gridfile.seek(bytes_per_slice*(iframe)) 
        L = np.fromfile(gridfile, dtype=np.int32, count=1)[0]
    except: # reached EOF
        break; 

    # Number of grids
    ngrids = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

    # Current MC sweep
    isweep = np.fromfile(gridfile, dtype=np.int32, count=1)[0]

    # Read the grid
    #gridfile.seek(igrid*(L*L//8),1)    # Move to igrid
    #gridbuffer = np.fromfile(gridfile, dtype=np.ubyte, count=L*L//8)
    #gridfile.seek((ngrids-igrid+1)*(L*L//8),1)   # skip remaining grids

    gridfile.seek(bytes_per_slice*iframe + 12 + igrid*(L*L//8))
    gridbuffer = np.fromfile(gridfile, dtype=np.ubyte, count=L*L//8)
        
gridfile.close()



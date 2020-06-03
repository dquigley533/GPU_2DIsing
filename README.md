# GPU_2DIsing

GPU accelerated simulations of the 2D Ising model for nucleation studies. Currently implements spin-flip moves with Metropolis acceptance probability. Simulations are performed on a LxL (L=64) lattice initialised with spin = -1 on all sites.

### Usage:

`GPU_2DIsing nsweeps nreplicas mag_output_int grid_output_int threadsPerBlock gpu_device gpu_method beta h itask`

nsweeps         : maximum number of MC sweeps (one sweep is L^2 attempted flips)  
nreplicas       : number of independent replicas to run in parallel on the GPU  
mag_output_int  : sweeps between checking magnetisation against nucleation/committor thresholds  
grid_output_int : sweeps between binary dumps of simulation grids to gridstates.bin  
threadsPerBlock : number of threads/replicas per GPU thread block, 32 is standard  
gpu_device      : index of GPU device to use, set to zero for single GPU system  
gpu_method      : 0, 1 or 2 as described below  
beta            : inverse temperature for simulation  
h               : magnetic field for simulation  
itask           : 0 for reversal from spin down configuration, 1 for committor calculation  

### gpu_method

0 : Uses GPU global memory only. This is slow but allows for much larger L.  
1 : Calculations on each grid are performed on a copy in device shared memory, stored as 1 bit per spin.  
2 : Each warp executes on an LxL array of 32-bit numbers, with each thread in a warp of 32 manipulating its own bit.  

Method 2 is typically fastest but cannot accomodate L much greater than 100. Limitations on shared memory also restrict occupancy (active warps per SM) but this doesn't seem to be the limiting factor to throughput. In most cases running with 3 or 4 warps per SM is optimal but this should be investigated with experimentation.

### Dependencies

On SCRTP systems:  

`module load GCC/8.3.0 CUDA/10.1.243 Python/3.7.4`

The visualisation script requires pygame,

`pip install --user pygame`

or similar.

### Workflow

Run with `itask=0` to generate a swarm of nucleating trajectories. For example, using the K20c (13 SMs) in brigitte.csc.warwick.ac.uk;  

`GPU_2DIsing 50000 1248 100 100 32 0 2 0.54 0.07 0`

which will print the fraction of nucleated trajectores (nucleation threshold is up_threshold in ising.cu) over time, measured in sweeps. This can be fitted to 1-exp(-k(t-t0)) to extract the nucleation rate k and the equilibration time t0.

The swarm can be visualised with;

`python vis_gridstates.py`

using 'p' to pause/play, and arrow keys to move between snapshots (left/right) and replicas (up/down). Pressing 'w' will save the current configuration to `gridinput.bin`. The committor pB for this configuration can then be calculated using;

`GPU_2DIsing 50000 1248 100 100 32 0 2 0.54 0.07 1`

which should be repeated several times to calculate a standard error on pB. Ultimately one will want to automate selection of interesting configurations rather than using the visualisation script. The committor calculation uses dn_threshold and up_threshold (defined in ising.cu) to define the boundaries of the A (initial) and B (reversed) macrostates.


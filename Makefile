#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary
prefix     = $(HOME)
bindir     = $(prefix)/bin

# Define objects in dependency order
OBJECTS   = mt19937ar.o gpu_tools.o mc_cpu.o mc_gpu.o io.o

CC    = gcc
NVCC  = nvcc
LD     = nvcc
CFLAGS =  -O3 -g 
NVFLAGS = -O3 --gpu-architecture sm_35 -g

.PRECIOUS: %.o
.PHONY:  clean

%: %.o
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<


GPU_2DIsing :  $(OBJECTS) ising.cu

	$(LD) -o $(bindir)/GPU_2DIsing $(OBJECTS) ising.cu $(NVFLAGS) 

clean : 

	rm -f *.mod *.d *.il *.o work.*
	rm -f $(bindir)/GPU_2DIsing


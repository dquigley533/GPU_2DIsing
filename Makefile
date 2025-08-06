#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary
prefix     = $(HOME)
bindir     = $(prefix)/bin

# Define objects in dependency order
OBJECTS   = mt19937ar.o grid.o gpu_tools.o io.o mc_cpu.o mc_gpu.o

CC    = gcc
NVCC  = nvcc
LD     = nvcc
CFLAGS =  -O3 
NVFLAGS = -O3 -gencode arch=compute_61,code=sm_61 --generate-line-info  #   Quadro P2000 in Telamon

.PRECIOUS: %.o
.PHONY:  clean

all : GPU_2DIsing

%: %.o
%.o: src/%.c include/%.h
	$(CC) $(CFLAGS) -c -o $@ $< -Iinclude/

%.o: src/%.cu include/%.h
	$(NVCC) $(NVFLAGS) -c -o $@ $< -Iinclude/


GPU_2DIsing :  $(OBJECTS) src/ising.cu

	$(LD) -o $(bindir)/GPU_2DIsing $(OBJECTS) src/ising.cu $(NVFLAGS) -Iinclude/

clean : 

	rm -f *.mod *.d *.il *.o work.*
	rm -f $(bindir)/GPU_2DIsing


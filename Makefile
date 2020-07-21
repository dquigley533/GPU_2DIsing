#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary
prefix     = $(HOME)
bindir     = $(prefix)/bin

# Define objects in dependency order
OBJECTS   =  mt19937ar.o gpu_tools.o mc_cpu.o mc_gpu.o io.o parser.o 

CC    = gcc
NVCC  = nvcc
LD     = nvcc
CFLAGS =  -O3 
NVFLAGS = -O3 -gencode arch=compute_35,code=sm_35 \
	          -gencode arch=compute_75,code=sm_75 \
			  -gencode arch=compute_60,code=sm_60 --generate-line-info

.PRECIOUS: %.o
.PHONY:  clean

all : GPU_2DIsing

%: %.o
%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu %.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<


GPU_2DIsing :  $(OBJECTS) ising.cu

	$(LD) -o $(bindir)/GPU_2DIsing $(OBJECTS) ising.cu $(NVFLAGS)

clean : 

	rm -f *.mod *.d *.il *.o work.*
	rm -f $(bindir)/GPU_2DIsing


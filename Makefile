#-*- mode: makefile; mode: font-lock; vc-back-end: Git -*-
SHELL = /bin/sh

# Where you want the binary
prefix     = $(HOME)
bindir     = $(prefix)/bin

# Object directory
objdir = obj

# Define objects with .obj extension in dependency order
OBJECTS = $(objdir)/mt19937ar.obj $(objdir)/grid.obj $(objdir)/gpu_tools.obj $(objdir)/io.obj $(objdir)/mc_cpu.obj $(objdir)/mc_gpu.obj $(objdir)/bootstrap.obj

CC    = gcc
NVCC  = nvcc
LD    = nvcc
CFLAGS =  -O3 -g
NVFLAGS = -O3 -gencode arch=compute_61,code=sm_61 --generate-line-info  -Wno-deprecated-gpu-targets # Quadro P2000 in Telamon

.PRECIOUS: $(objdir)/%.obj
.PHONY:  clean all

all : GPU_2DIsing

# Create object directory if it does not exist
$(objdir):
	mkdir -p $(objdir)

# Compile C files to obj
$(objdir)/%.obj: src/%.c include/%.h | $(objdir)
	$(CC) $(CFLAGS) -c -o $@ $< -Iinclude/

# Compile CUDA files to obj
$(objdir)/%.obj: src/%.cu include/%.h | $(objdir)
	$(NVCC) $(NVFLAGS) -c -o $@ $< -Iinclude/

GPU_2DIsing : $(OBJECTS) src/ising.cu | $(objdir)
	$(LD) -o $(bindir)/GPU_2DIsing $(OBJECTS) src/ising.cu $(NVFLAGS) -Iinclude/

clean :
	rm -rf $(objdir)
	rm -f $(bindir)/GPU_2DIsing
	rm -f *.mod *.d *.il work.*


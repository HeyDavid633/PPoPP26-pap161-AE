dir = $(shell pwd)

VPATH = $(dir)/utils/ $(dir)/ACSR/ $(dir)/experimental/windowed/ $(dir)/experimental/blocked/

## The path to eigen may be different depending on the machine you are working on. PLEASE CHANGE ACCORDINGLY. ##
# CUDAFLAGS =-O3 -I /srv/local/shared/eigen --expt-relaxed-constexpr -std=c++14
CUDAFLAGS = -O3 -w -I /usr/include/eigen3 --expt-relaxed-constexpr -std=c++14

# SPLAT test fp16
splat_my_test_fp16 : splat_my_test_fp16.o acsr.o utils.o
	nvcc $(CUDAFLAGS) $^ -o splat_my_test_fp16_exec

splat_my_test_fp16.o : splat_my_test_fp16.cu
	nvcc $(CUDAFLAGS) -c $< -o splat_my_test_fp16.o

# SPLAT test fp32
splat_my_test : splat_my_test.o acsr.o utils.o
	nvcc $(CUDAFLAGS) $^ -o splat_my_test_exec

splat_my_test.o : splat_my_test.cu
	nvcc $(CUDAFLAGS) -c $< -o splat_my_test.o

## Recipes for the blocked pattern.
# blocked : blocked.o acsr.o utils.o
# 	nvcc $(CUDAFLAGS) $^ -o blocked_exec 
# blocked.o : blocked.cu
# 	nvcc $(CUDAFLAGS) -c $< -o blocked.o

## Recipes for the windowed pattern.
# windowed : windowed.o acsr.o utils.o
# 	nvcc $(CUDAFLAGS) $^ -o windowed_exec

# windowed.o : window.cu
# 	nvcc $(CUDAFLAGS) -c $< -o windowed.o

## Helper recipes. ##
acsr.o : acsr.cc
	nvcc $(CUDAFLAGS) -c $< -o acsr.o

utils.o : utils.cc 
	nvcc $(CUDAFLAGS) -c $< -o utils.o

clean : 
	rm *.o *_exec sddmm_kernel.cu block_placement.cc

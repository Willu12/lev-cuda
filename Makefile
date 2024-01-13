# Makefile for CUDA project

# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDAFLAGS := -lineinfo

# Source files
SRCS := src/main.cpp src/utils.cpp src/gpu_lev.cpp src/kernels.cu

# Target executable
TARGET := cuda_lev

# Build executable
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) $^ -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) *_results
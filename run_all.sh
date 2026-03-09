#!/bin/bash
set -e
echo "Compiling and running kernels 01 through 09..."

# Base NVCC flags
NVCC_FLAGS="-O3 -std=c++17 -lcublas"

# Array of files to compile
FILES=(
    "01. Build Naive SGEMM.cu"
    "02. Shared Memory Tiling.cu"
    "03. Register Tiling - 1 side.cu"
    "04. Register Tiling - 2 side.cu"
    "05. Vectorized Register Tiling.cu"
    "06. Warp Tiling.cu"
)

# Array of files requiring arch=sm_70 or higher (Tensor Cores)
TC_FILES=(
    "07. Tensor Cores (Async TMA + WGMMA).cu"
    "08. Tensor Cores - Shared Memory WMMA.cu"
    "09. Async Producer–Consumer Pipeline.cu"
)

# Detect architecture if possible (defaulting to sm_75 as T4 was mentioned in comments)
ARCH_FLAG="-arch=sm_75"

echo "=== Standard Kernels ==="
for file in "${FILES[@]}"; do
    echo "Compiling $file..."
    nvcc $NVCC_FLAGS "$file" -o "bin_${file%.cu}"
    echo "Running ${file%.cu}:"
    ./"bin_${file%.cu}"
    echo "----------------------------------------"
done

echo "=== Tensor Core Kernels ==="
for file in "${TC_FILES[@]}"; do
    echo "Compiling $file..."
    nvcc $NVCC_FLAGS $ARCH_FLAG "$file" -o "bin_${file%.cu}"
    echo "Running ${file%.cu}:"
    ./"bin_${file%.cu}"
    echo "----------------------------------------"
done

%%writefile 02_Shared_Memory_Tiling.cu
// each block represent TILE (32.32) or (16x16) - tilling and calculate using its SRAM instead of accessing global memory 
// reuse TILE times each element
#include "/content/runner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>   // std::max
#include <iomanip>     // std::setw

template <const int TILE_SIZE>
__global__ void sgemm_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t)
    {
        // load tile A
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
          sA[ty][tx] = A[row * N + a_col];
        } else {
          sA[ty][tx] = 0.0f;
        }

        // Load tile B
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
          sB[ty][tx] = B[b_row * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Inner accumulation
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = alpha * acc + beta * C[row * K + col];
    }
}

void run_02_shared(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_shared<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    run_benchmark(run_02_shared, M, N, K, "02_Shared_Memory_Tiling");
    return 0;
}

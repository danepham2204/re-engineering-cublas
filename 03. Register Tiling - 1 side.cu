%%writefile 03_Register_Tiling.cu
// %%cuda
#include "/content/runner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 – 1D Register Tiling (Corrected & Optimized)
// Each thread computes THREAD_TILE consecutive elements of C (along columns)
// ─────────────────────────────────────────────────────────────────────────────
template <const int TILE_SIZE, const int THREAD_TILE>
__global__ void sgemm_register_1d(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // sA is square tile
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    
    // sB needs to be wide enough for THREAD_TILE columns per thread
    __shared__ float sB[TILE_SIZE][TILE_SIZE * THREAD_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row (same as Kernel 2)
    int row = blockIdx.y * TILE_SIZE + ty;

    // Each thread owns THREAD_TILE columns
    int col_base = blockIdx.x * (TILE_SIZE * THREAD_TILE) + tx * THREAD_TILE;

    // Register tile – private accumulators
    float acc[THREAD_TILE] = {0.0f};

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t)
    {
        // 1. Load tile A (standard row-major)
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
            sA[ty][tx] = A[row * N + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // 2. Load tile B – each thread loads THREAD_TILE elements
        #pragma unroll
        for (int m = 0; m < THREAD_TILE; ++m)
        {
            int b_row = t * TILE_SIZE + ty;
            int b_col = col_base + m;

            if (b_row < N && b_col < K) {
                sB[ty][tx + m * TILE_SIZE] = B[b_row * K + b_col];
            } else {
                sB[ty][tx + m * TILE_SIZE] = 0.0f;
            }
        }

        __syncthreads();

        // 3. Inner loop – broadcast A, multiply with wide B tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            float a_val = sA[ty][k];

            #pragma unroll
            for (int m = 0; m < THREAD_TILE; ++m)
            {
                acc[m] += a_val * sB[k][tx + m * TILE_SIZE];
            }
        }

        __syncthreads();
    }

    // 4. Write back
    #pragma unroll
    for (int m = 0; m < THREAD_TILE; ++m)
    {
        int final_col = col_base + m;
        if (row < M && final_col < K) {
            C[row * K + final_col] = alpha * acc[m] + beta * C[row * K + final_col];
        }
    }
}

#include "runner.h"

void run_03_register_1d(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    constexpr int TILE_SIZE = 32;
    constexpr int THREAD_TILE = 4;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((K + TILE_SIZE * THREAD_TILE - 1) / (TILE_SIZE * THREAD_TILE), (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_register_1d<TILE_SIZE, THREAD_TILE><<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    run_benchmark(run_03_register_1d, M, N, K, "03_Register_Tiling_1_Side");
    return 0;
}
%%cuda
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

template <const int TILE_SIZE, const int REG_M, const int REG_N>
__global__ void sgemm_register_2d_optimized(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, float alpha, float beta)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = blockIdx.y * TILE_SIZE + ty * REG_M;
    int col_start = blockIdx.x * TILE_SIZE + tx * REG_N;

    float acc[REG_M][REG_N];
    #pragma unroll
    for (int i = 0; i < REG_M; ++i)
        for (int j = 0; j < REG_N; ++j)
            acc[i][j] = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load Global -> Shared (Each thread loads multiple elements)
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += (TILE_SIZE / REG_M)) {
            for (int j = 0; j < TILE_SIZE; j += (TILE_SIZE / REG_N)) {
                int l_row = ty + i;
                int l_col = tx + j;
                if (blockIdx.y * TILE_SIZE + l_row < M && t * TILE_SIZE + l_col < N)
                    sA[l_row][l_col] = A[(blockIdx.y * TILE_SIZE + l_row) * N + (t * TILE_SIZE + l_col)];
                else sA[l_row][l_col] = 0.0f;

                if (t * TILE_SIZE + l_row < N && blockIdx.x * TILE_SIZE + l_col < K)
                    sB[l_row][l_col] = B[(t * TILE_SIZE + l_row) * K + (blockIdx.x * TILE_SIZE + l_col)];
                else sB[l_row][l_col] = 0.0f;
            }
        }
        __syncthreads();

        // Compute 2D Register Tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float reg_A[REG_M];
            float reg_B[REG_N];
            #pragma unroll
            for (int m = 0; m < REG_M; ++m) reg_A[m] = sA[ty * REG_M + m][k];
            #pragma unroll
            for (int n = 0; n < REG_N; ++n) reg_B[n] = sB[k][tx * REG_N + n];

            #pragma unroll
            for (int m = 0; m < REG_M; ++m)
                for (int n = 0; n < REG_N; ++n)
                    acc[m][n] += reg_A[m] * reg_B[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        for (int n = 0; n < REG_N; ++n) {
            int r = row_start + m;
            int c = col_start + n;
            if (r < M && c < K) C[r * K + c] = alpha * acc[m][n] + beta * C[r * K + c];
        }
    }
}

#include "runner.h"

void run_04_register_2d(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    const int TILE_SIZE = 32, REG_M = 4, REG_N = 4;
    dim3 block(TILE_SIZE / REG_N, TILE_SIZE / REG_M);
    dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_register_2d_optimized<TILE_SIZE, REG_M, REG_N><<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    run_benchmark(run_04_register_2d, M, N, K, "04_Register_Tiling_2_Side");
    return 0;
}
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

int main() {
    const int M = 2048, N = 2048, K = 2048; // Increased size for better GFLOPS measurement
    const int TILE_SIZE = 32, REG_M = 4, REG_N = 4;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(TILE_SIZE / REG_N, TILE_SIZE / REG_M);
    dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    sgemm_register_2d_optimized<TILE_SIZE, REG_M, REG_N><<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);

    cudaEventRecord(start);
    for(int i=0; i<10; i++) // Run 10 times for average
        sgemm_register_2d_optimized<TILE_SIZE, REG_M, REG_N><<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f; // Average time

    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (ms * 1e-3);

    std::cout << "Matrix Size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << ms << " ms\n";
    std::cout << "Performance: " << std::setprecision(2) << gflops << " GFLOP/s\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

// output:
// Matrix Size: 2048x2048x2048
// Avg Time: 12.228 ms
// Performance: 1404.96 GFLOP/s
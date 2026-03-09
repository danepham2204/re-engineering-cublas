%%writefile 09_Async_Pipeline.cu
// Kernel 9 - Producer/Consumer Pipeline + Epilogue Shared Memory Staging
//
// Progression from version 8:
// - Version 8: one shared-memory stage for A/B, then WMMA compute, then direct global store
// - Version 9: ping-pong shared-memory stages for A/B plus shared-memory epilogue staging for C
//
// Important note:
// This is a software-pipelined teaching version. It uses a producer/consumer structure
// and double buffering, but it is NOT true hardware-async cp.async/TMA yet.
// The goal is to show the kernel structure that later evolves into fully async pipelines.

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>



constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_WARPS_M = 2;
constexpr int BLOCK_WARPS_N = 4;
constexpr int WARPS_PER_BLOCK = BLOCK_WARPS_M * BLOCK_WARPS_N;  // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;         // 256

constexpr int BLOCK_TILE_M = BLOCK_WARPS_M * WMMA_M;            // 32
constexpr int BLOCK_TILE_N = BLOCK_WARPS_N * WMMA_N;            // 64

__device__ void load_stage_tiles(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half sA[2][BLOCK_TILE_M][WMMA_K],
    __half sB[2][WMMA_K][BLOCK_TILE_N],
    int M, int N, int K,
    int k0,
    int tid,
    int stage)
{
    for (int idx = tid; idx < BLOCK_TILE_M * WMMA_K; idx += THREADS_PER_BLOCK) {
        const int row = idx / WMMA_K;
        const int col = idx % WMMA_K;
        const int g_row = blockIdx.y * BLOCK_TILE_M + row;
        const int g_col = k0 + col;
        sA[stage][row][col] = (g_row < M && g_col < K)
            ? A[g_row * K + g_col]
            : __float2half(0.0f);
    }

    for (int idx = tid; idx < WMMA_K * BLOCK_TILE_N; idx += THREADS_PER_BLOCK) {
        const int row = idx / BLOCK_TILE_N;
        const int col = idx % BLOCK_TILE_N;
        const int g_row = k0 + row;
        const int g_col = blockIdx.x * BLOCK_TILE_N + col;
        sB[stage][row][col] = (g_row < K && g_col < N)
            ? B[g_row * N + g_col]
            : __float2half(0.0f);
    }
}

__global__ void sgemm_tensor_core_pipeline_epilogue(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{

    __shared__ __half sA[2][BLOCK_TILE_M][WMMA_K];
    __shared__ __half sB[2][WMMA_K][BLOCK_TILE_N];
    __shared__ float sC[BLOCK_TILE_M][BLOCK_TILE_N];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int warp_m = warp_id / BLOCK_WARPS_N;
    const int warp_n = warp_id % BLOCK_WARPS_N;

    const int block_row = blockIdx.y * BLOCK_TILE_M;
    const int block_col = blockIdx.x * BLOCK_TILE_N;
    const int c_row = block_row + warp_m * WMMA_M;
    const int c_col = block_col + warp_n * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    load_stage_tiles(A, B, sA, sB, M, N, K, 0, tid, 0);
    __syncthreads();

    int read_stage = 0;
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const int next_k = k0 + WMMA_K;
        const int write_stage = read_stage ^ 1;

        if (next_k < K) {
            load_stage_tiles(A, B, sA, sB, M, N, K, next_k, tid, write_stage);
        }

        if (c_row < M && c_col < N) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;

            const __half* tile_a = &sA[read_stage][warp_m * WMMA_M][0];
            const __half* tile_b = &sB[read_stage][0][warp_n * WMMA_N];

            nvcuda::wmma::load_matrix_sync(a_frag, tile_a, WMMA_K);
            nvcuda::wmma::load_matrix_sync(b_frag, tile_b, BLOCK_TILE_N);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
        read_stage = write_stage;
    }

    if (c_row < M && c_col < N) {
        nvcuda::wmma::store_matrix_sync(&sC[warp_m * WMMA_M][warp_n * WMMA_N], c_frag, BLOCK_TILE_N, nvcuda::wmma::mem_row_major);
    }

    __syncthreads();

    // Epilogue staging:
    // all threads cooperatively write the staged C tile from shared memory to global memory.
    for (int idx = tid; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += THREADS_PER_BLOCK) {
        const int row = idx / BLOCK_TILE_N;
        const int col = idx % BLOCK_TILE_N;
        const int g_row = block_row + row;
        const int g_col = block_col + col;

        if (g_row < M && g_col < N) {
            C[g_row * N + g_col] = alpha * sC[row][col] + beta * C[g_row * N + g_col];
        }
    }
}

#include "/content/runner_half.h"

void run_09_tensor_core_pipeline_epilogue(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
              (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    sgemm_tensor_core_pipeline_epilogue<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This kernel requires Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_09_tensor_core_pipeline_epilogue, M, N, K, "09_Async_Producer_Consumer_Pipeline");
    return 0;
}

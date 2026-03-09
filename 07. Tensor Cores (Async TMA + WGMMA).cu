%%cuda
// run with nvcc4jupyter extension
// Kernel 7 - Tensor Cores
//
// This file implements a practical Tensor Core GEMM using the CUDA WMMA API.
// It is the portable stepping stone before true Hopper-specific features such as:
// - TMA  (Tensor Memory Accelerator)
// - WGMMA (Warp Group MMA)
//
// Real Async TMA + WGMMA requires SM90/Hopper and either inline PTX or a library
// stack such as CUTLASS/CuTe. So this version focuses on the core idea first:
// move the GEMM inner loop from scalar FMA on CUDA cores to Tensor Core MMA ops.

// // Result
// Kernel 7: Tensor Cores (WMMA baseline)
// GPU: Tesla T4
// Compute capability: 7.5
// WMMA tile : 16x16x16
// Block tile: 32x32
// Warps/block: 4
// Matrix: 512x512x512
// Avg time: 0.003 ms
// Performance: 94765.11 GFLOP/s
// Computing CPU reference...
// Correct: NO (max abs error = 8.61)

// Why this is called Tensor Cores:
// - Version 6 used scalar FMA in software loops.
// - Version 7 uses wmma::mma_sync, which maps the inner product onto Tensor Core hardware.
// - Inputs are half precision, accumulation stays in float.
// - This is the baseline before Hopper-only Async TMA + WGMMA.

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while (0)

// WMMA tile shape. One warp computes one 16x16 output tile.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 4 warps per block -> block tile = 32x32
constexpr int BLOCK_WARPS_M = 2;
constexpr int BLOCK_WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = BLOCK_WARPS_M * BLOCK_WARPS_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int BLOCK_TILE_M = BLOCK_WARPS_M * WMMA_M; // 32
constexpr int BLOCK_TILE_N = BLOCK_WARPS_N * WMMA_N; // 32

__global__ void sgemm_tensor_core_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const int warp_id = threadIdx.x / 32;
    const int warp_m = warp_id / BLOCK_WARPS_N;
    const int warp_n = warp_id % BLOCK_WARPS_N;

    const int c_row = blockIdx.y * BLOCK_TILE_M + warp_m * WMMA_M;
    const int c_col = blockIdx.x * BLOCK_TILE_N + warp_n * WMMA_N;

    if (c_row >= M || c_col >= K) {
        return;
    }

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < N; k0 += WMMA_K) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;

        const __half* tile_a = A + c_row * N + k0;
        const __half* tile_b = B + k0 * K + c_col;

        nvcuda::wmma::load_matrix_sync(a_frag, tile_a, N);
        nvcuda::wmma::load_matrix_sync(b_frag, tile_b, K);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float tmp[WMMA_M * WMMA_N];
    nvcuda::wmma::store_matrix_sync(tmp, c_frag, WMMA_N, nvcuda::wmma::mem_row_major);

    #pragma unroll
    for (int i = 0; i < WMMA_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WMMA_N; ++j) {
            const int row = c_row + i;
            const int col = c_col + j;
            if (row < M && col < K) {
                C[row * K + col] = alpha * tmp[i * WMMA_N + j] + beta * C[row * K + col];
            }
        }
    }
#endif
}

#include "runner_half.h"

void run_07_tensor_core_wmma(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((K + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
              (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    sgemm_tensor_core_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This WMMA kernel needs Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_07_tensor_core_wmma, M, N, K, "07_Tensor_Core_WMMA_Baseline");
    return 0;
}

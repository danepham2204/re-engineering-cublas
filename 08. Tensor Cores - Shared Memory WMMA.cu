%%writefile 08_Tensor_Cores_Smem_WMMA.cu
// Kernel 8 - Tensor Cores with Shared Memory WMMA
//
// Progression from version 7:
// - Version 7: each warp loads WMMA fragments directly from global memory
// - Version 8: the whole block first stages tiles into shared memory,
//   then each warp loads its own 16x16 fragment from shared memory
//
// Why this matters:
// - better locality than direct global fragment loads
// - less redundant global-memory traffic
// - closer to production Tensor Core kernels
// - practical stepping stone before cp.async / TMA / WGMMA

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

__global__ void sgemm_tensor_core_smem_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{

    __shared__ __half sA[BLOCK_TILE_M][WMMA_K];
    __shared__ __half sB[WMMA_K][BLOCK_TILE_N];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_m = warp_id / BLOCK_WARPS_N;
    const int warp_n = warp_id % BLOCK_WARPS_N;

    const int c_row = blockIdx.y * BLOCK_TILE_M + warp_m * WMMA_M;
    const int c_col = blockIdx.x * BLOCK_TILE_N + warp_n * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        for (int idx = tid; idx < BLOCK_TILE_M * WMMA_K; idx += THREADS_PER_BLOCK) {
            const int row = idx / WMMA_K;
            const int col = idx % WMMA_K;
            const int g_row = blockIdx.y * BLOCK_TILE_M + row;
            const int g_col = k0 + col;

            sA[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2half(0.0f);
        }

        for (int idx = tid; idx < WMMA_K * BLOCK_TILE_N; idx += THREADS_PER_BLOCK) {
            const int row = idx / BLOCK_TILE_N;
            const int col = idx % BLOCK_TILE_N;
            const int g_row = k0 + row;
            const int g_col = blockIdx.x * BLOCK_TILE_N + col;

            sB[row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
        }

        __syncthreads();

        if (c_row < M && c_col < N) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;

            const __half* tile_a = &sA[warp_m * WMMA_M][0];
            const __half* tile_b = &sB[0][warp_n * WMMA_N];

            nvcuda::wmma::load_matrix_sync(a_frag, tile_a, WMMA_K);
            nvcuda::wmma::load_matrix_sync(b_frag, tile_b, BLOCK_TILE_N);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    if (c_row < M && c_col < N) {
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = c_frag.x[i] * alpha;
        }

        if (beta != 0.0f) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_orig;
            nvcuda::wmma::load_matrix_sync(c_orig, C + c_row * N + c_col, N, nvcuda::wmma::mem_row_major);
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] = c_frag.x[i] + beta * c_orig.x[i];
            }
        }

        nvcuda::wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, nvcuda::wmma::mem_row_major);
    }

}

#include "/content/runner_half.h"

void run_08_tensor_core_smem_wmma(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
              (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    sgemm_tensor_core_smem_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This kernel requires Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_08_tensor_core_smem_wmma, M, N, K, "08_Tensor_Core_Smem_WMMA");
    return 0;
}

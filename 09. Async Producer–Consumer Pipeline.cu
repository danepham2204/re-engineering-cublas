%%writefile 09_Async_Pipeline.cu
// Kernel 9 - Async Producer/Consumer Double-Buffered Pipeline
//
// Progression from version 8:
// - Version 8: synchronous load-then-compute (Tensor Core stalls waiting for SMEM to fill)
// - Version 9: two SMEM buffers (Ping-Pong) — while Tensor Core consumes buffer A,
//              threads are loading the NEXT tile into buffer B simultaneously
//
// Why this matters:
// - Hides the latency of global memory loads behind Tensor Core computation
// - Software pipelining: Load[k+1] and Compute[k] overlap in the same cycle window
// - This is the fundamental pattern used in every production GEMM (CUTLASS, cuBLAS)
//
// Note: This uses __syncthreads() (software pipeline). True hardware async requires
//       cp.async (SM80+) / TMA (SM90+). On T4 (SM75) this is the best we can do.

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WM = 32;
constexpr int WN = 32;

constexpr int WARPS_M = BM / WM;            // 2
constexpr int WARPS_N = BN / WN;            // 2
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 128

constexpr int FRAGS_M = WM / WMMA_M;        // 2
constexpr int FRAGS_N = WN / WMMA_N;        // 2

constexpr int PAD = 8; // Avoid SMEM bank conflicts during load_matrix_sync

__device__ void load_stage_tiles(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half sA[2][BM][BK + PAD],
    __half sB[2][BK][BN + PAD],
    int M, int N, int K,
    int k0, int tid, int stage)
{
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
        const int row = idx / BK;
        const int col = idx % BK;
        const int g_row = block_row + row;
        const int g_col = k0 + col;
        sA[stage][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2half(0.0f);
    }

    for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
        const int row = idx / BN;
        const int col = idx % BN;
        const int g_row = k0 + row;
        const int g_col = block_col + col;
        sB[stage][row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
    }
}

__global__ void sgemm_tensor_core_pipeline(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Double-buffered SMEM — 2 stages for Ping-Pong
    __shared__ __half sA[2][BM][BK + PAD];
    __shared__ __half sB[2][BK][BN + PAD];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int warp_row = warp_m * WM;
    const int warp_col = warp_n * WN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_M][FRAGS_N];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // --- Prefetch first tile into buffer 0 ---
    load_stage_tiles(A, B, sA, sB, M, N, K, 0, tid, 0);
    __syncthreads();

    int read_stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next_k = k0 + BK;
        const int write_stage = read_stage ^ 1;

        // --- PRODUCER: Load next tile while current tile is being consumed ---
        if (next_k < K) {
            load_stage_tiles(A, B, sA, sB, M, N, K, next_k, tid, write_stage);
        }

        // --- CONSUMER: Compute on tiles in read_stage ---
        if (block_row + warp_row < M && block_col + warp_col < N) {
            #pragma unroll
            for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[FRAGS_N];

                #pragma unroll
                for (int i = 0; i < FRAGS_M; ++i) {
                    wmma::load_matrix_sync(a_frag[i], &sA[read_stage][warp_row + i * WMMA_M][k_step], BK + PAD);
                }

                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    wmma::load_matrix_sync(b_frag[j], &sB[read_stage][k_step][warp_col + j * WMMA_N], BN + PAD);
                }

                #pragma unroll
                for (int i = 0; i < FRAGS_M; ++i) {
                    #pragma unroll
                    for (int j = 0; j < FRAGS_N; ++j) {
                        wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }
        }

        __syncthreads();
        read_stage = write_stage;
    }

    // --- Store results to Global Memory ---
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_r = block_row + warp_row + i * WMMA_M;
            const int c_c = block_col + warp_col + j * WMMA_N;

            if (c_r < M && c_c < N) {
                #pragma unroll
                for (int t = 0; t < c_frag[i][j].num_elements; t++) {
                    c_frag[i][j].x[t] *= alpha;
                }

                if (beta != 0.0f) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_orig;
                    wmma::load_matrix_sync(c_orig, C + c_r * N + c_c, N, wmma::mem_row_major);
                    #pragma unroll
                    for (int t = 0; t < c_frag[i][j].num_elements; t++) {
                        c_frag[i][j].x[t] += beta * c_orig.x[t];
                    }
                }

                wmma::store_matrix_sync(C + c_r * N + c_c, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

#include "/content/runner_half.h"

void run_09_tensor_core_pipeline(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tensor_core_pipeline<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This kernel requires Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_09_tensor_core_pipeline, M, N, K, "09_Async_Pipeline_4096");
    return 0;
}

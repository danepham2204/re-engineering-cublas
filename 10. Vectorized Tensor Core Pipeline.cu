%%writefile 10_Vectorized_TC.cu
// Kernel 10 - Vectorized Tensor Core Pipeline
//
// Progression from version 9:
// - Version 9: scalar __half loads from global to shared memory, double-buffered pipeline
// - Version 10: int4 (128-bit) vectorized loads from global to shared, float4 vectorized epilogue
//
// Why this matters:
// In Kernels 07-09, Tensor Cores finish their 16x16x16 MMA in 1-2 cycles, then the
// entire warp stalls waiting for the next tile to arrive from global memory.
// Scalar __half loads issue 8x. more memory instructions than necessary.
// By using int4 loads (128 bits = 8 __half per instruction), we drastically reduce
// the number of memory transactions and better saturate the T4's 320 GB/s bandwidth.
//
// Additionally, the epilogue (shared memory -> global C) is vectorized using float4
// (128 bits = 4 floats per instruction), further reducing store instruction pressure.
// 
// Architecture requirement: SM75+ (Turing Tensor Cores)
// Compile with: nvcc -O3 -arch=sm_75 -lcublas 10_Vectorized_TC.cu -o 10_vtc

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

// Number of int4 (128-bit) vector loads needed per shared-memory tile
// Each int4 carries 8 __half values (8 * 16 bits = 128 bits)
constexpr int SA_VEC_COUNT = (BLOCK_TILE_M * WMMA_K) / 8;       // 32*16/8 = 64
constexpr int SB_VEC_COUNT = (WMMA_K * BLOCK_TILE_N) / 8;       // 16*64/8 = 128

// ─── Vectorized tile loader ───────────────────────────────────────────────────
// Replaces the scalar loader from Kernel 09.
// Each thread issues one int4 load (128 bits = 8 __half) instead of eight
// separate 16-bit loads. This reduces memory instruction count by 8x.
__device__ void load_stage_tiles_vectorized(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half sA[2][BLOCK_TILE_M][WMMA_K],
    __half sB[2][WMMA_K][BLOCK_TILE_N],
    int M, int N, int K,
    int k0, int tid, int stage)
{
    // ── Load A tile: 64 int4 loads (32 rows × 16 cols / 8) ──
    for (int idx = tid; idx < SA_VEC_COUNT; idx += THREADS_PER_BLOCK) {
        const int flat = idx * 8;             // flat index in __half space
        const int row  = flat / WMMA_K;
        const int col  = flat % WMMA_K;       // always 0 or 8 (since WMMA_K=16)
        const int g_row = blockIdx.y * BLOCK_TILE_M + row;
        const int g_col = k0 + col;

        int4 val;
        if (g_row < M && (g_col + 8) <= K) {
            val = reinterpret_cast<const int4*>(&A[g_row * K + g_col])[0];
        } else {
            val = make_int4(0, 0, 0, 0);
        }
        reinterpret_cast<int4*>(&sA[stage][row][col])[0] = val;
    }

    // ── Load B tile: 128 int4 loads (16 rows × 64 cols / 8) ──
    for (int idx = tid; idx < SB_VEC_COUNT; idx += THREADS_PER_BLOCK) {
        const int flat = idx * 8;
        const int row  = flat / BLOCK_TILE_N;
        const int col  = flat % BLOCK_TILE_N; // always a multiple of 8
        const int g_row = k0 + row;
        const int g_col = blockIdx.x * BLOCK_TILE_N + col;

        int4 val;
        if (g_row < K && (g_col + 8) <= N) {
            val = reinterpret_cast<const int4*>(&B[g_row * N + g_col])[0];
        } else {
            val = make_int4(0, 0, 0, 0);
        }
        reinterpret_cast<int4*>(&sB[stage][row][col])[0] = val;
    }
}

// ─── Main kernel ──────────────────────────────────────────────────────────────
__global__ void sgemm_tensor_core_vectorized_pipeline(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ __half sA[2][BLOCK_TILE_M][WMMA_K];
    __shared__ __half sB[2][WMMA_K][BLOCK_TILE_N];
    __shared__ float  sC[BLOCK_TILE_M][BLOCK_TILE_N];

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_m = warp_id / BLOCK_WARPS_N;
    const int warp_n = warp_id % BLOCK_WARPS_N;

    const int block_row = blockIdx.y * BLOCK_TILE_M;
    const int block_col = blockIdx.x * BLOCK_TILE_N;
    const int c_row     = block_row + warp_m * WMMA_M;
    const int c_col     = block_col + warp_n * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // ── Prefetch first tile using vectorized loads ──
    load_stage_tiles_vectorized(A, B, sA, sB, M, N, K, 0, tid, 0);
    __syncthreads();

    // ── Double-buffered main loop ──
    int read_stage = 0;
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        const int next_k      = k0 + WMMA_K;
        const int write_stage = read_stage ^ 1;

        // Producer: prefetch next tile while current tile is being consumed
        if (next_k < K) {
            load_stage_tiles_vectorized(A, B, sA, sB, M, N, K, next_k, tid, write_stage);
        }

        // Consumer: WMMA compute on current tile
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

    // ── Store accumulator to shared memory via WMMA ──
    if (c_row < M && c_col < N) {
        nvcuda::wmma::store_matrix_sync(
            &sC[warp_m * WMMA_M][warp_n * WMMA_N],
            c_frag, BLOCK_TILE_N, nvcuda::wmma::mem_row_major);
    }

    __syncthreads();

    // ── Vectorized epilogue: float4 stores from shared to global ──
    // sC is 32×64 floats = 2048 floats. float4 = 4 floats → 512 float4 stores.
    constexpr int SC_VEC_COUNT = (BLOCK_TILE_M * BLOCK_TILE_N) / 4;  // 512
    for (int idx = tid; idx < SC_VEC_COUNT; idx += THREADS_PER_BLOCK) {
        const int flat = idx * 4;
        const int row  = flat / BLOCK_TILE_N;
        const int col  = flat % BLOCK_TILE_N;
        const int g_row = block_row + row;
        const int g_col = block_col + col;

        if (g_row < M && (g_col + 4) <= N) {
            float4 sc_val = reinterpret_cast<float4*>(&sC[row][col])[0];

            if (beta != 0.0f) {
                float4 c_val = reinterpret_cast<float4*>(&C[g_row * N + g_col])[0];
                sc_val.x = alpha * sc_val.x + beta * c_val.x;
                sc_val.y = alpha * sc_val.y + beta * c_val.y;
                sc_val.z = alpha * sc_val.z + beta * c_val.z;
                sc_val.w = alpha * sc_val.w + beta * c_val.w;
            } else {
                sc_val.x *= alpha;
                sc_val.y *= alpha;
                sc_val.z *= alpha;
                sc_val.w *= alpha;
            }

            reinterpret_cast<float4*>(&C[g_row * N + g_col])[0] = sc_val;
        }
    }
}

#include "/content/runner_half.h"

void run_10_vectorized_tc(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
              (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    sgemm_tensor_core_vectorized_pipeline<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This kernel requires Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_10_vectorized_tc, M, N, K, "10_Vectorized_TC_Pipeline");
    return 0;
}

%%writefile 11_Ldmatrix_TC.cu
// Kernel 11 - ldmatrix + int4 Vectorized Tensor Core
//
// Progression from version 10:
// - Version 10: double-buffered SMEM (37KB), int4 global loads, wmma::load_matrix_sync
// - Version 11: single-buffer SMEM (19KB) ← key occupancy unlock
//               + ldmatrix PTX instruction for SMEM→Fragment (SM75+)
//               + int4 global→SMEM (from v10)
//
// Why this matters:
// v10's 37KB double-buffer SMEM allows only 1 Block per SM on T4 (64KB total).
// Switching to a single buffer: sA[128][40]*2 + sB[32][136]*2 ≈ 19 KB
// → 2 Blocks per SM (register-file-limited at ~116 registers/thread)
// → 2x more warps → better latency hiding → Warp Active from 25% → ~50%
//
// ldmatrix.sync.aligned.m8n8.x4.shared.b16 (SM75+):
// Replaces wmma::load_matrix_sync for SMEM→Register path.
// Each thread provides ONE SMEM address; hardware loads 8 consecutive __half
// and distributes them to all 32 threads in the correct Tensor Core register layout.
// Zero software overhead for thread→element mapping vs load_matrix_sync.
//
// Important: if Max Err > 0 after running, it means ldmatrix register mapping
// needs adjustment. Fallback to wmma::load_matrix_sync is always safe.

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WM = 32;
constexpr int WN = 64;

constexpr int WARPS_M = BM / WM;             // 4
constexpr int WARPS_N = BN / WN;             // 2
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 256

constexpr int FRAGS_M = WM / WMMA_M;         // 2
constexpr int FRAGS_N = WN / WMMA_N;         // 4

constexpr int PAD = 8; // Stride = BK+PAD = 40 halves = 80 bytes, 16-byte aligned

constexpr int SA_VEC = (BM * BK) / 8; // 512 int4 loads
constexpr int SB_VEC = (BK * BN) / 8; // 512 int4 loads

// ─────────────────────────────────────────────────────────────────────────────
// ldmatrix helper for matrix_a (row_major, 16x16x16)
// SM75: each warp lane provides the address of ITS row in SMEM.
// Lane t provides: smem[row_base + (t & 0xF)][col_base + (t >> 4) * 8]
// Hardware then loads 8 consecutive __half from each of the 32 addresses
// and distributes to all threads in EXACTLY the register layout needed for mma_sync.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void ldmatrix_a(
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>& frag,
    const __half* smem,    // pointer to [warp_row][k_step] in SMEM
    int stride)            // stride in __half  (= BK + PAD = 40)
{
    const int lane = threadIdx.x & 31;
    const int row = lane & 0xF;         // 0..15
    const int col = (lane >> 4) * 8;   // 0 or 8
    uint32_t ptr = __cvta_generic_to_shared(smem + row * stride + col);
    uint32_t* r = reinterpret_cast<uint32_t*>(frag.x);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(ptr));
}

// ─────────────────────────────────────────────────────────────────────────────
// ldmatrix helper for matrix_b (row_major K×N tile: sB[BK][BN+PAD])
// For B stored as [k_step][warp_col]: row = k-dimension, col = N-dimension
// Lane t provides: smem[(t & 0xF)][warp_col + (t >> 4) * 8]
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void ldmatrix_b(
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>& frag,
    const __half* smem,    // pointer to [k_step][warp_col_offset] in SMEM
    int stride)            // stride in __half  (= BN + PAD = 136)
{
    const int lane = threadIdx.x & 31;
    const int row = lane & 0xF;         // 0..15
    const int col = (lane >> 4) * 8;   // 0 or 8
    uint32_t ptr = __cvta_generic_to_shared(smem + row * stride + col);
    uint32_t* r = reinterpret_cast<uint32_t*>(frag.x);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(ptr));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sgemm_tc_ldmatrix(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // SINGLE buffer: sA ≈ 10KB, sB ≈ 8.7KB → total ~19KB
    // T4 has 64KB SMEM per SM → supports 3 blocks (register-limited to 2)
    __shared__ __half sA[BM][BK + PAD];
    __shared__ __half sB[BK][BN + PAD];

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

    for (int k0 = 0; k0 < K; k0 += BK) {

        // --- Phase 1: int4 Vectorized Global → Shared Memory ---
        for (int idx = tid; idx < SA_VEC; idx += THREADS_PER_BLOCK) {
            const int flat = idx * 8;
            const int row  = flat / BK;
            const int col  = flat % BK;
            const int g_row = block_row + row;
            const int g_col = k0 + col;
            int4 val;
            if (g_row < M && g_col + 7 < K) {
                val = reinterpret_cast<const int4*>(&A[g_row * K + g_col])[0];
            } else {
                __half tmp[8] = {};
                for (int t = 0; t < 8 && g_col + t < K && g_row < M; ++t)
                    tmp[t] = A[g_row * K + g_col + t];
                val = reinterpret_cast<int4*>(tmp)[0];
            }
            reinterpret_cast<int4*>(&sA[row][col])[0] = val;
        }

        for (int idx = tid; idx < SB_VEC; idx += THREADS_PER_BLOCK) {
            const int flat = idx * 8;
            const int row  = flat / BN;
            const int col  = flat % BN;
            const int g_row = k0 + row;
            const int g_col = block_col + col;
            int4 val;
            if (g_row < K && g_col + 7 < N) {
                val = reinterpret_cast<const int4*>(&B[g_row * N + g_col])[0];
            } else {
                __half tmp[8] = {};
                for (int t = 0; t < 8 && g_col + t < N && g_row < K; ++t)
                    tmp[t] = B[g_row * N + g_col + t];
                val = reinterpret_cast<int4*>(tmp)[0];
            }
            reinterpret_cast<int4*>(&sB[row][col])[0] = val;
        }

        __syncthreads();

        // --- Phase 2: ldmatrix SMEM→Fragment + MMA ---
        if (block_row + warp_row < M && block_col + warp_col < N) {
            #pragma unroll
            for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[FRAGS_N];

                // ldmatrix for A: 2 fragments of 16×16 (covering WM=32 rows)
                #pragma unroll
                for (int i = 0; i < FRAGS_M; ++i) {
                    ldmatrix_a(a_frag[i], &sA[warp_row + i * WMMA_M][k_step], BK + PAD);
                }

                // ldmatrix for B: 4 fragments covering WN=64 columns
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    ldmatrix_b(b_frag[j], &sB[k_step][warp_col + j * WMMA_N], BN + PAD);
                }

                // 8 MMA operations (FRAGS_M × FRAGS_N)
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
    }

    // --- Phase 3: Store results ---
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_r = block_row + warp_row + i * WMMA_M;
            const int c_c = block_col + warp_col + j * WMMA_N;
            if (c_r < M && c_c < N) {
                #pragma unroll
                for (int t = 0; t < c_frag[i][j].num_elements; t++)
                    c_frag[i][j].x[t] *= alpha;
                if (beta != 0.0f) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_orig;
                    wmma::load_matrix_sync(c_orig, C + c_r * N + c_c, N, wmma::mem_row_major);
                    #pragma unroll
                    for (int t = 0; t < c_frag[i][j].num_elements; t++)
                        c_frag[i][j].x[t] += beta * c_orig.x[t];
                }
                wmma::store_matrix_sync(C + c_r * N + c_c, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

#include "/content/runner_half.h"

void run_11_tc_ldmatrix(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tc_ldmatrix<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 4096, N = 4096, K = 4096;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This kernel requires Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_11_tc_ldmatrix, M, N, K, "11_Ldmatrix_TC_4096");
    return 0;
}

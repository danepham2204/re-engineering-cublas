%%writefile 06_Warp_Tiling.cu
// kiểu 5 là mỗi thread độc lập 1 block tile lớn. 
// Mỗi block tính tile 128 x 128
// Mỗi thread tính 8 x 8 = 64 output
// Block có 16 x 16 = 256 threads
// sang kiểu 6 là mỗi warp chịu trách nhiệm một sub-tile rõ ràng.
// mỗi block tính tile 64 x 64
// mỗi warp tính 32 x 16
// mỗi thread tính 8 x 2 = 16 output
// block vẫn là 256 threads, nhưng tổ chức thành 8 warps rất rõ ràng

// So với version 5, version 6 đang tối ưu chủ yếu các điểm sau:
// giảm register pressure rất mạnh
// tổ chức compute theo đúng đơn vị warp của GPU
// cải thiện khả năng tăng occupancy
// làm dataflow trong shared memory rõ theo warp
// tạo nền tảng để thêm các tối ưu cao cấp hơn
// Nhưng đánh đổi là:
// mỗi thread làm ít việc hơn
// block tile nhỏ hơn
// bản hiện tại chưa còn lợi thế float4 của version 5

// Result:
// Kernel 6: Warp Tiling
// Block tile: 64x64x8
// Warp tile : 32x16
// Thread tile: 8x2
// Matrix: 512x512x512
// Avg time: 0.259 ms
// Performance: 1035.91 GFLOP/s
// Computing CPU reference...
// Correct: YES (max abs error = 0.00)

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

// Kernel 6 - Warp Tiling
// One block computes a 64x64 tile of C.
// The block is split into 8 warps, and each warp computes one 32x16 tile.
// Inside a warp, each thread accumulates an 8x2 micro-tile in registers.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;

constexpr int WM = 32;
constexpr int WN = 16;

constexpr int WARPS_M = BM / WM;   // 2
constexpr int WARPS_N = BN / WN;   // 4
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * WARP_SIZE; // 256

constexpr int TM = 8;
constexpr int TN = 2;

__global__ void sgemm_warp_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const int lane_row_group = lane / 8; // 0..3
    const int lane_col_group = lane % 8; // 0..7

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int warp_row = warp_m * WM;
    const int warp_col = warp_n * WN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
            const int row = idx / BK;
            const int col = idx % BK;
            const int g_row = block_row + row;
            const int g_col = k0 + col;
            sA[row][col] = (g_row < M && g_col < N) ? A[g_row * N + g_col] : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
            const int row = idx / BN;
            const int col = idx % BN;
            const int g_row = k0 + row;
            const int g_col = block_col + col;
            sB[row][col] = (g_row < N && g_col < K) ? B[g_row * K + g_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = warp_row + lane_row_group * TM + i;
                regA[i] = sA[row][kk];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = warp_col + lane_col_group * TN + j;
                regB[j] = sB[kk][col];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = block_row + warp_row + lane_row_group * TM + i;
        if (row >= M) {
            continue;
        }

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = block_col + warp_col + lane_col_group * TN + j;
            if (col < K) {
                C[row * K + col] = alpha * acc[i][j] + beta * C[row * K + col];
            }
        }
    }
}

#include "/content/runner.h"

void run_06_warp_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_warp_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    run_benchmark(run_06_warp_tiled, M, N, K, "06_Warp_Tiling");
    return 0;
}

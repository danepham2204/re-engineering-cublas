%%writefile 05_Vectorized_Register_Tiling.cu
// run with nvcc4jupyter extension
#include "/content/runner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_vectorized_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float threadResults[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k = 0; k < K; k += BK) {
        int row_a = tid / 2;
        int col_a = (tid % 2) * 4;
        float4 val_a = reinterpret_cast<float4*>(&A[(cRow + row_a) * K + (k + col_a)])[0];
        sA[row_a][col_a + 0] = val_a.x;
        sA[row_a][col_a + 1] = val_a.y;
        sA[row_a][col_a + 2] = val_a.z;
        sA[row_a][col_a + 3] = val_a.w;

        int row_b = tid / 32;
        int col_b = (tid % 32) * 4;
        float4 val_b = reinterpret_cast<float4*>(&B[(k + row_b) * N + (cCol + col_b)])[0];
        sB[row_b][col_b + 0] = val_b.x;
        sB[row_b][col_b + 1] = val_b.y;
        sB[row_b][col_b + 2] = val_b.z;
        sB[row_b][col_b + 3] = val_b.w;

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i)
                regA[i] = sA[threadIdx.y * TM + i][dotIdx];
            for (int i = 0; i < TN; ++i)
                regB[i] = sB[dotIdx][threadIdx.x * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm][rn] += regA[rm] * regB[rn];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            float4 res;
            res.x = threadResults[i][j + 0];
            res.y = threadResults[i][j + 1];
            res.z = threadResults[i][j + 2];
            res.w = threadResults[i][j + 3];
            int out_r = cRow + threadIdx.y * TM + i;
            int out_c = cCol + threadIdx.x * TN + j;
            reinterpret_cast<float4*>(&C[out_r * N + out_c])[0] = res;
        }
    }
}

#include "/content/runner.h"

void run_05_vectorized(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(N / BN, M / BM);
    sgemm_vectorized_kernel<<<gridDim, blockDim>>>((float*)d_A, (float*)d_B, d_C, M, N, K);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    run_benchmark(run_05_vectorized, M, N, K, "05_Vectorized_Register_Tiling");
    return 0;
}

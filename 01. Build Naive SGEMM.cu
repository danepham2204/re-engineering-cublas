%%cuda
#include <iostream>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include "runner.h"

// achieve 4.2 trillion individual math problems solved every single second.
// GEMM is fundamental operation in linear algebra and ML, computing the matrix product $C = \alpha.A.B + \beta.C
// where A= MxN, B = NxK, C is a MxK matrix.

constexpr uint BLOCK_SIZE = 32;

// tell the compiler A,B,C not overlapped with restrict
__global__ void sgemm_coalesced(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
  uint row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  uint column = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (row >= M || column >= K) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * K + column];
    }

    C[row * K + column] = alpha * sum + beta * C[row * K + column];
}

void run_01_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // float alpha = 1.0f, beta = 0.0f; -> handled implicitly or directly in kernel if needed 
    // In our runner, we initialize C to 0, so beta=0 is standard.
    // The kernel itself uses alpha/beta, but let's hardcode 1.0, 0.0 for benchmarking like other kernels
    sgemm_coalesced<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 2048, N = 2048, K = 2048;
    run_benchmark(run_01_naive, M, N, K, "01_Naive_SGEMM");
    return 0;
}
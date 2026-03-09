%%cuda
#include <iostream>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#include <random>

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

// Can we format code auto in cu ( can used AI so no need)
void launch_sgemm(int M, int N, int K, float alpha, float beta) {
  // Allocate device memory ( GPU)
  float *d_A, *d_B, *d_C;

  // assign memory to device 
  cudaMalloc(&d_A, M*N*sizeof(float));
  cudaMalloc(&d_B, N * K * sizeof(float));
  cudaMalloc(&d_C, M * K * sizeof(float));

  // Init host matrices 
  std::vector<float> h_A(M * N);   // A is M rows × N columns
  std::vector<float> h_B(N * K);   // B is N rows × K columns
  std::vector<float> h_C(M * K);   // C is M rows × K columns (result)

  // rnadom seed
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto&val: h_A) val = dist(gen);
  for (auto&val: h_B) val = dist(gen);
  for (auto&val: h_C) val = dist(gen);

  // Copy from host to device
  cudaMemcpy(d_A, h_A.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

  // Define block and grid
  // fixed size of block
  // block_size= 32
  // block = 32 x 32 = 1024,1,1
  // grid = ceil(K/block), ceil(M/block)
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 2D grid

    // sgemm calculation
    sgemm_coalesced<<<gridDim, blockDim>>> (d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();  // Wait for completion

    // Copy back and verify (optional)
    cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    // ... Add verification logic against CPU GEMM ...

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  
    std::cout << "\nFirst few elements of result C:\n";
    for (int i = 0; i < std::min(10, M*K); ++i) {
        std::cout << h_C[i] << " ";
        if ((i+1) % 5 == 0) std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    int M = 1024, N = 1024, K = 1024;  // Example dimensions
    float alpha = 1.0f, beta = 0.0f;
    launch_sgemm(M, N, K, alpha, beta);
    std::cout << "GEMM completed!" << std::endl;

    
    return 0;
}
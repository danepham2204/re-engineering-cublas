// each block represent TILE (32.32) or (16x16) - tilling and calculate using its SRAM instead of accessing global memory 
// reuse TILE times each element
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>   // std::max
#include <iomanip>     // std::setw

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

template <const int TILE_SIZE>
__global__ void sgemm_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t)
    {
        // load tile A
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col <N) {
          sA[ty][tx] = A[row * N + a_col]
        } else {
          sA[ty][tx] = 0
        }

        // Load tile B (transposed layout)
        int b_row = t * TILE_SIZE + ty;
        if (row < M && a_col <N) {
          sB[tx][ty] = B[b_row * K +col]
        } else {
            sB[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Inner accumulation
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = alpha * acc + beta * C[row * K + col];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference (slow but correct)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> cpu_gemm(
    const std::vector<float>& h_A,
    const std::vector<float>& h_B,
    int M, int N, int K,
    float alpha, float beta)
{
    std::vector<float> h_C(M * K, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * K + j];
            }
            h_C[i * K + j] = alpha * sum + beta * h_C[i * K + j];
        }
    }
    return h_C;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // ──── Configuration ─────────────────────────────────────────────────────
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    constexpr uint TILE_SIZE = 32;

    // ──── Host data ─────────────────────────────────────────────────────────
    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C_gpu(M * K);

    // Fixed seed → reproducible
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::generate(h_A.begin(), h_A.end(), [&]() { return dist(gen); });
    std::generate(h_B.begin(), h_B.end(), [&]() { return dist(gen); });
    std::fill(h_C_gpu.begin(), h_C_gpu.end(), 0.0f);  // start with zeros

    // ──── Device memory ─────────────────────────────────────────────────────
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C_gpu.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    // ──── Timing events ─────────────────────────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ──── Choose which kernel to run ────────────────────────────────────────
    // Comment / uncomment the one you want to test

    std::cout << "\nRunning shared-memory tiling kernel...\n";

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (K + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_shared<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e-3) / 1e9;

    std::cout << "Kernel time: " << ms << " ms\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(2)
              << gflops << " GFLOP/s\n";

    // ──── Get result back ───────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // ──── CPU reference for verification ────────────────────────────────────
    std::cout << "Computing CPU reference (slow for large matrices)...\n";
    auto h_C_ref = cpu_gemm(h_A, h_B, M, N, K, alpha, beta);

    // ──── Compare ───────────────────────────────────────────────────────────
    float max_diff = 0.0f;
    bool correct = true;

    for (size_t i = 0; i < h_C_gpu.size(); ++i) {
        float diff = std::abs(h_C_gpu[i] - h_C_ref[i]);
        if (diff > 1e-4f) {
            correct = false;
            max_diff = std::max(max_diff, diff);
        }
    }

    std::cout << "\nGPU vs CPU correct: " << (correct ? "YES" : "NO")
              << "   (max abs diff = " << max_diff << ")\n";

    // ──── Show a few elements ───────────────────────────────────────────────
    std::cout << "\nFirst 5 elements of GPU C: ";
    for (int i = 0; i < 5 && i < M*K; ++i) {
        std::cout << h_C_gpu[i] << " ";
    }
    std::cout << "\n";

    // ──── Cleanup ───────────────────────────────────────────────────────────
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\nDone.\n";
    return 0;
}
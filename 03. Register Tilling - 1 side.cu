%%cuda

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 – 1D Register Tiling (Corrected & Optimized)
// Each thread computes THREAD_TILE consecutive elements of C (along columns)
// ─────────────────────────────────────────────────────────────────────────────
template <const int TILE_SIZE, const int THREAD_TILE>
__global__ void sgemm_register_1d(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // sA is square tile
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    
    // sB needs to be wide enough for THREAD_TILE columns per thread
    __shared__ float sB[TILE_SIZE][TILE_SIZE * THREAD_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row (same as Kernel 2)
    int row = blockIdx.y * TILE_SIZE + ty;

    // Each thread owns THREAD_TILE columns
    int col_base = blockIdx.x * (TILE_SIZE * THREAD_TILE) + tx * THREAD_TILE;

    // Register tile – private accumulators
    float acc[THREAD_TILE] = {0.0f};

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t)
    {
        // 1. Load tile A (standard row-major)
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
            sA[ty][tx] = A[row * N + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // 2. Load tile B – each thread loads THREAD_TILE elements
        #pragma unroll
        for (int m = 0; m < THREAD_TILE; ++m)
        {
            int b_row = t * TILE_SIZE + ty;
            int b_col = col_base + m;

            if (b_row < N && b_col < K) {
                sB[ty][tx + m * TILE_SIZE] = B[b_row * K + b_col];
            } else {
                sB[ty][tx + m * TILE_SIZE] = 0.0f;
            }
        }

        __syncthreads();

        // 3. Inner loop – broadcast A, multiply with wide B tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            float a_val = sA[ty][k];

            #pragma unroll
            for (int m = 0; m < THREAD_TILE; ++m)
            {
                acc[m] += a_val * sB[k][tx + m * TILE_SIZE];
            }
        }

        __syncthreads();
    }

    // 4. Write back
    #pragma unroll
    for (int m = 0; m < THREAD_TILE; ++m)
    {
        int final_col = col_base + m;
        if (row < M && final_col < K) {
            C[row * K + final_col] = alpha * acc[m] + beta * C[row * K + final_col];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference GEMM
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> cpu_gemm(
    const std::vector<float>& h_A,
    const std::vector<float>& h_B,
    int M, int N, int K,
    float alpha, float beta)
{
    std::vector<float> C(M * K, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < N; ++kk) {
                sum += h_A[i * N + kk] * h_B[kk * K + j];
            }
            C[i * K + j] = alpha * sum + beta * C[i * K + j];
        }
    }
    return C;
}

int main()
{
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    constexpr int TILE_SIZE   = 32;
    constexpr int THREAD_TILE = 4;   // ← 1D register tile size

    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::generate(h_A.begin(), h_A.end(), [&]() { return dist(gen); });
    std::generate(h_B.begin(), h_B.end(), [&]() { return dist(gen); });

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Kernel 3: 1D Register Tiling (TILE=" << TILE_SIZE << ", THREAD_TILE=" << THREAD_TILE << ")\n";
    std::cout << "Matrix: " << M << "×" << N << "×" << K << "\n";

    // Grid adjusted for wider output per block
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (K + TILE_SIZE * THREAD_TILE - 1) / (TILE_SIZE * THREAD_TILE),
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_register_1d<TILE_SIZE, THREAD_TILE><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = flops / (ms / 1000.0) / 1e9;

    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(2)
              << gflops << " GFLOP/s\n";

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Computing CPU reference...\n";
    auto h_C_ref = cpu_gemm(h_A, h_B, M, N, K, alpha, beta);

    float max_err = 0.0f;
    bool ok = true;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float err = std::abs(h_C[i] - h_C_ref[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-4f) ok = false;
    }

    std::cout << "\nCorrect: " << (ok ? "YES" : "NO")
              << "   (max abs error = " << max_err << ")" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    std::cout << "\nFinished.\n";
    return 0;
}
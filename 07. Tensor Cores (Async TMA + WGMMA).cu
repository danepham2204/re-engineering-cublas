%%cuda
// run with nvcc4jupyter extension
// Kernel 7 - Tensor Cores
//
// This file implements a practical Tensor Core GEMM using the CUDA WMMA API.
// It is the portable stepping stone before true Hopper-specific features such as:
// - TMA  (Tensor Memory Accelerator)
// - WGMMA (Warp Group MMA)
//
// Real Async TMA + WGMMA requires SM90/Hopper and either inline PTX or a library
// stack such as CUTLASS/CuTe. So this version focuses on the core idea first:
// move the GEMM inner loop from scalar FMA on CUDA cores to Tensor Core MMA ops.

// // Result
// Kernel 7: Tensor Cores (WMMA baseline)
// GPU: Tesla T4
// Compute capability: 7.5
// WMMA tile : 16x16x16
// Block tile: 32x32
// Warps/block: 4
// Matrix: 512x512x512
// Avg time: 0.003 ms
// Performance: 94765.11 GFLOP/s
// Computing CPU reference...
// Correct: NO (max abs error = 8.61)

// Why this is called Tensor Cores:
// - Version 6 used scalar FMA in software loops.
// - Version 7 uses wmma::mma_sync, which maps the inner product onto Tensor Core hardware.
// - Inputs are half precision, accumulation stays in float.
// - This is the baseline before Hopper-only Async TMA + WGMMA.

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while (0)

// WMMA tile shape. One warp computes one 16x16 output tile.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 4 warps per block -> block tile = 32x32
constexpr int BLOCK_WARPS_M = 2;
constexpr int BLOCK_WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = BLOCK_WARPS_M * BLOCK_WARPS_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int BLOCK_TILE_M = BLOCK_WARPS_M * WMMA_M; // 32
constexpr int BLOCK_TILE_N = BLOCK_WARPS_N * WMMA_N; // 32

__global__ void sgemm_tensor_core_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const int warp_id = threadIdx.x / 32;
    const int warp_m = warp_id / BLOCK_WARPS_N;
    const int warp_n = warp_id % BLOCK_WARPS_N;

    const int c_row = blockIdx.y * BLOCK_TILE_M + warp_m * WMMA_M;
    const int c_col = blockIdx.x * BLOCK_TILE_N + warp_n * WMMA_N;

    if (c_row >= M || c_col >= K) {
        return;
    }

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < N; k0 += WMMA_K) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;

        const __half* tile_a = A + c_row * N + k0;
        const __half* tile_b = B + k0 * K + c_col;

        nvcuda::wmma::load_matrix_sync(a_frag, tile_a, N);
        nvcuda::wmma::load_matrix_sync(b_frag, tile_b, K);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float tmp[WMMA_M * WMMA_N];
    nvcuda::wmma::store_matrix_sync(tmp, c_frag, WMMA_N, nvcuda::wmma::mem_row_major);

    #pragma unroll
    for (int i = 0; i < WMMA_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WMMA_N; ++j) {
            const int row = c_row + i;
            const int col = c_col + j;
            if (row < M && col < K) {
                C[row * K + col] = alpha * tmp[i * WMMA_N + j] + beta * C[row * K + col];
            }
        }
    }
#endif
}

std::vector<float> cpu_gemm_from_half(
    const std::vector<__half>& h_A,
    const std::vector<__half>& h_B,
    int M, int N, int K,
    float alpha, float beta)
{
    std::vector<float> h_C(M * K, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += __half2float(h_A[i * N + k]) * __half2float(h_B[k * K + j]);
            }
            h_C[i * K + j] = alpha * sum + beta * h_C[i * K + j];
        }
    }

    return h_C;
}

int main() {
    // WMMA requires dimensions aligned to 16 for the simple version below.
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    static_assert(M % WMMA_M == 0, "M must be multiple of 16");
    static_assert(N % WMMA_K == 0, "N must be multiple of 16");
    static_assert(K % WMMA_N == 0, "K must be multiple of 16");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Kernel 7: Tensor Cores (WMMA baseline)\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";

    if (prop.major < 7) {
        std::cerr << "This WMMA kernel needs Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    std::vector<float> h_A_float(M * N);
    std::vector<float> h_B_float(N * K);
    std::vector<__half> h_A(M * N);
    std::vector<__half> h_B(N * K);
    std::vector<float> h_C(M * K, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::generate(h_A_float.begin(), h_A_float.end(), [&]() { return dist(gen); });
    std::generate(h_B_float.begin(), h_B_float.end(), [&]() { return dist(gen); });

    for (int i = 0; i < M * N; ++i) {
        h_A[i] = __float2half(h_A_float[i]);
    }
    for (int i = 0; i < N * K; ++i) {
        h_B[i] = __float2half(h_B_float[i]);
    }

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((K + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
              (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    sgemm_tensor_core_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int RUNS = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; ++i) {
        sgemm_tensor_core_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= RUNS;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "WMMA tile : " << WMMA_M << "x" << WMMA_N << "x" << WMMA_K << "\n";
    std::cout << "Block tile: " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << "\n";
    std::cout << "Warps/block: " << WARPS_PER_BLOCK << "\n";
    std::cout << "Matrix: " << M << "x" << N << "x" << K << "\n";

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double gflops = flops / (ms / 1000.0) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Avg time: " << ms << " ms\n";
    std::cout << std::setprecision(2);
    std::cout << "Performance: " << gflops << " GFLOP/s\n";

    std::cout << "Computing CPU reference...\n";
    auto h_C_ref = cpu_gemm_from_half(h_A, h_B, M, N, K, alpha, beta);

    float max_err = 0.0f;
    bool ok = true;
    for (size_t i = 0; i < h_C.size(); ++i) {
        const float err = std::abs(h_C[i] - h_C_ref[i]);
        max_err = std::max(max_err, err);
        if (err > 2e-1f) {
            ok = false;
        }
    }

    std::cout << "Correct: " << (ok ? "YES" : "NO")
              << " (max abs error = " << max_err << ")\n";

    std::cout << "\nWhy this is called Tensor Cores:\n";
    std::cout << "- Version 6 used scalar FMA in software loops.\n";
    std::cout << "- Version 7 uses wmma::mma_sync, which maps the inner product onto Tensor Core hardware.\n";
    std::cout << "- Inputs are half precision, accumulation stays in float.\n";
    std::cout << "- This is the baseline before Hopper-only Async TMA + WGMMA.\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}

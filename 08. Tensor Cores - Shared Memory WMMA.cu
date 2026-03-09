%%cuda
// Kernel 8 - Tensor Cores with Shared Memory WMMA
//
// Progression from version 7:
// - Version 7: each warp loads WMMA fragments directly from global memory
// - Version 8: the whole block first stages tiles into shared memory,
//   then each warp loads its own 16x16 fragment from shared memory
//
// Why this matters:
// - better locality than direct global fragment loads
// - less redundant global-memory traffic
// - closer to production Tensor Core kernels
// - practical stepping stone before cp.async / TMA / WGMMA

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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_WARPS_M = 2;
constexpr int BLOCK_WARPS_N = 4;
constexpr int WARPS_PER_BLOCK = BLOCK_WARPS_M * BLOCK_WARPS_N;  // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;         // 256

constexpr int BLOCK_TILE_M = BLOCK_WARPS_M * WMMA_M;            // 32
constexpr int BLOCK_TILE_N = BLOCK_WARPS_N * WMMA_N;            // 64

__global__ void sgemm_tensor_core_smem_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __shared__ __half sA[BLOCK_TILE_M][WMMA_K];
    __shared__ __half sB[WMMA_K][BLOCK_TILE_N];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_m = warp_id / BLOCK_WARPS_N;
    const int warp_n = warp_id % BLOCK_WARPS_N;

    const int c_row = blockIdx.y * BLOCK_TILE_M + warp_m * WMMA_M;
    const int c_col = blockIdx.x * BLOCK_TILE_N + warp_n * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < N; k0 += WMMA_K) {
        for (int idx = tid; idx < BLOCK_TILE_M * WMMA_K; idx += THREADS_PER_BLOCK) {
            const int row = idx / WMMA_K;
            const int col = idx % WMMA_K;
            const int g_row = blockIdx.y * BLOCK_TILE_M + row;
            const int g_col = k0 + col;

            sA[row][col] = (g_row < M && g_col < N) ? A[g_row * N + g_col] : __float2half(0.0f);
        }

        for (int idx = tid; idx < WMMA_K * BLOCK_TILE_N; idx += THREADS_PER_BLOCK) {
            const int row = idx / BLOCK_TILE_N;
            const int col = idx % BLOCK_TILE_N;
            const int g_row = k0 + row;
            const int g_col = blockIdx.x * BLOCK_TILE_N + col;

            sB[row][col] = (g_row < N && g_col < K) ? B[g_row * K + g_col] : __float2half(0.0f);
        }

        __syncthreads();

        if (c_row < M && c_col < K) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag;

            const __half* tile_a = &sA[warp_m * WMMA_M][0];
            const __half* tile_b = &sB[0][warp_n * WMMA_N];

            nvcuda::wmma::load_matrix_sync(a_frag, tile_a, WMMA_K);
            nvcuda::wmma::load_matrix_sync(b_frag, tile_b, BLOCK_TILE_N);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    if (c_row < M && c_col < K) {
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

    std::cout << "Kernel 8: Tensor Cores - Shared Memory WMMA\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";

    if (prop.major < 7) {
        std::cerr << "This kernel requires Tensor Core capable hardware (SM70+).\n";
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

    sgemm_tensor_core_smem_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_C, 0, M * K * sizeof(float)));

    constexpr int RUNS = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; ++i) {
        sgemm_tensor_core_smem_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= RUNS;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double gflops = flops / (ms / 1000.0) / 1e9;

    std::cout << "WMMA tile : " << WMMA_M << "x" << WMMA_N << "x" << WMMA_K << "\n";
    std::cout << "Block tile: " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << "\n";
    std::cout << "Warps/block: " << WARPS_PER_BLOCK << "\n";
    std::cout << "Avg time: " << std::fixed << std::setprecision(3) << ms << " ms\n";
    std::cout << "Performance: " << std::setprecision(2) << gflops << " GFLOP/s\n";

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

    std::cout << "\nWhat changed vs version 7:\n";
    std::cout << "- Version 7 loaded each WMMA fragment directly from global memory.\n";
    std::cout << "- Version 8 stages A/B tiles into shared memory first.\n";
    std::cout << "- This improves locality and reuse across warps inside the block.\n";
    std::cout << "- This is much closer to how optimized Tensor Core GEMM pipelines are structured.\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}

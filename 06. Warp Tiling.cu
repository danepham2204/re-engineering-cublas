%%cuda
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while (0)

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

int main() {
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::generate(h_A.begin(), h_A.end(), [&]() { return dist(gen); });
    std::generate(h_B.begin(), h_B.end(), [&]() { return dist(gen); });

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    sgemm_warp_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int RUNS = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; ++i) {
        sgemm_warp_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= RUNS;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Kernel 6: Warp Tiling\n";
    std::cout << "Block tile: " << BM << "x" << BN << "x" << BK << "\n";
    std::cout << "Warp tile : " << WM << "x" << WN << "\n";
    std::cout << "Thread tile: " << TM << "x" << TN << "\n";
    std::cout << "Matrix: " << M << "x" << N << "x" << K << "\n";

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double gflops = flops / (ms / 1000.0) / 1e9;
    std::cout << "Avg time: " << std::fixed << std::setprecision(3) << ms << " ms\n";
    std::cout << "Performance: " << std::setprecision(2) << gflops << " GFLOP/s\n";

    std::cout << "Computing CPU reference...\n";
    auto h_C_ref = cpu_gemm(h_A, h_B, M, N, K, alpha, beta);

    float max_err = 0.0f;
    bool ok = true;
    for (size_t i = 0; i < h_C.size(); ++i) {
        const float err = std::abs(h_C[i] - h_C_ref[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-4f) {
            ok = false;
        }
    }

    std::cout << "Correct: " << (ok ? "YES" : "NO")
              << " (max abs error = " << max_err << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}

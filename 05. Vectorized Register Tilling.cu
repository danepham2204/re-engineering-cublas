%%cuda
// run with nvcc4jupyter extension
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call) {                                                  \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1);                                                            \
    }                                                                       \
}

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

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const int sizeA = M * K, sizeB = K * N, sizeC = M * N;

    std::vector<float> h_A(sizeA), h_B(sizeB), h_C(sizeC, 0.0f);
    for (int i = 0; i < sizeA; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < sizeB; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(N / BN, M / BM);

    sgemm_vectorized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int RUNS = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; i++)
        sgemm_vectorized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= RUNS;

    double flops = 2.0 * M * N * K;
    double gflops = (flops / (ms / 1000.0)) / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Matrix size : " << M << "x" << N << "x" << K << "\n";
    std::cout << "Avg time    : " << ms << " ms\n";
    std::cout << "Performance : " << gflops << " GFLOPS\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

inline void run_benchmark(
    void (*kernel_launcher)(const float*, const float*, float*, int, int, int),
    int M, int N, int K, const char* kernel_name) 
{
    float *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    std::mt19937 gen(42);
    // For large matrices, large values will cause catastrophic cancellation in float32. 
    // Small values (-1 to 1) are better for numerical stability when checking errors.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) h_A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dist(gen);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Reference using cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    // cublas assumes column major by default. To do C = A * B in row major, it's equivalent to doing C^T = B^T * A^T in column major.
    // So we pass B, A instead of A, B and swap M and N dimensions
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, d_B, N, d_A, K,
                             &beta, d_C_ref, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    kernel_launcher(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iters = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++) {
        kernel_launcher(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= num_iters;

    // Validation
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(h_C[i] - h_C_ref[i]);
        if (err > max_err) max_err = err;
    }

    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (ms * 1e-3);

    std::cout << std::left << std::setw(35) << kernel_name 
              << " | Size: " << M << "x" << N << "x" << K 
              << " | Time: " << std::fixed << std::setprecision(3) << std::setw(6) << ms << " ms"
              << " | Perf: " << std::setprecision(2) << std::setw(8) << gflops << " GFLOP/s"
              << " | Max Err: " << std::scientific << std::setprecision(2) << max_err << "\n";

    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
}

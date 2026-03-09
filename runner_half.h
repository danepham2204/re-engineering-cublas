#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

inline void run_benchmark_half(
    void (*kernel_launcher)(const __half*, const __half*, float*, int, int, int),
    int M, int N, int K, const char* kernel_name) 
{
    __half *d_A, *d_B;
    float *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

    std::vector<float> h_A_float(M * K);
    std::vector<float> h_B_float(K * N);
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) {
        h_A_float[i] = dist(gen);
        h_A[i] = __float2half(h_A_float[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_float[i] = dist(gen);
        h_B[i] = __float2half(h_B_float[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Reference using cuBLAS (cublasGemmEx for mixed precision: A,B half, C float, compute float)
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    // C^T = B^T * A^T
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K,
                              &beta,
                              d_C_ref, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
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

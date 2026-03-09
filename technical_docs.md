# Rebuilding cuBLAS: Technical Documentation

## 1. System Environment
* **Platform:** Google Colab
* **GPU Architecture:** _(To be determined, likely T4 (sm_75) or V100 (sm_70))_
* **CUDA Version:** _(To be determined)_
* **Benchmark Parameters:** M=N=K=2048 (All tests use standard single/half precision parameters against cuBLAS baseline).

## 2. Performance Tracking
The following table documents the raw GFLOP/s and execution times of the kernels benchmarking a matrix of $2048 \times 2048 \times 2048$.

| Kernel | Time (ms) | Performance (GFLOP/s) | Max Absolute Error | Status |
| :--- | :--- | :--- | :--- | :--- |
| **01. Naive SGEMM** | 36.896 | 465.63 | 1.83e-04 | ✅ Pass |
| **02. Shared Memory Tiling** | 20.195 | 850.68 | 1.83e-04 | ✅ Pass |
| **03. Register Tiling (1D)** | - | - | - | ⏳ Pending |
| **04. Register Tiling (2D)** | - | - | - | ⏳ Pending |
| **05. Vectorized Register Tiling**| - | - | - | ⏳ Pending |
| **06. Warp Tiling** | - | - | - | ⏳ Pending |
| **07. Tensor Cores (WMMA)** | - | - | - | ⏳ Pending |
| **08. Tensor Cores SMEM WMMA** | - | - | - | ⏳ Pending |
| **09. Async Pipeline WMMA** | - | - | - | ⏳ Pending |

## 3. Bugs and Issues Log
This section captures math validation failures compared to cuBLAS or compilation errors found during testing.

* **Kernel 01:** No issues. Math behaves as expected.
* **Kernel 02:** Initial implementation had a high `Max Error` (1.15e+02) due to incorrect bounds checking and indexing when loading Tile B into Shared Memory. The issue was fixed by separating the correctly-bounded transposed element coordinates vs. block local coordinates, leading to a passing math check and a performance bump to ~850 GFLOP/s.

## 4. Architectural Bottleneck Summary
_(This section will be populated once all 9 kernels are run to compare memory bandwidth and arithmetic intensity)._


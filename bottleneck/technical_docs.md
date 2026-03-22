# Rebuilding cuBLAS: Technical Documentation

## System Environment

- **Platform:** Google Colab
- **GPU Architecture:** T4 (sm_75)
- **CUDA Version:** cuda_12.8.r12.8/compiler.35583870_0
- **Benchmark Parameters:** M=N=K=2048 (All tests use standard single/half precision parameters against cuBLAS baseline).

| Spec                  | Tesla T4 (Ours) | A100 (Reference) |
| --------------------- | --------------- | ---------------- |
| FP32 Peak             | 8.1 TFLOP/s     | 19.5 TFLOP/s     |
| FP16 Tensor Core Peak | 65 TFLOP/s      | 312 TFLOP/s      |
| Memory Bandwidth      | 320 GB/s        | 2,039 GB/s       |
| cuBLAS SGEMM          | ~8 TFLOP/s      | ~23 TFLOP/s      |

I only use old version T4 -> What can the T4 (SM75) actually do?
Technique Min Architecture T4 Support? Impact
Vectorized int4 loads Any SM ✅ Yes! Load 8 \_\_half per instruction instead of 1. This alone could 3-4x your TC kernel throughput.
cp.async SM80 (Ampere) ❌ No Hardware-async Global→Shared copy. T4 is SM75, one generation too old.
TMA SM90 (Hopper) ❌ No Dedicated DMA engine. Two generations too new for T4.

## Bugs and Issues Log

This section captures math validation failures compared to cuBLAS or compilation errors found during testing.

- **Kernel 01:** No issues. Math behaves as expected.
- **Kernel 02:** Initial implementation had a high `Max Error` (1.15e+02) due to incorrect bounds checking and indexing when loading Tile B into Shared Memory. The issue was fixed by separating the correctly-bounded transposed element coordinates vs. block local coordinates, leading to a passing math check and a performance bump to ~850 GFLOP/s.
- **Kernel 07, 08, 09:** Initial compilation failed entirely due to missing the `-arch=sm_75` NVCC compiler flag, making the compiler default to SM52, which wiped out all `nvcuda::wmma` operations. After fixing compilation, the kernel reported high `Max Error` (7.76e+01) and incorrect results. This was diagnosed as a race condition caused by assigning a thread-local uninitialized tracking array `float tmp[WMMA_M * WMMA_N]` to `wmma::store_matrix_sync`. Since WMMA is warp-synchronous, all 32 threads simultaneously tried to overwrite the same thread-local pointer, producing garbage computations. This was solved by directly feeding global/shared memory pointers (`C` and `sC`) to the `store` fragment correctly.

## Architectural Bottleneck Summary

### The Tensor Core Memory Wall (Tesla T4)

The highest performing FP32 kernel (Vectorized Register Tiling) achieved **~3.3 TFLOP/s**, nearing the Tesla T4's theoretical FP32 peak of 8.1 TFLOP/s.

However, the Tensor Core (FP16) kernels peaked at **~2.7 TFLOP/s**, despite the T4 having a theoretical FP16 Tensor Core peak of **65 TFLOP/s**. This is an expected artifact of the memory wall.

In Kernels 07-09, thread blocks load standard `__half` elements individually from global memory into shared memory. The Tesla T4 is severely constrained by its 320 GB/s global memory bandwidth. To unleash the true 65 TFLOP/s potential of the hardware, the kernels must fully saturate the memory bus utilizing **vectorized 128-bit memory accesses** (`float4` or `int4` casted to multiple `__half` elements) to drastically reduce the number of memory transactions issued per warp. Furthermore, modern Hopper scaling relies natively on asynchronous TMA (Tensor Memory Accelerator) rather than the synchronous `wmma` pipeline demonstrated here.

# Rebuilding cuBLAS: From a Naive CUDA Kernel to a Tensor Core Pipeline

## The Problem

A correct GEMM kernel is easy to write. A fast one is not — and the gap between the two is not filled by a single clever trick. It is filled by understanding *why* the hardware keeps rejecting your current design.

This project starts with a kernel that works but runs at ~465 GFLOP/s on hardware with a 65 TFLOP/s FP16 Tensor Core peak. The distance between those two numbers is the actual subject of this repository. Each kernel in this series identifies the dominant reason the previous one was slow, introduces one targeted structural change to address it, and then asks: *what broke next?*

The sequence is not a tour of optimization techniques. It is the output of a specific diagnostic loop applied repeatedly: **profile → identify bottleneck → intervene → re-measure**. The decisions made at each stage — what to tile, what to vectorize, when to move data into shared memory, when to pipeline — are driven by what the hardware exposed, not by a predetermined recipe.

By the end of this series you will have seen memory coalescing, shared-memory tiling, register tiling, vectorized loads, warp-level data ownership, Tensor Core WMMA, shared-memory operand staging, and software pipelining with a producer-consumer structure. The endpoint is not the fastest possible kernel. It is a complete account of *why* each transformation was necessary and what it cost.

![NVIDIA H100 GPU](img/GPU-NVIDIA-H100-SXM5-with-note.png)


![NVIDIA H100 GPU](img/SXM5-design-detail.svg)


Key notes:
+ Kernels are written from the perspective of a single thread.
+ All threads in grid running or executing the same kernel function

## Performance Tracking

The following table documents the raw GFLOP/s and execution times of the kernels benchmarking a matrix of $2048 \times 2048 \times 2048$.

| Kernel                             | Time (ms) | Performance (GFLOP/s) | Max Absolute Error | Status  |
| :--------------------------------- | :-------- | :-------------------- | :----------------- | :------ |
| **01. Naive SGEMM**                | 36.896    | 465.63                | 1.83e-04           | ✅ Pass |
| **02. Shared Memory Tiling**       | 20.195    | 850.68                | 1.83e-04           | ✅ Pass |
| **03. Register Tiling (1D)**       | 16.164    | 1062.82               | 1.83e-04           | ✅ Pass |
| **04. Register Tiling (2D)**       | 12.473    | 1377.36               | 1.83e-04           | ✅ Pass |
| **05. Vectorized Register Tiling** | 5.199     | 3304.42               | 1.83e-04           | ✅ Pass |
| **06. Warp Tiling**                | 13.326    | 1289.19               | 1.83e-04           | ✅ Pass |
| **07. Tensor Cores (WMMA)**        | 7.780     | 2208.25               | 0.00e+00           | ✅ Pass |
| **08. Tensor Cores SMEM WMMA**     | 7.052     | 2436.32               | 0.00e+00           | ✅ Pass |
| **09. Async Pipeline WMMA**        | 6.340     | 2709.72               | 0.00e+00           | ✅ Pass |

## Development Backlog

### Compilation & Infrastructure — The NVCC Story

One thing that became clear while working on this project is that `nvcc` is not just a compiler — it is a **multi-stage compilation driver** that quietly orchestrates several very different tools. Understanding this unlocked why a missing flag could cause a kernel to silently produce completely wrong results. This stage of compiling through `nvcc` is very important to understand:

+ **Stage 1 — Split.** NVCC splits the `.cu` file into two worlds. Host code (`main()`, memory allocation, kernel launches) is handed off to your system C++ compiler (clang or gcc). Device code (`__global__`, `__device__` functions) is routed to NVIDIA's own compiler backend. This split is why `#if defined(__CUDA_ARCH__)` guards exist — `__CUDA_ARCH__` is only defined during the device compilation pass, so host-side compilation never tries to process WMMA fragment types it has no concept of.

+ **Stage 2 — PTX generation.** Device kernels compile to **PTX (Parallel Thread Execution)**, NVIDIA's architecture-neutral virtual assembly. This is where the WMMA API gets lowered:
```
nvcuda::wmma::mma_sync(...)   →   wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32
nvcuda::wmma::load_matrix_sync →  wmma.load.a.sync.aligned.m16n16k16.global.row
```

+ **Stage 3 — SASS compilation via `ptxas`.** PTX is translated to **SASS**, the real binary machine instructions the hardware executes. This is where `-arch=sm_75` becomes critical. Without it, `ptxas` defaults to sm_52 (Maxwell), which has no WMMA opcodes. The `#if __CUDA_ARCH__ >= 700` block compiles to nothing, and every benchmark call returns in `0.006 ms` reporting 2.7 PFLOP/s — a mathematically perfect empty kernel. This was the root cause of the ghost performance numbers seen in Kernels 07–09 before the flag was identified.

**Stage 4 — Fatbinary packaging.** The final executable contains both the PTX source (for JIT compilation on future unknown GPUs) and the compiled sm_75 cubin (native code for the T4). At runtime, the CUDA driver picks the right one based on the actual hardware.

This investigation reinforced something important: **compiler flags are not just optimization hints — they are architecture contracts.** Specifying the wrong (or no) architecture doesn't just produce slower code. It produces silently wrong code that passes compilation, runs without error, and reports plausible-looking (but fabricated) performance numbers.

### Performance backlog

After getting all nine kernels running and validated against cuBLAS, a question came up that I couldn't quite shake: **why did our Tensor Core kernels (07–09) top out at ~2.7 TFLOP/s, slower than the pure FP32 vectorized kernel (05) at 3.3 TFLOP/s?**

The T4 has a theoretical FP16 Tensor Core peak of **65 TFLOP/s**. We were using less than 5% of it. The math operations themselves are correct — Max Error is 0 — the hardware is just not getting fed fast enough.

### The Discovery: It Was Never About Compute

Tensor Cores finish a 16×16×16 matrix multiply in 1–2 clock cycles. Then the warp sits completely idle, waiting for the next tile to arrive through the 320 GB/s global memory bus. Meanwhile, Kernel 05 was using `float4` (128-bit) vectorized loads, issuing **8× fewer memory instructions** per tile and saturating the bus much more efficiently — which is why a pure FP32 kernel was beating our FP16 Tensor Core implementation on the same hardware.

This is the classic **memory wall** problem.

### What Was Attempted

**Kernel 10: Vectorized TC Pipeline**  
I replaced the scalar `__half` loads with `int4` (128-bit) vectorized loads, loading 8 `__half` values per instruction, and vectorized the epilogue writes with `float4`. The result was actually *slightly slower* (2466 vs 2709 GFLOP/s).

The reason: our tiles (`BLOCK_TILE_M=32, BLOCK_TILE_N=64`) are too small. With 256 threads but only 64 int4 loads needed for the A tile, **75% of threads were sitting idle** during each load phase. We reduced instruction count by 8× but also killed warp-level memory parallelism. The GPU memory controller needs many simultaneous outstanding requests to saturate bandwidth — not fewer, serialized ones.

*Vectorization and larger tiles must go together.*

### What's Next

**The real unlock on Turing (T4, SM75) is combining vectorized int4 loads with significantly larger block tiles.** A `128×128` output tile with 16 warps per block would generate enough simultaneous memory transactions to justify the int4 packing and approach true memory bandwidth saturation.

Beyond that, the story continues with newer hardware:

- **SM80 (Ampere / A100):** `cp.async` enables hardware-async Global→Shared transfers, decoupling memory loads from the warp scheduler entirely. The warp doesn't stall; it issues a load and moves on to compute. This is the architecture that finally breaks the memory wall for Tensor Core kernels.

- **SM90 (Hopper / H100):** The **Tensor Memory Accelerator (TMA)** is a dedicated, physical DMA engine that bulk-copies tiles from global to shared memory autonomously. Combined with `wgmma` (Warp Group MMA, which has 4 warps cooperate on a single MMA), this is how H100 reaches 80+ TFLOP/s of sustained throughput — not just theoretical peak.

The next kernels in this series would be:
- **Kernel 11:** 128×128 tiles with `int4` vectorized loads (T4 compatible, closing the memory wall)
- **Kernel 12:** `cp.async` pipeline (Ampere+), true hardware-async prefetch
- **Kernel 13:** TMA + WGMMA (Hopper only) — the foundation of modern FlashAttention and cuBLAS internals

## 1. Problem Statement

The core computation studied in this repository is GEMM:

`C = alpha * A * B + beta * C`

Following standard conventions, the matrices are defined as:

- `A ∈ ℝ^(M × K)`
- `B ∈ ℝ^(K × N)`
- `C ∈ ℝ^(M × N)`

The central problem addressed is that a naive GPU implementation of GEMM fails to exploit the memory and execution hierarchies of NVIDIA GPUs. Despite GEMM's extremely high theoretical parallelism, a simple kernel typically suffers from excessive global-memory traffic, poor data reuse, low arithmetic intensity, high load/store instruction overhead, register pressure, insufficient overlap between memory and compute, and underutilization of Tensor Cores. This repository investigates how these inefficiencies can be systematically eliminated through kernel restructuring.

## 2. Research Objective

This project aims to answer the following question:

> **How does a CUDA GEMM kernel need to be transformed, stage by stage, to progress from a correctness-oriented baseline to a structured, high-performance GPU matrix multiplication?**

Specifically, the project aims to:

1. Identify the dominant bottleneck at each optimization stage.
2. Introduce one isolated structural optimization to mitigate that bottleneck.
3. Explain the resulting dataflow and execution model.
4. Analyze the trade-offs introduced by each architectural change.
5. Construct a logical progression from scalar CUDA core execution to asynchronous Tensor Core pipelines.

## 3. Why GEMM Matters

GEMM is a foundational kernel in scientific computing, numerical linear algebra, and deep learning. Many complex higher-level operations ultimately reduce to matrix multiplication. Consequently, GEMM is not merely a benchmark, but a central primitive dictating end-to-end performance in compute-bound workloads.

For GPU performance engineering, GEMM is an ideal case study because it exposes the full interaction between thread hierarchy, warp scheduling, global and shared memory bandwidth, register reuse, instruction issue throughput, and specialized hardware units like Tensor Cores.

## 4. Conceptual Dataflow

At a high level, optimized GEMM kernels progressively transform the dataflow from a direct global-memory computation model into a staged, hierarchical pipeline.

```mermaid
flowchart LR
    A[Global Memory A/B] --> B[Shared Memory Tiles]
    B --> C[Register Fragments]
    C --> D[Accumulator Registers]
    D --> E[Shared Memory Epilogue Optionally]
    E --> F[Global Memory C]
```

The optimization sequence in this repository systematically improves each data transition:

- Global memory to shared memory
- Shared memory to registers
- Registers to accumulators
- Accumulators to the final output validation

## 5. Execution Hierarchy

A core principle underlying this project is that an optimized GEMM must align perfectly with the GPU's execution hierarchy.

```mermaid
flowchart TD
    A[Grid] --> B[Block Tile]
    B --> C[Warp Tile]
    C --> D[Thread Tile]
    D --> E[Tensor Core Fragment]
```

Early kernels operate primarily at the block and thread levels. Subsequent kernels introduce warp-aware tiling and Tensor Core mapping, progressively tuning the work granularity to match the actual hardware scheduling and execution models.

## 6. Optimization Roadmap

| Version | File                                       | Core Optimization              | Main Bottleneck Targeted                           |
| :------ | :----------------------------------------- | :----------------------------- | :------------------------------------------------- |
| **01**  | `01. Build Naive SGEMM.cu`                 | Baseline CUDA SGEMM            | Establish correctness and baseline mapping         |
| **02**  | `02. Shared Memory Tiling.cu`              | Shared-memory tiling           | Repeated global-memory accesses                    |
| **03**  | `03. Register Tiling - 1 side.cu`          | 1D register tiling             | Low arithmetic intensity                           |
| **04**  | `04. Register Tiling - 2 side.cu`          | 2D register tiling             | Excessive shared-memory traffic per output         |
| **05**  | `05. Vectorized Register Tiling.cu`        | Vectorized loads/stores        | Load/store instruction pressure                    |
| **06**  | `06. Warp Tiling.cu`                       | Warp tiling                    | Scheduler alignment, locality, register pressure   |
| **07**  | `07. Tensor Cores (Async TMA + WGMMA).cu`  | WMMA Tensor Core baseline      | Transition compute from scalar FMA to Tensor Cores |
| **08**  | `08. Tensor Cores - Shared Memory WMMA.cu` | Shared-memory staged WMMA      | Tensor Core operand reuse and feed efficiency      |
| **09**  | `09. Async Producer–Consumer Pipeline.cu`  | Software pipelining & epilogue | Pipeline bubbles and load/compute serialization    |

## 7. Deep Explanation of Each Stage

### 7.1 Version 1: Naive SGEMM

The baseline kernel maps output elements directly to threads. Each thread computes one output by traversing the entire reduction dimension.

- **Demonstrates:** The basic GEMM structure and standard grid-to-matrix mapping.
- **Bottlenecks:** Every thread redundantly fetches from global memory, resulting in negligible arithmetic intensity and massive memory bandwidth bottlenecks.

### 7.2 Version 2: Shared-Memory Tiling

Introduces block-cooperative loading of `A` and `B` tiles into shared memory.

- **Improvements:** A single tile loaded from global memory is reused across the block, drastically reducing global bandwidth requirements.
- **New Bottleneck:** Shared-memory traffic scales up; each output still demands repeated shared-memory reads.

### 7.3 Versions 3 & 4: Register Tiling

Shifts data reuse deeper into the hierarchy by assigning multiple output elements to each thread (1D, then 2D).

- **Improvements:** Operands loaded into registers are reused across multiple FMA instructions, increasing arithmetic intensity and reducing shared-memory reads.
- **Trade-offs:** Higher output per thread increases register pressure, potentially limiting occupancy if untuned.

### 7.4 Version 5: Vectorized Register Tiling

Targets instruction-level inefficiencies in the memory path.

- **Improvements:** Vector instructions (e.g., `float4`) fetch multiple elements per issued instruction. This lowers the instruction pressure in the load/store units and improves frontend efficiency independently of memory transaction coalescing.

### 7.5 Version 6: Warp Tiling

Introduces warp-level ownership of output sub-tiles, bridging the gap between block-wide sharing and thread-local computation.

- **Improvements:** Matches the hardware’s true execution unit (the warp), ensuring exceptional scheduler alignment, spatial locality, and a reduction in redundant register usage. This serves as the critical conceptual bridge toward Tensor Cores.

### 7.6 Version 7: Tensor Core WMMA Baseline

Replaces scalar FMA instructions on CUDA cores with warp-wide matrix multiply-accumulate operations on Tensor Cores.

- **Improvements:** Unlocks specialized hardware for dense math. The optimization focus officially shifts from organizing scalar arithmetic to efficiently feeding matrix hardware geometry.

### 7.7 Version 8: Shared-Memory Staged WMMA

Refines the Tensor Core dataflow by staging operands through shared memory rather than relying solely on global memory fragment loads.

- **Improvements:** Restores operand reuse across warps, vastly improving locality and bringing the dataflow closer to production CUDA capabilities.

### 7.8 Version 9: Producer-Consumer Pipeline and Epilogue Staging

Implements software pipelining and structured shared-memory epilogues.

- **Improvements:** Overlaps memory fetches (producer) with computation (consumer) to hide latency and eliminate Tensor Core stalls. Staging the epilogue in shared memory allows the output matrix to be efficiently coalesced prior to the final global-memory write.

## 8. Experimental Methodology

To evaluate the progress of kernel optimization systematically, the project employs the following benchmarking protocol across implementations:

1. **Fixed Problem Dimensions:** Matrices are evaluated at standardized sizes (e.g., M=N=K=4096) to accurately measure caching effects.
2. **Warmup & Repeated Execution:** Cold-start anomalies are masked via warmup iterations followed by averaged timing runs.
3. **Reference Validation:** Maximum absolute error is calculated against a trusted CPU/GPU reference (e.g., cuBLAS) to guarantee mathematical correctness.
4. **Performance Profiling:** Throughput is reported in GFLOP/s, complemented by key Nsight Compute metrics identifying stall reasons (e.g., Memory Dependency, Execution Dependency) at each stage.

## 9. Conclusion

This project serves as an architectural study demonstrating that high-performance GEMM is not achieved through a single algorithmic trick, but via a sequence of deliberate, hardware-aligned transformations. By exposing the bottlenecks from naive memory accesses down to asynchronous Tensor Core pipelines, this repository provides clear visibility into the trade-offs that dictate modern high-performance GPU programming.

## 10. Inspiration and Acknowledgement

This work draws inspiration from performance engineering worklogs that treat GEMM optimization as a sequence of isolated structural improvements rather than as a single opaque final kernel. In particular, Hamza Elshafie's H100 GEMM optimization study helped shape the methodological framing of this repository: analyze one optimization at a time, identify the bottleneck it targets, and explain the hardware consequences of each change.

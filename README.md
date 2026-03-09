# Rebuilding cuBLAS: CUDA GEMM Kernel Optimization from Naive SGEMM to Tensor Cores

This project documents a hands-on journey of optimizing GEMM on NVIDIA GPUs, starting from a naive SGEMM kernel and progressively introducing the techniques used in high-performance matrix multiplication kernels:

- global-memory coalescing
- shared-memory tiling
- register tiling
- vectorized loads/stores
- warp tiling
- Tensor Core WMMA
- shared-memory staged Tensor Core kernels
- software pipelining with producer/consumer structure

The goal is not only to get a faster kernel. The goal is to understand what each optimization changes, what bottleneck it attacks, what trade-off it introduces, and how modern GEMM kernels evolve toward cuBLAS/CUTLASS-style design.

## What Is This?

This folder is a kernel-by-kernel study of GEMM:

`C = alpha * A * B + beta * C`

where:

- `A` is `MxK` or `MxN` depending on the kernel version and notation used
- `B` is `KxN` or `NxK`
- `C` is the output matrix

Each numbered file introduces one major optimization step and keeps the code close enough to the previous version that the effect of the new idea is visible.

This repository is best read as:

1. a learning project for CUDA performance engineering
2. a reproducible optimization ladder from simple to advanced GEMM kernels
3. a foundation for a future polished open-source CUDA GEMM project

## What Problem Does It Solve?

Naive matrix multiplication on GPU wastes most of the hardware.

Typical early bottlenecks are:

- too many global memory accesses
- poor data reuse
- too many scalar memory instructions
- too much shared-memory traffic
- register pressure
- low arithmetic intensity
- Tensor Cores sitting idle because the data path is not fast enough

This project solves those problems step by step by restructuring both:

- how data moves: global -> shared -> registers -> accumulators
- how work is partitioned: block -> warp -> thread -> Tensor Core fragment

In short, this project teaches how to turn:

- a correct but slow GEMM

into:

- a structured, high-throughput GEMM pipeline that is much closer to production kernels

## Optimization Journey

The current folder follows this progression:

| Version | File                                                                                 | Main Idea                           | Bottleneck It Targets                             |
| ------- | ------------------------------------------------------------------------------------ | ----------------------------------- | ------------------------------------------------- |
| 01      | `01. Build Naive SGEMM.cu`                                                           | Baseline SGEMM                      | correctness, basic launch structure               |
| 02      | `02. Shared Memory Tilling.cu`                                                       | Shared-memory tiling                | repeated global-memory access                     |
| 03      | `03. Register Tilling - 1 side.cu`                                                   | 1D register tiling                  | low arithmetic intensity                          |
| 04      | `04. Register Tilling - 2 side - Less register but product same FMA.cu`              | 2D register tiling                  | shared-memory traffic per output                  |
| 05      | `05. Vectorized Register Tilling.cu`                                                 | `float4` loads/stores               | instruction pressure in load/store path           |
| 06      | `06. Warp tilling.cu`                                                                | Warp-level tile ownership           | scheduler alignment, register pressure, locality  |
| 07      | `07. Tensor Cores (Async TMA + WGMMA).cu`                                            | WMMA Tensor Core baseline           | move GEMM compute from CUDA cores to Tensor Cores |
| 08      | `08. Tensor Cores - Shared Memory WMMA.cu`                                           | shared-memory staged WMMA           | Tensor Core feed path and reuse                   |
| 09      | `09. Asynchronous Producer–Consumer Pipeline with Epilogue Shared Memory Staging.cu` | software pipeline + staged epilogue | pipeline bubbles, load/compute overlap            |

## Important Scope Note

Versions `07`, `08`, and `09` are currently best understood as educational Tensor Core steps, not final production Hopper kernels.

More specifically:

- `07` uses WMMA as a practical Tensor Core baseline
- `08` stages Tensor Core operands through shared memory
- `09` demonstrates the software structure of producer/consumer pipelining

They are useful and real optimization steps, but they are not yet the final form of:

- true Hopper TMA kernels
- true WGMMA warp-group kernels
- fully async `cp.async` / TMA pipelines
- CUTLASS-level epilogue and swizzle design

That means the repository is already strong as a learning/performance project, but there is still meaningful work left before it can honestly claim "cuBLAS-like" engineering depth.

## Why This Project Is Interesting

Most GEMM tutorials stop at one of these points:

- naive CUDA
- shared memory tiling
- one Tensor Core example

This project is more ambitious:

- it preserves the full optimization ladder
- it keeps the kernels as standalone files
- it lets readers compare design choices version by version
- it makes performance engineering visible instead of magical

That is exactly what makes it a good GitHub project:

- educational value
- systems/performance depth
- clear progression
- easy-to-demo kernels

## What Has Already Improved

Across the versions, the project progressively improves:

### 1. Memory Reuse

- tile data is reused from shared memory instead of repeatedly fetched from global memory
- thread-level register tiles reuse loaded operands for multiple outputs
- warp and block mapping improve locality and data ownership

### 2. Arithmetic Intensity

- each thread computes more than one output
- more FMA work is done per byte loaded
- later kernels do more math per scheduling/event overhead

### 3. Instruction Efficiency

- vectorized loads/stores reduce scalar instruction count
- warp tiling organizes compute closer to the hardware execution model
- Tensor Core kernels replace scalar FMA loops with WMMA matrix operations

### 4. Pipeline Structure

- later kernels separate data staging from compute more clearly
- producer/consumer design prepares the project for true async copy pipelines
- epilogue staging introduces a more realistic writeback path

## Current Gaps

The project is strong, but it is not finished yet. The main missing pieces are:

### 1. Real Benchmarking Discipline

The project still needs a consistent benchmarking framework:

- fixed benchmark sizes
- warmup runs
- averaged timings
- correctness checks for every kernel
- unified performance reporting
- optional cuBLAS baseline comparison

### 2. Cleaner Mathematical Consistency

Some kernels use different dimension notation conventions across the series.

To make the repository easier to read, all kernels should eventually be standardized around one convention, ideally:

- `A = MxK`
- `B = KxN`
- `C = MxN`

### 3. Stronger Correctness Infrastructure

Every kernel should ideally have:

- CPU reference implementation
- max absolute error reporting
- tolerance policy by datatype
- pass/fail output

This becomes especially important once mixed precision and Tensor Cores are involved.

### 4. Real Hopper-Specific Kernels

If the final goal is to impress as a serious performance project, the biggest missing step is:

- true `cp.async` or TMA-based async loading
- true WGMMA warp-group implementation
- proper barrier choreography for async stages
- a more production-style epilogue

### 5. Better Documentation and Results

The repository should clearly explain:

- what each version optimizes
- why each change was made
- what new bottleneck appears after that change
- measured performance before/after

That is what turns "a folder of CUDA files" into a real public project.

## What Still Needs To Be Done

If the goal is to publish this as a strong GitHub project, here is the most practical finish checklist.

### High Priority

- Standardize matrix dimension notation across all kernels
- Add correctness validation to every version
- Add a consistent benchmark harness and output format
- Create a results table for all kernels on one GPU
- Add Nsight Compute screenshots or metric summaries
- Rename some files for consistency and spelling, for example `Tilling` -> `Tiling`

### Medium Priority

- Add one real Ampere/Hopper async copy version
- Add one true WGMMA/TMA-oriented kernel or annotated skeleton
- Add a benchmark script that runs all versions sequentially
- Add one diagram per major optimization step
- Add notes on register pressure, occupancy, and memory bottlenecks

### Nice To Have

- Add CUTLASS/cuBLAS comparison plots
- Add support for multiple datatypes such as FP32, FP16, BF16
- Add command-line build/run workflow outside notebook mode
- Add PTX/SASS inspection notes for selected kernels
- Add architecture notes for T4, A100, H100 differences

## Recommended Project Structure

To make this publishable, the folder should eventually grow into something like:

```text
GEMM-cuda-kernel-optimization/
├── README.md
├── kernels/
│   ├── 01_naive_sgemm.cu
│   ├── 02_shared_memory_tiling.cu
│   ├── 03_register_tiling_1d.cu
│   ├── 04_register_tiling_2d.cu
│   ├── 05_vectorized_register_tiling.cu
│   ├── 06_warp_tiling.cu
│   ├── 07_tensor_core_wmma.cu
│   ├── 08_tensor_core_smem_wmma.cu
│   └── 09_pipeline_epilogue_staging.cu
├── benchmarks/
│   └── run_all.py
├── docs/
│   ├── profiling.md
│   ├── roofline.md
│   ├── tensor-cores.md
│   └── images/
└── results/
    └── benchmark_results.csv
```

You do not need to do this immediately, but it is a strong target structure.

## Suggested GitHub Positioning

If you want the project to feel impressive, position it as:

> A from-scratch CUDA GEMM optimization worklog that rebuilds the path from naive SGEMM to Tensor Core pipelines, explaining not only what gets faster, but why.

That framing is strong because it communicates:

- technical depth
- educational value
- systems thinking
- performance engineering credibility

## Suggested Titles

Best recommended title:

**Rebuilding cuBLAS: CUDA GEMM Kernel Optimization from Naive SGEMM to Tensor Cores**

Other strong options:

- **From Naive SGEMM to Tensor Cores: A CUDA GEMM Optimization Worklog**
- **CUDA GEMM Optimization Journey: Tiling, Warp Mapping, WMMA, and Pipelining**
- **How GEMM Gets Fast: Rebuilding High-Performance CUDA Matrix Multiplication**
- **Inside High-Performance GEMM: A Step-by-Step CUDA Kernel Optimization Project**

If you want the most impressive and clickable title, use the first one.

If you want the most educational and honest title, use the second one.

## Suggested README Sections For a Public Repo

When you publish the full project, your top-level README should answer these questions fast:

1. What is this project?
2. Why should someone care?
3. What kernels are included?
4. What did each optimization improve?
5. What hardware was used?
6. How do I run it?
7. What is still incomplete?

This file already gives you most of that structure.

## A Good Honest Project Summary

This repository is a learning-first, performance-oriented CUDA GEMM project that walks through the real engineering steps between a basic SGEMM and modern Tensor Core kernels. It is not just about making GEMM faster. It is about understanding the bottleneck at each stage, restructuring the kernel accordingly, and building intuition for how high-performance GPU math actually works.

## Final Recommendation

To make this genuinely impressive on GitHub, do these next:

1. Clean up naming and notation.
2. Add benchmark and correctness consistency across all versions.
3. Add one polished results table with performance progression.
4. Be explicit about which Tensor Core kernels are educational approximations versus true Hopper implementations.
5. Add one final advanced kernel that uses real async-copy/TMA/WGMMA concepts or a carefully labeled architecture-specific skeleton.

If you do that, this stops being just a practice folder and becomes a strong portfolio-quality GPU performance project.

This project is inspired by performance worklogs such as Hamza Elshafie's H100 GEMM optimization write-up, especially the idea of building performance incrementally and explaining what each kernel buys you in practice.

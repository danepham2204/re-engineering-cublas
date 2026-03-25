# Execution and Profiling Guide

## Environment
- **Platform**: Google Colab (NVIDIA T4 GPU, SM75) / Local
- **CUDA**: 12.8
- **Compiler**: `nvcc`

## Benchmark Execution

In Colab, write sources to disk via `%%writefile`. Ensure the benchmark harness (`runner.h`, or `runner_half.h` for FP16) is available in the compilation path (e.g., `/content/runner.h`).

### Compilation
Mandatory compilation flags for NVIDIA T4 (SM 7.5):
```bash
nvcc -O3 -arch=sm_75 -lcublas <kernel_file>.cu -o <executable>
```
*Note: Omitting `-arch=sm_75` defaults to Maxwell (SM 5.2), which silently compiles away WMMA intrinsics.*

### Execution
```bash
./<executable>
```
The harness (`runner.h`) handles component warmup, iterates 20 times to compute average GFLOP/s, and validates the maximum error bound against the `cuBLAS` baseline.

## Hardware Profiling (Nsight Compute)

Profile the compiled binary to analyze hardware bottlenecks. Since `ncu` instrumentation severely degrades execution speed, exclusively use telemetry metrics for performance analysis and disregard instrumented execution latency.

### Configuration & Telemetry Collection
Bypass Colab privilege restrictions using `--target-processes all` and `--clock-control none`. Limit analysis to a single execution via `--launch-skip 1 --launch-count 1` and isolate the target kernel utilizing `--kernel-name`.

```bash
ncu --target-processes all \
    --clock-control none \
    --kernel-name <kernel_name> \
    --launch-skip 1 \
    --launch-count 1 \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum \
    ./<executable>
```
*(Colab limitation constraint: Keep the `--metrics` string on a single line without spaces or continuations `\`)*

### Legacy Profiling Fallback
If `ncu` is inaccessible, fallback to `nvprof`:
```bash
nvprof --metrics sm_efficiency,dram_read_throughput,flop_count_sp ./<executable>
```

### Key Diagnostic Metrics

| Hardware Metric | Diagnostic Focus |
| :--- | :--- |
| `sm__warps_active` | Warp occupancy, thread idling, and tile sizing efficacy. |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld` | Global memory pressure, L1/Tex cache locality, and miss rates. |
| `sm__sass_thread_inst_executed_op_ffma_pred_on` | Active FMA instruction counts and arithmetic intensity limits. |
| `dram__bytes_read` | DRAM traffic and bandwidth saturation levels. |

*Optional: Append `--csv --log-file profile.csv` to `ncu` for cross-kernel analysis and baseline comparisons.*
# RoPE Benchmark and Implementation

This project implements the RoFormer rotary position embedding (RoPE) paper:

> Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"

It provides both a CPU-bound serial implementation and a GPU-accelerated parallel implementation using CUDA, alongside a comprehensive benchmark suite to verify correctness and measure performance.

### Features
- Base interfaces for attention kernels and rotary embedding.
- **Serial Implementation**: CPU-based scaled dot-product attention and RoPE application.
- **Parallel Implementation**: CUDA-accelerated RoPE application (`ParallelRotaryEmbedding` and `ParallelRoPEAttention`).
- **Correctness Verification**: Automated element-wise equality checking between serial and parallel runs.
- **Performance Metrics**: Timing (ms), estimated throughput (GFLOP/s), and memory bandwidth (GB/s).
- Unit tests for RoPE correctness properties and attention behavior.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Unit Tests

Run the basic test suite to verify foundational matrix and math correctness:
```bash
ctest --test-dir build --output-on-failure
```

## Benchmarks

You can run individual benchmarks directly:
```bash
./build/rope_benchmark --seq-len 128 --head-dim 64 --iterations 10
```

For a comprehensive test across multiple configurations (e.g. sequence lengths from 512 up to 16384), a test suite script is provided:
```bash
./test.sh
```

The benchmark suite will automatically:
1. Run both the serial CPU and parallel GPU implementations side-by-side.
2. Verify that their outputs are mathematically identical.
3. Compute average execution times, speedups, compute throughput (GFLOP/s), and memory bandwidth (GB/s).


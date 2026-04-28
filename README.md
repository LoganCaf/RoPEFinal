# RoPE Serial Baseline

This is the project scaffold for the RoFormer rotary position embedding paper:

> Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"

The current code intentionally stops before the parallel implementation. It provides:

- Base interfaces for attention kernels and rotary embedding.
- Serial scaled dot-product attention.
- Serial RoPE attention that rotates queries and keys before attention.
- Timing and estimated throughput metrics.
- Unit tests for RoPE correctness properties and attention behavior.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Benchmark

```bash
./build/rope_benchmark --seq-len 128 --head-dim 64 --iterations 10
```

The benchmark prints CSV so runs can be redirected into files for later graphing:

```bash
./build/rope_benchmark --seq-len 256 --head-dim 64 --iterations 20 > results.csv
```

## Parallel Work Boundary

The assignment allows LLM help for the serial implementation and tests, but not for the parallel implementation or performance analysis. Add your OpenMP or CUDA implementation in a separate kernel class that implements `rope::AttentionKernel`, then compare it against `rope::SerialRoPEAttention`.


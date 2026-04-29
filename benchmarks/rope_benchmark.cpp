#include "rope/attention.hpp"
#include "rope/matrix.hpp"
#include "rope/metrics.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
  std::size_t seq_len{128};
  std::size_t head_dim{64};
  std::size_t iterations{10};
  unsigned seed{7};
};

std::size_t parse_size(const char *value, const std::string &name) {
  const long parsed = std::strtol(value, nullptr, 10);
  if (parsed <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--seq-len" && i + 1 < argc) {
      args.seq_len = parse_size(argv[++i], arg);
    } else if (arg == "--head-dim" && i + 1 < argc) {
      args.head_dim = parse_size(argv[++i], arg);
    } else if (arg == "--iterations" && i + 1 < argc) {
      args.iterations = parse_size(argv[++i], arg);
    } else if (arg == "--seed" && i + 1 < argc) {
      args.seed = static_cast<unsigned>(parse_size(argv[++i], arg));
    } else if (arg == "--help") {
      std::cout << "Usage: rope_benchmark [--seq-len N] [--head-dim D] [--iterations I] [--seed S]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (args.head_dim % 2 != 0) {
    throw std::invalid_argument("--head-dim must be even for RoPE");
  }

  return args;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    const rope::AttentionInput input{
        rope::make_random_matrix(args.seq_len, args.head_dim, args.seed),
        rope::make_random_matrix(args.seq_len, args.head_dim, args.seed + 1),
        rope::make_random_matrix(args.seq_len, args.head_dim, args.seed + 2),
    };

    const rope::SerialScaledDotProductAttention baseline;
    const rope::SerialRoPEAttention serial_rope_attention;
    const rope::ParallelRoPEAttention parallel_rope_attention;

    std::cout << "Running " << args.iterations << " iterations\n";

    int equal_count = 0;
    rope::PerformanceMetrics serial_total;
    rope::PerformanceMetrics parallel_total;

    // Warm up CUDA
    rope::PerformanceMetrics dummy;
    parallel_rope_attention.compute(input, &dummy);

    for (std::size_t i = 0; i < args.iterations; ++i) {
        rope::PerformanceMetrics serial_sample;
        rope::Matrix serial_output = serial_rope_attention.compute(input, &serial_sample);
        serial_total.elapsed_ms += serial_sample.elapsed_ms;
        serial_total.estimated_flops += serial_sample.estimated_flops;
        serial_total.estimated_bytes += serial_sample.estimated_bytes;

        rope::PerformanceMetrics parallel_sample;
        rope::Matrix parallel_output = parallel_rope_attention.compute(input, &parallel_sample);
        parallel_total.elapsed_ms += parallel_sample.elapsed_ms;
        parallel_total.estimated_flops += parallel_sample.estimated_flops;
        parallel_total.estimated_bytes += parallel_sample.estimated_bytes;

        bool equal = true;
        for (size_t j = 0; j < serial_output.size(); ++j) {
            if (std::abs(serial_output.values()[j] - parallel_output.values()[j]) > 1e-5) {
                equal = false;
                break;
            }
        }
        if (equal) {
            equal_count++;
        }
    }

    std::cout << "Parallel and Sequential implementations are equal for " << equal_count << "/" << args.iterations << " iterations\n";
    
    double avg_serial_ms = serial_total.elapsed_ms / args.iterations;
    double avg_parallel_ms = parallel_total.elapsed_ms / args.iterations;
    
    std::cout << "Average Serial execution time: " << avg_serial_ms << " ms\n";
    std::cout << "Average Parallel execution time: " << avg_parallel_ms << " ms\n";
    std::cout << "Average Speedup: " << avg_serial_ms / avg_parallel_ms << "x\n";

    double serial_gflops = (serial_total.estimated_flops / 1e6) / serial_total.elapsed_ms;
    double parallel_gflops = (parallel_total.estimated_flops / 1e6) / parallel_total.elapsed_ms;
    double serial_bw = (serial_total.estimated_bytes / 1e6) / serial_total.elapsed_ms;
    double parallel_bw = (parallel_total.estimated_bytes / 1e6) / parallel_total.elapsed_ms;

    std::cout << "Serial Compute: " << serial_gflops << " GFLOP/s, Memory Bandwidth: " << serial_bw << " GB/s\n";
    std::cout << "Parallel Compute: " << parallel_gflops << " GFLOP/s, Memory Bandwidth: " << parallel_bw << " GB/s\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}


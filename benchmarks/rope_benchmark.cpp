#include "rope/attention.hpp"
#include "rope/matrix.hpp"
#include "rope/metrics.hpp"

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
    const rope::SerialRoPEAttention rope_attention;

    rope::write_metrics_csv_header(std::cout);

    for (const rope::AttentionKernel *kernel : {static_cast<const rope::AttentionKernel *>(&baseline),
                                                static_cast<const rope::AttentionKernel *>(&rope_attention)}) {
      rope::PerformanceMetrics total;
      total.kernel_name = kernel->name();
      total.seq_len = args.seq_len;
      total.head_dim = args.head_dim;
      total.iterations = args.iterations;

      for (std::size_t i = 0; i < args.iterations; ++i) {
        rope::PerformanceMetrics sample;
        (void)kernel->compute(input, &sample);
        total.elapsed_ms += sample.elapsed_ms;
        total.estimated_flops += sample.estimated_flops;
        total.estimated_bytes += sample.estimated_bytes;
        total.checksum = sample.checksum;
      }

      rope::write_metrics_csv_row(std::cout, total);
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}


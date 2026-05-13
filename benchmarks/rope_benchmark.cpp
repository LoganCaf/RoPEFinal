#include "rope/attention.hpp"
#include "rope/matrix.hpp"
#include "rope/metrics.hpp"

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
  std::size_t seq_len{128};
  std::size_t head_dim{64};
  std::size_t iterations{10};
  unsigned seed{7};
  int threads_per_block{128};
  std::string mode{"compare"};
  std::string csv_output{""};
};

bool is_valid_mode(const std::string &mode) {
  return mode == "compare" ||
         mode == "serial-only" ||
         mode == "cuda-rope-serial-attention" ||
         mode == "cuda-full-no-preload" ||
         mode == "parallel-only";
}

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
    } else if (arg == "--csv-output" && i + 1 < argc) {
      args.csv_output = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      args.mode = argv[++i];
      if (!is_valid_mode(args.mode)) {
        throw std::invalid_argument(
            "--mode must be 'compare', 'serial-only', "
            "'cuda-rope-serial-attention', 'cuda-full-no-preload', "
            "or 'parallel-only'");
      }
    } else if (arg == "--threads-per-block" && i + 1 < argc) {
      args.threads_per_block = static_cast<int>(parse_size(argv[++i], arg));
    } else if (arg == "--help") {
      std::cout << "Usage: rope_benchmark [--seq-len N] [--head-dim D] "
                   "[--iterations I] [--seed S] [--threads-per-block T] "
                   "[--mode compare|serial-only|cuda-rope-serial-attention|cuda-full-no-preload|parallel-only] "
                   "[--csv-output FILE]\n";
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

rope::Matrix compute_cuda_rope_serial_attention(const rope::AttentionInput &input,
                                                int threads_per_block,
                                                rope::PerformanceMetrics *metrics) {
  rope::validate_attention_input(input);

  const auto start = std::chrono::steady_clock::now();
  rope::AttentionInput rotated{input.query, input.key, input.value};
  const rope::ParallelRotaryEmbedding rotary(10000.0, threads_per_block);
  rotary.apply_in_place(rotated.query);
  rotary.apply_in_place(rotated.key);

  const rope::SerialScaledDotProductAttention attention;
  rope::Matrix output = attention.compute(rotated);

  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  rope::fill_metrics(metrics,
                     "cuda_rope_serial_attention",
                     output,
                     input.query.rows(),
                     input.query.cols(),
                     elapsed_ms,
                     true);
  return output;
}

void accumulate_metrics(rope::PerformanceMetrics &total, const rope::PerformanceMetrics &sample) {
  total.elapsed_ms += sample.elapsed_ms;
  total.estimated_flops += sample.estimated_flops;
  total.estimated_bytes += sample.estimated_bytes;
}

void finish_metrics(rope::PerformanceMetrics &metrics,
                    const std::string &name,
                    const Args &args) {
  metrics.kernel_name = name;
  metrics.seq_len = args.seq_len;
  metrics.head_dim = args.head_dim;
  metrics.iterations = args.iterations;
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
    const rope::ParallelRoPEAttention parallel_rope_attention(nullptr, args.threads_per_block);
    const rope::ParallelRoPEAttention parallel_no_preload_attention(nullptr, args.threads_per_block, false);

    std::cout << "Running " << args.iterations << " iterations\n";

    int equal_count = 0;
    rope::PerformanceMetrics serial_total;
    rope::PerformanceMetrics cuda_rope_serial_total;
    rope::PerformanceMetrics parallel_no_preload_total;
    rope::PerformanceMetrics parallel_total;

    // Warm up
    const std::size_t warmup_iterations = 5;
    for (std::size_t i = 0; i < warmup_iterations; ++i) {
        rope::PerformanceMetrics dummy;
        if (args.mode == "compare" || args.mode == "serial-only") {
            serial_rope_attention.compute(input, &dummy);
        }
        if (args.mode == "compare" || args.mode == "parallel-only") {
            parallel_rope_attention.compute(input, &dummy);
        }
        if (args.mode == "cuda-full-no-preload") {
            parallel_no_preload_attention.compute(input, &dummy);
        }
        if (args.mode == "cuda-rope-serial-attention") {
            compute_cuda_rope_serial_attention(input, args.threads_per_block, &dummy);
        }
    }

    for (std::size_t i = 0; i < args.iterations; ++i) {
        rope::Matrix serial_output(1, 1);
        
        if (args.mode == "compare" || args.mode == "serial-only") {
            rope::PerformanceMetrics serial_sample;
            serial_output = serial_rope_attention.compute(input, &serial_sample);
            accumulate_metrics(serial_total, serial_sample);
        }

        rope::Matrix parallel_output(1, 1);
        if (args.mode == "compare" || args.mode == "parallel-only") {
            rope::PerformanceMetrics parallel_sample;
            parallel_output = parallel_rope_attention.compute(input, &parallel_sample);
            accumulate_metrics(parallel_total, parallel_sample);
        }

        if (args.mode == "cuda-rope-serial-attention") {
            rope::PerformanceMetrics cuda_rope_serial_sample;
            compute_cuda_rope_serial_attention(input, args.threads_per_block, &cuda_rope_serial_sample);
            accumulate_metrics(cuda_rope_serial_total, cuda_rope_serial_sample);
        }

        if (args.mode == "cuda-full-no-preload") {
            rope::PerformanceMetrics parallel_no_preload_sample;
            parallel_no_preload_attention.compute(input, &parallel_no_preload_sample);
            accumulate_metrics(parallel_no_preload_total, parallel_no_preload_sample);
        }

        if (args.mode == "compare") {
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
    }

    if (args.mode == "compare") {
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
    } else if (args.mode == "serial-only") {
        double avg_serial_ms = serial_total.elapsed_ms / args.iterations;
        std::cout << "Average Serial execution time: " << avg_serial_ms << " ms\n";

        double serial_gflops = (serial_total.estimated_flops / 1e6) / serial_total.elapsed_ms;
        double serial_bw = (serial_total.estimated_bytes / 1e6) / serial_total.elapsed_ms;
        std::cout << "Serial Compute: " << serial_gflops << " GFLOP/s, Memory Bandwidth: " << serial_bw << " GB/s\n";
    } else if (args.mode == "cuda-rope-serial-attention") {
        double avg_ms = cuda_rope_serial_total.elapsed_ms / args.iterations;
        std::cout << "Average CUDA RoPE + Serial Attention execution time: " << avg_ms << " ms\n";

        double gflops = (cuda_rope_serial_total.estimated_flops / 1e6) / cuda_rope_serial_total.elapsed_ms;
        double bw = (cuda_rope_serial_total.estimated_bytes / 1e6) / cuda_rope_serial_total.elapsed_ms;
        std::cout << "CUDA RoPE + Serial Attention Compute: " << gflops
                  << " GFLOP/s, Memory Bandwidth: " << bw << " GB/s\n";
    } else if (args.mode == "cuda-full-no-preload") {
        double avg_parallel_ms = parallel_no_preload_total.elapsed_ms / args.iterations;
        std::cout << "Average CUDA Full No-Preload execution time: " << avg_parallel_ms << " ms\n";
        
        double parallel_gflops = (parallel_no_preload_total.estimated_flops / 1e6) / parallel_no_preload_total.elapsed_ms;
        double parallel_bw = (parallel_no_preload_total.estimated_bytes / 1e6) / parallel_no_preload_total.elapsed_ms;
        std::cout << "CUDA Full No-Preload Compute: " << parallel_gflops
                  << " GFLOP/s, Memory Bandwidth: " << parallel_bw << " GB/s\n";
    } else {
        double avg_parallel_ms = parallel_total.elapsed_ms / args.iterations;
        std::cout << "Average Parallel execution time: " << avg_parallel_ms << " ms\n";
        
        double parallel_gflops = (parallel_total.estimated_flops / 1e6) / parallel_total.elapsed_ms;
        double parallel_bw = (parallel_total.estimated_bytes / 1e6) / parallel_total.elapsed_ms;
        std::cout << "Parallel Compute: " << parallel_gflops << " GFLOP/s, Memory Bandwidth: " << parallel_bw << " GB/s\n";
    }

    // Write to CSV if arg is provided
    if (!args.csv_output.empty()) {
      bool write_header = false;
      {
        std::ifstream f(args.csv_output);
        write_header = !f.good();
      }
      
      std::ofstream out_csv(args.csv_output, std::ios_base::app);
      if (write_header) {
        rope::write_metrics_csv_header(out_csv);
      }
      
      if (args.mode == "compare") {
        finish_metrics(serial_total, "sequential", args);
        rope::write_metrics_csv_row(out_csv, serial_total);
      }
      
      if (args.mode == "serial-only") {
        finish_metrics(serial_total, "sequential", args);
        rope::write_metrics_csv_row(out_csv, serial_total);
      } else if (args.mode == "cuda-rope-serial-attention") {
        finish_metrics(cuda_rope_serial_total, "cuda_rope_serial_attention", args);
        rope::write_metrics_csv_row(out_csv, cuda_rope_serial_total);
      } else if (args.mode == "cuda-full-no-preload") {
        finish_metrics(parallel_no_preload_total,
                       "parallel_no_preload_" + std::to_string(args.threads_per_block),
                       args);
        rope::write_metrics_csv_row(out_csv, parallel_no_preload_total);
      } else {
        finish_metrics(parallel_total, "parallel_" + std::to_string(args.threads_per_block), args);
        rope::write_metrics_csv_row(out_csv, parallel_total);
      }
    }

  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}

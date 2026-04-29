#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

namespace rope {

struct PerformanceMetrics {
  std::string kernel_name;
  std::size_t seq_len{0};
  std::size_t head_dim{0};
  std::size_t iterations{0};
  double elapsed_ms{0.0};
  double estimated_flops{0.0};
  double estimated_bytes{0.0};
  double checksum{0.0};

  double gflops() const;
  double bandwidth_gb_s() const;
};

double estimate_scaled_dot_product_attention_flops(std::size_t seq_len, std::size_t head_dim);
double estimate_scaled_dot_product_attention_bytes(std::size_t seq_len, std::size_t head_dim);
double estimate_rope_attention_flops(std::size_t seq_len, std::size_t head_dim);
double estimate_rope_attention_bytes(std::size_t seq_len, std::size_t head_dim);

class Matrix;

void fill_metrics(PerformanceMetrics *metrics,
                  const std::string &name,
                  const Matrix &output,
                  std::size_t seq_len,
                  std::size_t head_dim,
                  double elapsed_ms,
                  bool includes_rope);

void write_metrics_csv_header(std::ostream &out);
void write_metrics_csv_row(std::ostream &out, const PerformanceMetrics &metrics);

} // namespace rope

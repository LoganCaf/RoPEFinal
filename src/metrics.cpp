#include "rope/metrics.hpp"

#include <ostream>

namespace rope {

double PerformanceMetrics::gflops() const {
  const double seconds = elapsed_ms / 1000.0;
  return seconds > 0.0 ? estimated_flops / seconds / 1.0e9 : 0.0;
}

double PerformanceMetrics::bandwidth_gb_s() const {
  const double seconds = elapsed_ms / 1000.0;
  return seconds > 0.0 ? estimated_bytes / seconds / 1.0e9 : 0.0;
}

double estimate_scaled_dot_product_attention_flops(std::size_t seq_len, std::size_t head_dim) {
  const double n = static_cast<double>(seq_len);
  const double d = static_cast<double>(head_dim);

  const double qk_dot = n * n * (2.0 * d - 1.0);
  const double softmax = n * (4.0 * n);
  const double weighted_sum = n * n * (2.0 * d - 1.0);

  return qk_dot + softmax + weighted_sum;
}

double estimate_scaled_dot_product_attention_bytes(std::size_t seq_len, std::size_t head_dim) {
  const double scalar = static_cast<double>(sizeof(double));
  const double n = static_cast<double>(seq_len);
  const double d = static_cast<double>(head_dim);

  const double qkv_read = 3.0 * n * d * scalar;
  const double score_write_read = 2.0 * n * n * scalar;
  const double output_write = n * d * scalar;

  return qkv_read + score_write_read + output_write;
}

double estimate_rope_attention_flops(std::size_t seq_len, std::size_t head_dim) {
  const double n = static_cast<double>(seq_len);
  const double d = static_cast<double>(head_dim);
  const double rope_rotation = 2.0 * n * (d / 2.0) * 6.0;

  return estimate_scaled_dot_product_attention_flops(seq_len, head_dim) + rope_rotation;
}

double estimate_rope_attention_bytes(std::size_t seq_len, std::size_t head_dim) {
  const double scalar = static_cast<double>(sizeof(double));
  const double n = static_cast<double>(seq_len);
  const double d = static_cast<double>(head_dim);
  const double rotated_qk_write = 2.0 * n * d * scalar;

  return estimate_scaled_dot_product_attention_bytes(seq_len, head_dim) + rotated_qk_write;
}

void write_metrics_csv_header(std::ostream &out) {
  out << "kernel,seq_len,head_dim,iterations,elapsed_ms,gflops,bandwidth_gb_s,checksum\n";
}

void write_metrics_csv_row(std::ostream &out, const PerformanceMetrics &metrics) {
  out << metrics.kernel_name << ','
      << metrics.seq_len << ','
      << metrics.head_dim << ','
      << metrics.iterations << ','
      << metrics.elapsed_ms << ','
      << metrics.gflops() << ','
      << metrics.bandwidth_gb_s() << ','
      << metrics.checksum << '\n';
}

} // namespace rope

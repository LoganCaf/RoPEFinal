#include "rope/attention.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rope {

namespace {
using Clock = std::chrono::steady_clock;
} // namespace

void fill_metrics(PerformanceMetrics *metrics,
                  const std::string &name,
                  const Matrix &output,
                  std::size_t seq_len,
                  std::size_t head_dim,
                  double elapsed_ms,
                  bool includes_rope) {
  if (metrics == nullptr) {
    return;
  }

  metrics->kernel_name = name;
  metrics->seq_len = seq_len;
  metrics->head_dim = head_dim;
  metrics->iterations = 1;
  metrics->elapsed_ms = elapsed_ms;
  metrics->estimated_flops = includes_rope
                                  ? estimate_rope_attention_flops(seq_len, head_dim)
                                  : estimate_scaled_dot_product_attention_flops(seq_len, head_dim);
  metrics->estimated_bytes = includes_rope
                                  ? estimate_rope_attention_bytes(seq_len, head_dim)
                                  : estimate_scaled_dot_product_attention_bytes(seq_len, head_dim);
  metrics->checksum = output.checksum();
}

void validate_attention_input(const AttentionInput &input) {
  if (input.query.empty() || input.key.empty() || input.value.empty()) {
    throw std::invalid_argument("Attention inputs must be non-empty");
  }
  if (input.query.rows() != input.key.rows() || input.query.rows() != input.value.rows()) {
    throw std::invalid_argument("Query, key, and value must have the same sequence length");
  }
  if (input.query.cols() != input.key.cols()) {
    throw std::invalid_argument("Query and key must have the same head dimension");
  }
  if (input.query.cols() != input.value.cols()) {
    throw std::invalid_argument("This baseline expects value dimension to equal head dimension");
  }
}

std::string SerialScaledDotProductAttention::name() const {
  return "serial_scaled_dot_product_attention";
}

Matrix SerialScaledDotProductAttention::compute(const AttentionInput &input,
                                                PerformanceMetrics *metrics) const {
  validate_attention_input(input);

  const auto start = Clock::now();
  const std::size_t seq_len = input.query.rows();
  const std::size_t dim = input.query.cols();
  const double inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(dim));

  Matrix output(seq_len, dim);
  std::vector<double> scores(seq_len);

  for (std::size_t row = 0; row < seq_len; ++row) {
    double max_score = -std::numeric_limits<double>::infinity();

    for (std::size_t col = 0; col < seq_len; ++col) {
      double dot = 0.0;
      for (std::size_t d = 0; d < dim; ++d) {
        dot += input.query(row, d) * input.key(col, d);
      }
      scores[col] = dot * inv_sqrt_dim;
      max_score = std::max(max_score, scores[col]);
    }

    double denominator = 0.0;
    for (double &score : scores) {
      score = std::exp(score - max_score);
      denominator += score;
    }

    for (std::size_t d = 0; d < dim; ++d) {
      double value = 0.0;
      for (std::size_t col = 0; col < seq_len; ++col) {
        value += (scores[col] / denominator) * input.value(col, d);
      }
      output(row, d) = value;
    }
  }

  const auto end = Clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  fill_metrics(metrics, name(), output, seq_len, dim, elapsed_ms, false);
  return output;
}

SerialRoPEAttention::SerialRoPEAttention(std::shared_ptr<const RotaryEmbedding> rotary)
    : rotary_(std::move(rotary)) {
  if (!rotary_) {
    throw std::invalid_argument("SerialRoPEAttention requires a rotary embedding");
  }
}

std::string SerialRoPEAttention::name() const {
  return "serial_rope_attention";
}

Matrix SerialRoPEAttention::compute(const AttentionInput &input, PerformanceMetrics *metrics) const {
  validate_attention_input(input);

  const auto start = Clock::now();
  AttentionInput rotated{input.query, input.key, input.value};
  rotary_->apply_in_place(rotated.query);
  rotary_->apply_in_place(rotated.key);
  Matrix output = attention_.compute(rotated);

  const auto end = Clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  fill_metrics(metrics, name(), output, input.query.rows(), input.query.cols(), elapsed_ms, true);
  return output;
}

} // namespace rope

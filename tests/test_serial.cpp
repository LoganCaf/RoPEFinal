#include "rope/attention.hpp"
#include "rope/matrix.hpp"
#include "rope/rotary_embedding.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kTolerance = 1.0e-10;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(double actual, double expected, double tolerance, const std::string &message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

double row_norm_sq(const rope::Matrix &matrix, std::size_t row) {
  double total = 0.0;
  for (std::size_t col = 0; col < matrix.cols(); ++col) {
    total += matrix(row, col) * matrix(row, col);
  }
  return total;
}

double dot_row(const rope::Matrix &lhs, std::size_t lhs_row,
               const rope::Matrix &rhs, std::size_t rhs_row) {
  double total = 0.0;
  for (std::size_t col = 0; col < lhs.cols(); ++col) {
    total += lhs(lhs_row, col) * rhs(rhs_row, col);
  }
  return total;
}

void test_matrix_basics() {
  rope::Matrix matrix(2, 3);
  matrix(0, 0) = 1.0;
  matrix(0, 1) = 2.0;
  matrix(0, 2) = 3.0;
  matrix(1, 0) = 4.0;
  matrix(1, 1) = 5.0;
  matrix(1, 2) = 6.0;

  require(matrix.rows() == 2, "matrix rows");
  require(matrix.cols() == 3, "matrix cols");
  require_near(matrix.checksum(), 21.0, kTolerance, "matrix checksum");
}

void test_rope_position_zero_identity() {
  rope::Matrix matrix(1, 4, {1.0, -2.0, 3.0, -4.0});
  rope::SerialRotaryEmbedding rotary;
  rotary.apply_in_place(matrix);

  require_near(matrix(0, 0), 1.0, kTolerance, "position zero keeps first value");
  require_near(matrix(0, 1), -2.0, kTolerance, "position zero keeps second value");
  require_near(matrix(0, 2), 3.0, kTolerance, "position zero keeps third value");
  require_near(matrix(0, 3), -4.0, kTolerance, "position zero keeps fourth value");
}

void test_rope_preserves_norm() {
  rope::Matrix matrix(3, 4, {
      1.0, 2.0, 3.0, 4.0,
      -2.0, 0.5, 7.0, -1.0,
      0.25, -0.75, 1.5, -2.5,
  });

  const double norm_before = row_norm_sq(matrix, 2);
  rope::SerialRotaryEmbedding rotary;
  rotary.apply_in_place(matrix);

  require_near(row_norm_sq(matrix, 2), norm_before, 1.0e-9, "RoPE preserves vector norm");
}

void test_rope_relative_dot_product_property() {
  constexpr double base = 10000.0;
  rope::Matrix query(1, 4, {0.5, -1.0, 2.0, 0.25});
  rope::Matrix key(1, 4, {-0.75, 0.125, 1.5, -2.0});
  rope::Matrix rotated_query = query;
  rope::Matrix rotated_key = key;

  rope::apply_rope_to_row(rotated_query.values().data(), 5, rotated_query.cols(), base);
  rope::apply_rope_to_row(rotated_key.values().data(), 2, rotated_key.cols(), base);

  rope::Matrix relative_query = query;
  rope::apply_rope_to_row(relative_query.values().data(), 5 - 2, relative_query.cols(), base);

  require_near(dot_row(rotated_query, 0, rotated_key, 0),
               dot_row(relative_query, 0, key, 0),
               1.0e-9,
               "RoPE dot product depends on relative position");
}

void test_attention_identity_like_case() {
  const rope::AttentionInput input{
      rope::Matrix(2, 2, {10.0, 0.0, 0.0, 10.0}),
      rope::Matrix(2, 2, {10.0, 0.0, 0.0, 10.0}),
      rope::Matrix(2, 2, {1.0, 2.0, 3.0, 4.0}),
  };

  rope::SerialScaledDotProductAttention attention;
  const rope::Matrix output = attention.compute(input);

  require_near(output(0, 0), 1.0, 1.0e-9, "first token attends to first value");
  require_near(output(0, 1), 2.0, 1.0e-9, "first token second output");
  require_near(output(1, 0), 3.0, 1.0e-9, "second token attends to second value");
  require_near(output(1, 1), 4.0, 1.0e-9, "second token second output");
}

void test_metrics_are_populated() {
  const rope::AttentionInput input{
      rope::make_random_matrix(4, 4, 1),
      rope::make_random_matrix(4, 4, 2),
      rope::make_random_matrix(4, 4, 3),
  };

  rope::SerialRoPEAttention attention;
  rope::PerformanceMetrics metrics;
  const rope::Matrix output = attention.compute(input, &metrics);

  require(output.rows() == 4, "output rows");
  require(output.cols() == 4, "output cols");
  require(metrics.kernel_name == "serial_rope_attention", "metrics kernel name");
  require(metrics.elapsed_ms >= 0.0, "metrics elapsed time");
  require(metrics.estimated_flops > 0.0, "metrics flops");
  require(metrics.estimated_bytes > 0.0, "metrics bytes");
}

} // namespace

int main() {
  try {
    test_matrix_basics();
    test_rope_position_zero_identity();
    test_rope_preserves_norm();
    test_rope_relative_dot_product_property();
    test_attention_identity_like_case();
    test_metrics_are_populated();
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "all serial tests passed\n";
  return EXIT_SUCCESS;
}

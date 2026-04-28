#include "rope/rotary_embedding.hpp"

#include <cmath>
#include <stdexcept>

namespace rope {

SerialRotaryEmbedding::SerialRotaryEmbedding(double base) : base_(base) {
  if (base <= 0.0) {
    throw std::invalid_argument("RoPE base must be positive");
  }
}

std::string SerialRotaryEmbedding::name() const {
  return "serial_rope";
}

void SerialRotaryEmbedding::apply_in_place(Matrix &matrix) const {
  if (matrix.cols() % 2 != 0) {
    throw std::invalid_argument("RoPE requires an even head dimension");
  }

  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    apply_rope_to_row(&matrix.values()[row * matrix.cols()], row, matrix.cols(), base_);
  }
}

void apply_rope_to_row(double *row, std::size_t position, std::size_t dim, double base) {
  if (dim % 2 != 0) {
    throw std::invalid_argument("RoPE requires an even dimension");
  }
  if (base <= 0.0) {
    throw std::invalid_argument("RoPE base must be positive");
  }

  const double d = static_cast<double>(dim);
  for (std::size_t i = 0; i < dim; i += 2) {
    const double pair_index = static_cast<double>(i / 2);
    const double theta = 1.0 / std::pow(base, (2.0 * pair_index) / d);
    const double angle = static_cast<double>(position) * theta;
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    const double x0 = row[i];
    const double x1 = row[i + 1];
    row[i] = x0 * c - x1 * s;
    row[i + 1] = x0 * s + x1 * c;
  }
}

} // namespace rope


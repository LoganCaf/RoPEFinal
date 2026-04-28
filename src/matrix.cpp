#include "rope/matrix.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace rope {

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), values_(rows * cols, 0.0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<double> values)
    : rows_(rows), cols_(cols), values_(std::move(values)) {
  if (values_.size() != rows_ * cols_) {
    throw std::invalid_argument("Matrix values size does not match rows * cols");
  }
}

double &Matrix::operator()(std::size_t row, std::size_t col) {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix index out of range");
  }
  return values_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix index out of range");
  }
  return values_[row * cols_ + col];
}

void Matrix::fill(double value) {
  std::fill(values_.begin(), values_.end(), value);
}

double Matrix::checksum() const {
  return std::accumulate(values_.begin(), values_.end(), 0.0);
}

Matrix make_random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
  Matrix matrix(rows, cols);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (double &value : matrix.values()) {
    value = dist(gen);
  }

  return matrix;
}

} // namespace rope

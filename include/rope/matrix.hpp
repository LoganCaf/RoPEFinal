#pragma once

#include <cstddef>
#include <vector>

namespace rope {

class Matrix {
public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols);
  Matrix(std::size_t rows, std::size_t cols, std::vector<double> values);

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t size() const noexcept { return values_.size(); }
  bool empty() const noexcept { return values_.empty(); }

  double &operator()(std::size_t row, std::size_t col);
  double operator()(std::size_t row, std::size_t col) const;

  std::vector<double> &values() noexcept { return values_; }
  const std::vector<double> &values() const noexcept { return values_; }

  void fill(double value);
  double checksum() const;

private:
  std::size_t rows_{0};
  std::size_t cols_{0};
  std::vector<double> values_;
};

Matrix make_random_matrix(std::size_t rows, std::size_t cols, unsigned seed);

} // namespace rope


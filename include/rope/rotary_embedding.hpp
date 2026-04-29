#pragma once

#include "rope/matrix.hpp"

#include <cstddef>
#include <string>

namespace rope {

class RotaryEmbedding {
public:
  virtual ~RotaryEmbedding() = default;

  virtual std::string name() const = 0;
  virtual void apply_in_place(Matrix &matrix) const = 0;
};

class SerialRotaryEmbedding final : public RotaryEmbedding {
public:
  explicit SerialRotaryEmbedding(double base = 10000.0);

  std::string name() const override;
  void apply_in_place(Matrix &matrix) const override;

  double base() const noexcept { return base_; }

private:
  double base_;
};

class ParallelRotaryEmbedding final : public RotaryEmbedding {
public:
  explicit ParallelRotaryEmbedding(double base = 10000.0);

  std::string name() const override;
  void apply_in_place(Matrix &matrix) const override;

  double base() const noexcept { return base_; }

private:
  double base_;
};

void apply_rope_to_row(double *row, std::size_t position, std::size_t dim, double base);

} // namespace rope


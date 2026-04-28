#pragma once

#include "rope/matrix.hpp"
#include "rope/metrics.hpp"
#include "rope/rotary_embedding.hpp"

#include <memory>
#include <string>

namespace rope {

struct AttentionInput {
  Matrix query;
  Matrix key;
  Matrix value;
};

class AttentionKernel {
public:
  virtual ~AttentionKernel() = default;

  virtual std::string name() const = 0;
  virtual Matrix compute(const AttentionInput &input, PerformanceMetrics *metrics = nullptr) const = 0;
};

class SerialScaledDotProductAttention final : public AttentionKernel {
public:
  std::string name() const override;
  Matrix compute(const AttentionInput &input, PerformanceMetrics *metrics = nullptr) const override;
};

class SerialRoPEAttention final : public AttentionKernel {
public:
  explicit SerialRoPEAttention(std::shared_ptr<const RotaryEmbedding> rotary =
                                   std::make_shared<SerialRotaryEmbedding>());

  std::string name() const override;
  Matrix compute(const AttentionInput &input, PerformanceMetrics *metrics = nullptr) const override;

private:
  std::shared_ptr<const RotaryEmbedding> rotary_;
  SerialScaledDotProductAttention attention_;
};

void validate_attention_input(const AttentionInput &input);

} // namespace rope


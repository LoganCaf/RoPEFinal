#include "rope/attention.hpp"
#include "rope/rotary_embedding.hpp"

#include <chrono>
#include <stdexcept>

namespace rope {
  Matrix cuda_scaled_dot_product_attention(const AttentionInput& input,
                                           int threads_per_block,
                                           bool preload_query);

ParallelRoPEAttention::ParallelRoPEAttention(std::shared_ptr<const RotaryEmbedding> rotary,
                                             int threads_per_block,
                                             bool preload_query)
    : rotary_(std::move(rotary)),
      threads_per_block_(threads_per_block),
      preload_query_(preload_query) {
  if (!rotary_) {
    rotary_ = std::make_shared<ParallelRotaryEmbedding>(10000.0, threads_per_block_);
  }
}

std::string ParallelRoPEAttention::name() const {
  return "parallel_rope_attention";
}

Matrix ParallelRoPEAttention::compute(const AttentionInput &input, PerformanceMetrics *metrics) const {
  validate_attention_input(input);

  const auto start = std::chrono::steady_clock::now();
  AttentionInput rotated{input.query, input.key, input.value};
  rotary_->apply_in_place(rotated.query);
  rotary_->apply_in_place(rotated.key);
  Matrix output = cuda_scaled_dot_product_attention(rotated, threads_per_block_, preload_query_);

  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  fill_metrics(metrics, name(), output, input.query.rows(), input.query.cols(), elapsed_ms, true);
  return output;
}


__global__ void rope_kernel(double* in, double* out, double base, int rows, int cols, int vals){

    int gid = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(gid >= vals) return;
    
    int row = gid / cols;
    int col = gid % cols;

    int pair_idx = col / 2;
    double theta = 1.0 / pow(base, (2.0 * pair_idx) / cols);
    double angle = row * theta;
    double c = cos(angle);
    double s = sin(angle);
    
    double x0 = in[gid];
    double x1 = in[gid + 1];

    out[gid] = x0 * c - x1 * s;
    out[gid + 1] = x0 * s + x1 * c;

    return;

}

__global__ void score_kernel_basic(double* Q,double* K, double* score, int rows, int cols){

    int query_row = blockIdx.y;
    int key_row = blockIdx.x * blockDim.x + threadIdx.x;


    if (key_row >= rows) {
        return;
    }

    double dot = 0.0;

    for (int d = 0; d < cols; ++d) {
        dot += Q[query_row * cols + d] * K[key_row * cols + d];
    }

    score[query_row * rows + key_row] = dot / sqrt((double)cols);
}

__global__ void score_kernel(double* Q,double* K, double* score, int rows, int cols){
    extern __shared__ double q_shared[];

    int query_row = blockIdx.y;
    int key_row = blockIdx.x * blockDim.x + threadIdx.x;

    for (int d = threadIdx.x; d < cols; d += blockDim.x) {
        q_shared[d] = Q[query_row * cols + d];
    }

    __syncthreads();

    if (key_row >= rows) {
        return;
    }

    double dot = 0.0;

    for (int d = 0; d < cols; ++d) {
        dot += q_shared[d] * K[key_row * cols + d];
    }

    score[query_row * rows + key_row] = dot / sqrt((double)cols);
}

//softmax[j] = exp(score[row, j] - max_score) / sum(exp(score[row, k] - max_score))

__global__ void softmax_kernel(double* score, int rows) {
      extern __shared__ double shared[];

      int row = blockIdx.x;
      int tid = threadIdx.x;

      double local_max = -INFINITY;

      for (int col = tid; col < rows; col += blockDim.x) {
          double value = score[row * rows + col];
          if (value > local_max) {
              local_max = value;
          }
      }

      shared[tid] = local_max;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
          if (tid < stride && shared[tid + stride] > shared[tid]) {
              shared[tid] = shared[tid + stride];
          }
          __syncthreads();
      }

      double max_score = shared[0];

      double local_sum = 0.0;

      for (int col = tid; col < rows; col += blockDim.x) {
          double value = exp(score[row * rows + col] - max_score);
          score[row * rows + col] = value;
          local_sum += value;
      }

      shared[tid] = local_sum;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
          if (tid < stride) {
              shared[tid] += shared[tid + stride];
          }
          __syncthreads();
      }

      double denominator = shared[0];

      for (int col = tid; col < rows; col += blockDim.x) {
          score[row * rows + col] /= denominator;
      }
  }

  __global__ void output_kernel(
      const double* score,
      const double* V,
      double* output,
      int rows,
      int cols) {

      int d = blockIdx.x * blockDim.x + threadIdx.x;
      int row = blockIdx.y * blockDim.y + threadIdx.y;

      if (row >= rows || d >= cols) {
          return;
      }

      double value = 0.0;

      for (int col = 0; col < rows; ++col) {
          value += score[row * rows + col] * V[col * cols + d];
      }

      output[row * cols + d] = value;
  }

Matrix cuda_scaled_dot_product_attention(const AttentionInput& input,
                                         int threads_per_block,
                                         bool preload_query) {
      std::size_t rows = input.query.rows();
      std::size_t cols = input.query.cols();

      std::size_t size = rows * cols;
      std::size_t num_bytes = size * sizeof(double);

      std::size_t sizeScore = rows * rows;
      std::size_t num_bytesScore = sizeScore * sizeof(double);

      double* Q;
      double* K;
      double* score;

      cudaMalloc(&Q, num_bytes);
      cudaMalloc(&K, num_bytes);
      cudaMalloc(&score, num_bytesScore);

      cudaMemcpy(Q, input.query.values().data(), num_bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(K, input.key.values().data(), num_bytes, cudaMemcpyHostToDevice);

      dim3 block(threads_per_block);
      dim3 grid((rows + block.x - 1) / block.x, rows);

      if (preload_query) {
          score_kernel<<<grid, block, cols * sizeof(double)>>>(Q, K, score, rows, cols);
      } else {
          score_kernel_basic<<<grid, block>>>(Q, K, score, rows, cols);
      }

      softmax_kernel<<<rows, threads_per_block, threads_per_block * sizeof(double)>>>(score, rows);

      cudaMemcpy(K, input.value.values().data(), num_bytes, cudaMemcpyHostToDevice);

      dim3 output_block(16, 16);
      dim3 output_grid((cols + output_block.x - 1) / output_block.x,
                       (rows + output_block.y - 1) / output_block.y);

      output_kernel<<<output_grid, output_block>>>(score, K, Q, rows, cols);

      Matrix output(rows, cols);
      cudaMemcpy(output.values().data(), Q, num_bytes, cudaMemcpyDeviceToHost);

      cudaFree(Q);
      cudaFree(K);
      cudaFree(score);

      return output;
  }


ParallelRotaryEmbedding::ParallelRotaryEmbedding(double base, int threads_per_block) : base_(base), threads_per_block_(threads_per_block) {}

std::string ParallelRotaryEmbedding::name() const {
  return "parallel_rotary_embedding";
}

void ParallelRotaryEmbedding::apply_in_place(Matrix &matrix) const {

    double* gpu_in; 
    double* gpu_out;
    
    int size = matrix.size();
    int num_bytes = matrix.size() * sizeof(double);
    
    cudaMalloc(&gpu_in, num_bytes);
    cudaMalloc(&gpu_out, num_bytes);

    cudaMemcpy(gpu_in, matrix.values().data(), num_bytes, cudaMemcpyHostToDevice);

    int num_blocks = ((size / 2) + threads_per_block_ - 1) / threads_per_block_;

    rope_kernel<<<num_blocks, threads_per_block_>>>(
        gpu_in,
        gpu_out,
        base_,
        matrix.rows(),
        matrix.cols(),
        size
    );

    cudaMemcpy(matrix.values().data(), gpu_out, num_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_in);
    cudaFree(gpu_out);
    
}





} // namespace rope

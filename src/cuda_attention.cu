#include "rope/attention.hpp"
#include "rope/rotary_embedding.hpp"

#include <chrono>
#include <stdexcept>

#define THREADS_PER_BLK 128


namespace rope {

  Matrix cuda_scaled_dot_product_attention(const AttentionInput& input) {
    // allocate GPU memory
    // copy input.query, input.key, input.value to GPU
    // launch score kernel
    // launch softmax kernel
    // launch output kernel
    // copy result back
    double* Q; 
    double* K;
    double* score;
    
    int size = input.query.rows() * input.query.cols();
    int num_bytes = size * sizeof(double);

    int sizeScore = input.query.rows() * input.query.rows();
    int num_bytesScore = sizeScore * sizeof(double);
    
    cudaMalloc(&Q, num_bytes);
    cudaMalloc(&K, num_bytes);
    cudaMalloc(&score, num_bytesScore);

    cudaMemcpy(Q, input.query.values().data(), num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(K, input.key.values().data(), num_bytes, cudaMemcpyHostToDevice);

    dim3 block(THREADS_PER_BLK); // 128 threads
    dim3 grid((rows + block.x - 1) / block.x, rows);

    score_kernel<<<grid, block>>>(Q, K,score, input.query.rows(), input.query.cols());

    Matrix scoreOut;
    cudaMemcpy(scoreOut.values().data(), score, num_bytesScore, cudaMemcpyDeviceToHost);

    cudaFree(Q);
    cudaFree(K);
  }

ParallelRoPEAttention::ParallelRoPEAttention(std::shared_ptr<const RotaryEmbedding> rotary)
    : rotary_(std::move(rotary)) {
  if (!rotary_) {
    throw std::invalid_argument("ParallelRoPEAttention requires a rotary embedding");
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
  Matrix output = cuda_scaled_dot_product_attention(rotated);

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


ParallelRotaryEmbedding::ParallelRotaryEmbedding(double base) : base_(base) {}

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

    int num_blocks = ((size / 2) + THREADS_PER_BLK - 1) / THREADS_PER_BLK;

    rope_kernel<<<num_blocks,THREADS_PER_BLK>>>(
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

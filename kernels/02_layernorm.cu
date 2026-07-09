// LayerNorm — naive one-thread-per-row vs. one-warp-per-row with shuffle
// reductions.
//
// The naive kernel is correct and catastrophically parallel-poor: at T=128
// rows it launches 128 threads total, so ~99% of the GPU idles. The warp
// version gives each row 32 threads that cooperate through registers
// (__shfl_down_sync) — no shared memory needed for a d=192 row's reduction.

#include "common.cuh"

// One THREAD per row: a serial loop dressed up as a kernel. (The contrast.)
__global__ void layernorm_naive(const float* __restrict__ x, const float* __restrict__ g,
                                const float* __restrict__ b, float* __restrict__ out,
                                int T, int D, float eps) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* row = x + t * D;
  float mean = 0.f;
  for (int i = 0; i < D; i++) mean += row[i];
  mean /= D;
  float var = 0.f;
  for (int i = 0; i < D; i++) {
    float d = row[i] - mean;
    var += d * d;
  }
  var /= D;  // biased, like PyTorch
  float rstd = rsqrtf(var + eps);
  for (int i = 0; i < D; i++) out[t * D + i] = (row[i] - mean) * rstd * g[i] + b[i];
}

// Sum across the 32 lanes of a warp, in registers, log2(32)=5 steps.
__device__ inline float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}

// One WARP per row: lanes stride the 192 columns (6 elements each),
// then reduce through registers. blockDim = 32.
__global__ void layernorm_warp(const float* __restrict__ x, const float* __restrict__ g,
                               const float* __restrict__ b, float* __restrict__ out,
                               int T, int D, float eps) {
  int t = blockIdx.x;       // one block (= one warp) per row
  int lane = threadIdx.x;   // 0..31
  const float* row = x + t * D;

  float partial = 0.f;
  for (int i = lane; i < D; i += 32) partial += row[i];
  float mean = warp_sum(partial) / D;
  mean = __shfl_sync(0xffffffffu, mean, 0);  // broadcast lane 0's total

  float vpart = 0.f;
  for (int i = lane; i < D; i += 32) {
    float d = row[i] - mean;
    vpart += d * d;
  }
  float var = warp_sum(vpart) / D;
  var = __shfl_sync(0xffffffffu, var, 0);
  float rstd = rsqrtf(var + eps);

  for (int i = lane; i < D; i += 32)
    out[t * D + i] = (row[i] - mean) * rstd * g[i] + b[i];
}

void run_layernorm_bench(int T, int D) {
  std::vector<float> x(T * D), g(D, 1.f), b(D, 0.f), out(T * D), ref(T * D);
  fill_random(x, 3, 2.f);

  for (int t = 0; t < T; t++) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < D; i++) mean += x[t * D + i];
    mean /= D;
    for (int i = 0; i < D; i++) {
      float d = x[t * D + i] - mean;
      var += d * d;
    }
    var /= D;
    float rstd = 1.f / sqrtf(var + 1e-5f);
    for (int i = 0; i < D; i++) ref[t * D + i] = (x[t * D + i] - mean) * rstd;
  }

  float *d_x, *d_g, *d_b, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x, x.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_g, g.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_b, b.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_out, out.size() * 4));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * 4, cudaMemcpyHostToDevice));

  float ms_naive = time_ms([&] {
    layernorm_naive<<<(T + 127) / 128, 128>>>(d_x, d_g, d_b, d_out, T, D, 1e-5f);
  });
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
  float diff_naive = max_abs_diff(out, ref);

  float ms_warp = time_ms([&] { layernorm_warp<<<T, 32>>>(d_x, d_g, d_b, d_out, T, D, 1e-5f); });
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
  float diff_warp = max_abs_diff(out, ref);

  printf("  {\"kernel\": \"layernorm_naive\", \"ms\": %.5f, \"max_diff\": %g},\n", ms_naive, diff_naive);
  printf("  {\"kernel\": \"layernorm_warp\", \"ms\": %.5f, \"max_diff\": %g},\n", ms_warp, diff_warp);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
}

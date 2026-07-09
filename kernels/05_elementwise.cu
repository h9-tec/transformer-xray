// Elementwise ops — GELU and the residual add. These are pure memory-bound
// streams: the roofline says performance = bandwidth, so the only lever is
// moving fewer, wider transactions. The float4 variant loads 16 bytes per
// instruction instead of 4.

#include "common.cuh"

__device__ inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// Grid-stride loop: launch a fixed grid, let each thread walk the array.
// The same kernel works for any n without recomputing launch geometry.
__global__ void gelu_forward(const float* __restrict__ x, float* __restrict__ y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = gelu_tanh(x[i]);
}

// Same op, 4 floats per thread per iteration — one 16-byte transaction.
__global__ void gelu_forward_f4(const float4* __restrict__ x, float4* __restrict__ y, int n4) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4 v = x[i];
    y[i] = make_float4(gelu_tanh(v.x), gelu_tanh(v.y), gelu_tanh(v.z), gelu_tanh(v.w));
  }
}

__global__ void residual_add(const float* __restrict__ a, const float* __restrict__ b,
                             float* __restrict__ out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = a[i] + b[i];
}

void run_elementwise_bench(int n) {
  std::vector<float> x(n), y(n), ref(n);
  fill_random(x, 9, 3.f);
  for (int i = 0; i < n; i++) {
    float v = x[i];
    ref[i] = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
  }

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, n * 4));
  CUDA_CHECK(cudaMalloc(&d_y, n * 4));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice));

  int threads = 256, blocks = 256;
  float ms = time_ms([&] { gelu_forward<<<blocks, threads>>>(d_x, d_y, n); });
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * 4, cudaMemcpyDeviceToHost));
  float diff = max_abs_diff(y, ref);

  float ms4 = time_ms(
      [&] { gelu_forward_f4<<<blocks, threads>>>((const float4*)d_x, (float4*)d_y, n / 4); });
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * 4, cudaMemcpyDeviceToHost));
  float diff4 = max_abs_diff(y, ref);

  printf("  {\"kernel\": \"gelu_forward\", \"ms\": %.5f, \"max_diff\": %g},\n", ms, diff);
  printf("  {\"kernel\": \"gelu_forward_f4\", \"ms\": %.5f, \"max_diff\": %g},\n", ms4, diff4);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
}

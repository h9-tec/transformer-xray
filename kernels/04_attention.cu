// Causal attention for one head — scores, mask, softmax, and the weighted
// sum of V, fused into a single kernel: one thread block per query row.
//
// The causal mask costs nothing here: row i simply never loops past column i.
// That also shows the work imbalance the mask creates — row 0 does 1 column,
// row T−1 does T. (Real flash-attention adds online softmax + K/V tiling;
// this kernel keeps the whole row in shared memory, fine for T ≤ 1024.)

#include "common.cuh"

// q,k,v: [T, hd] for ONE head, row-major. out: [T, hd].
// grid: T blocks × 128 threads. smem: T floats of scores.
__global__ void attention_row(const float* __restrict__ q, const float* __restrict__ k,
                              const float* __restrict__ v, float* __restrict__ out,
                              int T, int hd, float scale) {
  extern __shared__ float s[];  // scores for this query row: T floats
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // 1. scores: s[j] = (q[row]·k[j]) · scale for j ≤ row (causal: skip the rest)
  for (int j = tid; j <= row; j += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < hd; i++) acc += q[row * hd + i] * k[j * hd + i];
    s[j] = acc * scale;
  }
  __syncthreads();

  // 2. softmax over s[0..row] — max-subtraction, single thread for clarity
  //    (a warp reduction here is the reader's exercise; row ≤ 256 makes
  //    the serial pass cheap relative to step 1 and 3).
  if (tid == 0) {
    float mx = -1e30f;
    for (int j = 0; j <= row; j++) mx = fmaxf(mx, s[j]);
    float sum = 0.f;
    for (int j = 0; j <= row; j++) {
      s[j] = expf(s[j] - mx);
      sum += s[j];
    }
    float inv = 1.f / sum;
    for (int j = 0; j <= row; j++) s[j] *= inv;
  }
  __syncthreads();

  // 3. out[row] = Σ_j s[j] · v[j] — threads split the hd output features
  for (int i = tid; i < hd; i += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j <= row; j++) acc += s[j] * v[j * hd + i];
    out[row * hd + i] = acc;
  }
}

void run_attention_bench(int T, int hd) {
  std::vector<float> q(T * hd), k(T * hd), v(T * hd), out(T * hd), ref(T * hd, 0.f);
  fill_random(q, 6, 1.f);
  fill_random(k, 7, 1.f);
  fill_random(v, 8, 1.f);
  float scale = 1.f / sqrtf((float)hd);

  std::vector<float> s(T);
  for (int r = 0; r < T; r++) {
    float mx = -1e30f;
    for (int j = 0; j <= r; j++) {
      float acc = 0.f;
      for (int i = 0; i < hd; i++) acc += q[r * hd + i] * k[j * hd + i];
      s[j] = acc * scale;
      mx = fmaxf(mx, s[j]);
    }
    float sum = 0.f;
    for (int j = 0; j <= r; j++) {
      s[j] = expf(s[j] - mx);
      sum += s[j];
    }
    for (int j = 0; j <= r; j++)
      for (int i = 0; i < hd; i++) ref[r * hd + i] += (s[j] / sum) * v[j * hd + i];
  }

  float *d_q, *d_k, *d_v, *d_out;
  CUDA_CHECK(cudaMalloc(&d_q, q.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_k, k.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_v, v.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_out, out.size() * 4));
  CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice));

  float ms = time_ms(
      [&] { attention_row<<<T, 128, T * sizeof(float)>>>(d_q, d_k, d_v, d_out, T, hd, scale); });
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));

  printf("  {\"kernel\": \"attention_row\", \"T\": %d, \"ms\": %.5f, \"max_diff\": %g},\n", T, ms,
         max_abs_diff(out, ref));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_out));
}

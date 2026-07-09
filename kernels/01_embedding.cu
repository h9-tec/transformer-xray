// Embedding lookup — forward gather, backward scatter-add.
//
// The forward is zero FLOPs: pure memory traffic. One thread per output
// element; consecutive threads read consecutive floats of a wte row, so
// every warp issues one coalesced 128-byte transaction.
//
// The backward is the atomics lesson: two positions holding the same token
// must ACCUMULATE into the same wte row. Plain writes would race and lose
// updates — atomicAdd serializes only the collisions.

#include "common.cuh"

// out[t, i] = wte[ids[t], i]           grid: ceil(T*D/256) blocks × 256 threads
__global__ void embedding_gather(const float* __restrict__ wte,
                                 const int* __restrict__ ids,
                                 float* __restrict__ out, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx / D;   // which position
  int i = idx % D;   // which feature
  out[idx] = wte[ids[t] * D + i];
}

// dWte[ids[t], i] += dOut[t, i]        the gather's transpose is a scatter
__global__ void embedding_scatter_add(const float* __restrict__ dOut,
                                      const int* __restrict__ ids,
                                      float* __restrict__ dWte, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx / D;
  int i = idx % D;
  // Two rows of dOut with the same token id target the same dWte row:
  // without atomics this is a lost-update race.
  atomicAdd(&dWte[ids[t] * D + i], dOut[idx]);
}

void run_embedding_bench(int T, int D, int V) {
  std::vector<float> wte(V * D), dOut(T * D), out(T * D), ref(T * D);
  std::vector<int> ids(T);
  fill_random(wte, 1, 0.02f);
  fill_random(dOut, 2, 1.f);
  for (int t = 0; t < T; t++) ids[t] = (t * 7 + 3) % V;

  for (int t = 0; t < T; t++)
    for (int i = 0; i < D; i++) ref[t * D + i] = wte[ids[t] * D + i];

  float *d_wte, *d_out;
  int* d_ids;
  CUDA_CHECK(cudaMalloc(&d_wte, wte.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_out, out.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_ids, ids.size() * 4));
  CUDA_CHECK(cudaMemcpy(d_wte, wte.data(), wte.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));

  int threads = 256, blocks = (T * D + threads - 1) / threads;
  auto launch = [&] { embedding_gather<<<blocks, threads>>>(d_wte, d_ids, d_out, T, D); };
  float ms = time_ms(launch);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));

  printf("  {\"kernel\": \"embedding_gather\", \"ms\": %.5f, \"max_diff\": %g},\n", ms,
         max_abs_diff(out, ref));

  CUDA_CHECK(cudaFree(d_wte));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_ids));
}

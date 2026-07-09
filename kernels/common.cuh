// Shared plumbing for the teaching kernels: error checking, event timing,
// and CPU-reference verification. Every kernel in this repo is verified
// against a plain CPU loop before it is timed.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Time a kernel launch (milliseconds), averaged over `iters` runs after one warmup.
template <typename F>
float time_ms(F launch, int iters = 50) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch();  // warmup + JIT
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); i++) m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

// Deterministic pseudo-random fill (mulberry32, same generator as the TS engine).
inline void fill_random(std::vector<float>& v, unsigned seed, float scale = 1.f) {
  unsigned a = seed;
  for (auto& x : v) {
    a += 0x6D2B79F5u;
    unsigned t = a;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + (t ^ (t >> 7)) * (t | 61u);
    unsigned r = t ^ (t >> 14);
    x = (r / 4294967296.0f - 0.5f) * 2.f * scale;
  }
}

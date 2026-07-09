// Verifies every teaching kernel against a CPU reference and times it,
// at THE MODEL'S REAL SHAPES (L=4, d=192, h=6, hd=32, V=65, T=128).
// Data is deterministic pseudo-random (same mulberry32 as the TS engine);
// kernel behavior does not depend on the weight values.
//
// Output is JSON — commit it to kernels/results/<gpu>.json and the web GPU
// panel can display it, labeled as measured on that GPU.

#include "common.cuh"

void run_embedding_bench(int T, int D, int V);
void run_layernorm_bench(int T, int D);
void run_matmul_bench(int M, int K, int N);
void run_attention_bench(int T, int hd);
void run_elementwise_bench(int n);

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int T = 128, D = 192, V = 65, HD = 32;

  printf("{\n\"gpu\": \"%s\",\n\"results\": [\n", prop.name);
  run_embedding_bench(T, D, V);
  run_layernorm_bench(T, D);
  run_matmul_bench(T, D, 3 * D);   // the QKV projection
  run_matmul_bench(T, D, 4 * D);   // the MLP up-projection
  run_attention_bench(T, HD);
  run_elementwise_bench(T * 4 * D);
  printf("  {\"kernel\": \"end\", \"ms\": 0, \"max_diff\": 0}\n]\n}\n");
  return 0;
}

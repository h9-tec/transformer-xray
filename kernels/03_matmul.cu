// Matmul — the op that IS the transformer (QKV, attention proj, MLP, logits
// are all this kernel with different shapes).
//
// naive:  one thread per output element, all reads from global memory.
//         Each C[i,j] re-reads a full row of A and column of B → for
//         [128×192]·[192×576] that's ~113 MB of traffic for a 28 MFLOP job.
// tiled:  each 32×32 thread block stages tiles of A and B in shared memory;
//         every loaded float is reused 32 times → traffic drops ~32×.
//         This is the whole idea behind every fast GEMM, cuBLAS included.

#include "common.cuh"

#define TILE 32

// C[i,j] = Σ_k A[i,k] · B[k,j]
__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
  if (i >= M || j >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
  C[i * N + j] = acc;
}

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int K, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int j = blockIdx.x * TILE + threadIdx.x;
  int i = blockIdx.y * TILE + threadIdx.y;
  float acc = 0.f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    // Cooperative load: each thread stages ONE element of each tile.
    // Both loads are coalesced (threadIdx.x walks contiguous memory).
    As[threadIdx.y][threadIdx.x] =
        (i < M && k0 + threadIdx.x < K) ? A[i * K + k0 + threadIdx.x] : 0.f;
    Bs[threadIdx.y][threadIdx.x] =
        (k0 + threadIdx.y < K && j < N) ? B[(k0 + threadIdx.y) * N + j] : 0.f;
    __syncthreads();  // tile must be complete before anyone reads it

    for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();  // done reading before the next overwrite
  }
  if (i < M && j < N) C[i * N + j] = acc;
}

void run_matmul_bench(int M, int K, int N) {
  std::vector<float> A(M * K), B(K * N), C(M * N), ref(M * N, 0.f);
  fill_random(A, 4, 1.f);
  fill_random(B, 5, 0.1f);

  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
      float a = A[i * K + k];
      for (int j = 0; j < N; j++) ref[i * N + j] += a * B[k * N + j];
    }

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, A.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_B, B.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_C, C.size() * 4));
  CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * 4, cudaMemcpyHostToDevice));

  dim3 threads(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  float ms_naive = time_ms([&] { matmul_naive<<<grid, threads>>>(d_A, d_B, d_C, M, K, N); });
  CUDA_CHECK(cudaMemcpy(C.data(), d_C, C.size() * 4, cudaMemcpyDeviceToHost));
  float diff_naive = max_abs_diff(C, ref);

  float ms_tiled = time_ms([&] { matmul_tiled<<<grid, threads>>>(d_A, d_B, d_C, M, K, N); });
  CUDA_CHECK(cudaMemcpy(C.data(), d_C, C.size() * 4, cudaMemcpyDeviceToHost));
  float diff_tiled = max_abs_diff(C, ref);

  printf("  {\"kernel\": \"matmul_naive\", \"shape\": \"%dx%dx%d\", \"ms\": %.5f, \"max_diff\": %g},\n",
         M, K, N, ms_naive, diff_naive);
  printf("  {\"kernel\": \"matmul_tiled\", \"shape\": \"%dx%dx%d\", \"ms\": %.5f, \"max_diff\": %g},\n",
         M, K, N, ms_tiled, diff_tiled);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

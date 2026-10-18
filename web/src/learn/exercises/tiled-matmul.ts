import { buf, int, noRaces, rand, trafficAtMost, type Exercise } from '../exercise';

const M = 64;
const K = 64;
const N = 64;

// Naive traffic at 64³ as the interpreter counts it: per warp per k, one
// A sector (broadcast) + four B sectors; plus the C stores.
const NAIVE_BYTES = ((M * N) / 32) * K * 5 * 32 + ((M * N) / 32) * 4 * 32;

export const tiledMatmul: Exercise = {
  slug: 'tiled-matmul',
  kernel: 'matmul',
  launch: { grid: [N / 32, M / 32, 1], block: [32, 32, 1] },
  starter: `// C = A · B, one thread per output. Every thread re-reads a whole row
// of A and a whole column of B from global memory. Stage 32×32 tiles of
// A and B in shared memory so each loaded float is reused 32 times.
#define TILE 32

__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
  if (i >= M || j >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
  C[i * N + j] = acc;
}
`,
  solution: `#define TILE 32

__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int j = blockIdx.x * TILE + threadIdx.x;
  int i = blockIdx.y * TILE + threadIdx.y;
  float acc = 0.f;

  for (int k0 = 0; k0 < K; k0 += TILE) {
    // Cooperative load: each thread stages ONE element of each tile.
    As[threadIdx.y][threadIdx.x] = (i < M && k0 + threadIdx.x < K) ? A[i * K + k0 + threadIdx.x] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < K && j < N) ? B[(k0 + threadIdx.y) * N + j] : 0.f;
    __syncthreads();  // tile must be complete before anyone reads it

    for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();  // done reading before the next overwrite
  }
  if (i < M && j < N) C[i * N + j] = acc;
}
`,
  build: (seed) => {
    const A = rand(M * K, seed, 1);
    const B = rand(K * N, seed + 1, 0.1);
    const C = new Float32Array(M * N);
    const ref = new Float32Array(M * N);
    for (let i = 0; i < M; i++)
      for (let k = 0; k < K; k++) {
        const a = A[i * K + k];
        for (let j = 0; j < N; j++) ref[i * N + j] = Math.fround(ref[i * N + j] + Math.fround(a * B[k * N + j]));
      }
    return { args: [buf('A', A), buf('B', B), buf('C', C), int(M), int(K), int(N)], outputs: [2], reference: [ref], tolerance: 1e-4 };
  },
  targets: [trafficAtMost(Math.round(NAIVE_BYTES / 12), 'at least 12× less than the naive kernel'), noRaces()],
  hint: ({ stats, races }) => {
    if (races.length) return `Race between lines ${races[0].firstLine} and ${races[0].secondLine}: a tile is overwritten or read before everyone is done. Two barriers per k-step.`;
    if (stats.globalBytes > NAIVE_BYTES / 12) return 'Still reading A and B straight from global memory. Load a TILE×TILE block of each into __shared__ arrays first.';
    return null;
  },
};

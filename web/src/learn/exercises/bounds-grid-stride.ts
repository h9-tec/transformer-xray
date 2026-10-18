import { buf, int, rand, type Exercise } from '../exercise';

const N = 5000; // not a multiple of the grid's 1024 threads: the point of the lesson

export const boundsGridStride: Exercise = {
  slug: 'bounds-grid-stride',
  kernel: 'residual_add',
  launch: { grid: [4, 1, 1], block: [256, 1, 1] },
  starter: `// The grid has 4 blocks × 256 threads = 1024 threads, but n = 5000.
// Make every element get added, with a fixed grid, for ANY n.
__global__ void residual_add(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = a[i] + b[i]; // TODO: this covers 1024 elements, then stops
}
`,
  solution: `__global__ void residual_add(const float* a, const float* b, float* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = a[i] + b[i];
}
`,
  build: (seed) => {
    const a = rand(N, seed, 1);
    const b = rand(N, seed + 1, 1);
    const out = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let i = 0; i < N; i++) ref[i] = Math.fround(a[i] + b[i]);
    return { args: [buf('a', a), buf('b', b), buf('out', out), int(N)], outputs: [2], reference: [ref], tolerance: 0 };
  },
  targets: [],
  hint: ({ firstMismatch }) =>
    firstMismatch && firstMismatch.index >= 1024
      ? `out[${firstMismatch.index}] was never touched: only 1024 threads exist. Loop with a stride of gridDim.x * blockDim.x.`
      : null,
};

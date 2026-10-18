import { buf, int, rand, type Exercise } from '../exercise';

const N = 4096;

export const helloThread: Exercise = {
  slug: 'hello-thread',
  kernel: 'vector_add',
  launch: { grid: [16, 1, 1], block: [256, 1, 1] },
  starter: `// Each thread owns ONE element of y. Which one?
// Hint: blockIdx.x counts blocks, blockDim.x is threads per block,
// threadIdx.x is this thread's slot inside its block.
__global__ void vector_add(const float* a, const float* b, float* y, int n) {
  int i = 0; // TODO: the global index of this thread
  y[i] = a[i] + b[i];
}
`,
  solution: `__global__ void vector_add(const float* a, const float* b, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}
`,
  build: (seed) => {
    const a = rand(N, seed, 1);
    const b = rand(N, seed + 1, 1);
    const y = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let i = 0; i < N; i++) ref[i] = Math.fround(a[i] + b[i]);
    return { args: [buf('a', a), buf('b', b), buf('y', y), int(N)], outputs: [2], reference: [ref], tolerance: 0 };
  },
  targets: [],
  hint: ({ firstMismatch }) =>
    firstMismatch && firstMismatch.index > 0 && firstMismatch.want !== 0
      ? `y[${firstMismatch.index}] was never written. Every thread computed the same index.`
      : null,
};

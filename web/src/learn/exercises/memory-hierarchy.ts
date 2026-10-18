import { buf, int, rand, requestsAtMost, type Exercise } from '../exercise';

const N = 4096;

export const memoryHierarchy: Exercise = {
  slug: 'memory-hierarchy',
  kernel: 'copy',
  launch: { grid: [16, 1, 1], block: [256, 1, 1] },
  starter: `// A copy is the purest memory-bound kernel: zero math, all traffic.
// This version moves 4 bytes per lane per instruction: 128 B per warp.
// Move 16 bytes per lane instead (float4), so each warp request is 512 B.
__global__ void copy(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}
`,
  solution: `__global__ void copy(const float4* x, float4* y, int n) {
  int n4 = n / 4;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) y[i] = x[i];
}
`,
  build: (seed) => {
    const x = rand(N, seed, 3);
    const y = new Float32Array(N);
    return { args: [buf('x', x), buf('y', y), int(N)], outputs: [1], reference: [Float32Array.from(x)], tolerance: 0 };
  },
  targets: [requestsAtMost(64, 'one 16-byte load and one 16-byte store per lane')],
  hint: ({ stats }) =>
    stats.globalRequests > 64
      ? `${stats.globalRequests} requests: each lane is still moving one float. Declare the pointers as float4* and copy n / 4 vectors.`
      : null,
};

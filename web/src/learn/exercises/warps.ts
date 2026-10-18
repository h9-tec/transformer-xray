import { buf, int, noDivergence, rand, type Exercise } from '../exercise';

const N = 1024;

export const warps: Exercise = {
  slug: 'warps',
  kernel: 'odd_even',
  launch: { grid: [4, 1, 1], block: [256, 1, 1] },
  starter: `// Odd elements get doubled, even elements get +1.
// This version splits every warp in two: lanes 0,2,4,… take one path,
// lanes 1,3,5,… the other, and the hardware runs both, one after the other.
// Compute the same result without a divergent branch.
__global__ void odd_even(const float* a, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i % 2) {
    y[i] = a[i] * 2.f;
  } else {
    y[i] = a[i] + 1.f;
  }
}
`,
  solution: `__global__ void odd_even(const float* a, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // A select, not a branch: both values are computed, one is kept.
  float odd = (float)(i & 1);
  y[i] = odd * (a[i] * 2.f) + (1.f - odd) * (a[i] + 1.f);
}
`,
  build: (seed) => {
    const a = rand(N, seed, 1);
    const y = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let i = 0; i < N; i++) ref[i] = i % 2 ? Math.fround(a[i] * 2) : Math.fround(a[i] + 1);
    return { args: [buf('a', a), buf('y', y), int(N)], outputs: [1], reference: [ref], tolerance: 0 };
  },
  targets: [noDivergence(0)],
  hint: ({ stats }) =>
    stats.divergentInstructions > 0
      ? 'The result is right but the warp still splits. Replace the if/else with arithmetic or a ?: select so every lane executes the same instructions.'
      : null,
};

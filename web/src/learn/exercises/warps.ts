import { buf, int, noDivergence, rand, type Exercise } from '../exercise';

const N = 1024;

// Two genuinely different, loop-heavy arms, so a split warp really does run
// both one after the other (a compiler cannot fold these into a select).
function oddArm(v: number): number {
  let x = v;
  for (let k = 0; k < 8; k++) x = Math.fround(Math.fround(x * 0.5) + 1);
  return x;
}
function evenArm(v: number): number {
  let x = v;
  for (let k = 0; k < 8; k++) x = Math.fround(Math.fround(x * x) * 0.25 - 0.5);
  return x;
}

export const warps: Exercise = {
  slug: 'warps',
  kernel: 'odd_even',
  launch: { grid: [4, 1, 1], block: [256, 1, 1] },
  starter: `// Odd elements run one 8-step recurrence, even elements another.
// Thread i handles element i, so every warp holds 16 odd and 16 even
// lanes and runs BOTH loops, half the lanes idle each time.
// Keep the math and the output identical, but regroup the work so each
// warp's 32 lanes all take the same arm. Hint: who says thread i must
// handle element i?
__global__ void odd_even(const float* a, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int j = i;                      // the element this thread handles
  float x = a[j];
  if (j % 2) {
    for (int k = 0; k < 8; k++) x = x * 0.5f + 1.f;
  } else {
    for (int k = 0; k < 8; k++) x = x * x * 0.25f - 0.5f;
  }
  y[j] = x;
}
`,
  solution: `__global__ void odd_even(const float* a, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // Threads 0..n/2-1 take the even elements, n/2..n-1 the odd ones:
  // the parity is now uniform across every warp, so no lane idles.
  int half = n / 2;
  int j = (i < half) ? 2 * i : 2 * (i - half) + 1;
  float x = a[j];
  if (j % 2) {
    for (int k = 0; k < 8; k++) x = x * 0.5f + 1.f;
  } else {
    for (int k = 0; k < 8; k++) x = x * x * 0.25f - 0.5f;
  }
  y[j] = x;
}
`,
  build: (seed) => {
    const a = rand(N, seed, 1);
    const y = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let j = 0; j < N; j++) ref[j] = j % 2 ? oddArm(a[j]) : evenArm(a[j]);
    return { args: [buf('a', a), buf('y', y), int(N)], outputs: [1], reference: [ref], tolerance: 1e-6 };
  },
  targets: [noDivergence(0)],
  hint: ({ stats }) =>
    stats.divergentInstructions > 0
      ? 'The result is right but warps still split: lanes in one warp disagree on j % 2. Change which element each thread handles so a warp sees only odd or only even j.'
      : null,
};

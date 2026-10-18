import { buf, int, noRaces, rand, type Exercise } from '../exercise';

const N = 256;
const B = 64;

export const sharedMemoryBarriers: Exercise = {
  slug: 'shared-memory-barriers',
  kernel: 'reverse_blocks',
  launch: { grid: [N / B, 1, 1], block: [B, 1, 1] },
  starter: `// Reverse each 64-element block: y[base + t] = x[base + 63 - t],
// but through shared memory, the way a tile is staged.
// Thread t writes s[t]; thread 63-t reads it. Nothing guarantees the
// write has happened when the read runs. Add what is missing.
__global__ void reverse_blocks(const float* x, float* y, int n) {
  __shared__ float s[64];
  int t = threadIdx.x;
  int base = blockIdx.x * blockDim.x;
  s[t] = x[base + t];
  y[base + t] = s[blockDim.x - 1 - t];
}
`,
  solution: `__global__ void reverse_blocks(const float* x, float* y, int n) {
  __shared__ float s[64];
  int t = threadIdx.x;
  int base = blockIdx.x * blockDim.x;
  s[t] = x[base + t];
  __syncthreads();  // every s[] write is visible before any thread reads
  y[base + t] = s[blockDim.x - 1 - t];
}
`,
  build: (seed) => {
    const x = rand(N, seed, 1);
    const y = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let b = 0; b < N / B; b++) for (let t = 0; t < B; t++) ref[b * B + t] = x[b * B + (B - 1 - t)];
    return { args: [buf('x', x), buf('y', y), int(N)], outputs: [1], reference: [ref], tolerance: 0 };
  },
  targets: [noRaces()],
  hint: ({ races }) =>
    races.length
      ? `Line ${races[0].secondLine} reads shared memory that line ${races[0].firstLine} writes from another thread, with no barrier between them.`
      : null,
};

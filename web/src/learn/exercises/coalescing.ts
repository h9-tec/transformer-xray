import { buf, coalesced, int, rand, type Exercise } from '../exercise';

const T = 16;
const D = 64;
const V = 65;

export const coalescing: Exercise = {
  slug: 'coalescing',
  kernel: 'embedding_gather',
  launch: { grid: [Math.ceil((T * D) / 256), 1, 1], block: [256, 1, 1] },
  starter: `// out[t, i] = wte[ids[t], i]. This version is CORRECT and slow:
// consecutive lanes get consecutive t, so each lane reads a different
// row of wte, 256 bytes apart. One warp load touches 32 sectors.
// Re-split idx so consecutive lanes read consecutive floats.
__global__ void embedding_gather(const float* wte, const int* ids, float* out, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx % T;   // position varies fastest across lanes
  int i = idx / T;   // feature
  out[t * D + i] = wte[ids[t] * D + i];
}
`,
  solution: `__global__ void embedding_gather(const float* wte, const int* ids, float* out, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx / D;   // which position
  int i = idx % D;   // which feature: varies fastest across lanes
  out[idx] = wte[ids[t] * D + i];
}
`,
  build: (seed) => {
    const wte = rand(V * D, seed, 0.02);
    const ids = Int32Array.from({ length: T }, (_, i) => (i * 7 + seed) % V);
    const out = new Float32Array(T * D);
    const ref = new Float32Array(T * D);
    for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) ref[t * D + i] = wte[ids[t] * D + i];
    return { args: [buf('wte', wte), buf('ids', ids), buf('out', out), int(T), int(D)], outputs: [2], reference: [ref], tolerance: 0 };
  },
  targets: [coalesced(4)],
  hint: ({ stats, maxDiff }) =>
    maxDiff === 0 && stats.globalSectors / stats.globalRequests > 4
      ? 'Correct, but lanes stride across rows. Make the feature index i = idx % D the one that changes between neighbouring lanes.'
      : null,
};

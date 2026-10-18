import { buf, int, noRaces, rand, type Exercise } from '../exercise';

const T = 16;
const D = 64;
const V = 65;

export const atomics: Exercise = {
  slug: 'atomics',
  kernel: 'embedding_scatter_add',
  launch: { grid: [Math.ceil((T * D) / 256), 1, 1], block: [256, 1, 1] },
  starter: `// The embedding backward: dWte[ids[t], i] += dOut[t, i].
// Several positions t hold the SAME token, so several threads add into
// the same dWte element. A plain += is read-modify-write: two threads
// read the old value, both add, one update is lost. On this simulator
// the blocks happen to run one after another, so the sum may even look
// right; the race detector still sees it. Make the add atomic.
__global__ void embedding_scatter_add(const float* dOut, const int* ids, float* dWte, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx / D;
  int i = idx % D;
  dWte[ids[t] * D + i] += dOut[idx];
}
`,
  solution: `__global__ void embedding_scatter_add(const float* dOut, const int* ids, float* dWte, int T, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * D) return;
  int t = idx / D;
  int i = idx % D;
  // Two rows of dOut with the same token id target the same dWte row:
  // atomicAdd serializes exactly the collisions and nothing else.
  atomicAdd(&dWte[ids[t] * D + i], dOut[idx]);
}
`,
  build: (seed) => {
    const dOut = rand(T * D, seed, 1);
    const ids = Int32Array.from({ length: T }, (_, i) => (i * 3 + seed) % 5); // heavy collisions
    const dWte = new Float32Array(V * D);
    const ref = new Float32Array(V * D);
    for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) ref[ids[t] * D + i] = Math.fround(ref[ids[t] * D + i] + dOut[t * D + i]);
    return { args: [buf('dOut', dOut), buf('ids', ids), buf('dWte', dWte), int(T), int(D)], outputs: [2], reference: [ref], tolerance: 1e-5 };
  },
  targets: [noRaces()],
  hint: ({ races }) =>
    races.length
      ? `Threads from different blocks write dWte at the same address (line ${races[0].firstLine}). On real hardware they run at the same time and updates are lost. Which builtin makes a read-modify-write indivisible?`
      : null,
};

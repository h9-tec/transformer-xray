import { buf, flt, int, noRaces, rand, type Exercise } from '../exercise';

const T = 16;
const HD = 32;

export const fusingAttention: Exercise = {
  slug: 'fusing-attention',
  kernel: 'attention_row',
  launch: { grid: [T, 1, 1], block: [128, 1, 1], sharedBytes: T * 4 },
  starter: `// Causal attention for one head, one block per query row:
//   1. scores s[j] = (q[row]·k[j]) · scale for j ≤ row
//   2. softmax over s[0..row]
//   3. out[row] = Σ_j s[j] · v[j]
// This version skips step 2 entirely and forgets the barrier after
// step 1, so step 3 reads scores other threads have not written yet.
// Add the softmax (max-subtract, exp, normalize) and the barriers.
__global__ void attention_row(const float* q, const float* k, const float* v, float* out,
                              int T, int hd, float scale) {
  extern __shared__ float s[];
  int row = blockIdx.x;
  int tid = threadIdx.x;

  for (int j = tid; j <= row; j += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < hd; i++) acc += q[row * hd + i] * k[j * hd + i];
    s[j] = acc * scale;
  }

  for (int i = tid; i < hd; i += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j <= row; j++) acc += s[j] * v[j * hd + i];
    out[row * hd + i] = acc;
  }
}
`,
  solution: `__global__ void attention_row(const float* q, const float* k, const float* v, float* out,
                              int T, int hd, float scale) {
  extern __shared__ float s[];  // scores for this query row: T floats
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // 1. scores: s[j] = (q[row]·k[j]) · scale for j ≤ row (causal: skip the rest)
  for (int j = tid; j <= row; j += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < hd; i++) acc += q[row * hd + i] * k[j * hd + i];
    s[j] = acc * scale;
  }
  __syncthreads();

  // 2. softmax over s[0..row], max-subtraction, single thread for clarity
  if (tid == 0) {
    float mx = -1e30f;
    for (int j = 0; j <= row; j++) mx = fmaxf(mx, s[j]);
    float sum = 0.f;
    for (int j = 0; j <= row; j++) {
      s[j] = expf(s[j] - mx);
      sum += s[j];
    }
    float inv = 1.f / sum;
    for (int j = 0; j <= row; j++) s[j] *= inv;
  }
  __syncthreads();

  // 3. out[row] = Σ_j s[j] · v[j], threads split the hd output features
  for (int i = tid; i < hd; i += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j <= row; j++) acc += s[j] * v[j * hd + i];
    out[row * hd + i] = acc;
  }
}
`,
  build: (seed) => {
    const q = rand(T * HD, seed, 1);
    const k = rand(T * HD, seed + 1, 1);
    const v = rand(T * HD, seed + 2, 1);
    const out = new Float32Array(T * HD);
    const scale = Math.fround(1 / Math.sqrt(HD));
    const ref = new Float32Array(T * HD);
    const s = new Float32Array(T);
    for (let r = 0; r < T; r++) {
      let mx = -Infinity;
      for (let j = 0; j <= r; j++) {
        let acc = 0;
        for (let i = 0; i < HD; i++) acc = Math.fround(acc + Math.fround(q[r * HD + i] * k[j * HD + i]));
        s[j] = Math.fround(acc * scale);
        mx = Math.max(mx, s[j]);
      }
      let sum = 0;
      for (let j = 0; j <= r; j++) {
        s[j] = Math.fround(Math.exp(s[j] - mx));
        sum = Math.fround(sum + s[j]);
      }
      const inv = Math.fround(1 / sum);
      for (let j = 0; j <= r; j++) s[j] = Math.fround(s[j] * inv);
      for (let i = 0; i < HD; i++) {
        let acc = 0;
        for (let j = 0; j <= r; j++) acc = Math.fround(acc + Math.fround(s[j] * v[j * HD + i]));
        ref[r * HD + i] = acc;
      }
    }
    return {
      args: [buf('q', q), buf('k', k), buf('v', v), buf('out', out), int(T), int(HD), flt(scale)],
      outputs: [3],
      reference: [ref],
      tolerance: 1e-4,
    };
  },
  targets: [noRaces()],
  hint: ({ races, maxDiff }) => {
    if (races.length) return `Line ${races[0].secondLine} reads s[] while other threads may still be writing it on line ${races[0].firstLine}. A __syncthreads() between the stages.`;
    if (maxDiff > 1e-4) return 'The weights must be a softmax: subtract the row max, exponentiate, divide by the sum. Then barrier again before step 3.';
    return null;
  },
};

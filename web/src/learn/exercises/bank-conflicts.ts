import { buf, noBankConflicts, rand, type Exercise } from '../exercise';

const W = 32;

export const bankConflicts: Exercise = {
  slug: 'bank-conflicts',
  kernel: 'transpose32',
  launch: { grid: [1, 1, 1], block: [32, 32, 1] },
  starter: `// Transpose a 32×32 tile through shared memory.
// The write tile[ty][tx] is fine: lanes vary tx, consecutive words.
// The read tile[tx][ty] is not: lanes vary tx, so they hit words
// 0, 32, 64, … — all in bank 0. A 32-way conflict. Fix it with the
// smallest possible change to the declaration.
__global__ void transpose32(const float* in, float* out) {
  __shared__ float tile[32][32];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  tile[ty][tx] = in[ty * 32 + tx];
  __syncthreads();
  out[ty * 32 + tx] = tile[tx][ty];
}
`,
  solution: `__global__ void transpose32(const float* in, float* out) {
  __shared__ float tile[32][33];  // one column of padding shifts every row by one bank
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  tile[ty][tx] = in[ty * 32 + tx];
  __syncthreads();
  out[ty * 32 + tx] = tile[tx][ty];
}
`,
  build: (seed) => {
    const inp = rand(W * W, seed, 1);
    const out = new Float32Array(W * W);
    const ref = new Float32Array(W * W);
    for (let r = 0; r < W; r++) for (let c = 0; c < W; c++) ref[r * W + c] = inp[c * W + r];
    return { args: [buf('in', inp), buf('out', out)], outputs: [1], reference: [ref], tolerance: 0 };
  },
  targets: [noBankConflicts()],
  hint: ({ stats }) =>
    stats.sharedWorstConflict > 1
      ? `${stats.sharedWorstConflict}-way conflict: the column read lands every lane in the same bank. Pad each row so row r starts in bank r.`
      : null,
};

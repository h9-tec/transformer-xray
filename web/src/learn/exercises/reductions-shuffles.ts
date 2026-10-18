import { buf, flt, int, noDivergence, rand, type Exercise } from '../exercise';

const T = 16;
const D = 64;
const EPS = 1e-5;

export const reductionsShuffles: Exercise = {
  slug: 'reductions-shuffles',
  kernel: 'layernorm_warp',
  launch: { grid: [T, 1, 1], block: [32, 1, 1] },
  starter: `// One warp per row, but only lane 0 works: it loops over all 64
// columns three times while 31 lanes idle. Make the lanes cooperate:
// each lane sums a strided slice, then reduce across the warp with
// __shfl_down_sync, then broadcast with __shfl_sync.
__global__ void layernorm_warp(const float* x, const float* g, const float* b, float* out,
                               int T, int D, float eps) {
  int t = blockIdx.x;
  int lane = threadIdx.x;
  const float* row = x + t * D;
  if (lane == 0) {
    float mean = 0.f;
    for (int i = 0; i < D; i++) mean += row[i];
    mean /= D;
    float var = 0.f;
    for (int i = 0; i < D; i++) { float d = row[i] - mean; var += d * d; }
    var /= D;
    float rstd = rsqrtf(var + eps);
    for (int i = 0; i < D; i++) out[t * D + i] = (row[i] - mean) * rstd * g[i] + b[i];
  }
}
`,
  solution: `// Sum across the 32 lanes of a warp, in registers, log2(32)=5 steps.
__device__ inline float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}

__global__ void layernorm_warp(const float* x, const float* g, const float* b, float* out,
                               int T, int D, float eps) {
  int t = blockIdx.x;
  int lane = threadIdx.x;
  const float* row = x + t * D;

  float partial = 0.f;
  for (int i = lane; i < D; i += 32) partial += row[i];
  float mean = warp_sum(partial) / D;
  mean = __shfl_sync(0xffffffffu, mean, 0);  // broadcast lane 0's total

  float vpart = 0.f;
  for (int i = lane; i < D; i += 32) {
    float d = row[i] - mean;
    vpart += d * d;
  }
  float var = warp_sum(vpart) / D;
  var = __shfl_sync(0xffffffffu, var, 0);
  float rstd = rsqrtf(var + eps);

  for (int i = lane; i < D; i += 32)
    out[t * D + i] = (row[i] - mean) * rstd * g[i] + b[i];
}
`,
  build: (seed) => {
    const x = rand(T * D, seed, 2);
    const g = new Float32Array(D).fill(1);
    const b = new Float32Array(D);
    const out = new Float32Array(T * D);
    const ref = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      let mean = 0;
      for (let i = 0; i < D; i++) mean += x[t * D + i];
      mean /= D;
      let v = 0;
      for (let i = 0; i < D; i++) v += (x[t * D + i] - mean) ** 2;
      v /= D;
      const rstd = 1 / Math.sqrt(v + EPS);
      for (let i = 0; i < D; i++) ref[t * D + i] = (x[t * D + i] - mean) * rstd;
    }
    return {
      args: [buf('x', x), buf('g', g), buf('b', b), buf('out', out), int(T), int(D), flt(EPS)],
      outputs: [3],
      reference: [ref],
      tolerance: 1e-4,
    };
  },
  targets: [noDivergence(4)],
  hint: ({ stats }) =>
    stats.divergentInstructions > 4
      ? `${stats.divergentInstructions} warp instructions ran with most lanes idle. Let every lane sum its own slice (i = lane; i < D; i += 32) and combine with shuffles.`
      : null,
};

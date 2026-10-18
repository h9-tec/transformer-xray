import { buf, int, rand, trafficAtMost, type Exercise } from '../exercise';

const N = 4096;

function geluRef(x: number): number {
  const c = Math.fround(0.7978845608028654);
  const x3 = Math.fround(Math.fround(x * x) * x);
  const inner = Math.fround(c * Math.fround(x + Math.fround(0.044715 * x3)));
  return Math.fround(Math.fround(0.5 * x) * Math.fround(1 + Math.fround(Math.tanh(inner))));
}

// unfused: a, b read + tmp write + tmp read + y write = 20 B/elem; fused: 12 B/elem
const FUSED_BYTES = N * 12;

export const rooflineFloat4: Exercise = {
  slug: 'roofline-float4',
  kernel: 'gelu_residual',
  launch: { grid: [16, 1, 1], block: [256, 1, 1] },
  starter: `// y = gelu(a + b), written as two passes through global memory:
// the sum goes out to tmp and comes straight back in. That is 20 bytes
// per element for about 22 FLOPs: deep in the memory-bound region.
// Fuse the two passes so each element is read twice and written once.
__device__ inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_residual(const float* a, const float* b, float* tmp, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  tmp[i] = a[i] + b[i];          // pass 1: residual add, out to global
  y[i] = gelu_tanh(tmp[i]);      // pass 2: back in from global, then GELU
}
`,
  solution: `__device__ inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// One pass: the sum never leaves the register file.
__global__ void gelu_residual(const float* a, const float* b, float* tmp, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = a[i] + b[i];
  y[i] = gelu_tanh(s);
}
`,
  build: (seed) => {
    const a = rand(N, seed, 2);
    const b = rand(N, seed + 1, 2);
    const tmp = new Float32Array(N);
    const y = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let i = 0; i < N; i++) ref[i] = geluRef(Math.fround(a[i] + b[i]));
    return { args: [buf('a', a), buf('b', b), buf('tmp', tmp), buf('y', y), int(N)], outputs: [3], reference: [ref], tolerance: 1e-5 };
  },
  targets: [trafficAtMost(FUSED_BYTES + 1024, 'read a and b once, write y once')],
  hint: ({ stats }) =>
    stats.globalBytes > FUSED_BYTES + 1024
      ? `${(stats.globalBytes / N).toFixed(0)} bytes per element moved. The sum does not need to visit global memory at all: keep it in a local variable between the add and the GELU.`
      : null,
};

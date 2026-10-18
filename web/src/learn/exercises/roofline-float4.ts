import { buf, int, rand, requestsAtMost, type Exercise } from '../exercise';

const N = 4096;
// scalar: 4096/32 warps × (1 load + 1 store) = 256 requests; float4: 64
const F4_REQUESTS = (N / 4 / 32) * 2;

function geluRef(x: number): number {
  const c = Math.fround(0.7978845608028654);
  const x3 = Math.fround(Math.fround(x * x) * x);
  const inner = Math.fround(c * Math.fround(x + Math.fround(0.044715 * x3)));
  return Math.fround(Math.fround(0.5 * x) * Math.fround(1 + Math.fround(Math.tanh(inner))));
}

export const rooflineFloat4: Exercise = {
  slug: 'roofline-float4',
  kernel: 'gelu',
  launch: { grid: [256, 1, 1], block: [256, 1, 1] },
  starter: `// GELU is a pure stream: ~20 FLOPs per float, 8 bytes moved. The
// roofline says it is memory-bound, so the only lever is moving fewer,
// wider transactions. Rewrite the kernel to load and store float4:
// one 16-byte transaction per lane instead of four 4-byte ones.
__device__ inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__global__ void gelu(const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = gelu_tanh(x[i]);
}
`,
  solution: `__device__ inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// Same op, 4 floats per thread per iteration: one 16-byte transaction.
__global__ void gelu(const float4* x, float4* y, int n) {
  int n4 = n / 4;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4 v = x[i];
    y[i] = make_float4(gelu_tanh(v.x), gelu_tanh(v.y), gelu_tanh(v.z), gelu_tanh(v.w));
  }
}
`,
  build: (seed) => {
    const x = rand(N, seed, 3);
    const y = new Float32Array(N);
    const ref = new Float32Array(N);
    for (let i = 0; i < N; i++) ref[i] = geluRef(x[i]);
    return { args: [buf('x', x), buf('y', y), int(N)], outputs: [1], reference: [ref], tolerance: 1e-5 };
  },
  targets: [requestsAtMost(F4_REQUESTS + 8, '16-byte loads and stores')],
  hint: ({ stats }) =>
    stats.globalRequests > F4_REQUESTS + 8
      ? `${stats.globalRequests} requests. Each lane should move a float4: declare x and y as float4*, loop to n / 4, and use make_float4 for the store.`
      : null,
};

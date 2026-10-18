import { describe, expect, it } from 'vitest';
import { compileProgram, launch, type ArgValue, type LaunchOptions } from '@/cuda';

const f32 = (name: string, n: number | ArrayLike<number>): ArgValue => ({
  kind: 'buffer',
  name,
  data: typeof n === 'number' ? new Float32Array(n) : Float32Array.from(n),
});
const i32 = (name: string, n: number | ArrayLike<number>): ArgValue => ({
  kind: 'buffer',
  name,
  data: typeof n === 'number' ? new Int32Array(n) : Int32Array.from(n),
});
const int = (value: number): ArgValue => ({ kind: 'int', value });
const flt = (value: number): ArgValue => ({ kind: 'float', value });

function run(src: string, kernel: string, opts: Partial<LaunchOptions> & { args: ArgValue[] }) {
  return launch(compileProgram(src), kernel, {
    grid: [1, 1, 1],
    block: [32, 1, 1],
    ...opts,
  });
}

const data = (a: ArgValue) => (a as { data: Float32Array | Int32Array }).data;

describe('runtime core', () => {
  it('vector add over 4096 elements with 256-thread blocks', () => {
    const n = 4096;
    const a = f32('a', Array.from({ length: n }, (_, i) => i));
    const b = f32('b', Array.from({ length: n }, (_, i) => 2 * i));
    const y = f32('y', n);
    const r = run(
      `__global__ void add(const float* a, const float* b, float* y, int n) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < n) y[i] = a[i] + b[i];
       }`,
      'add',
      { grid: [16, 1, 1], block: [256, 1, 1], args: [a, b, y, int(n)] },
    );
    for (let i = 0; i < n; i += 97) expect(data(y)[i]).toBe(3 * i);
    expect(r.stats.warpInstructions).toBeGreaterThan(0);
  });

  it('divergent if/else reconverges and both branches ran', () => {
    const y = f32('y', 64);
    run(
      `__global__ void k(float* y) { int i = threadIdx.x; if (i % 2) y[i] = 1.f; else y[i] = 2.f; y[i] += 10.f; }`,
      'k',
      { block: [64, 1, 1], args: [y] },
    );
    expect(Array.from(data(y)).slice(0, 4)).toEqual([12, 11, 12, 11]);
  });

  it('loops with break and continue per lane', () => {
    const y = f32('y', 32);
    run(
      `__global__ void k(float* y) {
         int i = threadIdx.x; float c = 0.f;
         for (int j = 0; j < i; j++) { if (j == 10) break; if (j % 3 == 0) continue; c += 1.f; }
         y[i] = c;
       }`,
      'k',
      { args: [y] },
    );
    // count of j in [0, min(i,10)) with j%3 != 0
    const expectRow = Array.from({ length: 32 }, (_, i) => {
      let c = 0;
      for (let j = 0; j < Math.min(i, 10); j++) if (j % 3 !== 0) c++;
      return c;
    });
    expect(Array.from(data(y))).toEqual(expectRow);
  });

  it('early return before a barrier does not deadlock', () => {
    const y = f32('y', 64);
    run(
      `__global__ void k(float* y, int n) { int i = threadIdx.x; if (i >= n) return; __syncthreads(); y[i] = 1.f; }`,
      'k',
      { block: [64, 1, 1], args: [y, int(40)] },
    );
    expect(data(y)[39]).toBe(1);
    expect(data(y)[40]).toBe(0);
  });

  it('barrier under a partial mask is a BarrierDivergenceError with line', () => {
    expect(() =>
      run(`__global__ void k(float* y) {\n if (threadIdx.x < 16) __syncthreads();\n y[threadIdx.x] = 1.f; }`, 'k', {
        args: [f32('y', 32)],
      }),
    ).toThrowError(expect.objectContaining({ kind: 'barrier', line: 2 }));
  });

  it('shared-memory tile with two barriers (reverse within block)', () => {
    const x = f32('x', Array.from({ length: 64 }, (_, i) => i));
    const y = f32('y', 64);
    run(
      `__global__ void k(const float* x, float* y) {
         __shared__ float s[64];
         int t = threadIdx.x;
         s[t] = x[t];
         __syncthreads();
         y[t] = s[blockDim.x - 1 - t];
         __syncthreads();
       }`,
      'k',
      { block: [64, 1, 1], args: [x, y] },
    );
    expect(data(y)[0]).toBe(63);
    expect(data(y)[63]).toBe(0);
  });

  it('warp_sum via __shfl_down_sync sums 32 lanes', () => {
    const y = f32('y', 32);
    run(
      `__device__ inline float warp_sum(float v) {
         for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
         return v;
       }
       __global__ void k(float* y) { float v = (float)threadIdx.x; y[threadIdx.x] = warp_sum(v); }`,
      'k',
      { args: [y] },
    );
    expect(data(y)[0]).toBe(496);
  });

  it('__shfl_sync broadcasts lane 0', () => {
    const y = f32('y', 32);
    run(
      `__global__ void k(float* y) { float v = (float)threadIdx.x + 5.f; y[threadIdx.x] = __shfl_sync(0xffffffffu, v, 0); }`,
      'k',
      { args: [y] },
    );
    expect(Array.from(data(y)).every((v) => v === 5)).toBe(true);
  });

  it('atomicAdd on colliding addresses accumulates all contributions', () => {
    const out = f32('out', 4);
    run(`__global__ void k(float* out) { atomicAdd(&out[threadIdx.x % 4], 1.f); }`, 'k', {
      block: [256, 1, 1],
      args: [out],
    });
    expect(Array.from(data(out))).toEqual([64, 64, 64, 64]);
  });

  it('float4 loads and make_float4 stores', () => {
    const x = f32('x', Array.from({ length: 128 }, (_, i) => i));
    const y = f32('y', 128);
    run(
      `__global__ void k(const float4* x, float4* y, int n4) {
         int i = threadIdx.x;
         if (i < n4) { float4 v = x[i]; y[i] = make_float4(v.x * 2.f, v.y * 2.f, v.z * 2.f, v.w * 2.f); }
       }`,
      'k',
      { args: [x, y, int(32)] },
    );
    expect(data(y)[5]).toBe(10);
    expect(data(y)[127]).toBe(254);
  });

  it('fp32 semantics: fround sums, int division truncates, unsigned wraps', () => {
    const y = f32('y', 4);
    const z = i32('z', 4);
    run(
      `__global__ void k(float* y, int* z) {
         if (threadIdx.x == 0) {
           y[0] = 0.1f + 0.2f;
           z[0] = -7 / 2;
           z[1] = -7 % 3;
           unsigned u = 0xffffffffu; u += 2u; z[2] = (int)u;
           z[3] = (int)(1e9f * 4.f > 3e9f);
         }
       }`,
      'k',
      { args: [y, z] },
    );
    expect(data(y)[0]).toBe(Math.fround(Math.fround(0.1) + Math.fround(0.2)));
    expect(Array.from(data(z))).toEqual([-3, -1, 1, 1]);
  });

  it('out-of-bounds store is an OutOfBoundsError naming index, buffer and thread', () => {
    expect(() =>
      run(`__global__ void k(float* y) {\n int idx = blockIdx.x * blockDim.x + threadIdx.x;\n y[idx] = 1.f; }`, 'k', {
        block: [64, 1, 1],
        args: [f32('y', 40)],
      }),
    ).toThrowError(
      expect.objectContaining({
        kind: 'oob',
        line: 3,
        thread: { block: [0, 0, 0], thread: [40, 0, 0] },
        message: expect.stringMatching(/y\[40\]/),
      }),
    );
  });

  it('an infinite loop hits the instruction budget as TimeoutError with the line', () => {
    expect(() =>
      run(`__global__ void k(float* y) {\n while (true) { y[0] = 1.f; }\n }`, 'k', { args: [f32('y', 1)], budget: 100000 }),
    ).toThrowError(expect.objectContaining({ kind: 'timeout', line: 2 }));
  });

  it('printf captures per-lane lines with %d and %f', () => {
    const r = run(`__global__ void k(float* y) { printf("tid %d x %f\\n", threadIdx.x, y[threadIdx.x]); }`, 'k', {
      block: [2, 1, 1],
      args: [f32('y', [1.5, 2.5])],
    });
    expect(r.printf).toEqual(['tid 0 x 1.500000', 'tid 1 x 2.500000']);
  });

  it('device function calls (gelu_tanh) and nested calls f(f(x))', () => {
    const y = f32('y', 32);
    run(
      `__device__ inline float gelu_tanh(float x) {
         const float c = 0.7978845608028654f;
         return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
       }
       __device__ float twice(float x) { return 2.f * x; }
       __global__ void k(float* y) { y[threadIdx.x] = twice(twice((float)threadIdx.x)) + gelu_tanh(0.f); }`,
      'k',
      { args: [y] },
    );
    expect(data(y)[3]).toBe(12);
    expect(data(y)[0]).toBe(0);
  });

  it('float args and gridDim are visible', () => {
    const y = f32('y', 32);
    run(`__global__ void k(float* y, float s) { y[threadIdx.x] = s * (float)gridDim.x; }`, 'k', {
      grid: [3, 1, 1],
      args: [y, flt(0.5)],
    });
    expect(data(y)[0]).toBe(1.5);
  });
});

import { describe, expect, it } from 'vitest';
import { compileProgram, launch, type ArgValue } from '@/cuda';

const f32 = (name: string, n: number): ArgValue => ({ kind: 'buffer', name, data: new Float32Array(n) });
const i32 = (name: string, v: number[]): ArgValue => ({ kind: 'buffer', name, data: Int32Array.from(v) });

const oneWarp = (src: string, kernel: string, args: ArgValue[], extra: Record<string, unknown> = {}) =>
  launch(compileProgram(src), kernel, { grid: [1, 1, 1], block: [32, 1, 1], args, ...extra });

describe('instrumentation', () => {
  it('row read: 1 request, 4 sectors, 128 bytes', () => {
    const r = oneWarp(`__global__ void k(const float* x, float* y) { y[threadIdx.x] = x[threadIdx.x]; }`, 'k', [
      f32('x', 32),
      f32('y', 32),
    ]);
    expect(r.stats.globalRequests).toBe(2); // one load + one store
    expect(r.stats.globalSectors).toBe(8);
    expect(r.stats.globalBytes).toBe(256);
  });

  it('stride-192 column read: 32 sectors per request', () => {
    const r = oneWarp(`__global__ void k(const float* x, float* y) { y[threadIdx.x] = x[threadIdx.x * 192]; }`, 'k', [
      f32('x', 32 * 192),
      f32('y', 32),
    ]);
    expect(r.stats.globalSectors).toBe(32 + 4);
  });

  it('shared stride-32 access reports a 32-way conflict; +1 padding reports 1', () => {
    const bad = oneWarp(
      `__global__ void k(float* y) { __shared__ float s[32][32]; s[threadIdx.x][0] = 1.f; __syncthreads(); y[threadIdx.x] = s[threadIdx.x][0]; }`,
      'k',
      [f32('y', 32)],
    );
    expect(bad.stats.sharedWorstConflict).toBe(32);
    const good = oneWarp(
      `__global__ void k(float* y) { __shared__ float s[32][33]; s[threadIdx.x][0] = 1.f; __syncthreads(); y[threadIdx.x] = s[threadIdx.x][0]; }`,
      'k',
      [f32('y', 32)],
    );
    expect(good.stats.sharedWorstConflict).toBe(1);
    expect(good.stats.sharedRequests).toBe(2);
  });

  it('divergentInstructions counts instructions under a partial mask', () => {
    const r = oneWarp(`__global__ void k(float* y) { if (threadIdx.x < 16) y[threadIdx.x] = 1.f; }`, 'k', [f32('y', 32)]);
    expect(r.stats.divergentInstructions).toBeGreaterThanOrEqual(1);
    const full = oneWarp(`__global__ void k(float* y) { y[threadIdx.x] = 1.f; }`, 'k', [f32('y', 32)]);
    expect(full.stats.divergentInstructions).toBe(0);
  });

  it('missing barrier between shared store and cross-thread load is a race naming both lines', () => {
    const r = oneWarp(
      `__global__ void k(const float* x, float* y) {
         __shared__ float s[64];
         int t = threadIdx.x;
         s[t] = x[t];
         y[t] = s[63 - t];
       }`,
      'k',
      [f32('x', 64), f32('y', 64)],
      { block: [64, 1, 1] },
    );
    expect(r.races.length).toBeGreaterThan(0);
    expect(r.races[0]).toMatchObject({ kind: 'shared', firstLine: 4, secondLine: 5 });
  });

  it('the same kernel with __syncthreads has no races', () => {
    const r = oneWarp(
      `__global__ void k(const float* x, float* y) {
         __shared__ float s[64];
         int t = threadIdx.x;
         s[t] = x[t];
         __syncthreads();
         y[t] = s[63 - t];
       }`,
      'k',
      [f32('x', 64), f32('y', 64)],
      { block: [64, 1, 1] },
    );
    expect(r.races).toEqual([]);
  });

  it('scatter-add without atomics across blocks is a global race; with atomicAdd it is not', () => {
    const ids = i32('ids', [3, 3, 3, 3]);
    const bad = launch(
      compileProgram(`__global__ void k(const int* ids, float* out) { int t = blockIdx.x; out[ids[t]] += 1.f; }`),
      'k',
      { grid: [4, 1, 1], block: [1, 1, 1], args: [ids, f32('out', 8)] },
    );
    expect(bad.races.some((r) => r.kind === 'global' && r.buffer === 'out')).toBe(true);
    const good = launch(
      compileProgram(`__global__ void k(const int* ids, float* out) { int t = blockIdx.x; atomicAdd(&out[ids[t]], 1.f); }`),
      'k',
      { grid: [4, 1, 1], block: [1, 1, 1], args: [ids, f32('out', 8)] },
    );
    expect(good.races).toEqual([]);
    expect(good.stats.atomics).toBe(4);
  });

  it('instrument:false still returns correct results and zeroed memory stats', () => {
    const y = f32('y', 32);
    const r = oneWarp(`__global__ void k(float* y) { y[threadIdx.x] = 2.f; }`, 'k', [y], { instrument: false });
    expect((y as { data: Float32Array }).data[7]).toBe(2);
    expect(r.stats.globalRequests).toBe(0);
    expect(r.stats.warpInstructions).toBeGreaterThan(0);
  });
});

import { describe, expect, it } from 'vitest';
import { handle } from '@/cuda/worker/handler';
import type { CudaRequest } from '@/cuda/worker/protocol';

function req(source: string, extra: Partial<CudaRequest> = {}): CudaRequest {
  const a = Float32Array.from([1, 2, 3, 4]);
  const b = Float32Array.from([10, 20, 30, 40]);
  const y = new Float32Array(4);
  return {
    id: 1,
    type: 'run',
    source,
    kernel: 'add',
    grid: [1, 1, 1],
    block: [4, 1, 1],
    args: [
      { kind: 'buffer', name: 'a', dtype: 'f32', buffer: a.buffer },
      { kind: 'buffer', name: 'b', dtype: 'f32', buffer: b.buffer },
      { kind: 'buffer', name: 'y', dtype: 'f32', buffer: y.buffer },
      { kind: 'int', value: 4 },
    ],
    instrument: true,
    ...extra,
  };
}

const ADD = `__global__ void add(const float* a, const float* b, float* y, int n) {
  int i = threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}`;

describe('cuda worker handler', () => {
  it('runs a kernel and returns mutated buffers', () => {
    const res = handle(req(ADD));
    expect(res.type).toBe('done');
    if (res.type !== 'done') return;
    expect(Array.from(new Float32Array(res.buffers[2]))).toEqual([11, 22, 33, 44]);
    expect(res.result.stats.globalRequests).toBeGreaterThan(0);
  });

  it('turns a TimeoutError into an error response with line', () => {
    const res = handle(req(`__global__ void add(const float* a, const float* b, float* y, int n) {\n while (true) {}\n}`, { budget: 10000 }));
    expect(res.type).toBe('error');
    if (res.type !== 'error') return;
    expect(res.error).toMatchObject({ kind: 'timeout', line: 2 });
  });

  it('turns a parse error into an error response, never throws', () => {
    const res = handle(req('__global__ void add(float* y) { y[0] = ; }'));
    expect(res.type).toBe('error');
    if (res.type !== 'error') return;
    expect(res.error.kind).toBe('parse');
    expect(res.error.line).toBe(1);
  });
});

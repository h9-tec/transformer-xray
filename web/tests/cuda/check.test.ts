import { expect, it } from 'vitest';
import { check } from '@/cuda/check';
import { parse } from '@/cuda/parser';
import { FILES, teachingSource } from './helpers';

it('computes the shared-memory layout of matmul_tiled', () => {
  const k = check(parse(teachingSource('03_matmul.cu'))).kernels.get('matmul_tiled')!;
  expect(k.staticSharedBytes).toBe(2 * 32 * 32 * 4);
  expect(k.sharedLayout.get('Bs')?.offset).toBe(4096);
  expect(k.sharedLayout.get('As')?.dims).toEqual([32, 32]);
});

it('marks attention_row as using extern shared', () => {
  expect(check(parse(teachingSource('04_attention.cu'))).kernels.get('attention_row')!.usesExternShared).toBe(true);
});

it('all five kernels pass the checker', () => {
  for (const f of FILES) expect(() => check(parse(teachingSource(f))), f).not.toThrow();
});

it('undeclared identifier is a type error at the use site', () => {
  expect(() => check(parse('__global__ void k(float* y){ y[0] = zz; }'))).toThrowError(
    expect.objectContaining({ kind: 'type', line: 1, col: 37 }),
  );
});

it('recursion and local arrays are unsupported', () => {
  expect(() =>
    check(parse('__device__ int f(int n){ return f(n-1); } __global__ void k(){}')),
  ).toThrowError(expect.objectContaining({ kind: 'unsupported' }));
  expect(() => check(parse('__global__ void k(){ float t[8]; }'))).toThrowError(
    expect.objectContaining({ kind: 'unsupported' }),
  );
});

it('__syncthreads inside a device function and writes through const pointers are rejected', () => {
  expect(() =>
    check(parse('__device__ void f(){ __syncthreads(); } __global__ void k(){ f(); }')),
  ).toThrowError(expect.objectContaining({ kind: 'unsupported' }));
  expect(() => check(parse('__global__ void k(const float* x){ x[0] = 1.f; }'))).toThrowError(
    expect.objectContaining({ kind: 'type' }),
  );
});

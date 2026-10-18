import { expect, it } from 'vitest';
import { parse } from '@/cuda/parser';
import { FILES, teachingSource } from './helpers';

it('parses all five real kernels', () => {
  for (const f of FILES) {
    const p = parse(teachingSource(f));
    expect(p.functions.length, f).toBeGreaterThan(0);
  }
});

it('collects #define numeric constants and ignores #include', () => {
  expect(parse('#include "x.h"\n#define TILE 32\n__global__ void k(){}').defines.get('TILE')).toBe(32);
});

it('builds the right tree for a grid-stride loop', () => {
  const p = parse(
    '__global__ void k(float* y, int n){ for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) y[i] = 0.f; }',
  );
  const f = p.functions[0];
  const loop = f.body.stmts[0];
  expect(loop.kind).toBe('For');
  expect(f.params[0].type).toEqual({ base: 'float', pointer: 1, const: false });
});

it('parses shared 2-D arrays, extern shared, casts, ternary and member access', () => {
  const p = parse(
    '__global__ void k(const float4* x){ __shared__ float As[32][32]; extern __shared__ float s[]; float4 v = x[0]; float a = (v.x > 0.f) ? (float)1 : -v.y; }',
  );
  const d = p.functions[0].body.stmts[0];
  expect(d.kind === 'VarDecl' && d.shared && d.dims.length).toBe(2);
  const e = p.functions[0].body.stmts[1];
  expect(e.kind === 'VarDecl' && e.externShared).toBe(true);
});

it('rejects templates and references as unsupported with a position', () => {
  expect(() => parse('template <typename T> __global__ void k(T& x){}')).toThrowError(
    expect.objectContaining({ kind: 'unsupported', line: 1 }),
  );
});

it('a missing semicolon is reported on the line that lacks it', () => {
  expect(() => parse('__global__ void k(float* y) {\n int i = threadIdx.x\n y[i] = 1.f;\n}')).toThrowError(
    expect.objectContaining({ kind: 'parse', line: 2 }),
  );
});

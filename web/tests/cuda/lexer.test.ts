import { expect, it } from 'vitest';
import { lex } from '@/cuda/lexer';

it('lexes a kernel signature with qualifiers and pointers', () => {
  const t = lex('__global__ void k(const float* __restrict__ x, int n) {}');
  expect(t.map((x) => x.text).slice(0, 6)).toEqual(['__global__', 'void', 'k', '(', 'const', 'float']);
});

it('lexes numeric literals with CUDA suffixes', () => {
  const v = lex('0xffffffffu 1e-30f 0.044715f 16 .5').filter((t) => t.kind !== 'eof');
  expect(v.map((t) => [t.kind, t.value, !!t.unsigned])).toEqual([
    ['int', 4294967295, true],
    ['float', 1e-30, false],
    ['float', 0.044715, false],
    ['int', 16, false],
    ['float', 0.5, false],
  ]);
});

it('keeps preprocessor lines as one directive token and skips comments', () => {
  const t = lex('#include "common.cuh"\n// hi\n#define TILE 32 /* c */\nint x;');
  expect(t.filter((x) => x.kind === 'directive').map((x) => x.text)).toEqual([
    '#include "common.cuh"',
    '#define TILE 32',
  ]);
  expect(t.find((x) => x.text === 'int')?.line).toBe(4);
});

it('longest-match punctuation', () => {
  expect(
    lex('a >>= b << c->x')
      .filter((t) => t.kind === 'punct')
      .map((t) => t.text),
  ).toEqual(['>>=', '<<', '->']);
});

it('reports position on a bad character', () => {
  expect(() => lex('int x = 1;\n  @')).toThrowError(
    expect.objectContaining({ kind: 'parse', line: 2, col: 3 }),
  );
});

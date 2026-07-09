/** Hand-computed unit checks for every op — small enough to verify on paper. */

import { describe, expect, it } from 'vitest';
import { geluTanh, geluTanhBwd } from '@/engine/ops/gelu';
import { layernorm } from '@/engine/ops/layernorm';
import { causalMask } from '@/engine/ops/mask';
import { matmul, matmulTA, matmulTB } from '@/engine/ops/matmul';
import { topKFilter } from '@/engine/ops/sample';
import { softmaxRows } from '@/engine/ops/softmax';
import { fromData } from '@/engine/tensor';

const t = (shape: number[], vals: number[]) => fromData(Float32Array.from(vals), shape);

describe('matmul family', () => {
  //  A = [1 2 3; 4 5 6]  (2x3),  B = [7 8; 9 10; 11 12]  (3x2)
  const A = t([2, 3], [1, 2, 3, 4, 5, 6]);
  const B = t([3, 2], [7, 8, 9, 10, 11, 12]);

  it('matmul', () => {
    expect(Array.from(matmul(A, B).data)).toEqual([58, 64, 139, 154]);
  });

  it('matmulTB(A, Bᵀ-stored) equals matmul(A, B)', () => {
    const Bt = t([2, 3], [7, 9, 11, 8, 10, 12]); // B transposed, stored row-major
    expect(Array.from(matmulTB(A, Bt).data)).toEqual([58, 64, 139, 154]);
  });

  it('matmulTA(Aᵀ-stored, B) equals matmul(A, B)', () => {
    const At = t([3, 2], [1, 4, 2, 5, 3, 6]); // A transposed
    expect(Array.from(matmulTA(At, B).data)).toEqual([58, 64, 139, 154]);
  });
});

describe('layernorm', () => {
  it('normalizes with biased variance', () => {
    // row [1,2,3,4]: mean 2.5, biased var 1.25
    const g = t([4], [1, 1, 1, 1]);
    const b = t([4], [0, 0, 0, 0]);
    const { out, mean, rstd } = layernorm(t([1, 4], [1, 2, 3, 4]), g, b, 0);
    expect(mean[0]).toBeCloseTo(2.5, 6);
    expect(rstd[0]).toBeCloseTo(1 / Math.sqrt(1.25), 5);
    expect(out.data[0]).toBeCloseTo(-1.5 / Math.sqrt(1.25), 5);
    // output has mean 0, variance 1
    const s = out.data.reduce((a, v) => a + v, 0);
    expect(s).toBeCloseTo(0, 5);
  });
});

describe('softmax + mask', () => {
  it('rows sum to 1 and masked cells are exactly 0', () => {
    const masked = causalMask(t([1, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]));
    expect(masked.data[1]).toBe(-Infinity); // (0,1)
    const sm = softmaxRows(masked);
    for (let i = 0; i < 3; i++) {
      let sum = 0;
      for (let j = 0; j < 3; j++) sum += sm.data[i * 3 + j];
      expect(sum).toBeCloseTo(1, 6);
    }
    expect(sm.data[0 * 3 + 1]).toBe(0);
    expect(sm.data[0 * 3 + 2]).toBe(0);
    expect(sm.data[0]).toBe(1); // first row: only the diagonal survives
  });
});

describe('gelu (tanh approximation)', () => {
  it('matches reference values', () => {
    const x = t([1, 3], [-3, 0, 1]);
    const y = geluTanh(x);
    expect(y.data[0]).toBeCloseTo(-0.0036, 3); // gelu(-3) ≈ -0.00363
    expect(y.data[1]).toBe(0);
    expect(y.data[2]).toBeCloseTo(0.8412, 4); // gelu(1) ≈ 0.84119
  });

  it('gradient is 0.5 at x=0', () => {
    const dy = t([1, 1], [1]);
    expect(geluTanhBwd(dy, t([1, 1], [0])).data[0]).toBeCloseTo(0.5, 6);
  });
});

describe('top-k filter', () => {
  it('keeps the k largest and −∞s the rest', () => {
    const filtered = topKFilter(Float32Array.from([1, 5, 3, 2]), 2);
    expect(Array.from(filtered)).toEqual([-Infinity, 5, 3, -Infinity]);
  });

  it('k ≥ length is a no-op; k = 1 keeps only the max', () => {
    expect(Array.from(topKFilter(Float32Array.from([1, 2]), 10))).toEqual([1, 2]);
    expect(Array.from(topKFilter(Float32Array.from([1, 2]), 1))).toEqual([-Infinity, 2]);
  });
});

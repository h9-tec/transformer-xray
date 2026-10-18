import { describe, expect, it } from 'vitest';
import { compileProgram, launch, type ArgValue, type LaunchResult } from '@/cuda';
import { gatherRows, gatherRowsBwd } from '@/engine/ops/embedding';
import { add } from '@/engine/ops/elementwise';
import { geluTanh } from '@/engine/ops/gelu';
import { layernorm } from '@/engine/ops/layernorm';
import { causalMask } from '@/engine/ops/mask';
import { matmul, matmulTB } from '@/engine/ops/matmul';
import { softmaxRows } from '@/engine/ops/softmax';
import type { Tensor } from '@/engine/tensor';
import { maxAbsDiff, randFill, teachingSource } from './helpers';

const buf = (name: string, data: Float32Array | Int32Array): ArgValue => ({ kind: 'buffer', name, data });
const int = (value: number): ArgValue => ({ kind: 'int', value });
const flt = (value: number): ArgValue => ({ kind: 'float', value });
const tensor = (data: Float32Array, shape: number[]): Tensor => ({ data, shape });

describe('the five real kernels match the TS engine', () => {
  const T = 16;
  const D = 64;
  const V = 65;

  it('embedding_gather == gatherRows', () => {
    const p = compileProgram(teachingSource('01_embedding.cu'));
    const wte = randFill(V * D, 1, 0.02);
    const ids = Int32Array.from({ length: T }, (_, i) => (i * 7) % V);
    const out = new Float32Array(T * D);
    launch(p, 'embedding_gather', {
      grid: [Math.ceil((T * D) / 256), 1, 1],
      block: [256, 1, 1],
      args: [buf('wte', wte), buf('ids', ids), buf('out', out), int(T), int(D)],
    });
    expect(maxAbsDiff(out, gatherRows(tensor(wte, [V, D]), ids).data)).toBe(0);
  });

  it('embedding_scatter_add == gatherRowsBwd (with repeated ids)', () => {
    const p = compileProgram(teachingSource('01_embedding.cu'));
    const dOut = randFill(T * D, 2, 1);
    const ids = Int32Array.from({ length: T }, (_, i) => (i * 3) % 5); // heavy collisions
    const dWte = new Float32Array(V * D);
    const r = launch(p, 'embedding_scatter_add', {
      grid: [Math.ceil((T * D) / 256), 1, 1],
      block: [256, 1, 1],
      args: [buf('dOut', dOut), buf('ids', ids), buf('dWte', dWte), int(T), int(D)],
    });
    expect(maxAbsDiff(dWte, gatherRowsBwd(tensor(dOut, [T, D]), ids, V).data)).toBeLessThan(1e-5);
    expect(r.races).toEqual([]);
  });

  it('layernorm_naive and layernorm_warp == layernorm().out', () => {
    const p = compileProgram(teachingSource('02_layernorm.cu'));
    const x = randFill(T * D, 3, 2);
    const g = new Float32Array(D).fill(1);
    const b = new Float32Array(D);
    const ref = layernorm(tensor(x, [T, D]), tensor(g, [D]), tensor(b, [D]), 1e-5).out.data;
    const out1 = new Float32Array(T * D);
    launch(p, 'layernorm_naive', {
      grid: [Math.ceil(T / 128), 1, 1],
      block: [128, 1, 1],
      args: [buf('x', x), buf('g', g), buf('b', b), buf('out', out1), int(T), int(D), flt(1e-5)],
    });
    expect(maxAbsDiff(out1, ref)).toBeLessThan(1e-5);
    const out2 = new Float32Array(T * D);
    launch(p, 'layernorm_warp', {
      grid: [T, 1, 1],
      block: [32, 1, 1],
      args: [buf('x', x), buf('g', g), buf('b', b), buf('out', out2), int(T), int(D), flt(1e-5)],
    });
    expect(maxAbsDiff(out2, ref)).toBeLessThan(1e-5);
  });

  function runMatmul(kernel: string): { C: Float32Array; r: LaunchResult; ref: Float32Array } {
    const p = compileProgram(teachingSource('03_matmul.cu'));
    const M = 64;
    const K = 64;
    const N = 64;
    const A = randFill(M * K, 4, 1);
    const B = randFill(K * N, 5, 0.1);
    const C = new Float32Array(M * N);
    const r = launch(p, kernel, {
      grid: [N / 32, M / 32, 1],
      block: [32, 32, 1],
      args: [buf('A', A), buf('B', B), buf('C', C), int(M), int(K), int(N)],
    });
    return { C, r, ref: matmul(tensor(A, [M, K]), tensor(B, [K, N])).data };
  }

  it('matmul_naive and matmul_tiled == matmul', () => {
    const naive = runMatmul('matmul_naive');
    expect(maxAbsDiff(naive.C, naive.ref)).toBeLessThan(1e-4);
    const tiled = runMatmul('matmul_tiled');
    expect(maxAbsDiff(tiled.C, tiled.ref)).toBeLessThan(1e-4);
    expect(tiled.r.races).toEqual([]);
  });

  it('tiled matmul moves ~32× less global traffic than naive at 64³', () => {
    const naive = runMatmul('matmul_naive');
    const tiled = runMatmul('matmul_tiled');
    const ratio = naive.r.stats.globalBytes / tiled.r.stats.globalBytes;
    expect(ratio).toBeGreaterThan(16);
    expect(ratio).toBeLessThan(40);
  });

  it('attention_row == causal softmax(QKᵀ·scale)·V', () => {
    const p = compileProgram(teachingSource('04_attention.cu'));
    const hd = 32;
    const q = randFill(T * hd, 6, 1);
    const k = randFill(T * hd, 7, 1);
    const v = randFill(T * hd, 8, 1);
    const out = new Float32Array(T * hd);
    const scale = 1 / Math.sqrt(hd);
    launch(p, 'attention_row', {
      grid: [T, 1, 1],
      block: [128, 1, 1],
      sharedBytes: T * 4,
      args: [buf('q', q), buf('k', k), buf('v', v), buf('out', out), int(T), int(hd), flt(scale)],
    });
    const scores = matmulTB(tensor(q, [T, hd]), tensor(k, [T, hd]));
    for (let i = 0; i < scores.data.length; i++) scores.data[i] = Math.fround(scores.data[i] * Math.fround(scale));
    // the engine's mask/softmax work on [heads, T, T]
    const probs = softmaxRows(causalMask(tensor(scores.data, [1, T, T])));
    const ref = matmul(tensor(probs.data, [T, T]), tensor(v, [T, hd])).data;
    expect(maxAbsDiff(out, ref)).toBeLessThan(1e-5);
  });

  it('gelu_forward, gelu_forward_f4 == geluTanh; residual_add == add', () => {
    const p = compileProgram(teachingSource('05_elementwise.cu'));
    const n = 4096;
    const x = randFill(n, 9, 3);
    const ref = geluTanh(tensor(x, [n])).data;
    const y1 = new Float32Array(n);
    launch(p, 'gelu_forward', { grid: [256, 1, 1], block: [256, 1, 1], args: [buf('x', x), buf('y', y1), int(n)] });
    expect(maxAbsDiff(y1, ref)).toBeLessThan(1e-5);
    const y2 = new Float32Array(n);
    launch(p, 'gelu_forward_f4', { grid: [256, 1, 1], block: [256, 1, 1], args: [buf('x', x), buf('y', y2), int(n / 4)] });
    expect(maxAbsDiff(y2, ref)).toBeLessThan(1e-5);
    const a = randFill(n, 10, 1);
    const b = randFill(n, 11, 1);
    const o = new Float32Array(n);
    launch(p, 'residual_add', { grid: [256, 1, 1], block: [256, 1, 1], args: [buf('a', a), buf('b', b), buf('out', o), int(n)] });
    expect(maxAbsDiff(o, add(tensor(a, [n]), tensor(b, [n])).data)).toBe(0);
  });
});

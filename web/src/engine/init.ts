/** Random initialization, GPT-2 scheme — so watch-it-learn can start from
 * actual noise: N(0, 0.02²) everywhere, residual projections scaled by
 * 1/√(2L), LayerNorm at identity (g=1, b=0). Deterministic under a seed. */

import type { Weights } from './gpt';
import type { Rng } from './rng';
import type { Config } from './weights';

function gaussian(rng: Rng): number {
  // Box–Muller; guard against log(0)
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normal(shape: number[], std: number, rng: Rng): { data: Float32Array; shape: number[] } {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = gaussian(rng) * std;
  return { data, shape };
}

function constant(shape: number[], value: number): { data: Float32Array; shape: number[] } {
  const n = shape.reduce((a, b) => a * b, 1);
  return { data: new Float32Array(n).fill(value), shape };
}

export function initWeights(cfg: Config, rng: Rng): Weights {
  const { nLayer: L, nEmbd: d, blockSize, vocabSize: V } = cfg;
  const std = 0.02;
  const projStd = 0.02 / Math.sqrt(2 * L);
  const w: Weights = new Map();

  w.set('wte', normal([V, d], std, rng));
  w.set('wpe', normal([blockSize, d], std, rng));
  for (let b = 0; b < L; b++) {
    const p = `block${b}`;
    w.set(`${p}.ln1.g`, constant([d], 1));
    w.set(`${p}.ln1.b`, constant([d], 0));
    w.set(`${p}.attn.qkv.w`, normal([d, 3 * d], std, rng));
    w.set(`${p}.attn.qkv.b`, constant([3 * d], 0));
    w.set(`${p}.attn.proj.w`, normal([d, d], projStd, rng));
    w.set(`${p}.attn.proj.b`, constant([d], 0));
    w.set(`${p}.ln2.g`, constant([d], 1));
    w.set(`${p}.ln2.b`, constant([d], 0));
    w.set(`${p}.mlp.fc.w`, normal([d, 4 * d], std, rng));
    w.set(`${p}.mlp.fc.b`, constant([4 * d], 0));
    w.set(`${p}.mlp.proj.w`, normal([4 * d, d], projStd, rng));
    w.set(`${p}.mlp.proj.b`, constant([d], 0));
  }
  w.set('final_ln.g', constant([d], 1));
  w.set('final_ln.b', constant([d], 0));
  return w;
}

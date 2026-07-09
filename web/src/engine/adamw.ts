/** AdamW, exactly PyTorch's algorithm and order of operations:
 *   1. decoupled decay:  p ← p·(1 − lr·λ)
 *   2. moments:          m ← β₁m + (1−β₁)g ;  v ← β₂v + (1−β₂)g²
 *   3. bias correction:  m̂ = m/(1−β₁ᵗ) ;  v̂ = v/(1−β₂ᵗ)
 *   4. update:           p ← p − lr·m̂/(√v̂ + ε)      (ε outside the sqrt)
 * One param group, decay on everything — matching golden.py exactly.
 */

import type { Tensor } from './tensor';

export interface AdamWHyper {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
}

export interface AdamWState {
  t: number;
  m: Map<string, Float32Array>;
  v: Map<string, Float32Array>;
}

export function adamwInit(): AdamWState {
  return { t: 0, m: new Map(), v: new Map() };
}

export function adamwStep(
  params: Map<string, Tensor>,
  grads: Map<string, Float32Array>,
  state: AdamWState,
  hp: AdamWHyper,
): void {
  state.t += 1;
  const bc1 = 1 - Math.pow(hp.beta1, state.t);
  const bc2 = 1 - Math.pow(hp.beta2, state.t);
  for (const [name, p] of params) {
    const g = grads.get(name);
    if (!g) throw new Error(`adamw: missing grad for ${name}`);
    let m = state.m.get(name);
    let v = state.v.get(name);
    if (!m) {
      m = new Float32Array(g.length);
      v = new Float32Array(g.length);
      state.m.set(name, m);
      state.v.set(name, v!);
    }
    const pd = p.data;
    const decay = 1 - hp.lr * hp.weightDecay;
    for (let i = 0; i < pd.length; i++) {
      pd[i] *= decay;
      m[i] = hp.beta1 * m[i] + (1 - hp.beta1) * g[i];
      v![i] = hp.beta2 * v![i] + (1 - hp.beta2) * g[i] * g[i];
      const mHat = m[i] / bc1;
      const vHat = v![i] / bc2;
      pd[i] -= (hp.lr * mHat) / (Math.sqrt(vHat) + hp.eps);
    }
  }
}

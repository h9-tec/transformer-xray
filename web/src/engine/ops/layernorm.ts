/** LayerNorm with PyTorch semantics: biased variance, eps inside the sqrt.
 * mean/rstd are returned — the backward needs them and the inspector shows them. */

import type { Tensor } from '../tensor';

export interface LayerNormOut {
  out: Tensor;
  mean: Float32Array; // [T]
  rstd: Float32Array; // [T]
}

export function layernorm(x: Tensor, g: Tensor, b: Tensor, eps: number): LayerNormOut {
  const [T, D] = x.shape;
  const out = new Float32Array(T * D);
  const mean = new Float32Array(T);
  const rstd = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const xt = t * D;
    let m = 0;
    for (let i = 0; i < D; i++) m += x.data[xt + i];
    m /= D;
    let v = 0;
    for (let i = 0; i < D; i++) {
      const d = x.data[xt + i] - m;
      v += d * d;
    }
    v /= D; // biased variance (divide by D, not D-1) — PyTorch LayerNorm semantics
    const r = 1 / Math.sqrt(v + eps);
    mean[t] = m;
    rstd[t] = r;
    for (let i = 0; i < D; i++) {
      out[xt + i] = (x.data[xt + i] - m) * r * g.data[i] + b.data[i];
    }
  }
  return { out: { data: out, shape: [T, D] }, mean, rstd };
}

export interface LayerNormGrads {
  dx: Tensor;
  dg: Float32Array;
  db: Float32Array;
}

/** The classic closed form. Per row, with x̂ = (x−μ)·rstd and dx̂ = dy⊙g:
 *    dx = rstd · (dx̂ − mean(dx̂) − x̂ · mean(dx̂ ⊙ x̂))
 * The two mean-subtractions are exactly "the gradient must not change the
 * row's mean or scale" — the senior content derives this. */
export function layernormBwd(
  dy: Tensor,
  x: Tensor,
  g: Tensor,
  mean: Float32Array,
  rstd: Float32Array,
): LayerNormGrads {
  const [T, D] = x.shape;
  const dx = new Float32Array(T * D);
  const dg = new Float32Array(D);
  const db = new Float32Array(D);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    const m = mean[t];
    const r = rstd[t];
    let m1 = 0; // mean(dx̂)
    let m2 = 0; // mean(dx̂ ⊙ x̂)
    for (let i = 0; i < D; i++) {
      const xhat = (x.data[o + i] - m) * r;
      const dxhat = dy.data[o + i] * g.data[i];
      m1 += dxhat;
      m2 += dxhat * xhat;
      dg[i] += dy.data[o + i] * xhat;
      db[i] += dy.data[o + i];
    }
    m1 /= D;
    m2 /= D;
    for (let i = 0; i < D; i++) {
      const xhat = (x.data[o + i] - m) * r;
      const dxhat = dy.data[o + i] * g.data[i];
      dx[o + i] = r * (dxhat - m1 - xhat * m2);
    }
  }
  return { dx: { data: dx, shape: [T, D] }, dg, db };
}

/** Row-wise softmax over the last axis, with max-subtraction — the standard
 * numerical-stability trick (e^{x−max} instead of e^x) the senior content
 * teaches. Safe with −Infinity entries (masked positions → probability 0);
 * the causal diagonal guarantees at least one finite entry per row. */

import type { Tensor } from '../tensor';
import { numel } from '../tensor';

export function softmaxRows(x: Tensor): Tensor {
  const shape = x.shape;
  const D = shape[shape.length - 1];
  const rows = numel(shape) / D;
  const out = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const o = r * D;
    let max = -Infinity;
    for (let i = 0; i < D; i++) if (x.data[o + i] > max) max = x.data[o + i];
    let sum = 0;
    for (let i = 0; i < D; i++) {
      const e = Math.exp(x.data[o + i] - max); // e^{−Inf − max} = 0 for masked entries
      out[o + i] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let i = 0; i < D; i++) out[o + i] *= inv;
  }
  return { data: out, shape: [...shape] };
}

/** dx = y ⊙ (dy − ⟨dy, y⟩) per row — the softmax Jacobian applied without
 * ever materializing the T×T Jacobian. */
export function softmaxRowsBwd(dy: Tensor, y: Tensor): Tensor {
  const shape = y.shape;
  const D = shape[shape.length - 1];
  const rows = numel(shape) / D;
  const dx = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const o = r * D;
    let dot = 0;
    for (let i = 0; i < D; i++) dot += dy.data[o + i] * y.data[o + i];
    for (let i = 0; i < D; i++) dx[o + i] = y.data[o + i] * (dy.data[o + i] - dot);
  }
  return { data: dx, shape: [...shape] };
}

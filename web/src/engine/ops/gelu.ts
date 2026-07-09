/** GELU, tanh approximation — the exact GPT-2 formula, and the exact same
 * formula the PyTorch side uses (approximate='tanh'). Never mix with erf-GELU:
 * they differ by ~1e-3 and parity would fail. */

import type { Tensor } from '../tensor';

const C = Math.sqrt(2 / Math.PI);
const A = 0.044715;

export function geluTanh(x: Tensor): Tensor {
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    out[i] = 0.5 * v * (1 + Math.tanh(C * (v + A * v * v * v)));
  }
  return { data: out, shape: [...x.shape] };
}

export function geluTanhBwd(dy: Tensor, x: Tensor): Tensor {
  const dx = new Float32Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    const u = C * (v + A * v * v * v);
    const t = Math.tanh(u);
    const sech2 = 1 - t * t;
    const du = C * (1 + 3 * A * v * v);
    dx[i] = dy.data[i] * (0.5 * (1 + t) + 0.5 * v * sech2 * du);
  }
  return { data: dx, shape: [...x.shape] };
}

import type { Tensor } from '../tensor';

/** c = a + b (same shape). Residual adds; backward is pass-through to both. */
export function add(a: Tensor, b: Tensor): Tensor {
  const out = new Float32Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
  return { data: out, shape: [...a.shape] };
}

/** x [T,D] + row [T',D] broadcast when T' rows align 1:1 (positional add uses
 * equal T, so this is a plain add over rows). */
export function addRows(x: Tensor, rows: Tensor): Tensor {
  const [T, D] = x.shape;
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    for (let i = 0; i < D; i++) out[o + i] = x.data[o + i] + rows.data[o + i];
  }
  return { data: out, shape: [T, D] };
}

export function scale(x: Tensor, f: number): Tensor {
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < out.length; i++) out[i] = x.data[i] * f;
  return { data: out, shape: [...x.shape] };
}

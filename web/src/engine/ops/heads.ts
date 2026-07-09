/** Split/merge between [T, 3d] qkv columns and per-head [h, T, hd] tensors.
 * In PyTorch these are views; here they are explicit copies so each traced
 * tensor is standalone — memory cost is tiny at this scale and the golden
 * trace exports q/k/v as real tensors anyway. */

import type { Tensor } from '../tensor';

/** Extract q (which=0), k (1) or v (2) from qkv [T,3d] into [h,T,hd]. */
export function splitHeads(qkv: Tensor, which: 0 | 1 | 2, nHead: number, headDim: number): Tensor {
  const [T, threeD] = qkv.shape;
  const d = threeD / 3;
  const out = new Float32Array(nHead * T * headDim);
  const base = which * d;
  for (let h = 0; h < nHead; h++) {
    const col = base + h * headDim;
    for (let t = 0; t < T; t++) {
      const src = t * threeD + col;
      const dst = (h * T + t) * headDim;
      for (let i = 0; i < headDim; i++) out[dst + i] = qkv.data[src + i];
    }
  }
  return { data: out, shape: [nHead, T, headDim] };
}

/** Accumulate the gradient of one split back into dQkv [T,3d]. */
export function splitHeadsBwd(
  dPart: Tensor,
  which: 0 | 1 | 2,
  dQkv: Tensor, // accumulated in place across q/k/v
): void {
  const [nHead, T, headDim] = dPart.shape;
  const threeD = dQkv.shape[1];
  const d = threeD / 3;
  const base = which * d;
  for (let h = 0; h < nHead; h++) {
    const col = base + h * headDim;
    for (let t = 0; t < T; t++) {
      const src = (h * T + t) * headDim;
      const dst = t * threeD + col;
      for (let i = 0; i < headDim; i++) dQkv.data[dst + i] += dPart.data[src + i];
    }
  }
}

/** [h,T,hd] → [T,d]: concatenate heads along columns. */
export function mergeHeads(wv: Tensor): Tensor {
  const [nHead, T, headDim] = wv.shape;
  const d = nHead * headDim;
  const out = new Float32Array(T * d);
  for (let h = 0; h < nHead; h++) {
    for (let t = 0; t < T; t++) {
      const src = (h * T + t) * headDim;
      const dst = t * d + h * headDim;
      for (let i = 0; i < headDim; i++) out[dst + i] = wv.data[src + i];
    }
  }
  return { data: out, shape: [T, d] };
}

export function mergeHeadsBwd(dY: Tensor, nHead: number, headDim: number): Tensor {
  const [T] = dY.shape;
  const d = dY.shape[1];
  const out = new Float32Array(nHead * T * headDim);
  for (let h = 0; h < nHead; h++) {
    for (let t = 0; t < T; t++) {
      const src = t * d + h * headDim;
      const dst = (h * T + t) * headDim;
      for (let i = 0; i < headDim; i++) out[dst + i] = dY.data[src + i];
    }
  }
  return { data: out, shape: [nHead, T, headDim] };
}

/** Per-head slice view helpers for [h,T,hd] / [h,T,T] tensors. */
export function headSlice(t: Tensor, h: number): Tensor {
  const [, T, D] = t.shape;
  const size = T * D;
  return { data: t.data.subarray(h * size, (h + 1) * size) as Float32Array, shape: [T, D] };
}

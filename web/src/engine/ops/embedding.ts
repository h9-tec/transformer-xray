import type { Tensor } from '../tensor';

/** Token embedding lookup: out[t] = wte[ids[t]]. A gather — zero FLOPs,
 * pure memory traffic (the GPU view's point exactly). */
export function gatherRows(wte: Tensor, ids: Int32Array): Tensor {
  const [, D] = wte.shape;
  const T = ids.length;
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const src = ids[t] * D;
    out.set(wte.data.subarray(src, src + D), t * D);
  }
  return { data: out, shape: [T, D] };
}

/** Backward of a gather is a scatter-add: rows that share a token id
 * accumulate. (On the GPU this is the atomicAdd lesson.) */
export function gatherRowsBwd(dOut: Tensor, ids: Int32Array, vocabSize: number): Tensor {
  const [T, D] = dOut.shape;
  const dWte = new Float32Array(vocabSize * D);
  for (let t = 0; t < T; t++) {
    const dst = ids[t] * D;
    const src = t * D;
    for (let i = 0; i < D; i++) dWte[dst + i] += dOut.data[src + i];
  }
  return { data: dWte, shape: [vocabSize, D] };
}

/** Positional embeddings: the first T rows of wpe. */
export function posRows(wpe: Tensor, T: number): Tensor {
  const [, D] = wpe.shape;
  return { data: wpe.data.slice(0, T * D), shape: [T, D] };
}

export function posRowsBwd(dPos: Tensor, blockSize: number): Tensor {
  const [T, D] = dPos.shape;
  const dWpe = new Float32Array(blockSize * D);
  dWpe.set(dPos.data.subarray(0, T * D));
  return { data: dWpe, shape: [blockSize, D] };
}

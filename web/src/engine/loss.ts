/** Cross-entropy over next-token targets, mean over positions.
 * Forward uses the log-sum-exp trick; backward is the famous fused form
 * dlogits = (softmax − onehot) / T — the senior content derives why the
 * softmax and the loss gradient collapse into "probabilities minus truth". */

import type { Tensor } from './tensor';

export interface LossResult {
  perToken: Float32Array; // [T]
  loss: number;
}

export function crossEntropy(logits: Tensor, targets: Int32Array): LossResult {
  const [T, V] = logits.shape;
  const perToken = new Float32Array(T);
  let total = 0;
  for (let t = 0; t < T; t++) {
    const o = t * V;
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (logits.data[o + i] > max) max = logits.data[o + i];
    let sum = 0;
    for (let i = 0; i < V; i++) sum += Math.exp(logits.data[o + i] - max);
    const lse = max + Math.log(sum);
    perToken[t] = lse - logits.data[o + targets[t]];
    total += perToken[t];
  }
  return { perToken, loss: total / T };
}

export function crossEntropyBwd(logits: Tensor, targets: Int32Array): Tensor {
  const [T, V] = logits.shape;
  const d = new Float32Array(T * V);
  const invT = 1 / T;
  for (let t = 0; t < T; t++) {
    const o = t * V;
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (logits.data[o + i] > max) max = logits.data[o + i];
    let sum = 0;
    for (let i = 0; i < V; i++) sum += Math.exp(logits.data[o + i] - max);
    const inv = 1 / sum;
    for (let i = 0; i < V; i++) {
      d[o + i] = Math.exp(logits.data[o + i] - max) * inv * invT; // p/T
    }
    d[o + targets[t]] -= invT; // (p − onehot)/T
  }
  return { data: d, shape: [T, V] };
}

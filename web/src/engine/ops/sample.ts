/** Sampling head: temperature → top-k → softmax → draw. Deterministic under
 * a seeded Rng; greedy path is exact argmax (ties broken by lowest id, which
 * matches PyTorch argmax). */

import type { Rng } from '../rng';
import type { Tensor } from '../tensor';

export function argmax(x: Float32Array): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

/** Keep the k largest logits, set the rest to −Infinity. k ≥ length = no-op. */
export function topKFilter(logits: Float32Array, k: number): Float32Array {
  const out = logits.slice();
  if (k >= logits.length) return out;
  const sorted = Array.from(logits).sort((a, b) => b - a);
  const threshold = sorted[k - 1];
  for (let i = 0; i < out.length; i++) if (out[i] < threshold) out[i] = -Infinity;
  return out;
}

export function softmax1d(x: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of x) if (v > max) max = v;
  const out = new Float32Array(x.length);
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    const e = Math.exp(x[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

export function sampleMultinomial(probs: Float32Array, rng: Rng): number {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1; // fp rounding: the tail owns the remainder
}

export interface SampleResult {
  scaled: Float32Array; // logits / temperature
  filtered: Float32Array; // after top-k
  probs: Float32Array;
  chosen: number;
}

export function sampleFromLogits(
  lastLogits: Tensor,
  temperature: number,
  topK: number,
  rng: Rng | null, // null = greedy
): SampleResult {
  const logits = lastLogits.data;
  if (rng === null) {
    const probs = softmax1d(logits);
    return { scaled: logits.slice(), filtered: logits.slice(), probs, chosen: argmax(logits) };
  }
  const t = Math.max(temperature, 1e-6);
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / t;
  const filtered = topKFilter(scaled, topK);
  const probs = softmax1d(filtered);
  return { scaled, filtered, probs, chosen: sampleMultinomial(probs, rng) };
}

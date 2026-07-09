/** Causal mask as its own op (not fused into softmax) — deliberately, so the
 * trace shows the "future is −∞" step explicitly. */

import type { Tensor } from '../tensor';

/** scores [h,T,T] → masked copy with j>i set to −Infinity. */
export function causalMask(scores: Tensor): Tensor {
  const [H, T, T2] = scores.shape;
  const out = scores.data.slice();
  for (let h = 0; h < H; h++) {
    const ho = h * T * T2;
    for (let i = 0; i < T; i++) {
      const row = ho + i * T2;
      for (let j = i + 1; j < T2; j++) out[row + j] = -Infinity;
    }
  }
  return { data: out, shape: [...scores.shape] };
}

/** Gradient flows only through unmasked positions. (Softmax already sends
 * exactly 0 gradient to masked cells; zeroing here keeps the invariant
 * explicit and the traced tensor clean.) */
export function causalMaskBwd(dMasked: Tensor): Tensor {
  const [H, T, T2] = dMasked.shape;
  const out = dMasked.data.slice();
  for (let h = 0; h < H; h++) {
    const ho = h * T * T2;
    for (let i = 0; i < T; i++) {
      const row = ho + i * T2;
      for (let j = i + 1; j < T2; j++) out[row + j] = 0;
    }
  }
  return { data: out, shape: [...dMasked.shape] };
}

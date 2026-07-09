/** The three matmul variants the whole model needs. Weights are stored
 * [in, out] (transposed at export), so forward never transposes:
 *   forward: y = x @ W          → matmul
 *   dx      = dy @ Wᵀ           → matmulTB
 *   dW      = xᵀ @ dy           → matmulTA
 * That the two backward matmuls are just "the same kernel with a transposed
 * operand" is itself part of the curriculum.
 */

import type { Tensor } from '../tensor';

/** C[M,N] = A[M,K] @ B[K,N]. i-k-j loop order: streams B and C rows linearly. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [M, K] = a.shape;
  const [Kb, N] = b.shape;
  if (K !== Kb) throw new Error(`matmul: inner dims ${K} != ${Kb}`);
  const c = new Float32Array(M * N);
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < M; i++) {
    const ai = i * K;
    const ci = i * N;
    for (let k = 0; k < K; k++) {
      const av = ad[ai + k];
      const bk = k * N;
      for (let j = 0; j < N; j++) c[ci + j] += av * bd[bk + j];
    }
  }
  return { data: c, shape: [M, N] };
}

/** C[M,N] = A[M,K] @ B[N,K]ᵀ — rows of both operands are contiguous dot products. */
export function matmulTB(a: Tensor, b: Tensor): Tensor {
  const [M, K] = a.shape;
  const [N, Kb] = b.shape;
  if (K !== Kb) throw new Error(`matmulTB: inner dims ${K} != ${Kb}`);
  const c = new Float32Array(M * N);
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < M; i++) {
    const ai = i * K;
    const ci = i * N;
    for (let j = 0; j < N; j++) {
      const bj = j * K;
      let acc = 0;
      for (let k = 0; k < K; k++) acc += ad[ai + k] * bd[bj + k];
      c[ci + j] = acc;
    }
  }
  return { data: c, shape: [M, N] };
}

/** C[K,N] = A[M,K]ᵀ @ B[M,N] — outer loop over M keeps all reads streaming. */
export function matmulTA(a: Tensor, b: Tensor): Tensor {
  const [M, K] = a.shape;
  const [Mb, N] = b.shape;
  if (M !== Mb) throw new Error(`matmulTA: outer dims ${M} != ${Mb}`);
  const c = new Float32Array(K * N);
  const ad = a.data;
  const bd = b.data;
  for (let m = 0; m < M; m++) {
    const am = m * K;
    const bm = m * N;
    for (let k = 0; k < K; k++) {
      const av = ad[am + k];
      const ck = k * N;
      for (let j = 0; j < N; j++) c[ck + j] += av * bd[bm + j];
    }
  }
  return { data: c, shape: [K, N] };
}

/** y = x @ W + b, W stored [in, out]. */
export function linear(x: Tensor, w: Tensor, b: Tensor): Tensor {
  const y = matmul(x, w);
  const [T, out] = y.shape;
  for (let t = 0; t < T; t++) {
    const yt = t * out;
    for (let j = 0; j < out; j++) y.data[yt + j] += b.data[j];
  }
  return y;
}

export interface LinearGrads {
  dx: Tensor;
  dw: Tensor;
  db: Tensor;
}

export function linearBwd(dy: Tensor, x: Tensor, w: Tensor): LinearGrads {
  const [T, out] = dy.shape;
  const db = new Float32Array(out);
  for (let t = 0; t < T; t++) {
    const dt = t * out;
    for (let j = 0; j < out; j++) db[j] += dy.data[dt + j];
  }
  return {
    dx: matmulTB(dy, w), // dy [T,out] @ Wᵀ [out,in]
    dw: matmulTA(x, dy), // xᵀ [in,T] @ dy [T,out] → [in,out], same layout as W
    db: { data: db, shape: [out] },
  };
}

/** GPU ops over `Buf`s, mirroring engine/ops/* one for one. Each op records
 * its dispatch on the encoder you pass; nothing runs until you submit. */

import { Buf, dispatch, uniform, type Gpu } from './device';

const wg = (n: number, size: number) => Math.max(1, Math.ceil(n / size));

export interface MatmulOpts {
  transA?: boolean;
  transB?: boolean;
  bias?: Buf;
  accumulate?: boolean;
  batch?: number;
  strideA?: number;
  strideB?: number;
  strideC?: number;
}

/** C[M,N] (= or +=) op(A) · op(B) (+ bias). With `batch`, strides select each slice. */
export function matmul(gpu: Gpu, enc: GPUCommandEncoder, A: Buf, B: Buf, C: Buf, M: number, K: number, N: number, o: MatmulOpts = {}): void {
  const batch = o.batch ?? 1;
  const params = uniform(gpu, [
    M, K, N, batch,
    o.strideA ?? 0, o.strideB ?? 0, o.strideC ?? 0,
    o.transA ? 1 : 0, o.transB ? 1 : 0, o.bias ? 1 : 0, o.accumulate ? 1 : 0, 0,
  ]);
  dispatch(gpu, enc, 'matmul_tiled', [A.buffer, B.buffer, (o.bias ?? A).buffer, C.buffer], params, [wg(N, 16), wg(M, 16), batch]);
}

export function layernorm(gpu: Gpu, enc: GPUCommandEncoder, x: Buf, g: Buf, b: Buf, out: Buf, mean: Buf, rstd: Buf, T: number, D: number, eps: number): void {
  dispatch(gpu, enc, 'layernorm_warp', [x.buffer, g.buffer, b.buffer, out.buffer, mean.buffer, rstd.buffer], uniform(gpu, [T, D, { f: eps }]), [T, 1, 1]);
}

export function embed(gpu: Gpu, enc: GPUCommandEncoder, wte: Buf, ids: Buf, wpe: Buf, out: Buf, T: number, D: number): void {
  dispatch(gpu, enc, 'embedding_gather', [wte.buffer, ids.buffer, wpe.buffer, out.buffer], uniform(gpu, [T, D]), [wg(T * D, 256), 1, 1]);
}

export function splitHeads(gpu: Gpu, enc: GPUCommandEncoder, qkv: Buf, out: Buf, T: number, d: number, H: number, hd: number, which: 0 | 1 | 2): void {
  dispatch(gpu, enc, 'split_heads', [qkv.buffer, out.buffer], uniform(gpu, [T, d, H, hd, which]), [wg(H * T * hd, 256), 1, 1]);
}

export function mergeHeads(gpu: Gpu, enc: GPUCommandEncoder, x: Buf, out: Buf, T: number, d: number, H: number, hd: number): void {
  dispatch(gpu, enc, 'merge_heads', [x.buffer, out.buffer], uniform(gpu, [T, d, H, hd]), [wg(T * d, 256), 1, 1]);
}

export function causalSoftmax(gpu: Gpu, enc: GPUCommandEncoder, scores: Buf, probs: Buf, H: number, T: number, scale: number): void {
  dispatch(gpu, enc, 'causal_softmax', [scores.buffer, probs.buffer], uniform(gpu, [H, T, { f: scale }]), [H * T, 1, 1]);
}

export function gelu(gpu: Gpu, enc: GPUCommandEncoder, x: Buf, y: Buf, n: number): void {
  dispatch(gpu, enc, 'gelu_forward', [x.buffer, y.buffer], uniform(gpu, [n]), [wg(n, 256), 1, 1]);
}

export function add(gpu: Gpu, enc: GPUCommandEncoder, a: Buf, b: Buf, out: Buf, n: number): void {
  dispatch(gpu, enc, 'residual_add', [a.buffer, b.buffer, out.buffer], uniform(gpu, [n]), [wg(n, 256), 1, 1]);
}

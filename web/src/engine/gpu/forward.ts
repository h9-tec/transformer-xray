/** The forward pass on the GPU, op for op in the order of engine/gpt.ts
 * forward(): embed → [ln1 → qkv → heads → scores → softmax → PV → merge →
 * proj → res1 → ln2 → fc → gelu → proj → res2]×L → final ln → logits.
 * One command encoder per call; only the logits come back. */

import { sampleFromLogits } from '@/engine/ops/sample';
import type { SampleOpts } from '@/engine/gpt';
import type { Config } from '@/engine/weights';
import { Buf, submitAndWait, type Gpu } from './device';
import type { GpuModel } from './model';
import * as g from './ops';

export interface GpuForwardResult {
  /** [T, V] logits */
  logits: Float32Array;
  ms: number;
}

/** Scratch buffers sized for the largest T seen, reused across calls. */
class Scratch {
  private pool = new Map<string, Buf>();
  constructor(private gpu: Gpu) {}
  get(name: string, floats: number): Buf {
    const key = `${name}:${floats}`;
    let b = this.pool.get(key);
    if (!b) {
      b = Buf.zeros(this.gpu, floats, name);
      this.pool.set(key, b);
    }
    return b;
  }
  destroy(): void {
    for (const b of this.pool.values()) b.destroy();
    this.pool.clear();
  }
}

const scratchByGpu = new WeakMap<Gpu, Scratch>();
const scratch = (gpu: Gpu) => {
  let s = scratchByGpu.get(gpu);
  if (!s) scratchByGpu.set(gpu, (s = new Scratch(gpu)));
  return s;
};

export async function gpuForward(gpu: Gpu, model: GpuModel, cfg: Config, ids: Int32Array): Promise<GpuForwardResult> {
  const t0 = performance.now();
  const { nLayer, nEmbd: d, nHead: H, headDim: hd, vocabSize: V, lnEps } = cfg;
  const T = ids.length;
  if (T > cfg.blockSize) throw new Error(`sequence length ${T} > block size ${cfg.blockSize}`);
  const S = scratch(gpu);
  const idsBuf = Buf.from(gpu, ids, 'ids');
  const enc = gpu.device.createCommandEncoder();

  // embeddings: x = wte[ids] + wpe[:T]
  let x = S.get('x', T * d);
  g.embed(gpu, enc, model.get('wte'), idsBuf, model.get('wpe'), x, T, d);

  const ln = S.get('ln', T * d);
  const mean = S.get('mean', T);
  const rstd = S.get('rstd', T);
  const qkv = S.get('qkv', T * 3 * d);
  const q = S.get('q', H * T * hd);
  const k = S.get('k', H * T * hd);
  const v = S.get('v', H * T * hd);
  const scores = S.get('scores', H * T * T);
  const probs = S.get('probs', H * T * T);
  const wv = S.get('wv', H * T * hd);
  const merged = S.get('merged', T * d);
  const proj = S.get('proj', T * d);
  const res1 = S.get('res1', T * d);
  const fc = S.get('fc', T * 4 * d);
  const act = S.get('act', T * 4 * d);
  const mlp = S.get('mlp', T * d);
  const res2 = S.get('res2', T * d);
  const scale = 1 / Math.sqrt(hd);

  for (let b = 0; b < nLayer; b++) {
    const p = `block${b}`;
    g.layernorm(gpu, enc, x, model.get(`${p}.ln1.g`), model.get(`${p}.ln1.b`), ln, mean, rstd, T, d, lnEps);
    g.matmul(gpu, enc, ln, model.get(`${p}.attn.qkv.w`), qkv, T, d, 3 * d, { bias: model.get(`${p}.attn.qkv.b`) });
    g.splitHeads(gpu, enc, qkv, q, T, d, H, hd, 0);
    g.splitHeads(gpu, enc, qkv, k, T, d, H, hd, 1);
    g.splitHeads(gpu, enc, qkv, v, T, d, H, hd, 2);
    // scores[h] = q[h] · k[h]ᵀ  (scale is applied inside the softmax)
    g.matmul(gpu, enc, q, k, scores, T, hd, T, { transB: true, batch: H, strideA: T * hd, strideB: T * hd, strideC: T * T });
    g.causalSoftmax(gpu, enc, scores, probs, H, T, scale);
    // wv[h] = probs[h] · v[h]
    g.matmul(gpu, enc, probs, v, wv, T, T, hd, { batch: H, strideA: T * T, strideB: T * hd, strideC: T * hd });
    g.mergeHeads(gpu, enc, wv, merged, T, d, H, hd);
    g.matmul(gpu, enc, merged, model.get(`${p}.attn.proj.w`), proj, T, d, d, { bias: model.get(`${p}.attn.proj.b`) });
    g.add(gpu, enc, x, proj, res1, T * d);
    g.layernorm(gpu, enc, res1, model.get(`${p}.ln2.g`), model.get(`${p}.ln2.b`), ln, mean, rstd, T, d, lnEps);
    g.matmul(gpu, enc, ln, model.get(`${p}.mlp.fc.w`), fc, T, d, 4 * d, { bias: model.get(`${p}.mlp.fc.b`) });
    g.gelu(gpu, enc, fc, act, T * 4 * d);
    g.matmul(gpu, enc, act, model.get(`${p}.mlp.proj.w`), mlp, T, 4 * d, d, { bias: model.get(`${p}.mlp.proj.b`) });
    g.add(gpu, enc, res1, mlp, res2, T * d);
    // dispatches run in order: the next block reads res2 (as x) before it
    // is overwritten at the end of that block
    x = res2;
  }

  g.layernorm(gpu, enc, x, model.get('final_ln.g'), model.get('final_ln.b'), ln, mean, rstd, T, d, lnEps);
  const logits = S.get('logits', T * V);
  // weight tying: logits = ln · wteᵀ, wte is [V, d]
  g.matmul(gpu, enc, ln, model.get('wte'), logits, T, d, V, { transB: true });
  await submitAndWait(gpu, enc);
  const out = await logits.read();
  idsBuf.destroy();
  return { logits: out.subarray(0, T * V), ms: performance.now() - t0 };
}

/** Full forward per generated token (no KV cache: at T ≤ 256 the GPU does
 * not need one). Sampling happens on the CPU with the engine's sampler so
 * seeds reproduce the CPU path token for token. */
export async function gpuGenerate(
  gpu: Gpu,
  model: GpuModel,
  cfg: Config,
  prompt: Int32Array,
  maxNewTokens: number,
  opts: SampleOpts,
  onToken?: (info: { token: number; logits: Float32Array; ms: number }) => void,
): Promise<Int32Array> {
  let ids = prompt;
  for (let i = 0; i < maxNewTokens; i++) {
    const cropped = ids.length > cfg.blockSize ? ids.subarray(ids.length - cfg.blockSize) : ids;
    const { logits, ms } = await gpuForward(gpu, model, cfg, cropped);
    const T = cropped.length;
    const last = logits.slice((T - 1) * cfg.vocabSize, T * cfg.vocabSize);
    const { chosen } = sampleFromLogits({ data: last, shape: [cfg.vocabSize] }, opts.temperature, opts.topK, opts.rng);
    const next = new Int32Array(ids.length + 1);
    next.set(ids);
    next[ids.length] = chosen;
    ids = next;
    onToken?.({ token: chosen, logits: last, ms });
  }
  return ids;
}

/** Untraced, KV-cached generation: one position at a time against cached
 * keys/values — milliseconds per token instead of a full re-forward.
 * The inspectors never read from this path; equivalence with the full
 * forward is enforced by tests (greedy decode must match token-for-token). */

import { sampleFromLogits } from './ops/sample';
import type { SampleOpts, Weights } from './gpt';
import type { Config } from './weights';
import type { Tensor } from './tensor';

export class KVCache {
  readonly k: Float32Array[];
  readonly v: Float32Array[];
  len = 0;

  constructor(cfg: Config) {
    this.k = [];
    this.v = [];
    for (let b = 0; b < cfg.nLayer; b++) {
      this.k.push(new Float32Array(cfg.nHead * cfg.blockSize * cfg.headDim));
      this.v.push(new Float32Array(cfg.nHead * cfg.blockSize * cfg.headDim));
    }
  }

  reset(): void {
    this.len = 0;
  }
}

function need(weights: Weights, name: string): Tensor {
  const t = weights.get(name);
  if (!t) throw new Error(`missing weight: ${name}`);
  return t;
}

function vecLinear(x: Float32Array, w: Tensor, b: Tensor): Float32Array {
  const [inDim, outDim] = w.shape;
  const out = b.data.slice();
  for (let i = 0; i < inDim; i++) {
    const xv = x[i];
    const row = i * outDim;
    for (let j = 0; j < outDim; j++) out[j] += xv * w.data[row + j];
  }
  return out;
}

function lnRow(x: Float32Array, g: Tensor, b: Tensor, eps: number): Float32Array {
  const D = x.length;
  let m = 0;
  for (let i = 0; i < D; i++) m += x[i];
  m /= D;
  let v = 0;
  for (let i = 0; i < D; i++) {
    const d = x[i] - m;
    v += d * d;
  }
  v /= D;
  const r = 1 / Math.sqrt(v + eps);
  const out = new Float32Array(D);
  for (let i = 0; i < D; i++) out[i] = (x[i] - m) * r * g.data[i] + b.data[i];
  return out;
}

const C_GELU = Math.sqrt(2 / Math.PI);

/** Run one token through the model; returns the logits row [V]. */
export function step(weights: Weights, cfg: Config, cache: KVCache, tokenId: number): Float32Array {
  const { nLayer, nEmbd: d, nHead, headDim, blockSize, vocabSize, lnEps } = cfg;
  const pos = cache.len;
  if (pos >= blockSize) throw new Error('KV cache full — re-prefill with a cropped context');
  const scaleF = 1 / Math.sqrt(headDim);

  const wte = need(weights, 'wte');
  const wpe = need(weights, 'wpe');
  const x = new Float32Array(d);
  for (let i = 0; i < d; i++) x[i] = wte.data[tokenId * d + i] + wpe.data[pos * d + i];

  for (let b = 0; b < nLayer; b++) {
    const p = `block${b}`;
    const h1 = lnRow(x, need(weights, `${p}.ln1.g`), need(weights, `${p}.ln1.b`), lnEps);
    const qkv = vecLinear(h1, need(weights, `${p}.attn.qkv.w`), need(weights, `${p}.attn.qkv.b`));

    const kCache = cache.k[b];
    const vCache = cache.v[b];
    const attnOut = new Float32Array(d);
    for (let h = 0; h < nHead; h++) {
      const qOff = h * headDim;
      const kOff = d + h * headDim;
      const vOff = 2 * d + h * headDim;
      const base = h * blockSize * headDim;
      // append this position's k/v
      for (let i = 0; i < headDim; i++) {
        kCache[base + pos * headDim + i] = qkv[kOff + i];
        vCache[base + pos * headDim + i] = qkv[vOff + i];
      }
      // attention over positions 0..pos
      const scores = new Float32Array(pos + 1);
      let max = -Infinity;
      for (let t = 0; t <= pos; t++) {
        let acc = 0;
        const kRow = base + t * headDim;
        for (let i = 0; i < headDim; i++) acc += qkv[qOff + i] * kCache[kRow + i];
        acc *= scaleF;
        scores[t] = acc;
        if (acc > max) max = acc;
      }
      let sum = 0;
      for (let t = 0; t <= pos; t++) {
        scores[t] = Math.exp(scores[t] - max);
        sum += scores[t];
      }
      const inv = 1 / sum;
      for (let t = 0; t <= pos; t++) {
        const s = scores[t] * inv;
        const vRow = base + t * headDim;
        for (let i = 0; i < headDim; i++) attnOut[qOff + i] += s * vCache[vRow + i];
      }
    }

    const proj = vecLinear(attnOut, need(weights, `${p}.attn.proj.w`), need(weights, `${p}.attn.proj.b`));
    for (let i = 0; i < d; i++) x[i] += proj[i];

    const h2 = lnRow(x, need(weights, `${p}.ln2.g`), need(weights, `${p}.ln2.b`), lnEps);
    const fc = vecLinear(h2, need(weights, `${p}.mlp.fc.w`), need(weights, `${p}.mlp.fc.b`));
    for (let i = 0; i < fc.length; i++) {
      const v = fc[i];
      fc[i] = 0.5 * v * (1 + Math.tanh(C_GELU * (v + 0.044715 * v * v * v)));
    }
    const mp = vecLinear(fc, need(weights, `${p}.mlp.proj.w`), need(weights, `${p}.mlp.proj.b`));
    for (let i = 0; i < d; i++) x[i] += mp[i];
  }

  const xf = lnRow(x, need(weights, 'final_ln.g'), need(weights, 'final_ln.b'), lnEps);
  const logits = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    let acc = 0;
    const row = v * d;
    for (let i = 0; i < d; i++) acc += xf[i] * wte.data[row + i];
    logits[v] = acc;
  }

  cache.len += 1;
  return logits;
}

export interface GenerateStepResult {
  token: number;
  logits: Float32Array; // pre-sampling logits for the position that produced `token`
  probs: Float32Array;
}

/** Generate with a KV cache; re-prefills on context overflow. */
export function generateKV(
  weights: Weights,
  cfg: Config,
  prompt: Int32Array,
  maxNewTokens: number,
  opts: SampleOpts,
  onToken?: (r: GenerateStepResult) => void,
): Int32Array {
  const cache = new KVCache(cfg);
  const ids: number[] = Array.from(prompt);

  const prefill = (from: number[]): Float32Array => {
    cache.reset();
    let logits: Float32Array = new Float32Array(cfg.vocabSize);
    for (const id of from) logits = step(weights, cfg, cache, id);
    return logits;
  };

  let logits = prefill(ids);
  for (let i = 0; i < maxNewTokens; i++) {
    const { probs, chosen } = sampleFromLogits(
      { data: logits, shape: [cfg.vocabSize] },
      opts.temperature,
      opts.topK,
      opts.rng,
    );
    onToken?.({ token: chosen, logits: logits.slice(), probs });
    ids.push(chosen);
    if (cache.len >= cfg.blockSize) {
      logits = prefill(ids.slice(-(cfg.blockSize - 1)));
    } else {
      logits = step(weights, cfg, cache, chosen);
    }
  }
  return Int32Array.from(ids);
}

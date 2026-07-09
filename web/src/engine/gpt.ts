/** The model: forward, loss, and a fully hand-written backward pass.
 *
 * Node ids on the tape match model/golden.py exactly — the parity suite
 * walks the golden manifest and compares tensor-for-tensor. Backward nodes
 * are "grad:<forward id>" and carry gradOf so the UI can mirror any forward
 * view into its gradient twin.
 */

import { adamwStep, type AdamWHyper, type AdamWState } from './adamw';
import { crossEntropy, crossEntropyBwd } from './loss';
import { add } from './ops/elementwise';
import { gatherRows, gatherRowsBwd, posRows, posRowsBwd } from './ops/embedding';
import { geluTanh, geluTanhBwd } from './ops/gelu';
import { headSlice, mergeHeads, mergeHeadsBwd, splitHeads, splitHeadsBwd } from './ops/heads';
import { layernorm, layernormBwd, type LayerNormOut } from './ops/layernorm';
import { causalMask, causalMaskBwd } from './ops/mask';
import { linear, linearBwd, matmul, matmulTA, matmulTB } from './ops/matmul';
import { sampleFromLogits } from './ops/sample';
import { softmaxRows, softmaxRowsBwd } from './ops/softmax';
import type { Rng } from './rng';
import { cost, Tape } from './tape';
import type { Tensor } from './tensor';
import { clone, zeros } from './tensor';
import type { Config } from './weights';

export type Weights = Map<string, Tensor>;

function w(weights: Weights, name: string): Tensor {
  const t = weights.get(name);
  if (!t) throw new Error(`missing weight: ${name}`);
  return t;
}

/* ─── forward ─────────────────────────────────────────────────────────── */

interface BlockCtx {
  input: Tensor; // x entering the block (needed by ln1 backward)
  ln1: LayerNormOut;
  qkv: Tensor;
  q: Tensor; // [h,T,hd]
  k: Tensor;
  v: Tensor;
  raw: Tensor; // [h,T,T] pre-scale scores
  scores: Tensor;
  masked: Tensor;
  att: Tensor; // post-softmax
  wv: Tensor; // [h,T,hd]
  merged: Tensor; // [T,d]
  proj: Tensor;
  res1: Tensor;
  ln2: LayerNormOut;
  fc: Tensor;
  gelu: Tensor;
  mlpProj: Tensor;
  res2: Tensor;
}

export interface ForwardCtx {
  ids: Int32Array;
  T: number;
  tok: Tensor;
  pos: Tensor;
  embSum: Tensor;
  blocks: BlockCtx[];
  finalLn: LayerNormOut;
  logits: Tensor;
}

/** Per-head A[h,T,X] × B[h,?,?] matmul helpers assembling [h,·,·] outputs. */
function perHead(
  fn: (a: Tensor, b: Tensor) => Tensor,
  a: Tensor,
  b: Tensor,
  outRows: number,
  outCols: number,
): Tensor {
  const H = a.shape[0];
  const out = zeros([H, outRows, outCols]);
  for (let h = 0; h < H; h++) {
    const r = fn(headSlice(a, h), headSlice(b, h));
    out.data.set(r.data, h * outRows * outCols);
  }
  return out;
}

export function forward(weights: Weights, cfg: Config, ids: Int32Array, tape?: Tape): ForwardCtx {
  const { nLayer, nEmbd: d, nHead, headDim, vocabSize, lnEps } = cfg;
  const T = ids.length;
  if (T > cfg.blockSize) throw new Error(`sequence length ${T} > block size ${cfg.blockSize}`);

  const rec = tape ? tape.rec.bind(tape) : null;

  const wte = w(weights, 'wte');
  const tok = gatherRows(wte, ids);
  rec?.({
    id: 'emb.token',
    phase: 'forward',
    kind: 'gather',
    tensor: tok,
    inputs: [{ weight: 'wte', role: 'weight' }],
    meta: { ids: Array.from(ids) },
    costs: cost.gather(T, d),
  });

  const pos = posRows(w(weights, 'wpe'), T);
  rec?.({
    id: 'emb.pos',
    phase: 'forward',
    kind: 'gather',
    tensor: pos,
    inputs: [{ weight: 'wpe', role: 'weight' }],
    costs: cost.gather(T, d),
  });

  let x = add(tok, pos);
  rec?.({
    id: 'emb.sum',
    phase: 'forward',
    kind: 'add',
    tensor: x,
    inputs: [
      { node: 'emb.token', role: 'input' },
      { node: 'emb.pos', role: 'input' },
    ],
    costs: cost.elementwise(T * d, 2, 1),
  });

  const scaleF = 1 / Math.sqrt(headDim);
  const blocks: BlockCtx[] = [];

  for (let b = 0; b < nLayer; b++) {
    const p = `block${b}`;
    const input = x;

    const ln1 = layernorm(input, w(weights, `${p}.ln1.g`), w(weights, `${p}.ln1.b`), lnEps);
    rec?.({
      id: `${p}.ln1`,
      phase: 'forward',
      kind: 'layernorm',
      layer: b,
      tensor: ln1.out,
      inputs: [
        { node: b === 0 ? 'emb.sum' : `block${b - 1}.res2`, role: 'input' },
        { weight: `${p}.ln1.g`, role: 'weight' },
        { weight: `${p}.ln1.b`, role: 'bias' },
      ],
      meta: { eps: lnEps },
      costs: cost.layernorm(T, d),
    });
    rec?.({
      id: `${p}.ln1.mean`,
      phase: 'forward',
      kind: 'layernorm',
      layer: b,
      tensor: { data: ln1.mean, shape: [T] },
      inputs: [{ node: `${p}.ln1`, role: 'input' }],
      costs: cost.none(),
    });
    rec?.({
      id: `${p}.ln1.rstd`,
      phase: 'forward',
      kind: 'layernorm',
      layer: b,
      tensor: { data: ln1.rstd, shape: [T] },
      inputs: [{ node: `${p}.ln1`, role: 'input' }],
      costs: cost.none(),
    });

    const qkv = linear(ln1.out, w(weights, `${p}.attn.qkv.w`), w(weights, `${p}.attn.qkv.b`));
    rec?.({
      id: `${p}.attn.qkv`,
      phase: 'forward',
      kind: 'linear',
      layer: b,
      tensor: qkv,
      inputs: [
        { node: `${p}.ln1`, role: 'input' },
        { weight: `${p}.attn.qkv.w`, role: 'weight' },
        { weight: `${p}.attn.qkv.b`, role: 'bias' },
      ],
      meta: { M: T, K: d, N: 3 * d },
      costs: cost.matmul(T, d, 3 * d),
    });

    const q = splitHeads(qkv, 0, nHead, headDim);
    const k = splitHeads(qkv, 1, nHead, headDim);
    const v = splitHeads(qkv, 2, nHead, headDim);
    for (const [name, t] of [
      ['q', q],
      ['k', k],
      ['v', v],
    ] as const) {
      rec?.({
        id: `${p}.attn.${name}`,
        phase: 'forward',
        kind: 'view',
        layer: b,
        tensor: t,
        inputs: [{ node: `${p}.attn.qkv`, role: 'input' }],
        meta: { headLayout: '[head, position, headDim]' },
        costs: cost.none(),
      });
    }

    const raw = perHead(matmulTB, q, k, T, T); // q @ kᵀ per head
    rec?.({
      id: `${p}.attn.scores_raw`,
      phase: 'forward',
      kind: 'matmul',
      layer: b,
      tensor: raw,
      inputs: [
        { node: `${p}.attn.q`, role: 'input' },
        { node: `${p}.attn.k`, role: 'input' },
      ],
      meta: { M: T, K: headDim, N: T, perHead: nHead, transposeB: true },
      costs: cost.matmul(T, headDim, T * nHead),
    });

    const scores = { data: raw.data.slice(), shape: [...raw.shape] };
    for (let i = 0; i < scores.data.length; i++) scores.data[i] *= scaleF;
    rec?.({
      id: `${p}.attn.scores`,
      phase: 'forward',
      kind: 'scale',
      layer: b,
      tensor: scores,
      inputs: [{ node: `${p}.attn.scores_raw`, role: 'input' }],
      meta: { factor: scaleF, headDim },
      costs: cost.elementwise(scores.data.length, 1, 1),
    });

    const masked = causalMask(scores);
    rec?.({
      id: `${p}.attn.masked`,
      phase: 'forward',
      kind: 'causal_mask',
      layer: b,
      tensor: masked,
      inputs: [{ node: `${p}.attn.scores`, role: 'input' }],
      costs: cost.elementwise(masked.data.length, 1, 0),
    });

    const att = softmaxRows(masked);
    rec?.({
      id: `${p}.attn.softmax`,
      phase: 'forward',
      kind: 'softmax',
      layer: b,
      tensor: att,
      inputs: [{ node: `${p}.attn.masked`, role: 'input' }],
      costs: cost.softmax(nHead * T, T),
    });

    const wv = perHead(matmul, att, v, T, headDim);
    rec?.({
      id: `${p}.attn.wv`,
      phase: 'forward',
      kind: 'matmul',
      layer: b,
      tensor: wv,
      inputs: [
        { node: `${p}.attn.softmax`, role: 'input' },
        { node: `${p}.attn.v`, role: 'input' },
      ],
      meta: { M: T, K: T, N: headDim, perHead: nHead },
      costs: cost.matmul(T, T, headDim * nHead),
    });

    const merged = mergeHeads(wv);
    const proj = linear(merged, w(weights, `${p}.attn.proj.w`), w(weights, `${p}.attn.proj.b`));
    rec?.({
      id: `${p}.attn.proj`,
      phase: 'forward',
      kind: 'linear',
      layer: b,
      tensor: proj,
      inputs: [
        { node: `${p}.attn.wv`, role: 'input' },
        { weight: `${p}.attn.proj.w`, role: 'weight' },
        { weight: `${p}.attn.proj.b`, role: 'bias' },
      ],
      meta: { M: T, K: d, N: d },
      costs: cost.matmul(T, d, d),
    });

    const res1 = add(input, proj);
    rec?.({
      id: `${p}.res1`,
      phase: 'forward',
      kind: 'add',
      layer: b,
      tensor: res1,
      inputs: [
        { node: b === 0 ? 'emb.sum' : `block${b - 1}.res2`, role: 'input' },
        { node: `${p}.attn.proj`, role: 'input' },
      ],
      costs: cost.elementwise(T * d, 2, 1),
    });

    const ln2 = layernorm(res1, w(weights, `${p}.ln2.g`), w(weights, `${p}.ln2.b`), lnEps);
    rec?.({
      id: `${p}.ln2`,
      phase: 'forward',
      kind: 'layernorm',
      layer: b,
      tensor: ln2.out,
      inputs: [
        { node: `${p}.res1`, role: 'input' },
        { weight: `${p}.ln2.g`, role: 'weight' },
        { weight: `${p}.ln2.b`, role: 'bias' },
      ],
      meta: { eps: lnEps },
      costs: cost.layernorm(T, d),
    });
    rec?.({
      id: `${p}.ln2.mean`,
      phase: 'forward',
      kind: 'layernorm',
      layer: b,
      tensor: { data: ln2.mean, shape: [T] },
      inputs: [{ node: `${p}.ln2`, role: 'input' }],
      costs: cost.none(),
    });
    rec?.({
      id: `${p}.ln2.rstd`,
      phase: 'forward',
      kind: 'layernorm',
      layer: b,
      tensor: { data: ln2.rstd, shape: [T] },
      inputs: [{ node: `${p}.ln2`, role: 'input' }],
      costs: cost.none(),
    });

    const fc = linear(ln2.out, w(weights, `${p}.mlp.fc.w`), w(weights, `${p}.mlp.fc.b`));
    rec?.({
      id: `${p}.mlp.fc`,
      phase: 'forward',
      kind: 'linear',
      layer: b,
      tensor: fc,
      inputs: [
        { node: `${p}.ln2`, role: 'input' },
        { weight: `${p}.mlp.fc.w`, role: 'weight' },
        { weight: `${p}.mlp.fc.b`, role: 'bias' },
      ],
      meta: { M: T, K: d, N: 4 * d },
      costs: cost.matmul(T, d, 4 * d),
    });

    const gelu = geluTanh(fc);
    rec?.({
      id: `${p}.mlp.gelu`,
      phase: 'forward',
      kind: 'gelu',
      layer: b,
      tensor: gelu,
      inputs: [{ node: `${p}.mlp.fc`, role: 'input' }],
      costs: cost.elementwise(gelu.data.length, 1, 14),
    });

    const mlpProj = linear(gelu, w(weights, `${p}.mlp.proj.w`), w(weights, `${p}.mlp.proj.b`));
    rec?.({
      id: `${p}.mlp.proj`,
      phase: 'forward',
      kind: 'linear',
      layer: b,
      tensor: mlpProj,
      inputs: [
        { node: `${p}.mlp.gelu`, role: 'input' },
        { weight: `${p}.mlp.proj.w`, role: 'weight' },
        { weight: `${p}.mlp.proj.b`, role: 'bias' },
      ],
      meta: { M: T, K: 4 * d, N: d },
      costs: cost.matmul(T, 4 * d, d),
    });

    const res2 = add(res1, mlpProj);
    rec?.({
      id: `${p}.res2`,
      phase: 'forward',
      kind: 'add',
      layer: b,
      tensor: res2,
      inputs: [
        { node: `${p}.res1`, role: 'input' },
        { node: `${p}.mlp.proj`, role: 'input' },
      ],
      costs: cost.elementwise(T * d, 2, 1),
    });

    blocks.push({
      input,
      ln1,
      qkv,
      q,
      k,
      v,
      raw,
      scores,
      masked,
      att,
      wv,
      merged,
      proj,
      res1,
      ln2,
      fc,
      gelu,
      mlpProj,
      res2,
    });
    x = res2;
  }

  const finalLn = layernorm(x, w(weights, 'final_ln.g'), w(weights, 'final_ln.b'), lnEps);
  rec?.({
    id: 'final.ln',
    phase: 'forward',
    kind: 'layernorm',
    tensor: finalLn.out,
    inputs: [
      { node: `block${nLayer - 1}.res2`, role: 'input' },
      { weight: 'final_ln.g', role: 'weight' },
      { weight: 'final_ln.b', role: 'bias' },
    ],
    meta: { eps: lnEps },
    costs: cost.layernorm(T, d),
  });
  rec?.({
    id: 'final.ln.mean',
    phase: 'forward',
    kind: 'layernorm',
    tensor: { data: finalLn.mean, shape: [T] },
    inputs: [{ node: 'final.ln', role: 'input' }],
    costs: cost.none(),
  });
  rec?.({
    id: 'final.ln.rstd',
    phase: 'forward',
    kind: 'layernorm',
    tensor: { data: finalLn.rstd, shape: [T] },
    inputs: [{ node: 'final.ln', role: 'input' }],
    costs: cost.none(),
  });

  // Weight tying: logits = x @ wteᵀ — the same 65×192 matrix that embedded
  // the tokens now scores them.
  const logits = matmulTB(finalLn.out, wte);
  rec?.({
    id: 'logits',
    phase: 'forward',
    kind: 'matmul',
    tensor: logits,
    inputs: [
      { node: 'final.ln', role: 'input' },
      { weight: 'wte', role: 'weight' },
    ],
    meta: { M: T, K: d, N: vocabSize, transposeB: true, tied: true },
    costs: cost.matmul(T, d, vocabSize),
  });

  return { ids, T, tok, pos, embSum: blocks[0].input, blocks, finalLn, logits };
}

/* ─── loss + backward ─────────────────────────────────────────────────── */

export interface BackwardResult {
  loss: number;
  perToken: Float32Array;
  grads: Map<string, Float32Array>; // keyed by canonical weight names
}

export function lossBackward(
  weights: Weights,
  cfg: Config,
  ctx: ForwardCtx,
  targets: Int32Array,
  tape?: Tape,
): BackwardResult {
  const { nLayer, nEmbd: d, nHead, headDim, vocabSize, blockSize } = cfg;
  const T = ctx.T;
  const rec = tape ? tape.rec.bind(tape) : null;
  const grads = new Map<string, Float32Array>();
  const scaleF = 1 / Math.sqrt(headDim);
  const wte = w(weights, 'wte');

  const { perToken, loss } = crossEntropy(ctx.logits, targets);
  rec?.({
    id: 'loss.per_token',
    phase: 'forward',
    kind: 'loss',
    tensor: { data: perToken, shape: [T] },
    inputs: [{ node: 'logits', role: 'input' }],
    meta: { targets: Array.from(targets) },
    costs: cost.softmax(T, vocabSize),
  });
  rec?.({
    id: 'loss',
    phase: 'forward',
    kind: 'loss',
    tensor: { data: Float32Array.of(loss), shape: [1] },
    inputs: [{ node: 'loss.per_token', role: 'input' }],
    costs: cost.none(),
  });

  // Fused softmax–cross-entropy gradient: (p − onehot)/T.
  const dLogits = crossEntropyBwd(ctx.logits, targets);
  rec?.({
    id: 'grad:logits',
    phase: 'backward',
    kind: 'loss',
    gradOf: 'logits',
    tensor: dLogits,
    inputs: [{ node: 'loss', role: 'input' }],
    meta: { fused: 'softmax+crossentropy' },
    costs: cost.softmax(T, vocabSize),
  });

  // Tied weight, contribution 1 of 2: from the logits matmul.
  const dWte = matmulTA(dLogits, ctx.finalLn.out); // [V,d]
  let dX = matmul(dLogits, wte); // [T,d]
  rec?.({
    id: 'grad:final.ln',
    phase: 'backward',
    kind: 'matmul',
    gradOf: 'final.ln',
    tensor: dX,
    inputs: [{ node: 'grad:logits', role: 'input' }],
    costs: cost.matmul(T, vocabSize, d),
  });

  {
    const lastRes2 = ctx.blocks[nLayer - 1].res2;
    const g = layernormBwd(dX, lastRes2, w(weights, 'final_ln.g'), ctx.finalLn.mean, ctx.finalLn.rstd);
    grads.set('final_ln.g', g.dg);
    grads.set('final_ln.b', g.db);
    dX = g.dx;
  }

  for (let b = nLayer - 1; b >= 0; b--) {
    const p = `block${b}`;
    const B = ctx.blocks[b];
    // dX is the gradient at res2.
    rec?.({
      id: `grad:${p}.res2`,
      phase: 'backward',
      kind: 'add',
      layer: b,
      gradOf: `${p}.res2`,
      tensor: dX,
      inputs: [],
      costs: cost.none(),
    });

    // res2 = res1 + mlpProj: gradient copies to both branches.
    const dMlpProj = dX;
    rec?.({
      id: `grad:${p}.mlp.proj`,
      phase: 'backward',
      kind: 'linear',
      layer: b,
      gradOf: `${p}.mlp.proj`,
      tensor: dMlpProj,
      inputs: [{ node: `grad:${p}.res2`, role: 'input' }],
      costs: cost.none(),
    });

    const mlpProjBwd = linearBwd(dMlpProj, B.gelu, w(weights, `${p}.mlp.proj.w`));
    grads.set(`${p}.mlp.proj.w`, mlpProjBwd.dw.data);
    grads.set(`${p}.mlp.proj.b`, mlpProjBwd.db.data);
    const dGelu = mlpProjBwd.dx;
    rec?.({
      id: `grad:${p}.mlp.gelu`,
      phase: 'backward',
      kind: 'gelu',
      layer: b,
      gradOf: `${p}.mlp.gelu`,
      tensor: dGelu,
      inputs: [{ node: `grad:${p}.mlp.proj`, role: 'input' }],
      costs: cost.matmul(T, d, 4 * d),
    });

    const dFc = geluTanhBwd(dGelu, B.fc);
    rec?.({
      id: `grad:${p}.mlp.fc`,
      phase: 'backward',
      kind: 'gelu',
      layer: b,
      gradOf: `${p}.mlp.fc`,
      tensor: dFc,
      inputs: [{ node: `grad:${p}.mlp.gelu`, role: 'input' }],
      costs: cost.elementwise(dFc.data.length, 2, 20),
    });

    const fcBwd = linearBwd(dFc, B.ln2.out, w(weights, `${p}.mlp.fc.w`));
    grads.set(`${p}.mlp.fc.w`, fcBwd.dw.data);
    grads.set(`${p}.mlp.fc.b`, fcBwd.db.data);
    const dLn2 = fcBwd.dx;
    rec?.({
      id: `grad:${p}.ln2`,
      phase: 'backward',
      kind: 'layernorm',
      layer: b,
      gradOf: `${p}.ln2`,
      tensor: dLn2,
      inputs: [{ node: `grad:${p}.mlp.fc`, role: 'input' }],
      costs: cost.matmul(T, 4 * d, d),
    });

    const ln2Bwd = layernormBwd(dLn2, B.res1, w(weights, `${p}.ln2.g`), B.ln2.mean, B.ln2.rstd);
    grads.set(`${p}.ln2.g`, ln2Bwd.dg);
    grads.set(`${p}.ln2.b`, ln2Bwd.db);

    // Gradient at res1 = skip contribution (dX) + ln2 path.
    const dRes1 = add(dX, ln2Bwd.dx);
    rec?.({
      id: `grad:${p}.res1`,
      phase: 'backward',
      kind: 'add',
      layer: b,
      gradOf: `${p}.res1`,
      tensor: dRes1,
      inputs: [
        { node: `grad:${p}.res2`, role: 'input' },
        { node: `grad:${p}.ln2`, role: 'input' },
      ],
      costs: cost.elementwise(T * d, 2, 1),
    });

    const dAttnProj = dRes1;
    rec?.({
      id: `grad:${p}.attn.proj`,
      phase: 'backward',
      kind: 'linear',
      layer: b,
      gradOf: `${p}.attn.proj`,
      tensor: dAttnProj,
      inputs: [{ node: `grad:${p}.res1`, role: 'input' }],
      costs: cost.none(),
    });

    const projBwd = linearBwd(dAttnProj, B.merged, w(weights, `${p}.attn.proj.w`));
    grads.set(`${p}.attn.proj.w`, projBwd.dw.data);
    grads.set(`${p}.attn.proj.b`, projBwd.db.data);

    const dWv = mergeHeadsBwd(projBwd.dx, nHead, headDim);
    rec?.({
      id: `grad:${p}.attn.wv`,
      phase: 'backward',
      kind: 'matmul',
      layer: b,
      gradOf: `${p}.attn.wv`,
      tensor: dWv,
      inputs: [{ node: `grad:${p}.attn.proj`, role: 'input' }],
      costs: cost.matmul(T, d, d),
    });

    // wv = att @ v  →  dAtt = dWv @ vᵀ ;  dV = attᵀ @ dWv (per head)
    const dAtt = perHead(matmulTB, dWv, B.v, T, T);
    const dV = perHead(matmulTA, B.att, dWv, T, headDim);
    rec?.({
      id: `grad:${p}.attn.softmax`,
      phase: 'backward',
      kind: 'softmax',
      layer: b,
      gradOf: `${p}.attn.softmax`,
      tensor: dAtt,
      inputs: [{ node: `grad:${p}.attn.wv`, role: 'input' }],
      costs: cost.matmul(T, headDim, T * nHead),
    });

    const dMasked = softmaxRowsBwd(dAtt, B.att);
    rec?.({
      id: `grad:${p}.attn.masked`,
      phase: 'backward',
      kind: 'softmax',
      layer: b,
      gradOf: `${p}.attn.masked`,
      tensor: dMasked,
      inputs: [{ node: `grad:${p}.attn.softmax`, role: 'input' }],
      costs: cost.softmax(nHead * T, T),
    });

    const dScores = causalMaskBwd(dMasked);
    rec?.({
      id: `grad:${p}.attn.scores`,
      phase: 'backward',
      kind: 'causal_mask',
      layer: b,
      gradOf: `${p}.attn.scores`,
      tensor: dScores,
      inputs: [{ node: `grad:${p}.attn.masked`, role: 'input' }],
      costs: cost.none(),
    });

    const dRaw = { data: dScores.data.slice(), shape: [...dScores.shape] };
    for (let i = 0; i < dRaw.data.length; i++) dRaw.data[i] *= scaleF;
    rec?.({
      id: `grad:${p}.attn.scores_raw`,
      phase: 'backward',
      kind: 'scale',
      layer: b,
      gradOf: `${p}.attn.scores_raw`,
      tensor: dRaw,
      inputs: [{ node: `grad:${p}.attn.scores`, role: 'input' }],
      meta: { factor: scaleF },
      costs: cost.elementwise(dRaw.data.length, 1, 1),
    });

    // raw = q @ kᵀ  →  dQ = dRaw @ k ;  dK = dRawᵀ @ q (per head)
    const dQ = perHead(matmul, dRaw, B.k, T, headDim);
    const dK = perHead(matmulTA, dRaw, B.q, T, headDim);
    for (const [name, t] of [
      ['q', dQ],
      ['k', dK],
      ['v', dV],
    ] as const) {
      rec?.({
        id: `grad:${p}.attn.${name}`,
        phase: 'backward',
        kind: 'view',
        layer: b,
        gradOf: `${p}.attn.${name}`,
        tensor: t,
        inputs: [{ node: `grad:${p}.attn.scores_raw`, role: 'input' }],
        costs: cost.matmul(T, T, headDim * nHead),
      });
    }

    const dQkv = zeros([T, 3 * d]);
    splitHeadsBwd(dQ, 0, dQkv);
    splitHeadsBwd(dK, 1, dQkv);
    splitHeadsBwd(dV, 2, dQkv);
    rec?.({
      id: `grad:${p}.attn.qkv`,
      phase: 'backward',
      kind: 'view',
      layer: b,
      gradOf: `${p}.attn.qkv`,
      tensor: dQkv,
      inputs: [
        { node: `grad:${p}.attn.q`, role: 'input' },
        { node: `grad:${p}.attn.k`, role: 'input' },
        { node: `grad:${p}.attn.v`, role: 'input' },
      ],
      costs: cost.none(),
    });

    const qkvBwd = linearBwd(dQkv, B.ln1.out, w(weights, `${p}.attn.qkv.w`));
    grads.set(`${p}.attn.qkv.w`, qkvBwd.dw.data);
    grads.set(`${p}.attn.qkv.b`, qkvBwd.db.data);
    const dLn1 = qkvBwd.dx;
    rec?.({
      id: `grad:${p}.ln1`,
      phase: 'backward',
      kind: 'layernorm',
      layer: b,
      gradOf: `${p}.ln1`,
      tensor: dLn1,
      inputs: [{ node: `grad:${p}.attn.qkv`, role: 'input' }],
      costs: cost.matmul(T, 3 * d, d),
    });

    const ln1Bwd = layernormBwd(dLn1, B.input, w(weights, `${p}.ln1.g`), B.ln1.mean, B.ln1.rstd);
    grads.set(`${p}.ln1.g`, ln1Bwd.dg);
    grads.set(`${p}.ln1.b`, ln1Bwd.db);

    // Gradient entering the block = skip (dRes1) + ln1 path.
    dX = add(dRes1, ln1Bwd.dx);
  }

  rec?.({
    id: 'grad:emb.sum',
    phase: 'backward',
    kind: 'add',
    gradOf: 'emb.sum',
    tensor: dX,
    inputs: [{ node: 'grad:block0.res1', role: 'input' }],
    costs: cost.none(),
  });
  // The add broadcast both ways: token and positional embeddings each receive dX.
  rec?.({
    id: 'grad:emb.token',
    phase: 'backward',
    kind: 'gather',
    gradOf: 'emb.token',
    tensor: dX,
    inputs: [{ node: 'grad:emb.sum', role: 'input' }],
    costs: cost.none(),
  });
  rec?.({
    id: 'grad:emb.pos',
    phase: 'backward',
    kind: 'gather',
    gradOf: 'emb.pos',
    tensor: dX,
    inputs: [{ node: 'grad:emb.sum', role: 'input' }],
    costs: cost.none(),
  });

  // Tied weight, contribution 2 of 2: scatter-add from the embedding lookup.
  const dWteGather = gatherRowsBwd(dX, ctx.ids, vocabSize);
  for (let i = 0; i < dWte.data.length; i++) dWte.data[i] += dWteGather.data[i];
  grads.set('wte', dWte.data);
  grads.set('wpe', posRowsBwd(dX, blockSize).data);

  return { loss, perToken, grads };
}

/* ─── training step ───────────────────────────────────────────────────── */

/** Deep-copy weights into a mutable parameter set (loaded weights are views
 * over the fetched ArrayBuffer and treated as read-only). */
export function toMutableParams(weights: Weights): Weights {
  const out: Weights = new Map();
  for (const [name, t] of weights) out.set(name, clone(t));
  return out;
}

export function trainStep(
  params: Weights,
  cfg: Config,
  ids: Int32Array,
  targets: Int32Array,
  state: AdamWState,
  hp: AdamWHyper,
  tape?: Tape,
): { loss: number; perToken: Float32Array; grads: Map<string, Float32Array> } {
  const ctx = forward(params, cfg, ids, tape);
  const res = lossBackward(params, cfg, ctx, targets, tape);
  adamwStep(params, res.grads, state, hp);
  return res;
}

/* ─── generation (full re-forward; the KV-cache path lives in kvcache.ts) ── */

export interface SampleOpts {
  temperature: number;
  topK: number;
  rng: Rng | null; // null = greedy
}

export function generateFull(
  weights: Weights,
  cfg: Config,
  prompt: Int32Array,
  maxNewTokens: number,
  opts: SampleOpts,
): Int32Array {
  let ids = prompt;
  for (let i = 0; i < maxNewTokens; i++) {
    const cropped = ids.length > cfg.blockSize ? ids.subarray(ids.length - cfg.blockSize) : ids;
    const ctx = forward(weights, cfg, cropped);
    const T = cropped.length;
    const last = {
      data: ctx.logits.data.subarray((T - 1) * cfg.vocabSize, T * cfg.vocabSize) as Float32Array,
      shape: [cfg.vocabSize],
    };
    const { chosen } = sampleFromLogits(last, opts.temperature, opts.topK, opts.rng);
    const next = new Int32Array(ids.length + 1);
    next.set(ids);
    next[ids.length] = chosen;
    ids = next;
  }
  return ids;
}

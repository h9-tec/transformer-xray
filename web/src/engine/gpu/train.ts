/** One training step on the GPU: forward with activations kept, the
 * backward pass op for op in the order of engine/gpt.ts lossBackward, and
 * AdamW in PyTorch's order. Parameters, moments and gradients all live on
 * the GPU; only the per-token loss comes back each step. */

import type { AdamWHyper } from '@/engine/adamw';
import type { Weights } from '@/engine/gpt';
import type { Config } from '@/engine/weights';
import { Buf, dispatch, submitAndWait, uniform, type Gpu } from './device';
import type { GpuModel } from './model';
import * as g from './ops';

const wg = (n: number, size: number) => Math.max(1, Math.ceil(n / size));

interface Acts {
  T: number;
  ids: Buf;
  targets: Buf;
  x0: Buf; // embedding sum (block 0 input)
  blocks: {
    ln1: Buf; m1: Buf; r1: Buf; qkv: Buf; q: Buf; k: Buf; v: Buf; probs: Buf; wv: Buf; merged: Buf; proj: Buf; res1: Buf;
    ln2: Buf; m2: Buf; r2: Buf; fc: Buf; act: Buf; mlp: Buf; res2: Buf;
  }[];
  lnf: Buf; mf: Buf; rf: Buf; logits: Buf; perToken: Buf;
  // backward scratch
  dLogits: Buf; dX: Buf; dX2: Buf; dRes1: Buf; dAct: Buf; dFc: Buf; dLn: Buf; dMerged: Buf; dWv: Buf; dAtt: Buf; dRaw: Buf;
  dQ: Buf; dK: Buf; dV: Buf; dQkv: Buf;
}

export class GpuTrainer {
  t = 0;
  private grads = new Map<string, Buf>();
  private m = new Map<string, Buf>();
  private v = new Map<string, Buf>();
  private acts: Acts | null = null;

  constructor(
    readonly gpu: Gpu,
    readonly model: GpuModel,
    readonly cfg: Config,
    public hp: AdamWHyper,
  ) {
    for (const [name, b] of model.bufs) {
      this.grads.set(name, Buf.zeros(gpu, b.floats, `d.${name}`));
      this.m.set(name, Buf.zeros(gpu, b.floats, `m.${name}`));
      this.v.set(name, Buf.zeros(gpu, b.floats, `v.${name}`));
    }
  }

  private alloc(T: number): Acts {
    if (this.acts && this.acts.T === T) return this.acts;
    this.acts?.ids.destroy();
    const { nLayer, nEmbd: d, nHead: H, headDim: hd, vocabSize: V } = this.cfg;
    const z = (n: number, label: string) => Buf.zeros(this.gpu, n, label);
    const blocks = Array.from({ length: nLayer }, (_, b) => ({
      ln1: z(T * d, `b${b}.ln1`), m1: z(T, 'm1'), r1: z(T, 'r1'), qkv: z(T * 3 * d, 'qkv'), q: z(H * T * hd, 'q'), k: z(H * T * hd, 'k'), v: z(H * T * hd, 'v'),
      probs: z(H * T * T, 'probs'), wv: z(H * T * hd, 'wv'), merged: z(T * d, 'merged'), proj: z(T * d, 'proj'), res1: z(T * d, 'res1'),
      ln2: z(T * d, 'ln2'), m2: z(T, 'm2'), r2: z(T, 'r2'), fc: z(T * 4 * d, 'fc'), act: z(T * 4 * d, 'act'), mlp: z(T * d, 'mlp'), res2: z(T * d, 'res2'),
    }));
    this.acts = {
      T,
      ids: new Buf(this.gpu, T * 4),
      targets: new Buf(this.gpu, T * 4),
      x0: z(T * d, 'x0'),
      blocks,
      lnf: z(T * d, 'lnf'), mf: z(T, 'mf'), rf: z(T, 'rf'), logits: z(T * V, 'logits'), perToken: z(T, 'perToken'),
      dLogits: z(T * V, 'dLogits'), dX: z(T * d, 'dX'), dX2: z(T * d, 'dX2'), dRes1: z(T * d, 'dRes1'), dAct: z(T * 4 * d, 'dAct'), dFc: z(T * 4 * d, 'dFc'),
      dLn: z(T * d, 'dLn'), dMerged: z(T * d, 'dMerged'), dWv: z(H * T * hd, 'dWv'), dAtt: z(H * T * T, 'dAtt'), dRaw: z(H * T * T, 'dRaw'),
      dQ: z(H * T * hd, 'dQ'), dK: z(H * T * hd, 'dK'), dV: z(H * T * hd, 'dV'), dQkv: z(T * 3 * d, 'dQkv'),
    };
    return this.acts;
  }

  /** forward + backward + AdamW on one (x, y) batch; returns the mean loss */
  async step(x: Int32Array, y: Int32Array): Promise<{ loss: number; ms: number }> {
    const t0 = performance.now();
    const { nLayer, nEmbd: d, nHead: H, headDim: hd, vocabSize: V, blockSize, lnEps } = this.cfg;
    const T = x.length;
    const A = this.alloc(T);
    const gpu = this.gpu;
    const M = this.model;
    A.ids.write(x);
    A.targets.write(y);
    const enc = gpu.device.createCommandEncoder();
    const scale = 1 / Math.sqrt(hd);

    // ── forward, keeping every activation ──
    g.embed(gpu, enc, M.get('wte'), A.ids, M.get('wpe'), A.x0, T, d);
    let xin = A.x0;
    for (let b = 0; b < nLayer; b++) {
      const p = `block${b}`;
      const B = A.blocks[b];
      g.layernorm(gpu, enc, xin, M.get(`${p}.ln1.g`), M.get(`${p}.ln1.b`), B.ln1, B.m1, B.r1, T, d, lnEps);
      g.matmul(gpu, enc, B.ln1, M.get(`${p}.attn.qkv.w`), B.qkv, T, d, 3 * d, { bias: M.get(`${p}.attn.qkv.b`) });
      g.splitHeads(gpu, enc, B.qkv, B.q, T, d, H, hd, 0);
      g.splitHeads(gpu, enc, B.qkv, B.k, T, d, H, hd, 1);
      g.splitHeads(gpu, enc, B.qkv, B.v, T, d, H, hd, 2);
      g.matmul(gpu, enc, B.q, B.k, A.dAtt, T, hd, T, { transB: true, batch: H, strideA: T * hd, strideB: T * hd, strideC: T * T }); // raw scores (scratch)
      g.causalSoftmax(gpu, enc, A.dAtt, B.probs, H, T, scale);
      g.matmul(gpu, enc, B.probs, B.v, B.wv, T, T, hd, { batch: H, strideA: T * T, strideB: T * hd, strideC: T * hd });
      g.mergeHeads(gpu, enc, B.wv, B.merged, T, d, H, hd);
      g.matmul(gpu, enc, B.merged, M.get(`${p}.attn.proj.w`), B.proj, T, d, d, { bias: M.get(`${p}.attn.proj.b`) });
      g.add(gpu, enc, xin, B.proj, B.res1, T * d);
      g.layernorm(gpu, enc, B.res1, M.get(`${p}.ln2.g`), M.get(`${p}.ln2.b`), B.ln2, B.m2, B.r2, T, d, lnEps);
      g.matmul(gpu, enc, B.ln2, M.get(`${p}.mlp.fc.w`), B.fc, T, d, 4 * d, { bias: M.get(`${p}.mlp.fc.b`) });
      g.gelu(gpu, enc, B.fc, B.act, T * 4 * d);
      g.matmul(gpu, enc, B.act, M.get(`${p}.mlp.proj.w`), B.mlp, T, 4 * d, d, { bias: M.get(`${p}.mlp.proj.b`) });
      g.add(gpu, enc, B.res1, B.mlp, B.res2, T * d);
      xin = B.res2;
    }
    g.layernorm(gpu, enc, xin, M.get('final_ln.g'), M.get('final_ln.b'), A.lnf, A.mf, A.rf, T, d, lnEps);
    g.matmul(gpu, enc, A.lnf, M.get('wte'), A.logits, T, d, V, { transB: true });

    // ── loss and backward ──
    const G = (name: string) => this.grads.get(name)!;
    dispatch(gpu, enc, 'cross_entropy', [A.logits.buffer, A.targets.buffer, A.perToken.buffer, A.dLogits.buffer], uniform(gpu, [T, V]), [T, 1, 1]);
    // tied weight, part 1: dWte = dLogitsᵀ · lnf   [V,d]
    g.matmul(gpu, enc, A.dLogits, A.lnf, G('wte'), V, T, d, { transA: true });
    // dX = dLogits · wte  [T,d]
    g.matmul(gpu, enc, A.dLogits, M.get('wte'), A.dX, T, V, d);
    // final layernorm backward
    const lastRes2 = A.blocks[nLayer - 1].res2;
    this.lnBwd(enc, A.dX, lastRes2, 'final_ln', A.mf, A.rf, A.dX2, T, d);
    let dX = A.dX2; // gradient at res2 of the last block
    let dXother = A.dX;

    for (let b = nLayer - 1; b >= 0; b--) {
      const p = `block${b}`;
      const B = A.blocks[b];
      // mlp.proj: dw = actᵀ·dX ; db = colsum dX ; dAct = dX · Wᵀ
      g.matmul(gpu, enc, B.act, dX, G(`${p}.mlp.proj.w`), 4 * d, T, d, { transA: true });
      this.colSum(enc, dX, G(`${p}.mlp.proj.b`), T, d);
      g.matmul(gpu, enc, dX, M.get(`${p}.mlp.proj.w`), A.dAct, T, d, 4 * d, { transB: true });
      dispatch(gpu, enc, 'gelu_bwd', [A.dAct.buffer, B.fc.buffer, A.dFc.buffer], uniform(gpu, [T * 4 * d]), [wg(T * 4 * d, 256), 1, 1]);
      // mlp.fc: dw = ln2ᵀ·dFc ; db ; dLn2 = dFc · Wᵀ
      g.matmul(gpu, enc, B.ln2, A.dFc, G(`${p}.mlp.fc.w`), d, T, 4 * d, { transA: true });
      this.colSum(enc, A.dFc, G(`${p}.mlp.fc.b`), T, 4 * d);
      g.matmul(gpu, enc, A.dFc, M.get(`${p}.mlp.fc.w`), A.dLn, T, 4 * d, d, { transB: true });
      // ln2 backward → dXother ; dRes1 = dX + dXother
      this.lnBwd(enc, A.dLn, B.res1, `${p}.ln2`, B.m2, B.r2, dXother, T, d);
      g.add(gpu, enc, dX, dXother, A.dRes1, T * d);
      // attn.proj: dw = mergedᵀ·dRes1 ; db ; dMerged = dRes1 · Wᵀ
      g.matmul(gpu, enc, B.merged, A.dRes1, G(`${p}.attn.proj.w`), d, T, d, { transA: true });
      this.colSum(enc, A.dRes1, G(`${p}.attn.proj.b`), T, d);
      g.matmul(gpu, enc, A.dRes1, M.get(`${p}.attn.proj.w`), A.dMerged, T, d, d, { transB: true });
      dispatch(gpu, enc, 'merged_to_heads', [A.dMerged.buffer, A.dWv.buffer], uniform(gpu, [T, d, H, hd]), [wg(H * T * hd, 256), 1, 1]);
      // dAtt = dWv · vᵀ ; dV = attᵀ · dWv   (per head)
      g.matmul(gpu, enc, A.dWv, B.v, A.dAtt, T, hd, T, { transB: true, batch: H, strideA: T * hd, strideB: T * hd, strideC: T * T });
      g.matmul(gpu, enc, B.probs, A.dWv, A.dV, T, T, hd, { transA: true, batch: H, strideA: T * T, strideB: T * hd, strideC: T * hd });
      // softmax + mask + scale backward
      dispatch(gpu, enc, 'softmax_bwd_causal', [A.dAtt.buffer, B.probs.buffer, A.dRaw.buffer], uniform(gpu, [H, T, { f: scale }]), [H * T, 1, 1]);
      // dQ = dRaw · k ; dK = dRawᵀ · q   (per head)
      g.matmul(gpu, enc, A.dRaw, B.k, A.dQ, T, T, hd, { batch: H, strideA: T * T, strideB: T * hd, strideC: T * hd });
      g.matmul(gpu, enc, A.dRaw, B.q, A.dK, T, T, hd, { transA: true, batch: H, strideA: T * T, strideB: T * hd, strideC: T * hd });
      for (const [which, buf] of [[0, A.dQ], [1, A.dK], [2, A.dV]] as const) {
        dispatch(gpu, enc, 'heads_to_qkv', [buf.buffer, A.dQkv.buffer], uniform(gpu, [T, d, H, hd, which]), [wg(H * T * hd, 256), 1, 1]);
      }
      // attn.qkv: dw = ln1ᵀ·dQkv ; db ; dLn1 = dQkv · Wᵀ
      g.matmul(gpu, enc, B.ln1, A.dQkv, G(`${p}.attn.qkv.w`), d, T, 3 * d, { transA: true });
      this.colSum(enc, A.dQkv, G(`${p}.attn.qkv.b`), T, 3 * d);
      g.matmul(gpu, enc, A.dQkv, M.get(`${p}.attn.qkv.w`), A.dLn, T, 3 * d, d, { transB: true });
      // ln1 backward → dXother ; dX = dRes1 + dXother
      const input = b === 0 ? A.x0 : A.blocks[b - 1].res2;
      this.lnBwd(enc, A.dLn, input, `${p}.ln1`, B.m1, B.r1, dXother, T, d);
      g.add(gpu, enc, A.dRes1, dXother, dX, T * d);
      // dX now holds the gradient entering this block (= res2 grad of the previous block)
    }
    // embeddings: tied weight part 2 (scatter-add into dWte), positions
    dispatch(gpu, enc, 'wte_bwd', [dX.buffer, A.ids.buffer, G('wte').buffer], uniform(gpu, [T, d, V]), [wg(V * d, 256), 1, 1]);
    dispatch(gpu, enc, 'wpe_bwd', [dX.buffer, G('wpe').buffer], uniform(gpu, [T, d, blockSize]), [wg(blockSize * d, 256), 1, 1]);

    // ── AdamW ──
    this.t += 1;
    const bc1 = 1 - Math.pow(this.hp.beta1, this.t);
    const bc2 = 1 - Math.pow(this.hp.beta2, this.t);
    const decay = 1 - this.hp.lr * this.hp.weightDecay;
    for (const [name, param] of M.bufs) {
      const n = param.floats;
      dispatch(
        gpu,
        enc,
        'adamw_step',
        [param.buffer, G(name).buffer, this.m.get(name)!.buffer, this.v.get(name)!.buffer],
        uniform(gpu, [n, { f: this.hp.lr }, { f: this.hp.beta1 }, { f: this.hp.beta2 }, { f: this.hp.eps }, { f: decay }, { f: bc1 }, { f: bc2 }]),
        [wg(n, 256), 1, 1],
      );
    }
    await submitAndWait(gpu, enc);
    const per = await A.perToken.read();
    let loss = 0;
    for (let i = 0; i < T; i++) loss += per[i];
    return { loss: loss / T, ms: performance.now() - t0 };
  }

  private lnBwd(enc: GPUCommandEncoder, dy: Buf, x: Buf, prefix: string, mean: Buf, rstd: Buf, dx: Buf, T: number, D: number): void {
    const gpu = this.gpu;
    dispatch(gpu, enc, 'layernorm_bwd_dx', [dy.buffer, x.buffer, this.model.get(`${prefix}.g`).buffer, mean.buffer, rstd.buffer, dx.buffer], uniform(gpu, [T, D]), [T, 1, 1]);
    dispatch(gpu, enc, 'layernorm_bwd_dgdb', [dy.buffer, x.buffer, mean.buffer, rstd.buffer, this.grads.get(`${prefix}.g`)!.buffer, this.grads.get(`${prefix}.b`)!.buffer], uniform(gpu, [T, D]), [wg(D, 64), 1, 1]);
  }

  private colSum(enc: GPUCommandEncoder, dy: Buf, db: Buf, T: number, N: number): void {
    dispatch(this.gpu, enc, 'bias_grad', [dy.buffer, db.buffer], uniform(this.gpu, [T, N]), [wg(N, 64), 1, 1]);
  }

  download(): Promise<Weights> {
    return this.model.download();
  }

  destroy(): void {
    for (const b of this.grads.values()) b.destroy();
    for (const b of this.m.values()) b.destroy();
    for (const b of this.v.values()) b.destroy();
  }
}

/** In-browser parity checks: every GPU op against its CPU twin at the
 * model's shapes. Node has no WebGPU, so this runs on /debug/gpu and a
 * Playwright project reads the JSON it prints. */

import { add as cpuAdd } from '@/engine/ops/elementwise';
import { gatherRows, posRows } from '@/engine/ops/embedding';
import { geluTanh } from '@/engine/ops/gelu';
import { mergeHeads as cpuMerge, splitHeads as cpuSplit } from '@/engine/ops/heads';
import { layernorm as cpuLayernorm } from '@/engine/ops/layernorm';
import { causalMask } from '@/engine/ops/mask';
import { matmul as cpuMatmul, matmulTA as cpuMatmulTA, matmulTB as cpuMatmulTB } from '@/engine/ops/matmul';
import { softmaxRows } from '@/engine/ops/softmax';
import { mulberry32 } from '@/engine/rng';
import type { Tensor } from '@/engine/tensor';
import { Buf, getGpu, submitAndWait, type Gpu } from './device';
import * as g from './ops';

export interface GpuReport {
  adapter: string | null;
  reason?: string;
  kernels?: Record<string, { maxDiff: number; ms: number }>;
  forwardMaxDiff?: number;
  argmaxMatch?: boolean;
  trainMaxDiff?: number;
  lossGpu?: number;
  lossCpu?: number;
}

const T = 128;
const D = 192;
const H = 6;
const HD = 32;
const V = 65;

function rnd(n: number, seed: number, scale = 1): Float32Array {
  const r = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (r() - 0.5) * 2 * scale;
  return out;
}
const t = (data: Float32Array, shape: number[]): Tensor => ({ data, shape });

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (Number.isNaN(d)) return Number.NaN;
    if (d > m) m = d;
  }
  return m;
}

type Case = { name: string; run: (gpu: Gpu) => Promise<{ got: Float32Array; want: Float32Array }> };

const CASES: Case[] = [
  {
    name: 'embedding_gather',
    run: async (gpu) => {
      const wte = rnd(V * D, 1, 0.02);
      const wpe = rnd(256 * D, 2, 0.02);
      const ids = Int32Array.from({ length: T }, (_, i) => (i * 7) % V);
      const out = Buf.zeros(gpu, T * D);
      const enc = gpu.device.createCommandEncoder();
      g.embed(gpu, enc, Buf.from(gpu, wte), Buf.from(gpu, ids), Buf.from(gpu, wpe), out, T, D);
      await submitAndWait(gpu, enc);
      const want = cpuAdd(gatherRows(t(wte, [V, D]), ids), posRows(t(wpe, [256, D]), T)).data;
      return { got: await out.read(), want };
    },
  },
  {
    name: 'layernorm_warp',
    run: async (gpu) => {
      const x = rnd(T * D, 3, 2);
      const gg = rnd(D, 4, 0.5).map((v) => 1 + v);
      const bb = rnd(D, 5, 0.1);
      const out = Buf.zeros(gpu, T * D);
      const enc = gpu.device.createCommandEncoder();
      g.layernorm(gpu, enc, Buf.from(gpu, x), Buf.from(gpu, gg), Buf.from(gpu, bb), out, Buf.zeros(gpu, T), Buf.zeros(gpu, T), T, D, 1e-5);
      await submitAndWait(gpu, enc);
      return { got: await out.read(), want: cpuLayernorm(t(x, [T, D]), t(gg, [D]), t(bb, [D]), 1e-5).out.data };
    },
  },
  {
    name: 'matmul_tiled',
    run: async (gpu) => {
      const A = rnd(T * D, 6, 1);
      const B = rnd(D * 3 * D, 7, 0.1);
      const bias = rnd(3 * D, 8, 0.1);
      const C = Buf.zeros(gpu, T * 3 * D);
      const enc = gpu.device.createCommandEncoder();
      g.matmul(gpu, enc, Buf.from(gpu, A), Buf.from(gpu, B), C, T, D, 3 * D, { bias: Buf.from(gpu, bias) });
      await submitAndWait(gpu, enc);
      const want = cpuMatmul(t(A, [T, D]), t(B, [D, 3 * D])).data;
      for (let i = 0; i < T; i++) for (let j = 0; j < 3 * D; j++) want[i * 3 * D + j] = Math.fround(want[i * 3 * D + j] + bias[j]);
      return { got: await C.read(), want };
    },
  },
  {
    name: 'matmul_tiled_TB',
    run: async (gpu) => {
      // scores = Q · Kᵀ per head: batched [H] × [T,hd]·[T,hd]ᵀ
      const Q = rnd(H * T * HD, 9, 1);
      const K = rnd(H * T * HD, 10, 1);
      const C = Buf.zeros(gpu, H * T * T);
      const enc = gpu.device.createCommandEncoder();
      g.matmul(gpu, enc, Buf.from(gpu, Q), Buf.from(gpu, K), C, T, HD, T, { transB: true, batch: H, strideA: T * HD, strideB: T * HD, strideC: T * T });
      await submitAndWait(gpu, enc);
      const want = new Float32Array(H * T * T);
      for (let h = 0; h < H; h++) {
        const r = cpuMatmulTB(t(Q.subarray(h * T * HD, (h + 1) * T * HD), [T, HD]), t(K.subarray(h * T * HD, (h + 1) * T * HD), [T, HD]));
        want.set(r.data, h * T * T);
      }
      return { got: await C.read(), want };
    },
  },
  {
    name: 'matmul_tiled_TA',
    run: async (gpu) => {
      // dW = Xᵀ · dY : [D,T]·[T,3D]
      const X = rnd(T * D, 11, 1);
      const dY = rnd(T * 3 * D, 12, 1);
      const C = Buf.zeros(gpu, D * 3 * D);
      const enc = gpu.device.createCommandEncoder();
      g.matmul(gpu, enc, Buf.from(gpu, X), Buf.from(gpu, dY), C, D, T, 3 * D, { transA: true });
      await submitAndWait(gpu, enc);
      return { got: await C.read(), want: cpuMatmulTA(t(X, [T, D]), t(dY, [T, 3 * D])).data };
    },
  },
  {
    name: 'split_heads',
    run: async (gpu) => {
      const qkv = rnd(T * 3 * D, 13, 1);
      const out = Buf.zeros(gpu, H * T * HD);
      const enc = gpu.device.createCommandEncoder();
      g.splitHeads(gpu, enc, Buf.from(gpu, qkv), out, T, D, H, HD, 1);
      await submitAndWait(gpu, enc);
      return { got: await out.read(), want: cpuSplit(t(qkv, [T, 3 * D]), 1, H, HD).data };
    },
  },
  {
    name: 'merge_heads',
    run: async (gpu) => {
      const x = rnd(H * T * HD, 14, 1);
      const out = Buf.zeros(gpu, T * D);
      const enc = gpu.device.createCommandEncoder();
      g.mergeHeads(gpu, enc, Buf.from(gpu, x), out, T, D, H, HD);
      await submitAndWait(gpu, enc);
      return { got: await out.read(), want: cpuMerge(t(x, [H, T, HD])).data };
    },
  },
  {
    name: 'causal_softmax',
    run: async (gpu) => {
      const s = rnd(H * T * T, 15, 3);
      const scale = 1 / Math.sqrt(HD);
      const out = Buf.zeros(gpu, H * T * T);
      const enc = gpu.device.createCommandEncoder();
      g.causalSoftmax(gpu, enc, Buf.from(gpu, s), out, H, T, scale);
      await submitAndWait(gpu, enc);
      const scaled = Float32Array.from(s, (v) => Math.fround(v * Math.fround(scale)));
      return { got: await out.read(), want: softmaxRows(causalMask(t(scaled, [H, T, T]))).data };
    },
  },
  {
    name: 'gelu_forward',
    run: async (gpu) => {
      const x = rnd(T * 4 * D, 16, 3);
      const y = Buf.zeros(gpu, T * 4 * D);
      const enc = gpu.device.createCommandEncoder();
      g.gelu(gpu, enc, Buf.from(gpu, x), y, T * 4 * D);
      await submitAndWait(gpu, enc);
      return { got: await y.read(), want: geluTanh(t(x, [T * 4 * D])).data };
    },
  },
  {
    name: 'residual_add',
    run: async (gpu) => {
      const a = rnd(T * D, 17, 1);
      const b = rnd(T * D, 18, 1);
      const out = Buf.zeros(gpu, T * D);
      const enc = gpu.device.createCommandEncoder();
      g.add(gpu, enc, Buf.from(gpu, a), Buf.from(gpu, b), out, T * D);
      await submitAndWait(gpu, enc);
      return { got: await out.read(), want: cpuAdd(t(a, [T * D]), t(b, [T * D])).data };
    },
  },
];

export async function runGpuSelfTest(level: 'kernels' | 'forward' | 'train' = 'kernels'): Promise<GpuReport> {
  const gpu = await getGpu();
  if (!gpu) return { adapter: null, reason: 'navigator.gpu is missing or no adapter was granted' };
  const report: GpuReport = { adapter: gpu.adapterInfo, kernels: {} };
  for (const c of CASES) {
    const t0 = performance.now();
    try {
      const { got, want } = await c.run(gpu);
      report.kernels![c.name] = { maxDiff: maxAbsDiff(got, want), ms: performance.now() - t0 };
    } catch (e) {
      report.kernels![c.name] = { maxDiff: Number.POSITIVE_INFINITY, ms: performance.now() - t0 };
      report.reason = `${c.name}: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
  if (level === 'kernels') return report;
  const more = await import('./selftest-model');
  return more.extendReport(gpu, report, level);
}

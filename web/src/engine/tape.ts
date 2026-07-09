/** The trace tape: every op the model executes records a TraceNode with a
 * snapshot of its output, provenance (inputs), and exact cost arithmetic.
 * The inspectors render nothing that didn't come off this tape. */

import type { Tensor } from './tensor';

export type Phase = 'forward' | 'backward' | 'update';

export type OpKind =
  | 'tokenize'
  | 'gather'
  | 'add'
  | 'layernorm'
  | 'linear'
  | 'matmul'
  | 'scale'
  | 'causal_mask'
  | 'softmax'
  | 'gelu'
  | 'view'
  | 'loss'
  | 'sample'
  | 'adamw';

export interface TensorRefIn {
  node?: string; // upstream trace node id
  weight?: string; // or a weight tensor name
  role: 'input' | 'weight' | 'bias' | 'mask' | 'target';
}

export interface Costs {
  flops: number;
  bytesRead: number;
  bytesWritten: number;
}

export interface Stats {
  min: number; // over finite values
  max: number;
  mean: number;
  absmax: number;
  nonFinite: number; // count of ±Inf entries (masked scores)
}

export interface TraceNode {
  id: string;
  step: number;
  phase: Phase;
  kind: OpKind;
  layer: number | null;
  head: number | null;
  gradOf?: string; // backward nodes point at their forward twin
  inputs: TensorRefIn[];
  shape: number[];
  data: Float32Array | null;
  meta: Record<string, unknown>;
  stats: Stats;
  costs: Costs;
}

const ZERO_COST: Costs = { flops: 0, bytesRead: 0, bytesWritten: 0 };

export function computeStats(data: Float32Array | null): Stats {
  if (!data || data.length === 0) return { min: 0, max: 0, mean: 0, absmax: 0, nonFinite: 0 };
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let absmax = 0;
  let nonFinite = 0;
  let n = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) {
      nonFinite++;
      continue;
    }
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    const a = Math.abs(v);
    if (a > absmax) absmax = a;
    n++;
  }
  return { min, max, mean: n ? sum / n : 0, absmax, nonFinite };
}

export interface RecInput {
  id: string;
  phase: Phase;
  kind: OpKind;
  tensor: Tensor | null;
  layer?: number | null;
  head?: number | null;
  gradOf?: string;
  inputs?: TensorRefIn[];
  meta?: Record<string, unknown>;
  costs?: Costs;
  /** Copy the tensor data (default true). Pass false when the caller
   * guarantees the buffer is never reused. */
  snapshot?: boolean;
}

export class Tape {
  nodes: TraceNode[] = [];
  private index = new Map<string, TraceNode>();

  rec(r: RecInput): void {
    const data = r.tensor ? (r.snapshot === false ? r.tensor.data : r.tensor.data.slice()) : null;
    const node: TraceNode = {
      id: r.id,
      step: this.nodes.length,
      phase: r.phase,
      kind: r.kind,
      layer: r.layer ?? null,
      head: r.head ?? null,
      gradOf: r.gradOf,
      inputs: r.inputs ?? [],
      shape: r.tensor ? [...r.tensor.shape] : [],
      data,
      meta: r.meta ?? {},
      stats: computeStats(data),
      costs: r.costs ?? ZERO_COST,
    };
    this.nodes.push(node);
    this.index.set(node.id, node);
  }

  get(id: string): TraceNode | undefined {
    return this.index.get(id);
  }
}

/** Exact cost arithmetic — the GPU panel's numbers all come from here. */
export const cost = {
  matmul(M: number, K: number, N: number): Costs {
    return { flops: 2 * M * K * N, bytesRead: 4 * (M * K + K * N), bytesWritten: 4 * M * N };
  },
  /** flopsPerEl is approximate for transcendental ops (documented as such). */
  elementwise(n: number, reads: number, flopsPerEl: number): Costs {
    return { flops: flopsPerEl * n, bytesRead: 4 * reads * n, bytesWritten: 4 * n };
  },
  layernorm(T: number, D: number): Costs {
    return { flops: 8 * T * D, bytesRead: 4 * (T * D + 2 * D), bytesWritten: 4 * T * D };
  },
  softmax(rows: number, D: number): Costs {
    return { flops: 5 * rows * D, bytesRead: 4 * rows * D, bytesWritten: 4 * rows * D };
  },
  gather(T: number, D: number): Costs {
    return { flops: 0, bytesRead: 4 * T * D + 4 * T, bytesWritten: 4 * T * D };
  },
  none(): Costs {
    return ZERO_COST;
  },
};

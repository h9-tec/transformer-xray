/** Typed messages across the worker boundary. Tensor payloads move as
 * transferable ArrayBuffers — zero copies. */

import type { Costs, OpKind, Phase, Stats, TensorRefIn } from '../tape';
import type { ModelConfigJson } from '../weights';

export interface GenerateParams {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topK: number;
  seed: number | null; // null = greedy
}

export type WorkerRequest =
  | { id: number; type: 'init'; base: string }
  | { id: number; type: 'generate'; params: GenerateParams }
  | { id: number; type: 'trace'; text: string; targets?: string };

/** TraceNode with its Float32Array detached into a transferable buffer. */
export interface WireNode {
  id: string;
  step: number;
  phase: Phase;
  kind: OpKind;
  layer: number | null;
  head: number | null;
  gradOf?: string;
  inputs: TensorRefIn[];
  shape: number[];
  buffer: ArrayBuffer | null;
  meta: Record<string, unknown>;
  stats: Stats;
  costs: Costs;
}

export type WorkerResponse =
  | { id: number; type: 'ready'; cfgJson: ModelConfigJson; chars: string }
  | {
      id: number;
      type: 'token';
      index: number;
      token: number;
      /** raw last-position logits (pre-temperature), transferable */
      logits: ArrayBuffer;
    }
  | { id: number; type: 'generated'; text: string }
  | { id: number; type: 'traced'; nodes: WireNode[]; loss?: number }
  | { id: number; type: 'error'; message: string };

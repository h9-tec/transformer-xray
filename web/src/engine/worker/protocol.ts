/** Typed messages across the worker boundary. Tensor payloads move as
 * transferable ArrayBuffers — zero copies. */

import type { BackendName, BackendStatus } from '../backend';
import type { Costs, OpKind, Phase, Stats, TensorRefIn } from '../tape';
import type { ModelConfigJson } from '../weights';

export interface GenerateParams {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topK: number;
  seed: number | null; // null = greedy
}

export interface TrainParams {
  steps: number;
  seqLen: number;
  lr: number;
  /** post a generation sample every N steps (0 = never) */
  sampleEvery: number;
}

export type WorkerRequest =
  | {
      id: number;
      type: 'init';
      /** weights transferred from the main thread — the worker never fetches */
      cfgJson: ModelConfigJson;
      chars: string;
      weightsBuffer: ArrayBuffer;
    }
  | { id: number; type: 'generate'; params: GenerateParams }
  | { id: number; type: 'trace'; text: string; targets?: string }
  | {
      id: number;
      type: 'trainInit';
      fromScratch: boolean;
      seed: number;
      /** training corpus text, fetched by the main thread */
      corpus: string;
    }
  | { id: number; type: 'trainRun'; params: TrainParams }
  | { id: number; type: 'trainStop' }
  | { id: number; type: 'trainReset' }
  | { id: number; type: 'setBackend'; backend: BackendName };

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

export interface WireParamGrad {
  name: string;
  shape: number[];
  buffer: ArrayBuffer;
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
      /** wall time of this step on the GPU path (WebGPU), when applicable */
      ms?: number;
    }
  | { id: number; type: 'generated'; text: string }
  | {
      id: number;
      type: 'traced';
      nodes: WireNode[];
      loss?: number;
      /** per-parameter gradients (present when targets were given) */
      paramGrads: WireParamGrad[];
      /** true when the trace ran against live training params, not the checkpoint */
      trainedParams: boolean;
    }
  | { id: number; type: 'trainReady'; fromScratch: boolean }
  | { id: number; type: 'trainProgress'; step: number; loss: number; sample?: string; ms?: number }
  | { id: number; type: 'trainDone'; step: number }
  /** id 0 = unsolicited (device lost) */
  | ({ id: number; type: 'backend' } & BackendStatus)
  | { id: number; type: 'error'; message: string };

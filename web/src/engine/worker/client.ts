/** Main-thread facade over the engine worker. */

import type { TraceNode } from '../tape';
import type { ModelConfigJson } from '../weights';
import type {
  GenerateParams,
  TrainParams,
  WireNode,
  WorkerRequest,
  WorkerResponse,
} from './protocol';
import { withBase } from '@/lib/basePath';
import { WORKER_URL } from './worker-url';

export interface TokenEvent {
  index: number;
  token: number;
  logits: Float32Array;
}

export interface ParamGrad {
  name: string;
  shape: number[];
  data: Float32Array;
}

export interface TraceResult {
  nodes: TraceNode[];
  byId: Map<string, TraceNode>;
  loss?: number;
  paramGrads: Map<string, ParamGrad>;
  /** true when traced against live training params rather than the checkpoint */
  trainedParams: boolean;
}

export interface TrainProgress {
  step: number;
  loss: number;
  sample?: string;
}

/** Omit distributed over the request union (plain Omit collapses unions). */
type OutgoingRequest = WorkerRequest extends infer R
  ? R extends WorkerRequest
    ? Omit<R, 'id'>
    : never
  : never;

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  onToken?: (t: TokenEvent) => void;
  onProgress?: (p: TrainProgress) => void;
}

function rehydrate(w: WireNode): TraceNode {
  return {
    id: w.id,
    step: w.step,
    phase: w.phase,
    kind: w.kind,
    layer: w.layer,
    head: w.head,
    gradOf: w.gradOf,
    inputs: w.inputs,
    shape: w.shape,
    data: w.buffer ? new Float32Array(w.buffer) : null,
    meta: w.meta,
    stats: w.stats,
    costs: w.costs,
  };
}

export class EngineClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  constructor() {
    // The worker is bundled by esbuild into public/ (see build:worker) —
    // deliberately outside the app bundler: bundler-worker interop broke
    // silently under static export, and a plain script URL cannot. The
    // filename is content-hashed so caches can never serve a stale engine.
    this.worker = new Worker(withBase(WORKER_URL));
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.dispatch(e.data);
    // A worker that fails to boot emits 'error', not a message — without this,
    // every pending promise would hang forever.
    this.worker.onerror = (e) => this.failAll(`engine worker error: ${e.message ?? 'unknown'}`);
    this.worker.onmessageerror = () => this.failAll('engine worker: message deserialization failed');
  }

  private failAll(message: string): void {
    for (const p of this.pending.values()) p.reject(new Error(message));
    this.pending.clear();
  }

  private dispatch(msg: WorkerResponse): void {
    const p = this.pending.get(msg.id);
    if (!p) return;
    switch (msg.type) {
      case 'token':
        p.onToken?.({ index: msg.index, token: msg.token, logits: new Float32Array(msg.logits) });
        break;
      case 'trainProgress':
        p.onProgress?.({ step: msg.step, loss: msg.loss, sample: msg.sample });
        break;
      case 'ready':
      case 'generated':
      case 'traced':
      case 'trainReady':
      case 'trainDone':
        this.pending.delete(msg.id);
        p.resolve(msg);
        break;
      case 'error':
        this.pending.delete(msg.id);
        p.reject(new Error(msg.message));
        break;
    }
  }

  private send<T>(
    req: OutgoingRequest,
    handlers: Pick<Pending, 'onToken' | 'onProgress'> = {},
    transfer: Transferable[] = [],
  ): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, ...handlers });
      this.worker.postMessage({ ...req, id }, transfer);
    });
  }

  /** Weights are loaded once on the main thread and copied into the worker
   * (transferred, so no serialization cost). The worker never fetches. */
  async init(
    cfgJson: ModelConfigJson,
    chars: string,
    weightsBuffer: ArrayBuffer,
  ): Promise<{ cfgJson: ModelConfigJson; chars: string }> {
    const r = await this.send<WorkerResponse & { type: 'ready' }>(
      { type: 'init', cfgJson, chars, weightsBuffer },
      {},
      [weightsBuffer],
    );
    return { cfgJson: r.cfgJson, chars: r.chars };
  }

  async generate(params: GenerateParams, onToken: (t: TokenEvent) => void): Promise<string> {
    const r = await this.send<WorkerResponse & { type: 'generated' }>(
      { type: 'generate', params },
      { onToken },
    );
    return r.text;
  }

  async trace(text: string, targets?: string): Promise<TraceResult> {
    const r = await this.send<WorkerResponse & { type: 'traced' }>({
      type: 'trace',
      text,
      targets,
    });
    const nodes = r.nodes.map(rehydrate);
    const paramGrads = new Map(
      r.paramGrads.map((g) => [
        g.name,
        { name: g.name, shape: g.shape, data: new Float32Array(g.buffer) },
      ]),
    );
    return {
      nodes,
      byId: new Map(nodes.map((n) => [n.id, n])),
      loss: r.loss,
      paramGrads,
      trainedParams: r.trainedParams,
    };
  }

  async trainInit(fromScratch: boolean, seed: number, corpus: string): Promise<void> {
    await this.send<WorkerResponse & { type: 'trainReady' }>({
      type: 'trainInit',
      fromScratch,
      seed,
      corpus,
    });
  }

  /** Resolves when the run finishes or trainStop is honored. */
  async trainRun(
    params: TrainParams,
    onProgress: (p: TrainProgress) => void,
  ): Promise<{ step: number }> {
    const r = await this.send<WorkerResponse & { type: 'trainDone' }>(
      { type: 'trainRun', params },
      { onProgress },
    );
    return { step: r.step };
  }

  trainStop(): void {
    this.worker.postMessage({ id: this.nextId++, type: 'trainStop' });
  }

  trainReset(): void {
    this.worker.postMessage({ id: this.nextId++, type: 'trainReset' });
  }

  terminate(): void {
    this.worker.terminate();
  }
}

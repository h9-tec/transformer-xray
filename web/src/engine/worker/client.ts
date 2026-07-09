/** Main-thread facade over the engine worker. */

import type { TraceNode } from '../tape';
import type { ModelConfigJson } from '../weights';
import type { GenerateParams, WireNode, WorkerRequest, WorkerResponse } from './protocol';

export interface TokenEvent {
  index: number;
  token: number;
  logits: Float32Array;
}

export interface TraceResult {
  nodes: TraceNode[];
  byId: Map<string, TraceNode>;
  loss?: number;
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
    this.worker = new Worker(new URL('./engine.worker.ts', import.meta.url));
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.dispatch(e.data);
  }

  private dispatch(msg: WorkerResponse): void {
    const p = this.pending.get(msg.id);
    if (!p) return;
    switch (msg.type) {
      case 'token':
        p.onToken?.({ index: msg.index, token: msg.token, logits: new Float32Array(msg.logits) });
        break;
      case 'ready':
      case 'generated':
      case 'traced':
        this.pending.delete(msg.id);
        p.resolve(msg);
        break;
      case 'error':
        this.pending.delete(msg.id);
        p.reject(new Error(msg.message));
        break;
    }
  }

  private send<T>(req: OutgoingRequest, onToken?: (t: TokenEvent) => void): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onToken });
      this.worker.postMessage({ ...req, id });
    });
  }

  async init(base = '/model'): Promise<{ cfgJson: ModelConfigJson; chars: string }> {
    const r = await this.send<WorkerResponse & { type: 'ready' }>({ type: 'init', base });
    return { cfgJson: r.cfgJson, chars: r.chars };
  }

  async generate(params: GenerateParams, onToken: (t: TokenEvent) => void): Promise<string> {
    const r = await this.send<WorkerResponse & { type: 'generated' }>(
      { type: 'generate', params },
      onToken,
    );
    return r.text;
  }

  async trace(text: string, targets?: string): Promise<TraceResult> {
    const r = await this.send<WorkerResponse & { type: 'traced' }>({ type: 'trace', text, targets });
    const nodes = r.nodes.map(rehydrate);
    return { nodes, byId: new Map(nodes.map((n) => [n.id, n])), loss: r.loss };
  }

  terminate(): void {
    this.worker.terminate();
  }
}

/** Main-thread handle on the CUDA worker. One request at a time per client
 * is the normal use (the editor's Run button); concurrent calls queue by id.
 * A crashed worker is restarted and every pending call rejects with
 * 'the kernel runner restarted'. */

import { withBase } from '@/lib/basePath';
import type { CudaRequest, CudaResponse } from './protocol';
import { WORKER_URL } from './worker-url';

type Pending = { resolve: (r: CudaResponse) => void; reject: (e: Error) => void };

export class CudaClient {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = new Worker(withBase(WORKER_URL));
    w.onmessage = (e: MessageEvent<CudaResponse>) => {
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      p.resolve(e.data);
    };
    w.onerror = () => this.restart();
    w.onmessageerror = () => this.restart();
    this.worker = w;
    return w;
  }

  private restart(): void {
    this.worker?.terminate();
    this.worker = null;
    const err = new Error('the kernel runner restarted');
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }

  run(req: Omit<CudaRequest, 'id' | 'type'>): Promise<CudaResponse> {
    const id = this.nextId++;
    const msg: CudaRequest = { ...req, id, type: 'run' };
    const transfer = msg.args.filter((a) => a.kind === 'buffer').map((a) => (a as { buffer: ArrayBuffer }).buffer);
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ensure().postMessage(msg, transfer);
    });
  }

  /** Kill a runaway kernel (the worker enforces a budget, but a user may not want to wait). */
  cancel(): void {
    this.restart();
  }
}

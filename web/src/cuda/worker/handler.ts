/** Pure request handler: the worker is a one-line wrapper around this so it
 * can be unit-tested in Node. Never throws — every failure is a response. */

import { CudaError } from '../errors';
import { compileProgram, launch, type ArgValue } from '../index';
import type { CudaRequest, CudaResponse } from './protocol';

export function handle(req: CudaRequest): CudaResponse {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const ms = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  try {
    const args: ArgValue[] = req.args.map((a) =>
      a.kind === 'buffer'
        ? { kind: 'buffer', name: a.name, data: a.dtype === 'i32' ? new Int32Array(a.buffer) : new Float32Array(a.buffer) }
        : { kind: a.kind, value: a.value },
    );
    const program = compileProgram(req.source);
    const result = launch(program, req.kernel, {
      grid: req.grid,
      block: req.block,
      sharedBytes: req.sharedBytes,
      args,
      instrument: req.instrument,
      budget: req.budget,
    });
    const buffers = req.args.filter((a) => a.kind === 'buffer').map((a) => (a as { buffer: ArrayBuffer }).buffer);
    return { id: req.id, type: 'done', buffers, result, ms: ms() };
  } catch (e) {
    if (e instanceof CudaError) {
      return {
        id: req.id,
        type: 'error',
        error: { kind: e.kind, message: e.message, line: e.line, col: e.col, thread: e.thread },
        ms: ms(),
      };
    }
    const message = e instanceof Error ? e.message : String(e);
    return { id: req.id, type: 'error', error: { kind: 'runtime', message, line: 1, col: 1 }, ms: ms() };
  }
}

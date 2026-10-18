/** Messages across the CUDA worker boundary. Buffers travel as transferable
 * ArrayBuffers and come back mutated, in the same order they were sent. */

import type { ErrorKind, ThreadId } from '../errors';
import type { LaunchResult } from '../index';

export type WireArg =
  | { kind: 'buffer'; name: string; dtype: 'f32' | 'i32'; buffer: ArrayBuffer }
  | { kind: 'int' | 'float'; value: number };

export interface CudaRequest {
  id: number;
  type: 'run';
  source: string;
  kernel: string;
  grid: [number, number, number];
  block: [number, number, number];
  sharedBytes?: number;
  args: WireArg[];
  instrument: boolean;
  budget?: number;
}

export interface WireError {
  kind: ErrorKind;
  message: string;
  line: number;
  col: number;
  thread?: ThreadId;
}

export type CudaResponse =
  | { id: number; type: 'done'; buffers: ArrayBuffer[]; result: LaunchResult; ms: number }
  | { id: number; type: 'error'; error: WireError; ms: number };

export type CudaClientErrorCode = 'cancelled' | 'crashed';

/** Why a run never produced a response: the user stopped it, or the worker died. */
export class CudaClientError extends Error {
  constructor(
    readonly code: CudaClientErrorCode,
    message: string,
  ) {
    super(message);
    this.name = 'CudaClientError';
  }
}

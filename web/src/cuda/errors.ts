/** Every failure the interpreter can report. All carry a 1-based source
 * position so the editor can highlight the line; runtime errors also carry
 * the thread that hit them. */

export type ErrorKind = 'parse' | 'unsupported' | 'type' | 'oob' | 'barrier' | 'timeout' | 'runtime';

export interface ThreadId {
  block: [number, number, number];
  thread: [number, number, number];
}

export class CudaError extends Error {
  readonly kind: ErrorKind;
  readonly line: number;
  readonly col: number;
  readonly thread?: ThreadId;
  constructor(kind: ErrorKind, message: string, line: number, col: number, thread?: ThreadId) {
    super(message);
    this.name = 'CudaError';
    this.kind = kind;
    this.line = line;
    this.col = col;
    if (thread) this.thread = thread;
  }
}

export class ParseError extends CudaError {
  constructor(message: string, line: number, col: number) {
    super('parse', message, line, col);
    this.name = 'ParseError';
  }
}
export class UnsupportedError extends CudaError {
  constructor(what: string, line: number, col: number) {
    super('unsupported', `${what} is not in the course subset`, line, col);
    this.name = 'UnsupportedError';
  }
}
export class CudaTypeError extends CudaError {
  constructor(message: string, line: number, col: number) {
    super('type', message, line, col);
    this.name = 'CudaTypeError';
  }
}
export class OutOfBoundsError extends CudaError {
  constructor(message: string, line: number, col: number, thread?: ThreadId) {
    super('oob', message, line, col, thread);
    this.name = 'OutOfBoundsError';
  }
}
export class BarrierDivergenceError extends CudaError {
  constructor(message: string, line: number, col: number, thread?: ThreadId) {
    super('barrier', message, line, col, thread);
    this.name = 'BarrierDivergenceError';
  }
}
export class TimeoutError extends CudaError {
  constructor(message: string, line: number, col: number, thread?: ThreadId) {
    super('timeout', message, line, col, thread);
    this.name = 'TimeoutError';
  }
}
export class RuntimeError extends CudaError {
  constructor(message: string, line: number, col: number, thread?: ThreadId) {
    super('runtime', message, line, col, thread);
    this.name = 'RuntimeError';
  }
}

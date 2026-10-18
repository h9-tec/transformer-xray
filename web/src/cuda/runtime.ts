/** Execution model shared by compile.ts and builtins.ts: device buffers,
 * pointer encoding, the per-warp context, masks, and the instrumentation
 * hook interface (implemented in instrument.ts). */

import type { ThreadId } from './errors';
import { OutOfBoundsError } from './errors';

export const LANES = 32;
export const FULL = 0xffffffff | 0; // all 32 lanes, as a signed int32

/** Pointer = bufId * 2^32 + elementIndex, stored in a double. bufId 0 is the
 * block's shared memory; bufId k ≥ 1 is argument buffer k-1. */
const PTR_SCALE = 4294967296;
export const makePtr = (bufId: number, idx: number): number => bufId * PTR_SCALE + idx;
export const ptrBuf = (p: number): number => Math.floor(p / PTR_SCALE);
export const ptrIdx = (p: number): number => p - Math.floor(p / PTR_SCALE) * PTR_SCALE;

export type Elem = 'float' | 'int';

export interface DeviceBuffer {
  id: number;
  name: string;
  elem: Elem;
  f32: Float32Array;
  i32: Int32Array;
  /** length in 4-byte elements */
  length: number;
}

export function bit(l: number): number {
  return (1 << l) | 0;
}
export function has(mask: number, l: number): boolean {
  return ((mask >>> l) & 1) === 1;
}
export function popcount(m: number): number {
  m = m - ((m >>> 1) & 0x55555555);
  m = (m & 0x33333333) + ((m >>> 2) & 0x33333333);
  return (((m + (m >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/** What instrument.ts observes. The runtime calls these; a no-op
 * implementation is used when instrumentation is off. */
export interface Hooks {
  onWarpInstruction(active: number): void;
  onGlobal(kind: 'load' | 'store' | 'atomic', buf: DeviceBuffer, byteAddrs: number[], line: number, threads: ThreadId[]): void;
  onShared(kind: 'load' | 'store' | 'atomic', byteAddrs: number[], line: number, threads: ThreadId[]): void;
  onBarrier(): void;
  onBlockStart(block: [number, number, number]): void;
}

export const NO_HOOKS: Hooks = {
  onWarpInstruction() {},
  onGlobal() {},
  onShared() {},
  onBarrier() {},
  onBlockStart() {},
};

/** Per-launch state shared by every warp. */
export interface LaunchCtx {
  buffers: DeviceBuffer[]; // index = bufId (0 = shared)
  grid: [number, number, number];
  block: [number, number, number];
  blockThreads: number;
  /** scalar/pointer params as 32-wide constant buffers */
  params: Float64Array[];
  hooks: Hooks;
  budget: number;
  ops: number;
  atomics: number;
  barriers: number;
  warpInstructions: number;
  printf: string[];
  printfTruncated: boolean;
  /** current block, for error reports */
  blockIdx: [number, number, number];
  /** per-block uniform buffers for blockIdx.x/y/z */
  blockIdxBuf: [Float64Array, Float64Array, Float64Array];
}

export interface WarpCtx {
  ctx: LaunchCtx;
  warp: number;
  /** lane → thread coords (as doubles for direct use as expression buffers) */
  tx: Float64Array;
  ty: Float64Array;
  tz: Float64Array;
  /** linear thread index per lane */
  linear: Int32Array;
  /** lanes that exist (block size not a multiple of 32 leaves some dead) */
  valid: number;
  active: number;
  exited: number;
  broke: number;
  cont: number;
  /** kernel locals: one 32-wide buffer per slot (float4 slots have length 128) */
  locals: Float64Array[];
  /** device-function frames by function name */
  frames: Map<string, Float64Array[]>;
}

export function threadIdOf(w: WarpCtx, lane: number): ThreadId {
  return { block: [...w.ctx.blockIdx] as [number, number, number], thread: [w.tx[lane], w.ty[lane], w.tz[lane]] };
}

export function checkBounds(w: WarpCtx, buf: DeviceBuffer, idx: number, lane: number, line: number, col: number, width = 1): void {
  if (idx < 0 || idx + width > buf.length || !Number.isInteger(idx)) {
    throw new OutOfBoundsError(
      `${buf.name}[${idx}] is out of bounds (size ${buf.length})`,
      line,
      col,
      threadIdOf(w, lane),
    );
  }
}

export function makeBuffer(id: number, name: string, elem: Elem, data: Float32Array | Int32Array): DeviceBuffer {
  const ab = data.buffer as ArrayBuffer;
  return {
    id,
    name,
    elem,
    f32: data instanceof Float32Array ? data : new Float32Array(ab, data.byteOffset, data.length),
    i32: data instanceof Int32Array ? data : new Int32Array(ab, data.byteOffset, data.length),
    length: data.length,
  };
}

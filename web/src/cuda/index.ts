/** Public API of the in-browser CUDA interpreter.
 *
 *   const p = compileProgram(source);          // lex → parse → check → compile (throws CudaError)
 *   const r = launch(p, 'matmul_tiled', {...}); // runs block by block, warp by warp; mutates arg buffers
 */

import { check } from './check';
import { compile, sharedBufferId, type CompiledProgram } from './compile';
import { BarrierDivergenceError, CudaError, RuntimeError, type ThreadId } from './errors';
import { Instrument } from './instrument';
import { parse } from './parser';
import {
  FULL,
  LANES,
  makeBuffer,
  makePtr,
  NO_HOOKS,
  type DeviceBuffer,
  type LaunchCtx,
  type WarpCtx,
} from './runtime';

export type { CompiledProgram } from './compile';
export { CudaError } from './errors';
export type { ThreadId } from './errors';
export { BUILTINS } from './check';

export type ArgValue =
  | { kind: 'buffer'; name: string; data: Float32Array | Int32Array }
  | { kind: 'int'; value: number }
  | { kind: 'float'; value: number };

export interface LaunchOptions {
  grid: [number, number, number];
  block: [number, number, number];
  sharedBytes?: number;
  args: ArgValue[];
  instrument?: boolean;
  budget?: number;
}

export interface LaunchStats {
  laneOps: number;
  warpInstructions: number;
  divergentInstructions: number;
  globalRequests: number;
  globalSectors: number;
  globalBytes: number;
  sharedRequests: number;
  sharedWorstConflict: number;
  atomics: number;
  barriers: number;
}

export interface RaceReport {
  kind: 'shared' | 'global';
  /** one example address */
  address: number;
  buffer: string;
  firstLine: number;
  secondLine: number;
  /** one example pair of threads */
  threads: [ThreadId, ThreadId];
  /** how many (address, pair) conflicts collapsed into this report */
  count: number;
}

export interface LaunchResult {
  stats: LaunchStats;
  races: RaceReport[];
  printf: string[];
}

export const DEFAULT_BUDGET = 50_000_000;

export function compileProgram(source: string): CompiledProgram {
  return compile(check(parse(source)));
}

export function launch(p: CompiledProgram, kernel: string, opts: LaunchOptions): LaunchResult {
  const k = p.kernels.get(kernel);
  if (!k) throw new RuntimeError(`no __global__ kernel named '${kernel}'`, 1, 1);
  const decl = k.info.decl;
  if (opts.args.length !== decl.params.length)
    throw new RuntimeError(`${kernel} takes ${decl.params.length} arguments, launch passed ${opts.args.length}`, decl.pos.line, decl.pos.col);

  const [bx, by, bz] = opts.block;
  const blockThreads = bx * by * bz;
  if (blockThreads <= 0 || blockThreads > 1024) throw new RuntimeError(`block size ${blockThreads} must be 1..1024`, decl.pos.line, decl.pos.col);
  const [gx, gy, gz] = opts.grid;
  if (gx * gy * gz <= 0) throw new RuntimeError('grid must have at least one block', decl.pos.line, decl.pos.col);

  // buffers: 0 = the extern __shared__ array (dynamic), then args in order,
  // then each static __shared__ array as its own bounded buffer. All shared
  // arrays are views into one per-block ArrayBuffer that is zeroed per block.
  const externFloats = Math.floor((opts.sharedBytes ?? 0) / 4);
  const sharedBytesTotal = k.info.staticSharedBytes + externFloats * 4;
  const sharedAB = new ArrayBuffer(Math.max(4, sharedBytesTotal));
  const sharedData = new Float32Array(sharedAB);
  const externView = new Float32Array(sharedAB, k.info.staticSharedBytes, externFloats);
  const buffers: DeviceBuffer[] = [makeBuffer(0, k.info.externName ?? 'shared', k.info.externElem ?? 'float', externView, true)];
  if (k.info.usesExternShared && externFloats === 0) {
    const d = k.info.decl;
    throw new RuntimeError(`${kernel} declares extern __shared__ memory: pass sharedBytes in the launch`, d.pos.line, d.pos.col);
  }

  const params: Float64Array[] = [];
  opts.args.forEach((a, i) => {
    const param = decl.params[i];
    const buf = new Float64Array(LANES);
    if (a.kind === 'buffer') {
      if (!param.type.pointer) throw new RuntimeError(`argument ${i + 1} of ${kernel} is a scalar but a buffer was passed`, param.pos.line, param.pos.col);
      const elem = a.data instanceof Int32Array ? 'int' : 'float';
      const db = makeBuffer(buffers.length, a.name, elem, a.data);
      buffers.push(db);
      buf.fill(makePtr(db.id, 0));
    } else {
      if (param.type.pointer) throw new RuntimeError(`argument ${i + 1} of ${kernel} is a pointer but a scalar was passed`, param.pos.line, param.pos.col);
      buf.fill(param.type.base === 'float' ? Math.fround(a.value) : a.value | 0);
    }
    params.push(buf);
  });
  for (const [name, entry] of k.info.sharedLayout) {
    const len = entry.dims.reduce((a, b) => a * b, 1);
    const view = new Float32Array(sharedAB, entry.offset, len);
    const id = sharedBufferId(k.info, name);
    buffers[id] = makeBuffer(id, name, entry.elem, view, true);
  }

  const instrument = new Instrument(opts.instrument ?? true, buffers);
  const ctx: LaunchCtx = {
    buffers,
    grid: opts.grid,
    block: opts.block,
    blockThreads,
    params,
    hooks: opts.instrument === false ? NO_HOOKS : instrument,
    budget: opts.budget ?? DEFAULT_BUDGET,
    ops: 0,
    atomics: 0,
    barriers: 0,
    warpInstructions: 0,
    printf: [],
    printfTruncated: false,
    blockIdx: [0, 0, 0],
    blockIdxBuf: [new Float64Array(LANES), new Float64Array(LANES), new Float64Array(LANES)],
  };

  const warpsPerBlock = Math.ceil(blockThreads / LANES);
  const warps: WarpCtx[] = [];
  for (let wi = 0; wi < warpsPerBlock; wi++) {
    const tx = new Float64Array(LANES);
    const ty = new Float64Array(LANES);
    const tz = new Float64Array(LANES);
    const linear = new Int32Array(LANES);
    let valid = 0;
    for (let l = 0; l < LANES; l++) {
      const t = wi * LANES + l;
      if (t >= blockThreads) break;
      linear[l] = t;
      tx[l] = t % bx;
      ty[l] = Math.floor(t / bx) % by;
      tz[l] = Math.floor(t / (bx * by));
      valid |= 1 << l;
    }
    warps.push({
      ctx,
      warp: wi,
      tx,
      ty,
      tz,
      linear,
      valid: valid | 0,
      active: 0,
      exited: 0,
      broke: 0,
      cont: 0,
      locals: Array.from({ length: k.localSlots }, (_, s) => new Float64Array(k.slotLen[s] ?? LANES)),
      params: params.slice(),
      barriers: 0,
      frames: new Map(),
    });
  }

  for (let bzi = 0; bzi < gz; bzi++)
    for (let byi = 0; byi < gy; byi++)
      for (let bxi = 0; bxi < gx; bxi++) {
        ctx.blockIdx = [bxi, byi, bzi];
        ctx.blockIdxBuf[0].fill(bxi);
        ctx.blockIdxBuf[1].fill(byi);
        ctx.blockIdxBuf[2].fill(bzi);
        sharedData.fill(0);
        ctx.hooks.onBlockStart(ctx.blockIdx);
        runBlock(k.body, warps, ctx);
      }

  const stats = instrument.stats();
  stats.laneOps = ctx.ops;
  stats.warpInstructions = ctx.warpInstructions;
  stats.atomics = ctx.atomics;
  stats.barriers = ctx.barriers;
  return { stats, races: instrument.races(), printf: ctx.printf };
}

/** Round-robin the block's warps; a warp parks at a barrier until every
 * warp that still has live lanes has arrived (exited threads don't count).
 * Two mistakes are errors rather than silent: warps parked at different
 * barrier sites, and a warp that ran off the end of the kernel with live
 * lanes while others wait at a barrier it never reached. */
function runBlock(
  body: (w: WarpCtx) => Generator<{ line: number; col: number }, void, void>,
  warps: WarpCtx[],
  ctx: LaunchCtx,
): void {
  const gens = warps.map((w) => {
    w.active = w.valid;
    w.exited = 0;
    w.broke = 0;
    w.cont = 0;
    w.barriers = 0;
    w.params = ctx.params.slice();
    return body(w);
  });
  const n = warps.length;
  const finished: boolean[] = Array.from({ length: n }, () => false);
  const fellThrough: boolean[] = Array.from({ length: n }, () => false);
  const site: ({ line: number; col: number } | null)[] = Array.from({ length: n }, () => null);
  let released = 0;
  for (;;) {
    let progressed = false;
    for (let i = 0; i < n; i++) {
      if (finished[i] || site[i]) continue;
      const r = gens[i].next();
      progressed = true;
      if (r.done) {
        finished[i] = true;
        fellThrough[i] = warps[i].active !== 0;
      } else site[i] = r.value;
    }
    const live = warps.map((_, i) => !finished[i]);
    if (!live.some(Boolean)) return;
    const allWaiting = live.every((l, i) => !l || site[i] !== null);
    if (allWaiting) {
      const parked = site.map((s, i) => (s ? i : -1)).filter((i) => i >= 0);
      const first = site[parked[0]]!;
      for (const i of parked) {
        const s = site[i]!;
        if (s.line !== first.line || s.col !== first.col) {
          throw new BarrierDivergenceError(
            `warps of one block are waiting at different __syncthreads() calls (line ${first.line} and line ${s.line}): every thread must reach the same barrier`,
            s.line,
            s.col,
            threadOf(warps[i]),
          );
        }
      }
      for (let i = 0; i < n; i++) {
        if (finished[i] && fellThrough[i] && warps[i].barriers <= released) {
          throw new BarrierDivergenceError(
            `some threads reached the end of the kernel without the __syncthreads() on line ${first.line} that other threads are waiting at`,
            first.line,
            first.col,
            threadOf(warps[i]),
          );
        }
      }
      released++;
      ctx.barriers++;
      ctx.hooks.onBarrier();
      for (const i of parked) {
        site[i] = null;
        warps[i].barriers++;
      }
      continue;
    }
    if (!progressed) throw new RuntimeError('scheduler stalled', 1, 1);
  }
}

function threadOf(w: WarpCtx): ThreadId {
  let lane = 0;
  while (lane < LANES - 1 && !((w.valid >>> lane) & 1)) lane++;
  return { block: [...w.ctx.blockIdx] as [number, number, number], thread: [w.tx[lane], w.ty[lane], w.tz[lane]] };
}

export { FULL };
export { CudaError as InterpreterError };

/** Public API of the in-browser CUDA interpreter.
 *
 *   const p = compileProgram(source);          // lex → parse → check → compile (throws CudaError)
 *   const r = launch(p, 'matmul_tiled', {...}); // runs block by block, warp by warp; mutates arg buffers
 */

import { check } from './check';
import { compile, type CompiledProgram } from './compile';
import { CudaError, RuntimeError, type ThreadId } from './errors';
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
  address: number;
  buffer: string;
  firstLine: number;
  secondLine: number;
  threads: [ThreadId, ThreadId];
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

  // buffers: 0 = shared (static + dynamic), then args in order
  const sharedFloats = (k.info.staticSharedBytes + (opts.sharedBytes ?? 0)) / 4;
  const sharedData = new Float32Array(Math.max(1, Math.ceil(sharedFloats)));
  const buffers: DeviceBuffer[] = [makeBuffer(0, 'shared', 'float', sharedData)];
  buffers[0].length = sharedFloats; // bounds against the real size, not the padded 1
  if (k.info.usesExternShared && (opts.sharedBytes ?? 0) === 0) {
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
 * warp that still has live lanes has arrived (exited threads don't count). */
function runBlock(body: (w: WarpCtx) => Generator<'barrier', void, void>, warps: WarpCtx[], ctx: LaunchCtx): void {
  const gens = warps.map((w) => {
    w.active = w.valid;
    w.exited = 0;
    w.broke = 0;
    w.cont = 0;
    return body(w);
  });
  const finished: boolean[] = Array.from({ length: warps.length }, () => false);
  const waiting: boolean[] = Array.from({ length: warps.length }, () => false);
  for (;;) {
    let progressed = false;
    for (let i = 0; i < warps.length; i++) {
      if (finished[i] || waiting[i]) continue;
      const r = gens[i].next();
      progressed = true;
      if (r.done) finished[i] = true;
      else waiting[i] = true;
    }
    const live = warps.map((_, i) => !finished[i]);
    if (!live.some(Boolean)) return;
    const allWaiting = live.every((l, i) => !l || waiting[i]);
    if (allWaiting) {
      ctx.barriers++;
      ctx.hooks.onBarrier();
      for (let i = 0; i < warps.length; i++) waiting[i] = false;
      continue;
    }
    if (!progressed) throw new RuntimeError('scheduler stalled', 1, 1);
  }
}

export { FULL };
export { CudaError as InterpreterError };

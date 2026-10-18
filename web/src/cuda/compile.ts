/* oxlint-disable require-yield -- statement generators only yield at barriers */
/** AST → closures. Every expression node owns a preallocated 32-wide lane
 * buffer and compiles to `(w) => buffer`, evaluated under `w.active`.
 * Statements compile to generator functions so `__syncthreads()` can
 * suspend a warp; control flow manipulates the active/exited/broke/cont
 * masks exactly as a SIMT machine does. */

import type { Block, Expr, FunctionDecl, Stmt, TypeRef } from './ast';
import { formatPrintf, MATH1, MATH2, shuffle, type PrintfKind, type ShuffleKind } from './builtins';
import type { Checked, KernelInfo } from './check';
import { BarrierDivergenceError, CudaTypeError, RuntimeError, TimeoutError, type ThreadId } from './errors';
import {
  checkBounds,
  FULL,
  has,
  LANES,
  makePtr,
  popcount,
  ptrBuf,
  ptrIdx,
  threadIdOf,
  type DeviceBuffer,
  type WarpCtx,
} from './runtime';

export type ExprFn = (w: WarpCtx) => Float64Array;
export type StmtGen = (w: WarpCtx) => Generator<'barrier', void, void>;

type NumKind = 'f' | 'i' | 'u' | 'p' | 'f4';

const fr = Math.fround;

function kindOf(t: TypeRef | undefined): NumKind {
  if (!t) return 'f';
  if (t.pointer) return 'p';
  switch (t.base) {
    case 'float':
      return 'f';
    case 'float4':
      return 'f4';
    case 'unsigned':
      return 'u';
    default:
      return 'i';
  }
}

function constBuf(v: number, len = LANES): Float64Array {
  const b = new Float64Array(len);
  b.fill(v);
  return b;
}

const truthy = (buf: Float64Array, mask: number): number => {
  let m = 0;
  for (let l = 0; l < LANES; l++) if (has(mask, l) && buf[l] !== 0) m |= 1 << l;
  return m | 0;
};

export interface CompiledKernel {
  info: KernelInfo;
  body: StmtGen;
  localSlots: number;
  /** slot → buffer length (32, or 128 for float4) */
  slotLen: number[];
}

export interface CompiledProgram {
  checked: Checked;
  kernels: Map<string, CompiledKernel>;
  /** device functions: frame slot lengths by name */
  deviceFrames: Map<string, number[]>;
}

interface FnCtx {
  fn: FunctionDecl;
  isKernel: boolean;
  kernel: KernelInfo | null;
  /** return buffer for device functions */
  ret: Float64Array | null;
  slotLen: number[];
  /** for device functions: the frame's local buffers (shared by all call sites, no recursion) */
  frame: Float64Array[] | null;
  /** index of local slot 0 inside the frame (params come first) */
  localBase: number;
}

export function compile(checked: Checked): CompiledProgram {
  const c = new Compiler(checked);
  return c.run();
}

class Compiler {
  private kernels = new Map<string, CompiledKernel>();
  private deviceFns = new Map<string, { body: StmtGen; ret: Float64Array | null; frame: Float64Array[]; decl: FunctionDecl }>();
  private deviceFrames = new Map<string, number[]>();

  constructor(private checked: Checked) {}

  run(): CompiledProgram {
    // device functions first (kernels call them)
    for (const f of this.checked.program.functions) {
      if (f.fnKind !== 'device') continue;
      const slotLen: number[] = [];
      const frame: Float64Array[] = f.params.map((p) => new Float64Array(p.type.base === 'float4' ? 128 : LANES));
      const ret = f.returnType.base === 'void' ? null : new Float64Array(f.returnType.base === 'float4' ? 128 : LANES);
      const ctx: FnCtx = { fn: f, isKernel: false, kernel: null, ret, slotLen, frame, localBase: f.params.length };
      const body = this.block(f.body, ctx);
      // locals allocated after compiling the body (slot lengths known), after the params
      for (let s = 0; s < slotLen.length; s++) frame[f.params.length + s] = new Float64Array(slotLen[s] ?? LANES);
      this.deviceFns.set(f.name, { body, ret, frame, decl: f });
      this.deviceFrames.set(f.name, slotLen);
    }
    for (const [name, info] of this.checked.kernels) {
      const slotLen: number[] = [];
      const ctx: FnCtx = { fn: info.decl, isKernel: true, kernel: info, ret: null, slotLen, frame: null, localBase: 0 };
      const body = this.block(info.decl.body, ctx);
      const localSlots = (info.decl as FunctionDecl & { localSlots?: number }).localSlots ?? slotLen.length;
      this.kernels.set(name, { info, body, localSlots: Math.max(localSlots, slotLen.length), slotLen });
    }
    return { checked: this.checked, kernels: this.kernels, deviceFrames: this.deviceFrames };
  }

  // ─── statements ───

  private block(b: Block, ctx: FnCtx): StmtGen {
    const stmts = b.stmts.map((s) => this.stmt(s, ctx));
    return function* (w) {
      for (let i = 0; i < stmts.length; i++) {
        if (w.active === 0) return;
        yield* stmts[i](w);
      }
    };
  }

  private tick(w: WarpCtx, line: number, col: number): void {
    const c = w.ctx;
    c.warpInstructions++;
    c.ops += popcount(w.active);
    c.hooks.onWarpInstruction(w.active);
    if (c.ops > c.budget) {
      let lane = 0;
      while (lane < LANES && !has(w.active, lane)) lane++;
      throw new TimeoutError(
        `kernel exceeded the instruction budget (${c.budget.toLocaleString()} lane-ops); is a loop not terminating?`,
        line,
        col,
        threadIdOf(w, Math.min(lane, LANES - 1)),
      );
    }
  }

  private stmt(s: Stmt, ctx: FnCtx): StmtGen {
    const { line, col } = s.pos;
    const tick = (w: WarpCtx) => this.tick(w, line, col);
    switch (s.kind) {
      case 'Block':
        return this.block(s, ctx);
      case 'VarDecl': {
        if (s.shared) return function* () {};
        const slot = this.slotOf(s, ctx);
        const len = s.type.base === 'float4' ? 128 : LANES;
        ctx.slotLen[slot] = len;
        if (!s.init) return function* () {};
        const init = this.expr(s.init, ctx);
        const k = kindOf(s.type);
        const vk = kindOf(s.init.type);
        const frame = ctx.frame;
        const fslot = ctx.localBase + slot;
        return function* (w) {
          tick(w);
          const v = init(w);
          const dst = frame ? frame[fslot] : w.locals[slot];
          storeLanes(dst, v, w.active, k, vk);
        };
      }
      case 'ExprStmt': {
        const e = this.expr(s.expr, ctx);
        return function* (w) {
          tick(w);
          e(w);
        };
      }
      case 'If': {
        const cond = this.expr(s.cond, ctx);
        const cons = this.stmt(s.consequent, ctx);
        const alt = s.alternate ? this.stmt(s.alternate, ctx) : null;
        return function* (w) {
          tick(w);
          const entry = w.active;
          const t = truthy(cond(w), entry);
          const e = entry & ~t;
          let survivors = 0;
          if (t !== 0) {
            w.active = t;
            yield* cons(w);
            survivors |= w.active;
          }
          if (e !== 0) {
            w.active = e;
            if (alt) yield* alt(w);
            survivors |= w.active;
          }
          w.active = survivors;
        };
      }
      case 'For':
      case 'While': {
        const init = s.kind === 'For' && s.init ? this.stmt(s.init, ctx) : null;
        const cond = s.cond ? this.expr(s.cond, ctx) : null;
        const step = s.kind === 'For' && s.step ? this.expr(s.step, ctx) : null;
        const body = this.stmt(s.body, ctx);
        return function* (w) {
          tick(w);
          if (init) yield* init(w);
          let live = w.active;
          let done = 0;
          const savedBroke = w.broke;
          const savedCont = w.cont;
          w.broke = 0;
          w.cont = 0;
          for (;;) {
            if (live === 0) break;
            w.active = live;
            if (cond) {
              tick(w);
              const t = truthy(cond(w), live);
              done |= live & ~t;
              live = t;
              if (live === 0) break;
              w.active = live;
            }
            yield* body(w);
            live = (w.active | w.cont) | 0;
            w.cont = 0;
            if (live === 0) break;
            w.active = live;
            if (step) {
              tick(w);
              step(w);
            }
          }
          w.active = (done | w.broke) | 0;
          w.broke = savedBroke;
          w.cont = savedCont;
        };
      }
      case 'Return': {
        const value = s.value ? this.expr(s.value, ctx) : null;
        const ret = ctx.ret;
        const k = ret && s.value ? kindOf(ctx.fn.returnType) : 'f';
        const vk = s.value ? kindOf(s.value.type) : 'f';
        return function* (w) {
          tick(w);
          if (value && ret) storeLanes(ret, value(w), w.active, k, vk);
          w.exited |= w.active;
          w.active = 0;
        };
      }
      case 'Break':
        return function* (w) {
          tick(w);
          w.broke |= w.active;
          w.active = 0;
        };
      case 'Continue':
        return function* (w) {
          tick(w);
          w.cont |= w.active;
          w.active = 0;
        };
      case 'Sync':
        return function* (w) {
          tick(w);
          const expected = (w.valid & ~w.exited) | 0;
          if (w.active !== expected) {
            let lane = 0;
            while (lane < LANES && !(has(expected, lane) && !has(w.active, lane))) lane++;
            throw new BarrierDivergenceError(
              '__syncthreads() reached by only part of a warp: every thread of the block must hit the same barrier',
              line,
              col,
              threadIdOf(w, Math.min(lane, LANES - 1)),
            );
          }
          yield 'barrier';
        };
    }
  }

  private slotOf(d: { name: string } & Stmt, ctx: FnCtx): number {
    // the checker numbered locals in declaration order; recover the slot via a per-ctx counter
    const anyCtx = ctx as FnCtx & { _slots?: number };
    anyCtx._slots = (anyCtx._slots ?? 0) + 1;
    return anyCtx._slots - 1;
  }

  // ─── expressions ───

  private expr(e: Expr, ctx: FnCtx): ExprFn {
    const { line, col } = e.pos;
    const k = kindOf(e.type);
    switch (e.kind) {
      case 'Num': {
        const b = constBuf(e.isFloat ? fr(e.value) : e.unsigned ? e.value >>> 0 : e.value | 0);
        return () => b;
      }
      case 'Str': {
        const b = constBuf(0);
        return () => b;
      }
      case 'Ident': {
        const r = e.resolved;
        if (!r) throw new CudaTypeError(`unresolved '${e.name}'`, line, col);
        switch (r.kind) {
          case 'param': {
            const idx = r.index;
            if (ctx.isKernel) return (w) => w.ctx.params[idx];
            // device function param = frame slot idx (params occupy the first slots)
            const frame = ctx.frame!;
            return () => frame[idx];
          }
          case 'local': {
            const slot = r.slot;
            if (ctx.frame) {
              const frame = ctx.frame;
              const fslot = ctx.localBase + slot;
              return () => frame[fslot];
            }
            return (w) => w.locals[slot];
          }
          case 'define': {
            const b = constBuf(Number.isInteger(r.value) ? r.value : fr(r.value));
            return () => b;
          }
          case 'shared': {
            const name = r.name;
            const info = ctx.kernel;
            if (!info) throw new CudaTypeError('shared memory outside a kernel', line, col);
            const entry = info.sharedLayout.get(name);
            const offsetElems = entry ? entry.offset / 4 : info.staticSharedBytes / 4; // extern after static
            const b = constBuf(makePtr(0, offsetElems));
            return () => b;
          }
          case 'builtinVar': {
            if (r.name === 'warpSize') {
              const b = constBuf(32);
              return () => b;
            }
            throw new CudaTypeError(`'${r.name}' needs .x, .y or .z`, line, col);
          }
        }
        break;
      }
      case 'Member': {
        const base = e.base;
        if (base.kind === 'Ident' && base.resolved?.kind === 'builtinVar') {
          const which = base.resolved.name;
          const axis = e.name === 'x' ? 0 : e.name === 'y' ? 1 : 2;
          switch (which) {
            case 'threadIdx':
              return axis === 0 ? (w) => w.tx : axis === 1 ? (w) => w.ty : (w) => w.tz;
            case 'blockIdx':
              return (w) => w.ctx.blockIdxBuf[axis];
            case 'blockDim': {
              let b: Float64Array | null = null;
              return (w) => (b ??= constBuf(w.ctx.block[axis]));
            }
            case 'gridDim': {
              let b: Float64Array | null = null;
              return (w) => (b ??= constBuf(w.ctx.grid[axis]));
            }
          }
        }
        // float4 component: a zero-copy view into the 128-wide buffer
        const bf = this.expr(base, ctx);
        const comp = e.name === 'x' ? 0 : e.name === 'y' ? 1 : e.name === 'z' ? 2 : 3;
        return (w) => bf(w).subarray(comp * LANES, comp * LANES + LANES);
      }
      case 'Unary': {
        const out = new Float64Array(k === 'f4' ? 128 : LANES);
        if (e.op === '&') {
          // address of an array element: pointer arithmetic, no load
          const ix = e.arg as Expr & { kind: 'Index' };
          const base = this.expr(ix.base, ctx);
          const index = this.expr(ix.index, ctx);
          const width = kindOf(ix.base.type) === 'p' && ix.base.type?.base === 'float4' ? 4 : 1;
          return (w) => {
            const b = base(w);
            const i = index(w);
            for (let l = 0; l < LANES; l++) if (has(w.active, l)) out[l] = b[l] + i[l] * width;
            return out;
          };
        }
        if (e.op === '*') {
          const ptr = this.expr(e.arg, ctx);
          const elem = e.arg.type?.base === 'int' || e.arg.type?.base === 'unsigned' ? 'int' : 'float';
          const f4 = e.arg.type?.base === 'float4';
          return (w) => {
            this.load(w, ptr(w), null, out, elem, f4, line, col, 'load');
            return out;
          };
        }
        if (e.op === '++pre' || e.op === '--pre' || e.op === 'post++' || e.op === 'post--') {
          const delta = e.op.includes('++') ? 1 : -1;
          const pre = e.op.endsWith('pre');
          const read = this.expr(e.arg, ctx);
          const ak = kindOf(e.arg.type);
          const store = this.storer(e.arg, ctx);
          const tmp = new Float64Array(LANES);
          return (w) => {
            const cur = read(w);
            for (let l = 0; l < LANES; l++) {
              if (!has(w.active, l)) continue;
              out[l] = cur[l];
              tmp[l] = ak === 'f' ? fr(cur[l] + delta) : ak === 'u' ? (cur[l] + delta) >>> 0 : (cur[l] + delta) | 0;
            }
            store(w, tmp);
            if (pre) for (let l = 0; l < LANES; l++) if (has(w.active, l)) out[l] = tmp[l];
            return out;
          };
        }
        const arg = this.expr(e.arg, ctx);
        const op = e.op;
        return (w) => {
          const a = arg(w);
          const m = w.active;
          for (let l = 0; l < LANES; l++) {
            if (!has(m, l)) continue;
            const v = a[l];
            switch (op) {
              case '-':
                out[l] = k === 'f' ? -v : k === 'u' ? -v >>> 0 : -v | 0;
                break;
              case '+':
                out[l] = v;
                break;
              case '!':
                out[l] = v === 0 ? 1 : 0;
                break;
              case '~':
                out[l] = k === 'u' ? ~v >>> 0 : ~v;
                break;
            }
          }
          return out;
        };
      }
      case 'Binary': {
        const out = new Float64Array(LANES);
        const lf = this.expr(e.l, ctx);
        const rf = this.expr(e.r, ctx);
        const lk = kindOf(e.l.type);
        const rk = kindOf(e.r.type);
        const op = e.op;
        if (op === '&&' || op === '||') {
          const isAnd = op === '&&';
          return (w) => {
            const m = w.active;
            const a = lf(w);
            const ta = truthy(a, m);
            const need = isAnd ? ta : m & ~ta;
            for (let l = 0; l < LANES; l++) if (has(m, l)) out[l] = isAnd ? 0 : 1;
            if (need !== 0) {
              w.active = need;
              const b = rf(w);
              w.active = m;
              for (let l = 0; l < LANES; l++) if (has(need, l)) out[l] = b[l] !== 0 ? 1 : 0;
            }
            return out;
          };
        }
        // pointer arithmetic
        if (lk === 'p' || rk === 'p') {
          const width = (lk === 'p' ? e.l.type?.base : e.r.type?.base) === 'float4' ? 4 : 1;
          return (w) => {
            const a = lf(w);
            const b = rf(w);
            const m = w.active;
            for (let l = 0; l < LANES; l++) {
              if (!has(m, l)) continue;
              switch (op) {
                case '+':
                  out[l] = lk === 'p' ? a[l] + b[l] * width : b[l] + a[l] * width;
                  break;
                case '-':
                  out[l] = rk === 'p' ? (a[l] - b[l]) / width : a[l] - b[l] * width;
                  break;
                case '==': out[l] = a[l] === b[l] ? 1 : 0; break;
                case '!=': out[l] = a[l] !== b[l] ? 1 : 0; break;
                case '<': out[l] = a[l] < b[l] ? 1 : 0; break;
                case '>': out[l] = a[l] > b[l] ? 1 : 0; break;
                case '<=': out[l] = a[l] <= b[l] ? 1 : 0; break;
                case '>=': out[l] = a[l] >= b[l] ? 1 : 0; break;
                default:
                  throw new RuntimeError(`operator ${op} on pointers`, line, col);
              }
            }
            return out;
          };
        }
        const isFloat = lk === 'f' || rk === 'f';
        const isUnsigned = !isFloat && (lk === 'u' || rk === 'u');
        const fn = binaryOp(op, isFloat, isUnsigned, line, col);
        return (w) => {
          const a = lf(w);
          const b = rf(w);
          const m = w.active;
          for (let l = 0; l < LANES; l++) if (has(m, l)) out[l] = fn(a[l], b[l]);
          return out;
        };
      }
      case 'Assign': {
        const value = this.expr(e.value, ctx);
        const store = this.storer(e.target, ctx);
        const tk = kindOf(e.target.type);
        const vk = kindOf(e.value.type);
        if (e.op === '=') {
          if (tk === 'f4') {
            return (w) => {
              const v = value(w);
              store(w, v);
              return v;
            };
          }
          const tmp = new Float64Array(LANES);
          return (w) => {
            const v = value(w);
            storeLanes(tmp, v, w.active, tk, vk);
            store(w, tmp);
            return tmp;
          };
        }
        const read = this.expr(e.target, ctx);
        const bop = e.op.slice(0, -1);
        const isFloat = tk === 'f' || vk === 'f';
        const fn = binaryOp(bop, isFloat, tk === 'u', line, col);
        const tmp = new Float64Array(LANES);
        const ptrWidth = tk === 'p' ? (e.target.type?.base === 'float4' ? 4 : 1) : 0;
        return (w) => {
          const v = value(w);
          const cur = read(w);
          const m = w.active;
          for (let l = 0; l < LANES; l++) {
            if (!has(m, l)) continue;
            tmp[l] = ptrWidth ? (bop === '+' ? cur[l] + v[l] * ptrWidth : cur[l] - v[l] * ptrWidth) : fn(cur[l], v[l]);
            if (tk === 'f') tmp[l] = fr(tmp[l]);
            else if (tk === 'i') tmp[l] = tmp[l] | 0;
            else if (tk === 'u') tmp[l] = tmp[l] >>> 0;
          }
          store(w, tmp);
          return tmp;
        };
      }
      case 'Cond': {
        const out = new Float64Array(k === 'f4' ? 128 : LANES);
        const c = this.expr(e.c, ctx);
        const a = this.expr(e.a, ctx);
        const b = this.expr(e.b, ctx);
        const ak = kindOf(e.a.type);
        const bk = kindOf(e.b.type);
        return (w) => {
          const m = w.active;
          const t = truthy(c(w), m);
          const f = m & ~t;
          if (t !== 0) {
            w.active = t;
            storeLanes(out, a(w), t, k, ak);
          }
          if (f !== 0) {
            w.active = f;
            storeLanes(out, b(w), f, k, bk);
          }
          w.active = m;
          return out;
        };
      }
      case 'Call':
        return this.call(e, ctx);
      case 'Index': {
        const base = this.expr(e.base, ctx);
        const index = this.expr(e.index, ctx);
        const bt = e.base.type;
        // 2-D shared row: As[i] → row pointer
        if (e.type?.pointer && e.base.kind === 'Ident' && e.base.resolved?.kind === 'shared') {
          const entry = ctx.kernel?.sharedLayout.get(e.base.resolved.name);
          const stride = entry?.dims[1] ?? 1;
          const out = new Float64Array(LANES);
          return (w) => {
            const b = base(w);
            const i = index(w);
            for (let l = 0; l < LANES; l++) if (has(w.active, l)) out[l] = b[l] + i[l] * stride;
            return out;
          };
        }
        const f4 = bt?.base === 'float4';
        const elem = bt?.base === 'int' || bt?.base === 'unsigned' ? 'int' : 'float';
        const out = new Float64Array(f4 ? 128 : LANES);
        return (w) => {
          this.load(w, base(w), index(w), out, elem, f4, line, col, 'load');
          return out;
        };
      }
      case 'Cast': {
        const arg = this.expr(e.arg, ctx);
        const ak = kindOf(e.arg.type);
        if (k === 'p' || ak === k) return arg;
        const out = new Float64Array(LANES);
        return (w) => {
          const a = arg(w);
          const m = w.active;
          for (let l = 0; l < LANES; l++) {
            if (!has(m, l)) continue;
            const v = a[l];
            out[l] = k === 'f' ? fr(v) : k === 'u' ? (ak === 'f' ? Math.trunc(v) >>> 0 : v >>> 0) : ak === 'f' ? Math.trunc(v) | 0 : v | 0;
          }
          return out;
        };
      }
    }
    throw new CudaTypeError('unsupported expression', line, col);
  }

  /** A store closure for an lvalue expression: `(w, lanes) => void`. */
  private storer(target: Expr, ctx: FnCtx): (w: WarpCtx, v: Float64Array) => void {
    const { line, col } = target.pos;
    const tk = kindOf(target.type);
    switch (target.kind) {
      case 'Ident': {
        const r = target.resolved;
        if (r?.kind === 'local') {
          const slot = r.slot;
          if (ctx.frame) {
            const frame = ctx.frame;
            const fslot = ctx.localBase + slot;
            return (w, v) => storeLanes(frame[fslot], v, w.active, tk, tk);
          }
          return (w, v) => storeLanes(w.locals[slot], v, w.active, tk, tk);
        }
        if (r?.kind === 'param' && !ctx.isKernel) {
          const frame = ctx.frame!;
          const idx = r.index;
          return (w, v) => storeLanes(frame[idx], v, w.active, tk, tk);
        }
        if (r?.kind === 'param' && ctx.isKernel) {
          // kernel params are uniform constants; writing one needs a per-warp copy
          const idx = r.index;
          return (w, v) => {
            const p = w.ctx.params[idx];
            const copy = new Float64Array(p);
            storeLanes(copy, v, w.active, tk, tk);
            w.ctx.params[idx] = copy;
          };
        }
        throw new CudaTypeError(`cannot assign to '${target.name}'`, line, col);
      }
      case 'Member': {
        // float4 component of a local / param
        const comp = target.name === 'x' ? 0 : target.name === 'y' ? 1 : target.name === 'z' ? 2 : 3;
        const baseRead = this.expr(target.base, ctx);
        return (w, v) => {
          const b = baseRead(w);
          for (let l = 0; l < LANES; l++) if (has(w.active, l)) b[comp * LANES + l] = fr(v[l]);
        };
      }
      case 'Index': {
        const base = this.expr(target.base, ctx);
        const index = this.expr(target.index, ctx);
        const bt = target.base.type;
        const f4 = bt?.base === 'float4';
        const elem = bt?.base === 'int' || bt?.base === 'unsigned' ? 'int' : 'float';
        return (w, v) => this.store(w, base(w), index(w), v, elem, f4, line, col, 'store');
      }
      case 'Unary': {
        if (target.op !== '*') break;
        const ptr = this.expr(target.arg, ctx);
        const bt = target.arg.type;
        const f4 = bt?.base === 'float4';
        const elem = bt?.base === 'int' || bt?.base === 'unsigned' ? 'int' : 'float';
        return (w, v) => this.store(w, ptr(w), null, v, elem, f4, line, col, 'store');
      }
      default:
    }
    throw new CudaTypeError('this expression cannot be assigned to', line, col);
  }

  // ─── memory ───

  private resolveAddr(
    w: WarpCtx,
    ptrs: Float64Array,
    index: Float64Array | null,
    width: number,
    line: number,
    col: number,
  ): { buf: DeviceBuffer; idxs: Float64Array } {
    // all active lanes must address the same buffer (true for every kernel we teach)
    const idxs = this.idxScratch;
    let buf: DeviceBuffer | null = null;
    for (let l = 0; l < LANES; l++) {
      if (!has(w.active, l)) continue;
      const p = ptrs[l];
      const b = ptrBuf(p);
      const idx = ptrIdx(p) + (index ? index[l] * width : 0);
      const db = w.ctx.buffers[b];
      if (!db) throw new RuntimeError('dereferencing an invalid pointer', line, col, threadIdOf(w, l));
      if (buf && db !== buf) throw new RuntimeError('one load mixes two different buffers across lanes', line, col, threadIdOf(w, l));
      buf = db;
      checkBounds(w, db, idx, l, line, col, width);
      idxs[l] = idx;
    }
    return { buf: buf as DeviceBuffer, idxs };
  }
  private idxScratch = new Float64Array(LANES);

  private emitAccess(
    w: WarpCtx,
    kind: 'load' | 'store' | 'atomic',
    buf: DeviceBuffer,
    idxs: Float64Array,
    width: number,
    line: number,
  ): void {
    const addrs: number[] = [];
    const threads: ThreadId[] = [];
    for (let l = 0; l < LANES; l++) {
      if (!has(w.active, l)) continue;
      for (let k = 0; k < width; k++) {
        addrs.push((idxs[l] + k) * 4);
        threads.push(threadIdOf(w, l));
      }
    }
    if (buf.id === 0) w.ctx.hooks.onShared(kind, addrs, line, threads);
    else w.ctx.hooks.onGlobal(kind, buf, addrs, line, threads);
  }

  private load(
    w: WarpCtx,
    ptrs: Float64Array,
    index: Float64Array | null,
    out: Float64Array,
    elem: 'float' | 'int',
    f4: boolean,
    line: number,
    col: number,
    kind: 'load',
  ): void {
    if (w.active === 0) return;
    const width = f4 ? 4 : 1;
    const { buf, idxs } = this.resolveAddr(w, ptrs, index, width, line, col);
    const src = elem === 'int' ? buf.i32 : buf.f32;
    for (let l = 0; l < LANES; l++) {
      if (!has(w.active, l)) continue;
      const i = idxs[l];
      if (f4) for (let c = 0; c < 4; c++) out[c * LANES + l] = src[i + c];
      else out[l] = src[i];
    }
    this.emitAccess(w, kind, buf, idxs, width, line);
  }

  private store(
    w: WarpCtx,
    ptrs: Float64Array,
    index: Float64Array | null,
    v: Float64Array,
    elem: 'float' | 'int',
    f4: boolean,
    line: number,
    col: number,
    kind: 'store',
  ): void {
    if (w.active === 0) return;
    const width = f4 ? 4 : 1;
    const { buf, idxs } = this.resolveAddr(w, ptrs, index, width, line, col);
    const dst = elem === 'int' ? buf.i32 : buf.f32;
    for (let l = 0; l < LANES; l++) {
      if (!has(w.active, l)) continue;
      const i = idxs[l];
      if (f4) for (let c = 0; c < 4; c++) dst[i + c] = v[c * LANES + l];
      else dst[i] = v[l];
    }
    this.emitAccess(w, kind, buf, idxs, width, line);
  }

  // ─── calls ───

  private call(e: Expr & { kind: 'Call' }, ctx: FnCtx): ExprFn {
    const { line, col } = e.pos;
    const name = e.callee;
    const args = e.args.map((a) => this.expr(a, ctx));
    const out = new Float64Array(e.type?.base === 'float4' ? 128 : LANES);

    if (MATH1[name]) {
      const f = MATH1[name];
      const a0 = args[0];
      return (w) => {
        const a = a0(w);
        for (let l = 0; l < LANES; l++) if (has(w.active, l)) out[l] = f(a[l]);
        return out;
      };
    }
    if (MATH2[name]) {
      const f = MATH2[name];
      const [a0, a1] = args;
      return (w) => {
        const a = a0(w);
        const b = a1(w);
        for (let l = 0; l < LANES; l++) if (has(w.active, l)) out[l] = f(a[l], b[l]);
        return out;
      };
    }
    if (name.startsWith('__shfl')) {
      const kind: ShuffleKind = name === '__shfl_sync' ? 'sync' : name === '__shfl_up_sync' ? 'up' : name === '__shfl_down_sync' ? 'down' : 'xor';
      const [, a1, a2] = args;
      return (w) => {
        args[0](w);
        const v = a1(w);
        const p = a2(w);
        shuffle(kind, v, p, out, w.active);
        return out;
      };
    }
    if (name === 'make_float4') {
      return (w) => {
        for (let c = 0; c < 4; c++) {
          const a = args[c](w);
          for (let l = 0; l < LANES; l++) if (has(w.active, l)) out[c * LANES + l] = fr(a[l]);
        }
        return out;
      };
    }
    if (name.startsWith('atomic')) {
      const ptrF = args[0];
      const valF = args[1];
      const pt = e.args[0].type;
      const elem: 'float' | 'int' = pt?.base === 'int' || pt?.base === 'unsigned' ? 'int' : 'float';
      const op = name;
      return (w) => {
        if (w.active === 0) return out;
        const ptrs = ptrF(w);
        const vals = valF(w);
        const { buf, idxs } = this.resolveAddr(w, ptrs, null, 1, line, col);
        const mem = elem === 'int' ? buf.i32 : buf.f32;
        for (let l = 0; l < LANES; l++) {
          if (!has(w.active, l)) continue;
          const i = idxs[l];
          const old = mem[i];
          const v = vals[l];
          let nv: number;
          switch (op) {
            case 'atomicAdd': nv = elem === 'float' ? fr(old + v) : (old + v) | 0; break;
            case 'atomicMax': nv = Math.max(old, v); break;
            case 'atomicMin': nv = Math.min(old, v); break;
            default: nv = v; // atomicExch
          }
          mem[i] = nv;
          out[l] = old;
          w.ctx.atomics++;
        }
        this.emitAccess(w, 'atomic', buf, idxs, 1, line);
        return out;
      };
    }
    if (name === 'printf') {
      const fmt = (e.args[0] as Expr & { kind: 'Str' }).value;
      const rest = args.slice(1);
      const kinds: PrintfKind[] = e.args.slice(1).map((a) => {
        const k = kindOf(a.type);
        return k === 'f' ? 'f' : k === 'u' ? 'u' : 'i';
      });
      return (w) => {
        const bufs = rest.map((f) => f(w));
        const c = w.ctx;
        for (let l = 0; l < LANES; l++) {
          if (!has(w.active, l)) continue;
          if (c.printf.length >= 200) {
            if (!c.printfTruncated) {
              c.printf.push('[printf: output truncated]');
              c.printfTruncated = true;
            }
            break;
          }
          const line1 = formatPrintf(fmt, bufs.map((b) => b[l]), kinds).replace(/\n$/, '');
          c.printf.push(line1);
          out[l] = line1.length;
        }
        return out;
      };
    }
    // device function
    const dev = this.deviceFns.get(name);
    if (!dev) throw new CudaTypeError(`unknown function '${name}'`, line, col);
    const paramKinds = dev.decl.params.map((p) => kindOf(p.type));
    const argKinds = e.args.map((a) => kindOf(a.type));
    const retK = kindOf(dev.decl.returnType);
    return (w) => {
      // evaluate args first (may themselves call this function), then copy into the frame
      const vals = args.map((f) => f(w));
      for (let i = 0; i < vals.length; i++) {
        const dst = dev.frame[i] ?? (dev.frame[i] = new Float64Array(paramKinds[i] === 'f4' ? 128 : LANES));
        storeLanes(dst, vals[i], w.active, paramKinds[i], argKinds[i]);
      }
      const savedActive = w.active;
      const savedExited = w.exited;
      const savedBroke = w.broke;
      const savedCont = w.cont;
      w.exited = 0;
      w.broke = 0;
      w.cont = 0;
      const gen = dev.body(w);
      for (;;) {
        const r = gen.next();
        if (r.done) break;
        throw new BarrierDivergenceError('__syncthreads() inside a device function', line, col);
      }
      w.active = savedActive;
      w.exited = savedExited;
      w.broke = savedBroke;
      w.cont = savedCont;
      if (dev.ret) storeLanes(out, dev.ret, w.active, retK, retK);
      return out;
    };
  }
}

/** Copy `src` lanes into `dst` under `mask`, converting to the target kind. */
function storeLanes(dst: Float64Array, src: Float64Array, mask: number, dk: NumKind, sk: NumKind): void {
  if (dk === 'f4') {
    for (let l = 0; l < LANES; l++) {
      if (!has(mask, l)) continue;
      for (let c = 0; c < 4; c++) dst[c * LANES + l] = src[c * LANES + l];
    }
    return;
  }
  for (let l = 0; l < LANES; l++) {
    if (!has(mask, l)) continue;
    const v = src[l];
    dst[l] = dk === 'f' ? fr(v) : dk === 'i' ? (sk === 'f' ? Math.trunc(v) | 0 : v | 0) : dk === 'u' ? (sk === 'f' ? Math.trunc(v) >>> 0 : v >>> 0) : v;
  }
}

function binaryOp(op: string, isFloat: boolean, isUnsigned: boolean, line: number, col: number): (a: number, b: number) => number {
  if (isFloat) {
    switch (op) {
      case '+': return (a, b) => fr(a + b);
      case '-': return (a, b) => fr(a - b);
      case '*': return (a, b) => fr(a * b);
      case '/': return (a, b) => fr(a / b);
      case '<': return (a, b) => (a < b ? 1 : 0);
      case '>': return (a, b) => (a > b ? 1 : 0);
      case '<=': return (a, b) => (a <= b ? 1 : 0);
      case '>=': return (a, b) => (a >= b ? 1 : 0);
      case '==': return (a, b) => (a === b ? 1 : 0);
      case '!=': return (a, b) => (a !== b ? 1 : 0);
    }
    throw new CudaTypeError(`operator '${op}' on floats`, line, col);
  }
  const wrap = isUnsigned ? (x: number) => x >>> 0 : (x: number) => x | 0;
  switch (op) {
    case '+': return (a, b) => wrap(a + b);
    case '-': return (a, b) => wrap(a - b);
    case '*': return (a, b) => wrap(Math.imul(a | 0, b | 0));
    case '/': return (a, b) => {
      if (b === 0) return 0;
      return wrap(Math.trunc(a / b));
    };
    case '%': return (a, b) => (b === 0 ? 0 : wrap(a % b));
    case '<': return (a, b) => (a < b ? 1 : 0);
    case '>': return (a, b) => (a > b ? 1 : 0);
    case '<=': return (a, b) => (a <= b ? 1 : 0);
    case '>=': return (a, b) => (a >= b ? 1 : 0);
    case '==': return (a, b) => (a === b ? 1 : 0);
    case '!=': return (a, b) => (a !== b ? 1 : 0);
    case '&': return (a, b) => wrap(a & b);
    case '|': return (a, b) => wrap(a | b);
    case '^': return (a, b) => wrap(a ^ b);
    case '<<': return (a, b) => wrap(a << b);
    case '>>': return isUnsigned ? (a, b) => a >>> b : (a, b) => a >> b;
  }
  throw new CudaTypeError(`operator '${op}'`, line, col);
}

export { FULL };

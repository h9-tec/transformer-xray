/** Semantic pass: resolve every name, assign a static type to every
 * expression, lay out shared memory, and reject what the course subset
 * does not cover — all with source positions. Annotates the AST in place. */

import type {
  Block,
  Expr,
  FunctionDecl,
  Ident,
  Program,
  Resolved,
  Stmt,
  TypeName,
  TypeRef,
  VarDecl,
} from './ast';
import { CudaTypeError, UnsupportedError } from './errors';

export interface BuiltinSig {
  params: (TypeName | 'any' | 'ptr')[];
  ret: TypeName;
  /** variadic (printf) */
  variadic?: boolean;
}

export const BUILTINS: Record<string, BuiltinSig> = {
  __shfl_sync: { params: ['unsigned', 'float', 'int'], ret: 'float' },
  __shfl_up_sync: { params: ['unsigned', 'float', 'int'], ret: 'float' },
  __shfl_down_sync: { params: ['unsigned', 'float', 'int'], ret: 'float' },
  __shfl_xor_sync: { params: ['unsigned', 'float', 'int'], ret: 'float' },
  atomicAdd: { params: ['ptr', 'any'], ret: 'float' },
  atomicMax: { params: ['ptr', 'any'], ret: 'float' },
  atomicMin: { params: ['ptr', 'any'], ret: 'float' },
  atomicExch: { params: ['ptr', 'any'], ret: 'float' },
  expf: { params: ['float'], ret: 'float' },
  logf: { params: ['float'], ret: 'float' },
  sqrtf: { params: ['float'], ret: 'float' },
  rsqrtf: { params: ['float'], ret: 'float' },
  tanhf: { params: ['float'], ret: 'float' },
  fmaxf: { params: ['float', 'float'], ret: 'float' },
  fminf: { params: ['float', 'float'], ret: 'float' },
  fabsf: { params: ['float'], ret: 'float' },
  powf: { params: ['float', 'float'], ret: 'float' },
  floorf: { params: ['float'], ret: 'float' },
  ceilf: { params: ['float'], ret: 'float' },
  make_float4: { params: ['float', 'float', 'float', 'float'], ret: 'float4' },
  printf: { params: ['any'], ret: 'int', variadic: true },
};

export const BUILTIN_VARS = new Set(['threadIdx', 'blockIdx', 'blockDim', 'gridDim', 'warpSize']);

export interface SharedEntry {
  /** byte offset inside the block's static shared buffer */
  offset: number;
  dims: number[];
  elem: 'float' | 'int';
  decl: VarDecl;
}

export interface KernelInfo {
  decl: FunctionDecl;
  staticSharedBytes: number;
  usesExternShared: boolean;
  sharedLayout: Map<string, SharedEntry>;
  /** name of the extern __shared__ array, if any */
  externName?: string;
  externElem?: 'float' | 'int';
}

export interface Checked {
  program: Program;
  kernels: Map<string, KernelInfo>;
  devices: Map<string, FunctionDecl>;
}

const T = (base: TypeName, pointer = 0, c = false): TypeRef => ({ base, pointer, const: c });
const INT = T('int');
const FLOAT = T('float');

export function check(program: Program): Checked {
  const devices = new Map<string, FunctionDecl>();
  const kernels = new Map<string, KernelInfo>();
  const byName = new Map<string, FunctionDecl>();
  for (const f of program.functions) {
    if (byName.has(f.name)) throw new CudaTypeError(`function '${f.name}' is defined twice`, f.pos.line, f.pos.col);
    byName.set(f.name, f);
    if (f.fnKind === 'device') devices.set(f.name, f);
  }
  // call graph for recursion detection
  const calls = new Map<string, Set<string>>();
  for (const f of program.functions) calls.set(f.name, collectCalls(f.body));
  for (const f of program.functions) {
    if (reaches(f.name, f.name, calls, new Set())) throw new UnsupportedError('recursion', f.pos.line, f.pos.col);
  }

  for (const f of program.functions) {
    const info = checkFunction(f, program, byName);
    if (f.fnKind === 'global') kernels.set(f.name, info);
  }
  return { program, kernels, devices };
}

function collectCalls(b: Block): Set<string> {
  const out = new Set<string>();
  const visitE = (e: Expr): void => {
    switch (e.kind) {
      case 'Call':
        out.add(e.callee);
        e.args.forEach(visitE);
        break;
      case 'Unary':
        visitE(e.arg);
        break;
      case 'Binary':
        visitE(e.l);
        visitE(e.r);
        break;
      case 'Assign':
        visitE(e.target);
        visitE(e.value);
        break;
      case 'Cond':
        visitE(e.c);
        visitE(e.a);
        visitE(e.b);
        break;
      case 'Index':
        visitE(e.base);
        visitE(e.index);
        break;
      case 'Member':
        visitE(e.base);
        break;
      case 'Cast':
        visitE(e.arg);
        break;
      default:
    }
  };
  const visitS = (s: Stmt): void => {
    switch (s.kind) {
      case 'Block':
        s.stmts.forEach(visitS);
        break;
      case 'VarDecl':
        if (s.init) visitE(s.init);
        break;
      case 'ExprStmt':
        visitE(s.expr);
        break;
      case 'If':
        visitE(s.cond);
        visitS(s.consequent);
        if (s.alternate) visitS(s.alternate);
        break;
      case 'For':
        if (s.init) visitS(s.init);
        if (s.cond) visitE(s.cond);
        if (s.step) visitE(s.step);
        visitS(s.body);
        break;
      case 'While':
        visitE(s.cond);
        visitS(s.body);
        break;
      case 'Return':
        if (s.value) visitE(s.value);
        break;
      default:
    }
  };
  visitS(b);
  return out;
}

function reaches(from: string, target: string, calls: Map<string, Set<string>>, seen: Set<string>): boolean {
  for (const c of calls.get(from) ?? []) {
    if (c === target) return true;
    if (!seen.has(c) && calls.has(c)) {
      seen.add(c);
      if (reaches(c, target, calls, seen)) return true;
    }
  }
  return false;
}

interface Scope {
  vars: Map<string, { resolved: Resolved; type: TypeRef }>;
}

class FnChecker {
  private scopes: Scope[] = [];
  private slots = 0;
  readonly sharedLayout = new Map<string, SharedEntry>();
  staticSharedBytes = 0;
  externName?: string;
  externElem?: 'float' | 'int';
  private loopDepth = 0;

  constructor(
    private fn: FunctionDecl,
    private program: Program,
    private fns: Map<string, FunctionDecl>,
  ) {}

  run(): void {
    this.push();
    this.fn.params.forEach((p, index) => {
      if (p.type.base === 'void' && p.type.pointer === 0) throw new CudaTypeError('a parameter cannot be void', p.pos.line, p.pos.col);
      if (p.type.base === 'dim3') throw new UnsupportedError('dim3 parameters', p.pos.line, p.pos.col);
      this.declare(p.name, { kind: 'param', index }, p.type, p.pos);
    });
    this.block(this.fn.body, false);
    this.pop();
    (this.fn as FunctionDecl & { localSlots?: number }).localSlots = this.slots;
  }

  private push(): void {
    this.scopes.push({ vars: new Map() });
  }
  private pop(): void {
    this.scopes.pop();
  }
  private declare(name: string, resolved: Resolved, type: TypeRef, pos: { line: number; col: number }): void {
    const top = this.scopes[this.scopes.length - 1];
    if (top.vars.has(name)) throw new CudaTypeError(`'${name}' is already declared in this scope`, pos.line, pos.col);
    top.vars.set(name, { resolved, type });
  }
  private lookup(name: string): { resolved: Resolved; type: TypeRef } | undefined {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const v = this.scopes[i].vars.get(name);
      if (v) return v;
    }
    const d = this.program.defines.get(name);
    if (d !== undefined) return { resolved: { kind: 'define', value: d }, type: Number.isInteger(d) ? INT : FLOAT };
    if (BUILTIN_VARS.has(name))
      return {
        resolved: { kind: 'builtinVar', name: name as 'threadIdx' },
        type: name === 'warpSize' ? INT : T('dim3'),
      };
    return undefined;
  }

  private block(b: Block, newScope = true): void {
    if (newScope) this.push();
    for (const s of b.stmts) this.stmt(s);
    if (newScope) this.pop();
  }

  private stmt(s: Stmt): void {
    switch (s.kind) {
      case 'Block':
        return this.block(s);
      case 'VarDecl':
        return this.varDecl(s);
      case 'ExprStmt':
        this.expr(s.expr);
        return;
      case 'If':
        this.expr(s.cond);
        this.push();
        this.stmt(s.consequent);
        this.pop();
        if (s.alternate) {
          this.push();
          this.stmt(s.alternate);
          this.pop();
        }
        return;
      case 'For':
        this.push();
        if (s.init) this.stmt(s.init);
        if (s.cond) this.expr(s.cond);
        if (s.step) this.expr(s.step);
        this.loopDepth++;
        this.push();
        this.stmt(s.body);
        this.pop();
        this.loopDepth--;
        this.pop();
        return;
      case 'While':
        this.expr(s.cond);
        this.loopDepth++;
        this.push();
        this.stmt(s.body);
        this.pop();
        this.loopDepth--;
        return;
      case 'Return':
        if (s.value) {
          const t = this.expr(s.value);
          if (this.fn.returnType.base === 'void') throw new CudaTypeError('returning a value from a void function', s.pos.line, s.pos.col);
          void t;
        } else if (this.fn.returnType.base !== 'void') throw new CudaTypeError('return without a value in a non-void function', s.pos.line, s.pos.col);
        return;
      case 'Break':
      case 'Continue':
        if (this.loopDepth === 0) throw new CudaTypeError(`'${s.kind.toLowerCase()}' outside a loop`, s.pos.line, s.pos.col);
        return;
      case 'Sync':
        if (this.fn.fnKind === 'device') throw new UnsupportedError('__syncthreads() inside a __device__ function', s.pos.line, s.pos.col);
        return;
    }
  }

  private varDecl(d: VarDecl): void {
    const { pos } = d;
    if (d.type.base === 'void') throw new CudaTypeError('a variable cannot be void', pos.line, pos.col);
    if (d.type.base === 'dim3') throw new UnsupportedError('dim3 variables', pos.line, pos.col);
    if (d.shared) {
      if (this.fn.fnKind !== 'global') throw new UnsupportedError('__shared__ inside a __device__ function', pos.line, pos.col);
      if (d.type.pointer) throw new UnsupportedError('__shared__ pointers', pos.line, pos.col);
      if (d.init) throw new UnsupportedError('initializing __shared__ memory at declaration', pos.line, pos.col);
      const elem = d.type.base === 'float' ? 'float' : d.type.base === 'int' || d.type.base === 'unsigned' ? 'int' : null;
      if (!elem) throw new UnsupportedError(`__shared__ ${d.type.base}`, pos.line, pos.col);
      if (d.externShared) {
        if (this.externName) throw new UnsupportedError('more than one extern __shared__ array', pos.line, pos.col);
        this.externName = d.name;
        this.externElem = elem;
        d.dimValues = [];
        this.declare(d.name, { kind: 'shared', name: d.name }, T(d.type.base, 1), pos);
        return;
      }
      if (d.dims.length === 0 || d.dims.length > 2) throw new UnsupportedError('__shared__ arrays other than 1-D or 2-D', pos.line, pos.col);
      const dims = d.dims.map((e) => this.constInt(e));
      d.dimValues = dims;
      const bytes = dims.reduce((a, b) => a * b, 1) * 4;
      this.sharedLayout.set(d.name, { offset: this.staticSharedBytes, dims, elem, decl: d });
      this.staticSharedBytes += bytes;
      this.declare(d.name, { kind: 'shared', name: d.name }, T(d.type.base, 1), pos);
      return;
    }
    if (d.dims.length) throw new UnsupportedError('local (per-thread) arrays', pos.line, pos.col);
    if (d.init) {
      const t = this.expr(d.init);
      if (t.pointer !== d.type.pointer && !(d.type.pointer && t.pointer))
        throw new CudaTypeError(`cannot initialize ${typeStr(d.type)} '${d.name}' from ${typeStr(t)}`, pos.line, pos.col);
    }
    this.declare(d.name, { kind: 'local', slot: this.slots++ }, d.type, pos);
  }

  private constInt(e: Expr): number {
    const v = this.constEval(e);
    if (v === undefined || !Number.isInteger(v) || v <= 0)
      throw new UnsupportedError('an array size that is not a positive integer constant', e.pos.line, e.pos.col);
    return v;
  }

  private constEval(e: Expr): number | undefined {
    switch (e.kind) {
      case 'Num':
        return e.value;
      case 'Ident': {
        const d = this.program.defines.get(e.name);
        return d;
      }
      case 'Binary': {
        const a = this.constEval(e.l);
        const b = this.constEval(e.r);
        if (a === undefined || b === undefined) return undefined;
        switch (e.op) {
          case '+': return a + b;
          case '-': return a - b;
          case '*': return a * b;
          case '/': return Math.trunc(a / b);
          default: return undefined;
        }
      }
      default:
        return undefined;
    }
  }

  /** Types an expression, annotating it; returns its static type. */
  expr(e: Expr): TypeRef {
    const t = this.exprInner(e);
    e.type = t;
    return t;
  }

  private exprInner(e: Expr): TypeRef {
    const { pos } = e;
    switch (e.kind) {
      case 'Num':
        return e.isFloat ? FLOAT : e.unsigned ? T('unsigned') : INT;
      case 'Str':
        return T('int'); // only valid as printf's format
      case 'Ident': {
        const v = this.lookup(e.name);
        if (!v) throw new CudaTypeError(`'${e.name}' is not declared`, pos.line, pos.col);
        (e as Ident).resolved = v.resolved;
        return v.type;
      }
      case 'Unary': {
        if (e.op === '&') {
          if (e.arg.kind !== 'Index') throw new UnsupportedError("'&' on anything but an array element", pos.line, pos.col);
          const t = this.expr(e.arg);
          return T(t.base, 1, t.const);
        }
        if (e.op === '*') {
          const t = this.expr(e.arg);
          if (!t.pointer) throw new CudaTypeError('dereferencing a non-pointer', pos.line, pos.col);
          return T(t.base, 0, t.const);
        }
        const t = this.expr(e.arg);
        if (e.op === '++pre' || e.op === '--pre' || e.op === 'post++' || e.op === 'post--') this.assertLvalue(e.arg);
        if (e.op === '!') return INT;
        return t;
      }
      case 'Binary': {
        const l = this.expr(e.l);
        const r = this.expr(e.r);
        if (l.pointer || r.pointer) {
          if ((e.op === '+' || e.op === '-') && (l.pointer !== r.pointer)) return l.pointer ? l : r;
          if (e.op === '-' && l.pointer && r.pointer) return INT;
          if (['==', '!=', '<', '>', '<=', '>='].includes(e.op)) return INT;
          throw new CudaTypeError(`operator '${e.op}' on a pointer`, pos.line, pos.col);
        }
        if (l.base === 'float4' || r.base === 'float4') throw new UnsupportedError('arithmetic on float4 (use .x .y .z .w)', pos.line, pos.col);
        if (['<', '>', '<=', '>=', '==', '!=', '&&', '||'].includes(e.op)) return INT;
        if (['&', '|', '^', '<<', '>>', '%'].includes(e.op)) {
          if (l.base === 'float' || r.base === 'float') throw new CudaTypeError(`operator '${e.op}' on a float`, pos.line, pos.col);
          return l.base === 'unsigned' || r.base === 'unsigned' ? T('unsigned') : INT;
        }
        if (l.base === 'float' || r.base === 'float') return FLOAT;
        return l.base === 'unsigned' || r.base === 'unsigned' ? T('unsigned') : INT;
      }
      case 'Assign': {
        const t = this.expr(e.target);
        this.assertLvalue(e.target);
        const v = this.expr(e.value);
        if (t.pointer && e.op === '=' && !v.pointer) throw new CudaTypeError('assigning a value to a pointer', pos.line, pos.col);
        if (!t.pointer && v.pointer) throw new CudaTypeError('assigning a pointer to a value', pos.line, pos.col);
        if (t.base === 'float4' && v.base !== 'float4') throw new CudaTypeError('assigning a scalar to a float4', pos.line, pos.col);
        return t;
      }
      case 'Cond': {
        this.expr(e.c);
        const a = this.expr(e.a);
        const b = this.expr(e.b);
        if (a.pointer || b.pointer) return a.pointer ? a : b;
        return a.base === 'float' || b.base === 'float' ? FLOAT : a;
      }
      case 'Call':
        return this.call(e);
      case 'Index': {
        const b = this.expr(e.base);
        const i = this.expr(e.index);
        if (i.base === 'float') throw new CudaTypeError('array index must be an integer', e.index.pos.line, e.index.pos.col);
        if (!b.pointer) throw new CudaTypeError('indexing a non-pointer', pos.line, pos.col);
        // 2-D shared: As[i] is a row pointer
        const sh = this.sharedOf(e.base);
        if (sh && sh.dims.length === 2) return T(b.base, 1, b.const);
        return T(b.base, 0, b.const);
      }
      case 'Member': {
        const b = this.expr(e.base);
        if (b.pointer) throw new CudaTypeError(`'.${e.name}' on a pointer`, pos.line, pos.col);
        if (b.base === 'dim3' || b.base === 'float4') {
          if (!['x', 'y', 'z', 'w'].includes(e.name) || (b.base === 'dim3' && e.name === 'w'))
            throw new CudaTypeError(`no member '${e.name}' on ${b.base}`, pos.line, pos.col);
          return b.base === 'dim3' ? INT : FLOAT;
        }
        throw new CudaTypeError(`'.${e.name}' on a ${b.base}`, pos.line, pos.col);
      }
      case 'Cast': {
        const a = this.expr(e.arg);
        if (e.type.pointer && !a.pointer) throw new UnsupportedError('casting a value to a pointer', pos.line, pos.col);
        if (!e.type.pointer && a.pointer) throw new UnsupportedError('casting a pointer to a value', pos.line, pos.col);
        return e.type;
      }
    }
  }

  /** The shared-array entry an expression names directly (Ident), if any. */
  private sharedOf(e: Expr): SharedEntry | undefined {
    if (e.kind === 'Ident' && e.resolved?.kind === 'shared') return this.sharedLayout.get(e.resolved.name);
    return undefined;
  }

  private assertLvalue(e: Expr): void {
    const { pos } = e;
    if (e.kind === 'Ident') {
      if (e.resolved?.kind === 'define' || e.resolved?.kind === 'builtinVar' || e.resolved?.kind === 'shared')
        throw new CudaTypeError(`cannot assign to '${e.name}'`, pos.line, pos.col);
      return;
    }
    if (e.kind === 'Index') {
      if (e.base.type?.const) throw new CudaTypeError('writing through a const pointer', pos.line, pos.col);
      return;
    }
    if (e.kind === 'Unary' && e.op === '*') {
      if (e.arg.type?.const) throw new CudaTypeError('writing through a const pointer', pos.line, pos.col);
      return;
    }
    if (e.kind === 'Member') return this.assertLvalue(e.base);
    throw new CudaTypeError('this expression cannot be assigned to', pos.line, pos.col);
  }

  private call(e: Expr & { kind: 'Call' }): TypeRef {
    const { pos } = e;
    const argTypes = e.args.map((a) => this.expr(a));
    const b = BUILTINS[e.callee];
    if (b) {
      if (!b.variadic && argTypes.length !== b.params.length)
        throw new CudaTypeError(`${e.callee} takes ${b.params.length} arguments, got ${argTypes.length}`, pos.line, pos.col);
      if (b.variadic && (e.args.length === 0 || e.args[0].kind !== 'Str'))
        throw new CudaTypeError('printf needs a string literal format', pos.line, pos.col);
      b.params.forEach((p, i) => {
        if (p === 'ptr' && !argTypes[i].pointer) throw new CudaTypeError(`${e.callee}: argument ${i + 1} must be an address (&x[i])`, pos.line, pos.col);
        if (p === 'ptr' && argTypes[i].const) throw new CudaTypeError(`${e.callee} through a const pointer`, pos.line, pos.col);
      });
      if (e.callee.startsWith('atomic')) return T(argTypes[0].base);
      return T(b.ret);
    }
    const f = this.fns.get(e.callee);
    if (!f) throw new UnsupportedError(`the function '${e.callee}'`, pos.line, pos.col);
    if (f.fnKind === 'global') throw new UnsupportedError('launching a kernel from device code', pos.line, pos.col);
    if (f.params.length !== argTypes.length)
      throw new CudaTypeError(`${e.callee} takes ${f.params.length} arguments, got ${argTypes.length}`, pos.line, pos.col);
    f.params.forEach((p, i) => {
      if (!!p.type.pointer !== !!argTypes[i].pointer)
        throw new CudaTypeError(`${e.callee}: argument ${i + 1} should be ${typeStr(p.type)}`, pos.line, pos.col);
    });
    return f.returnType;
  }
}

function checkFunction(f: FunctionDecl, program: Program, fns: Map<string, FunctionDecl>): KernelInfo {
  const c = new FnChecker(f, program, fns);
  c.run();
  return {
    decl: f,
    staticSharedBytes: c.staticSharedBytes,
    usesExternShared: !!c.externName,
    sharedLayout: c.sharedLayout,
    externName: c.externName,
    externElem: c.externElem,
  };
}

export function typeStr(t: TypeRef): string {
  return `${t.const ? 'const ' : ''}${t.base}${'*'.repeat(t.pointer)}`;
}

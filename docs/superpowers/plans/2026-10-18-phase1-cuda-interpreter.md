# Phase 1 — CUDA-Subset Interpreter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A TypeScript interpreter that parses a CUDA subset, runs kernels warp-by-warp in lockstep with barriers, shuffles, atomics and shared memory, instruments every memory access, and runs all five real `kernels/*.cu` files unchanged with results matching the TS engine.

**Architecture:** `web/src/cuda/` is a pure-TS pipeline: `lexer → parser (AST) → check (names, types, layout) → compile (AST to closures over 32-lane typed arrays) → runtime (blocks sequential; warps are generators parked at barriers by a round-robin scheduler)`. Expressions evaluate per warp into preallocated 32-wide buffers under a 32-bit active mask; statements are generator functions so `__syncthreads()` can suspend a warp. `instrument.ts` observes every load/store and turns addresses into sectors, banks, divergence and race reports using `lib/memory.ts`. A worker wrapper runs launches off the UI thread.

**Tech Stack:** TypeScript (strict), Vitest, esbuild (worker bundle via the existing `scripts/build-worker.mjs` pattern). No parser libraries.

**Spec:** `docs/superpowers/specs/2026-10-18-cuda-course-webgpu-pages-design.md` §4.2.

## Global Constraints

- Subset is fixed by the forcing function: all five `kernels/0[1-5]_*.cu` parse and run **unchanged** (text after `void run_` stripped, as `lib/kernels.ts` already does). Anything outside the subset throws `UnsupportedError` with `line`/`col` and the message `… is not in the course subset`.
- Float math follows fp32: every float result passes through `Math.fround`; int arithmetic wraps to 32 bits (`| 0`, unsigned `>>> 0`); int division truncates toward zero; `%` on ints is C remainder.
- Blocks execute sequentially; within a block warps of 32 lanes run in lockstep under an active mask; barrier mismatch raises `BarrierDivergenceError`.
- Default instruction budget 50,000,000 lane-ops → `TimeoutError`.
- All errors extend `CudaError { kind, message, line, col, thread? }` where `thread` is `{ block: [x,y,z], thread: [x,y,z] }` when known.
- No DOM access in `web/src/cuda/**` except `cuda/worker/`.
- Gates stay green: `tsc`, `oxlint`, vitest, build, Playwright 26/26 root and prefix. Commands in `/home/hesham/llm/web` with `npx pnpm`. Work on `main`; push in the last task.

## Review Focus

1. A learner writes `for (int i = 0; i < n; i++)` with `n` never reached (or `while (true)`): the Run button must return a `TimeoutError` naming the line, not hang the tab. Pinned in Task 4 (budget test) and Task 7 (worker returns the error).
2. A learner indexes `out[idx]` without a bounds check on the last block: `OutOfBoundsError` with the index, buffer name and thread, not silent garbage. Pinned in Task 4.
3. A learner omits `__syncthreads()` between a shared-memory store and a load by another thread: the race report names both lines; results may still be "correct" by luck, so the report must not depend on the value. Pinned in Task 5.
4. A learner puts `__syncthreads()` inside `if (threadIdx.x < 16)`: `BarrierDivergenceError`, not a deadlock. Pinned in Task 4.
5. A learner's kernel returns early for some lanes (`if (idx >= n) return;`) and then hits a barrier: lanes that returned must not block the barrier (CUDA semantics: exited threads don't participate). Pinned in Task 4.

---

### Task 1: Lexer

**Files:**
- Create: `web/src/cuda/errors.ts`, `web/src/cuda/lexer.ts`
- Test: `web/tests/cuda/lexer.test.ts`

**Interfaces:**
- Produces (`errors.ts`): `export type ErrorKind = 'parse' | 'unsupported' | 'type' | 'oob' | 'barrier' | 'timeout' | 'runtime'`; `export interface ThreadId { block: [number, number, number]; thread: [number, number, number] }`; `export class CudaError extends Error { kind: ErrorKind; line: number; col: number; thread?: ThreadId; constructor(kind, message, line, col, thread?) }`; subclasses `ParseError`, `UnsupportedError`, `CudaTypeError`, `OutOfBoundsError`, `BarrierDivergenceError`, `TimeoutError`, `RuntimeError`, each fixing `kind`.
- Produces (`lexer.ts`): `export type TokKind = 'ident' | 'int' | 'float' | 'string' | 'punct' | 'directive' | 'eof'`; `export interface Token { kind: TokKind; text: string; value?: number; unsigned?: boolean; line: number; col: number }`; `export function lex(source: string): Token[]`. Rules: `//` and `/* */` comments skipped; a line starting with `#` is one `directive` token whose `text` is the whole line trimmed; ints accept decimal and `0x` hex with optional `u`/`U` suffix (`unsigned: true`); floats accept `1.`, `.5`, `1e-30`, `0.044715f`, suffix `f` dropped; strings are double-quoted with `\n`/`\"`/`\\` escapes; puncts are the longest match from `<<= >>= ... += -= *= /= %= &= |= ^= == != <= >= && || << >> ++ -- -> :: ( ) [ ] { } ; , . + - * / % < > = ! ~ & | ^ ? :`.

- [ ] **Step 1: Write failing tests**

```ts
// tests/cuda/lexer.test.ts
it('lexes a kernel signature with qualifiers and pointers', () => {
  const t = lex('__global__ void k(const float* __restrict__ x, int n) {}');
  expect(t.map((x) => x.text).slice(0, 6)).toEqual(['__global__', 'void', 'k', '(', 'const', 'float']);
});
it('lexes numeric literals with CUDA suffixes', () => {
  const v = lex('0xffffffffu 1e-30f 0.044715f 16 .5').filter((t) => t.kind !== 'eof');
  expect(v.map((t) => [t.kind, t.value, !!t.unsigned])).toEqual([
    ['int', 4294967295, true], ['float', 1e-30, false], ['float', 0.044715, false], ['int', 16, false], ['float', 0.5, false],
  ]);
});
it('keeps preprocessor lines as one directive token and skips comments', () => {
  const t = lex('#include "common.cuh"\n// hi\n#define TILE 32 /* c */\nint x;');
  expect(t.filter((x) => x.kind === 'directive').map((x) => x.text)).toEqual(['#include "common.cuh"', '#define TILE 32']);
  expect(t.find((x) => x.text === 'int')?.line).toBe(4);
});
it('longest-match punctuation', () => {
  expect(lex('a >>= b << c->x').filter((t) => t.kind === 'punct').map((t) => t.text)).toEqual(['>>=', '<<', '->']);
});
it('reports position on a bad character', () => {
  expect(() => lex('int x = 1;\n  @')).toThrowError(expect.objectContaining({ kind: 'parse', line: 2, col: 3 }));
});
```

- [ ] **Step 2: Run** — `npx pnpm vitest run tests/cuda/lexer.test.ts` → FAIL (module missing).
- [ ] **Step 3: Implement** `errors.ts` and `lex()` (hand-written scanner; 1-based line/col).
- [ ] **Step 4: Run** → 5 passed.
- [ ] **Step 5: Commit** — `feat(cuda): lexer and error types`

---

### Task 2: Parser and AST

**Files:**
- Create: `web/src/cuda/ast.ts`, `web/src/cuda/parser.ts`
- Test: `web/tests/cuda/parser.test.ts`

**Interfaces:**
- Produces (`ast.ts`): position `Pos { line; col }` on every node. `TypeName = 'void' | 'int' | 'unsigned' | 'float' | 'bool' | 'float4' | 'dim3'`; `TypeRef { base: TypeName; pointer: number /* 0,1 */; const: boolean }`; `Program { defines: Map<string, number>; functions: FunctionDecl[] }`; `FunctionDecl { name; kind: 'global' | 'device'; returnType: TypeRef; params: Param[]; body: Block; pos }`; `Param { name; type: TypeRef }`; statements: `Block{stmts}`, `VarDecl{ type: TypeRef; name; dims: Expr[] /* [] scalar, [N] or [N][M] arrays */; init?: Expr; shared: boolean; externShared: boolean }`, `ExprStmt{expr}`, `If{cond; then; else?}`, `For{init?: Stmt; cond?: Expr; step?: Expr; body}`, `While{cond; body}`, `Return{value?}`, `Break`, `Continue`, `Sync` (for `__syncthreads()`); expressions: `Num{value; isFloat; unsigned}`, `Str{value}`, `Ident{name}`, `Unary{op: '-'|'!'|'~'|'++pre'|'--pre'|'post++'|'post--'|'&'|'*'; arg}`, `Binary{op; l; r}`, `Assign{op: '='|'+='|…; target; value}`, `Cond{c; a; b}`, `Call{callee: string; args}`, `Index{base; index}`, `Member{base; name}`, `Cast{type: TypeRef; arg}`.
- Produces (`parser.ts`): `export function parse(source: string): Program`. Handles: `#include` ignored; `#define NAME <int|float literal>` → `defines`; `#pragma` ignored; other directives → `UnsupportedError`. Qualifiers `__restrict__`, `inline`, `static` ignored; `extern __shared__ T s[]` → `VarDecl{externShared:true}`; `__shared__ T a[N][M]`; `const` on params; C casts `(float)x`, `(const float4*)p`; `make_float4(...)` is an ordinary `Call`. Precedence: standard C. Templates, classes, structs, `new`, references (`T&`), recursion are detected later (check) or are parse errors with `UnsupportedError` where syntactically visible (`template`, `class`, `struct`, `&` in a param type).

- [ ] **Step 1: Write failing tests**

```ts
it('parses all five real kernels', () => {
  for (const f of ['01_embedding.cu','02_layernorm.cu','03_matmul.cu','04_attention.cu','05_elementwise.cu']) {
    const src = teachingSource(f);   // helper in tests/cuda/helpers.ts: read kernels/<f>, cut at 'void run_'
    const p = parse(src);
    expect(p.functions.length, f).toBeGreaterThan(0);
  }
});
it('collects #define numeric constants and ignores #include', () => {
  expect(parse('#include "x.h"\n#define TILE 32\n__global__ void k(){}').defines.get('TILE')).toBe(32);
});
it('builds the right tree for a grid-stride loop', () => {
  const p = parse('__global__ void k(float* y, int n){ for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += gridDim.x*blockDim.x) y[i] = 0.f; }');
  const f = p.functions[0]; const loop = f.body.stmts[0];
  expect(loop.kind).toBe('For'); expect(f.params[0].type).toEqual({ base: 'float', pointer: 1, const: false });
});
it('parses shared 2-D arrays, extern shared, casts, ternary and member access', () => {
  const p = parse('__global__ void k(const float4* x){ __shared__ float As[32][32]; extern __shared__ float s[]; float4 v = x[0]; float a = (v.x > 0.f) ? (float)1 : -v.y; }');
  const d = p.functions[0].body.stmts[0]; expect(d.kind === 'VarDecl' && d.shared && d.dims.length).toBe(2);
});
it('rejects templates and references as unsupported with a position', () => {
  expect(() => parse('template <typename T> __global__ void k(T& x){}')).toThrowError(expect.objectContaining({ kind: 'unsupported', line: 1 }));
});
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** recursive-descent parser over `lex()` tokens.
- [ ] **Step 4: Run** → 5 passed.
- [ ] **Step 5: Commit** — `feat(cuda): parser for the course subset; all five kernels parse`

---

### Task 3: Checker (names, types, layout)

**Files:**
- Create: `web/src/cuda/check.ts`
- Test: `web/tests/cuda/check.test.ts`

**Interfaces:**
- Produces: `export interface Checked { program: Program; kernels: Map<string, KernelInfo>; devices: Map<string, FunctionDecl> }`; `KernelInfo { decl: FunctionDecl; staticSharedBytes: number; usesExternShared: boolean; sharedLayout: Map<string, { offset: number; dims: number[]; elem: 'float' | 'int' }> }`; `export function check(program: Program): Checked`. Resolves every `Ident` to a param, local, shared array, `#define`, or builtin (`threadIdx blockIdx blockDim gridDim warpSize`), annotating nodes in place with `resolved` and a static `TypeRef` (`node.type`). Builtin function table (`BUILTINS` exported, name → `{ params: TypeName[]; ret: TypeName }`): `__shfl_sync __shfl_up_sync __shfl_down_sync __shfl_xor_sync atomicAdd atomicMax atomicMin atomicExch expf logf sqrtf rsqrtf tanhf fmaxf fminf fabsf powf floorf ceilf make_float4 printf`. Errors: undeclared identifier (`type`), assignment to a `const` pointee (`type`), call to unknown function (`unsupported`), recursion (`unsupported`), `__syncthreads` inside a device function (`unsupported`), local (non-shared) arrays (`unsupported` in v1), pointer-to-pointer (`unsupported`).

- [ ] **Step 1: Write failing tests**

```ts
it('computes the shared-memory layout of matmul_tiled', () => {
  const k = check(parse(teachingSource('03_matmul.cu'))).kernels.get('matmul_tiled')!;
  expect(k.staticSharedBytes).toBe(2 * 32 * 32 * 4);
  expect(k.sharedLayout.get('Bs')?.offset).toBe(4096);
});
it('marks attention_row as using extern shared', () => {
  expect(check(parse(teachingSource('04_attention.cu'))).kernels.get('attention_row')!.usesExternShared).toBe(true);
});
it('all five kernels pass the checker', () => { for (const f of FILES) expect(() => check(parse(teachingSource(f)))).not.toThrow(); });
it('undeclared identifier is a type error at the use site', () => {
  expect(() => check(parse('__global__ void k(float* y){ y[0] = zz; }'))).toThrowError(expect.objectContaining({ kind: 'type', line: 1, col: 36 }));
});
it('recursion and local arrays are unsupported', () => {
  expect(() => check(parse('__device__ int f(int n){ return f(n-1); } __global__ void k(){}'))).toThrowError(expect.objectContaining({ kind: 'unsupported' }));
  expect(() => check(parse('__global__ void k(){ float t[8]; }'))).toThrowError(expect.objectContaining({ kind: 'unsupported' }));
});
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → 5 passed.
- [ ] **Step 5: Commit** — `feat(cuda): checker with symbol resolution, types, shared layout`

---

### Task 4: Runtime core — compile to closures, warps, scheduler, builtins

**Files:**
- Create: `web/src/cuda/runtime.ts` (memory model + launch), `web/src/cuda/compile.ts` (AST → closures), `web/src/cuda/builtins.ts`, `web/src/cuda/index.ts` (public API)
- Test: `web/tests/cuda/runtime.test.ts`

**Interfaces:**
- Produces (`index.ts`): `export type ArgValue = { kind: 'buffer'; name: string; data: Float32Array | Int32Array } | { kind: 'int'; value: number } | { kind: 'float'; value: number }`; `export interface LaunchOptions { grid: [number, number, number]; block: [number, number, number]; sharedBytes?: number; args: ArgValue[]; instrument?: boolean; budget?: number }`; `export interface LaunchStats { laneOps: number; warpInstructions: number; divergentInstructions: number; globalRequests: number; globalSectors: number; globalBytes: number; sharedRequests: number; sharedWorstConflict: number; atomics: number; barriers: number }`; `export interface RaceReport { kind: 'shared' | 'global'; address: number; buffer: string; firstLine: number; secondLine: number; threads: [ThreadId, ThreadId] }`; `export interface LaunchResult { stats: LaunchStats; races: RaceReport[]; printf: string[] }`; `export function compileProgram(source: string): CompiledProgram` (= lex+parse+check+compile; throws `CudaError`); `export function launch(p: CompiledProgram, kernel: string, opts: LaunchOptions): LaunchResult` (mutates `args` buffers in place; throws `CudaError` at runtime).
- Memory model (`runtime.ts`): a pointer value is a double `bufId * 2**32 + elemIndex`; `bufId 0` = static+extern shared (one `ArrayBuffer` per block, zeroed per block), `bufId ≥ 1` = args in order. `float4*` deref reads 4 consecutive floats at `4*idx`. Bounds are checked on every indexed access.
- Lane values: per expression node one preallocated `Float64Array(32)` (float4 → four); active mask is a 32-bit int; float results `Math.fround`ed, ints `| 0`.
- Scheduler: per block, create one generator per warp (`function* runKernel(warp)`), round-robin: run a warp until it yields `'barrier'` or finishes; a barrier releases when every *live* warp (one with ≥1 lane not exited) has arrived; a warp finishing while others wait at a barrier with lanes still active → `BarrierDivergenceError`; a barrier reached under a partial mask (some lanes active, some inactive but not exited) → `BarrierDivergenceError` (CUDA undefined behaviour, we make it a hard error with the line).
- Builtins (`builtins.ts`): shuffles operate on the current warp's lane buffer (`down`: `l+off<32 ? v[l+off] : v[l]`; `up`; `xor`: `v[l^m]`; `sync`: broadcast `v[src]`); atomics are sequentially consistent per lane in lane order and count in `stats.atomics`; `printf` formats `%d %i %u %f %g %e %x %s %%` per active lane, capped at 200 lines then `[printf: output truncated]`.

- [ ] **Step 1: Write failing tests** (helpers: `run(src, kernel, opts)` = `launch(compileProgram(src), …)`; `f32(n)`/`i32(n)` make arg buffers)

```ts
it('vector add over 4096 elements with 256-thread blocks', ...)            // y[i] = a[i]+b[i] for all i; stats.warpInstructions > 0
it('divergent if/else reconverges and both branches ran', ...)             // y[i] = (i%2) ? 1 : 2; all lanes written
it('loops with break and continue per lane', ...)                          // y[i] = count of j<i with j%3!=0 using continue; break at j==10
it('early return before a barrier does not deadlock', ...)                 // if (i >= n) return; __syncthreads(); y[i]=1  with n=40, block=64 → ok
it('barrier under a partial mask is a BarrierDivergenceError with line', ...) // if (threadIdx.x < 16) __syncthreads();  → kind 'barrier', line 1
it('shared-memory tile with two barriers (reverse within block)', ...)     // s[tid]=x[i]; sync; y[i]=s[blockDim.x-1-tid]
it('warp_sum via __shfl_down_sync sums 32 lanes', ...)                     // port of layernorm warp_sum; result in lane 0 == 496
it('__shfl_sync broadcasts lane 0', ...)
it('atomicAdd on colliding addresses accumulates all contributions', ...)  // 256 threads add 1.0 to out[tid % 4] → each 64
it('float4 loads and make_float4 stores', ...)                             // y4[i] = make_float4(v.x*2, …)
it('fp32 semantics: 0.1f+0.2f equals Math.fround sum, int division truncates, unsigned wraps', ...)
it('out-of-bounds store is an OutOfBoundsError naming index, buffer and thread', ...)   // y[idx] with no bounds check, n=40, block=64 → oob, buffer 'y', index 40, thread {block:[0,0,0],thread:[40,0,0]}
it('an infinite loop hits the instruction budget as TimeoutError with the line', ...)   // while (true) {}  budget: 100000
it('printf captures per-lane lines with %d and %f', ...)                   // 2 lanes → ['tid 0 x 1.500000', 'tid 1 x 2.500000']
it('device function calls (gelu_tanh) and nested calls f(f(x))', ...)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `compile.ts`, `runtime.ts`, `builtins.ts`, `index.ts`. Compile statements as generator functions `(w: WarpCtx) => Generator<'barrier', void>`; `Block` iterates children with `yield*`; expressions as plain closures returning the node's lane buffer. Track per-warp `exited`, `broken`, `continued` masks for `return/break/continue`.
- [ ] **Step 4: Run** → 15 passed.
- [ ] **Step 5: Commit** — `feat(cuda): runtime — lockstep warps, barriers, shuffles, atomics, float4, budget`

---

### Task 5: Instrumentation and race detection

**Files:**
- Create: `web/src/cuda/instrument.ts`
- Modify: `web/src/cuda/runtime.ts` (call hooks on every global/shared load/store/atomic, on every warp instruction, on barriers)
- Test: `web/tests/cuda/instrument.test.ts`

**Interfaces:**
- Consumes: `sectorsTouched`, `bankConflicts` from `@/lib/memory`.
- Produces: `export class Instrument { constructor(enabled: boolean); onWarpInstruction(activeMask: number): void; onGlobal(kind: 'load'|'store'|'atomic', bufId: number, bufName: string, byteAddrs: number[], line: number, threads: ThreadId[]): void; onShared(kind, byteAddrs, line, threads): void; onBarrier(block: number): void; stats(): LaunchStats; races(): RaceReport[] }`. Race rule: per (space, address) keep `{ lastWriter: ThreadId, lastWriterLine, epoch, block }`; a store or load by a *different thread* in the *same block and epoch* after a store, or any plain store/load touching an address another *block* stored (atomics exempt both ways) → one `RaceReport` per address (first occurrence only, max 50 reports).

- [ ] **Step 1: Write failing tests**

```ts
it('row read: 1 request, 4 sectors, 128 bytes', ...)                       // y[i]=x[i], one warp
it('stride-192 column read: 32 sectors per request', ...)
it('shared stride-32 access reports a 32-way conflict; +1 padding reports 1', ...)
it('divergentInstructions counts instructions under a partial mask', ...)  // if (tid<16) y[tid]=1 → ≥1 divergent
it('missing barrier between shared store and cross-thread load is a race naming both lines', ...)  // kind 'shared', firstLine=store line, secondLine=load line
it('the same kernel with __syncthreads has no races', ...)
it('scatter-add without atomics across blocks is a global race; with atomicAdd it is not', ...)
it('instrument:false still returns correct results and zeroed stats', ...)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → 8 passed; re-run `tests/cuda` → all green.
- [ ] **Step 5: Commit** — `feat(cuda): instrumentation — sectors, banks, divergence, races`

---

### Task 6: Fixtures — the five real kernels match the engine; performance budget

**Files:**
- Create: `web/tests/cuda/helpers.ts` (`teachingSource(file)`, `FILES`, `randFill(seed, scale)` using `mulberry32` from `@/engine/rng`)
- Test: `web/tests/cuda/fixtures.test.ts`, `web/tests/cuda/perf.test.ts`

**Interfaces:**
- Consumes: engine ops `gatherRows`, `gatherRowsBwd` (`@/engine/ops/embedding`), `layernorm` (`@/engine/ops/layernorm`), `matmul` (`@/engine/ops/matmul`), `causalMask` + `softmaxRows` + `matmulTB`/`matmul` for attention reference, `geluTanh`, `add`.

- [ ] **Step 1: Write failing tests** (shapes: T=16, D=64, V=65 for embedding; T=16, D=64 layernorm; 64×64×64 matmul; T=16, hd=32 attention with `sharedBytes: T*4`; n=4096 elementwise, f4 with n4=1024)

```ts
it('embedding_gather == gatherRows', ...)                     // launch grid ceil(T*D/256), block 256; maxAbsDiff === 0
it('embedding_scatter_add == gatherRowsBwd (with repeated ids)', ...)   // ids with duplicates; diff < 1e-5
it('layernorm_naive and layernorm_warp == layernorm().y', ...) // naive: grid ceil(T/128),block 128; warp: grid T, block 32; diff < 1e-5
it('matmul_naive and matmul_tiled == matmul', ...)             // grid [2,2,1], block [32,32,1]; diff < 1e-4
it('attention_row == causal softmax(QKᵀ·scale)·V', ...)        // grid T, block 128, sharedBytes T*4; diff < 1e-5
it('gelu_forward, gelu_forward_f4 == geluTanh; residual_add == add', ...)  // grid 256, block 256 (grid-stride)
it('tiled matmul moves ~32× less global traffic than naive at 64³', ...)   // stats.globalBytes ratio between 16 and 40
// perf.test.ts
it('matmul_tiled 64×64×64 instrumented completes under 2 s', ...)
it('attention_row T=16 instrumented completes under 1 s', ...)
```

- [ ] **Step 2: Run** → FAIL (helpers missing / mismatches).
- [ ] **Step 3: Fix whatever the fixtures expose** in `compile.ts` / `runtime.ts` / `builtins.ts`; no changes to the `.cu` files. If perf misses, optimize the hot path (avoid per-lane closures; keep `for (l…)` loops with mask tests) before widening the budget; record the measured time in the ledger.
- [ ] **Step 4: Run** `npx pnpm vitest run tests/cuda` → all pass.
- [ ] **Step 5: Commit** — `test(cuda): five real kernels match the TS engine; perf budget`

---

### Task 7: Worker, protocol, bundle

**Files:**
- Create: `web/src/cuda/worker/protocol.ts`, `web/src/cuda/worker/handler.ts` (pure: `handle(req: CudaRequest): CudaResponse`), `web/src/cuda/worker/cuda.worker.ts` (thin: `onmessage → handle`), `web/src/cuda/worker/client.ts` (`CudaClient` with `run(req): Promise<CudaResponse>`, restarts the worker on `error`, rejects pending with `'the kernel runner restarted'`), `web/src/cuda/worker/worker-url.ts` (generated)
- Modify: `web/scripts/build-worker.mjs` (bundle both workers: generalize to a list `[{entry, outPrefix, urlFile}]`), `.gitignore` (`web/public/cuda.worker.*.js`, `web/src/cuda/worker/worker-url.ts`)
- Test: `web/tests/cuda/handler.test.ts`

**Interfaces:**
- Produces: `CudaRequest = { id: number; type: 'run'; source: string; kernel: string; grid; block; sharedBytes?; args: WireArg[]; instrument: boolean; budget?: number }` where `WireArg = { kind: 'buffer'; name: string; dtype: 'f32' | 'i32'; buffer: ArrayBuffer } | { kind: 'int' | 'float'; value: number }`; `CudaResponse = { id; type: 'done'; buffers: ArrayBuffer[] /* same order as buffer args */; result: LaunchResult } | { id; type: 'error'; error: { kind; message; line; col; thread? } }`. `handle()` never throws: every `CudaError` becomes an `'error'` response; a non-`CudaError` becomes `kind: 'runtime'`.

- [ ] **Step 1: Write failing tests**

```ts
it('handle runs a kernel and returns mutated buffers', ...)       // vector add through WireArg buffers
it('handle turns a TimeoutError into an error response with line', ...)
it('handle turns a parse error into an error response, never throws', ...)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**; generalize `build-worker.mjs`; `withBase()` the URL in `client.ts` exactly like the engine client.
- [ ] **Step 4: Run** tests → pass; `npx pnpm build` → prints both `bundled engine.worker.*.js` and `bundled cuda.worker.*.js`; `ls public/cuda.worker.*.js`.
- [ ] **Step 5: Commit** — `feat(cuda): worker + client; build bundles both workers`

---

### Task 8: Subset reference and ship

**Files:**
- Create: `web/src/content/en/cuda-subset.mdx` (what the course subset supports, with the one-line rule for each `UnsupportedError`; linked from lesson pages in Phase 2) and `docs/cuda-subset.md` (same content for the repo)
- Modify: `README.md` (layout: add `web/src/cuda/` line; "What's inside": the interpreter paragraph)

- [ ] **Step 1:** Write both docs from `BUILTINS` and the checker's unsupported list (source of truth: the code; the doc must name every builtin in `BUILTINS`). Add a test `tests/cuda/doc.test.ts`: every key of `BUILTINS` appears in `docs/cuda-subset.md`.
- [ ] **Step 2:** Run the full gate set: `tsc`, `lint`, `vitest run`, `build`, `playwright test`, prefixed build + `BASE_PATH=/transformer-xray playwright test`, local rebuild. All green.
- [ ] **Step 3:** Commit `docs(cuda): subset reference; README`, push `main`, poll the deploy and CI runs to success, `curl` the live `/en/` → 200.

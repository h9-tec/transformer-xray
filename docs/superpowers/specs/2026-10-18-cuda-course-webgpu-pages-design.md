# Transformer X-Ray → full CUDA course, WebGPU engine, GitHub Pages

Date: 2026-10-18 · Status: approved; amended with §4.0 (visual identity). Phases 0, R, 1, 2 shipped 2026-10-18; Phase 3 implemented 2026-10-18 (traces stay on the CPU engine; GPU serves generation and training).

## 1. Goal

Turn Transformer X-Ray from "a transformer tour with CUDA side panels" into a
CUDA-first interactive course that is live on GitHub Pages:

1. **Learners write real CUDA in the browser.** A TypeScript interpreter runs
   a CUDA subset warp-by-warp in lockstep, exactly as the hardware schedules
   it, and because it sees every memory access it can *show* coalescing,
   bank conflicts, divergence, races and barrier bugs in the learner's own
   code.
2. **A 12-lesson track at `/learn`**, from "what is a thread" to fused
   attention and the roofline. Each lesson: concept (junior/senior), a
   simulator, a write-the-kernel exercise checked against a reference, a
   3-question quiz, and the real `.cu` file it builds toward.
3. **The model actually runs on the viewer's GPU** via a WebGPU backend whose
   WGSL shaders mirror the CUDA kernels one-to-one, selectable in the
   playground and the trainer, parity-tested against the TypeScript engine.
4. **Deployed to GitHub Pages** at `https://h9-tec.github.io/transformer-xray/`
   from a public repo, built by GitHub Actions on every push to `main`.

Success: a learner with no CUDA background can open the page, work through
the 12 lessons in a weekend, write a tiled matmul that passes, and then flip
the playground to "your GPU" and watch the same ideas run for real.

## 2. Non-goals (v1)

- Compiling the learner's CUDA to WGSL or running it on the real GPU. The
  interpreter is the execution model for exercises; WebGPU runs *our*
  shaders only.
- Full C++ (templates, classes, structs beyond `float4`/`dim3`, pointers to
  pointers, dynamic allocation, recursion).
- Tensor cores, bf16, multi-GPU, streams, cooperative groups. Mentioned in
  lesson 12 as "what comes next", not taught.
- Arabic lesson content. Chrome stays bilingual; lesson MDX is English only
  with the content directory laid out so `content/ar/learn/` can be added.
- Accounts or server-side progress. Progress lives in `localStorage`.

## 3. What exists (constraints we inherit)

- `web/`: Next 16 static export (`output: 'export'`, `trailingSlash`),
  next-intl with `localePrefix: 'always'` (`/en/...`, `/ar/...`), MDX+KaTeX,
  Tailwind 4 with the oscilloscope design tokens, Zustand stores, Vitest
  (457 parity tests), Playwright smoke suite, oxlint.
- Engine in `web/src/engine/` runs in a Web Worker bundled by esbuild to a
  content-hashed `public/engine.worker.<hash>.js`; the URL is an absolute
  path (`/engine.worker...`), as are `fetch('/model/...')` calls.
- `kernels/`: five `.cu` files; `web/src/lib/kernels.ts` reads them at build
  time so the GPU panels can never drift from compiled source; measured
  timings in `web/src/content/gpu-results.json`.
- No git remote; not on GitHub yet. Local machine has nvcc (CUDA 13.4) and an
  RTX 5080 laptop GPU for re-benchmarking.

## 4. Architecture

```
web/src/
  cuda/                 Phase 1 — CUDA-subset interpreter (pure TS, no DOM)
    lexer.ts  parser.ts  ast.ts  check.ts  compile.ts  runtime.ts
    builtins.ts  instrument.ts  errors.ts  worker/  index.ts
  learn/                Phase 2 — course data + checkers
    lessons.ts  exercises/<slug>.ts  quizzes.ts
  components/learn/     Phase 2 — lesson UI
    CourseMap  LessonShell  Editor  RunPanel  Quiz  sims/*
  content/en/learn/     Phase 2 — 12 lesson MDX files
  engine/backend/       Phase 3 — Backend interface, cpu.ts, gpu.ts
  engine/gpu/shaders/   Phase 3 — one .wgsl per CUDA kernel
  lib/basePath.ts       Phase 0 — asset URL helper
.github/workflows/deploy.yml   Phase 0
```

The four phases are independent enough to ship one at a time; each lands on
the live Pages site when its PR merges.

### 4.0 Phase R — visual identity and site structure (added after Phase 0)

The deployed Phase 0 site was rejected as "dull, dark, no learning path, too
dense". Approved direction: **premium dark with glow**, learning path first,
teaching over density. Mockup approved 2026-10-18:
`https://claude.ai/artifact/JWqk8DHzW4trQKDv6gL2r4`. Phase R ships before
Phase 1 so every later phase lands inside the new look.

**Design system** (`web/src/app/globals.css` `@theme`, existing token names
kept so component classes keep working; values remapped):

| token | value | role |
|---|---|---|
| `bg` | `#06080f` | page ground ("ink") |
| `surface` | `#0b0f1a` | raised surface |
| `raised` | `#121829` | panel |
| `line` | `rgba(160,180,230,.12)` | hairlines |
| `ink` / `muted` / `faint` | `#e8ecf7` / `#9aa5bf` / `#5f6a85` | text levels |
| `accent` | `#ff9a4d` (deep `#ff6a1f`) | copper: compute, threads, warps, interactive, CTA |
| `fwd` | `#74c7ff` (deep `#2f9bff`) | ice: memory, data, forward pass |
| `bwd` | `#ff7ab6` | backward pass / gradients |
| `update` | `#b59bff` | optimizer |
| `ok` / `danger` | `#5ee6a8` / `#ff6b7a` | pass / fail |

Type: display **Syne** (700/800, tight tracking), body **Instrument Sans**,
code **JetBrains Mono**, Arabic body IBM Plex Sans Arabic (unchanged). Glow
is used in exactly three places: the hero instrument, lit lesson tiles, and
primary buttons. No uppercase eyebrow labels, no middle-dot separators.

**Site structure**

- `/` landing: hero (headline + CTA "Start lesson 1" + live **warp
  instrument**: 32 lanes, three access patterns, sectors-touched readout
  computed from real addresses), the **lesson map** (12 tiles laid out as a
  GPU die; tile state from `learnStore`), "How a lesson works" (see / write
  / find it in the model), the three labs with live thumbnails.
- `/learn` course index: intro + the same lesson map.
- `/learn/[slug]` lesson shell (full content arrives in Phase 2): header
  (number, minutes, kernel it builds toward), goals, "what you will build",
  the real kernel source for lessons that have one, link into the matching
  inspector. Exercise, simulator and quiz slots state plainly that they open
  with the interpreter release.
- Labs (`/playground`, `/train`, `/inspect/*`) keep their routes, restyled
  through the tokens; inspectors default to Intuition depth and the GPU panel
  moves into a collapsible "Under the hood" section so the first screen is
  the story, not the numbers.
- Shared: `web/src/learn/lessons.ts` (the 12-lesson table: slug, number,
  title, blurb, minutes, kernelFile, inspectorHref) and
  `web/src/stores/learnStore.ts` (Zustand + persist; `passed: slug[]`,
  `current: slug`), both consumed by Phase 2.
- `web/src/lib/memory.ts`: `sectorsTouched(byteAddrs, sectorBytes=32)` and
  `bankConflicts(byteAddrs)` — pure, unit-tested, used by the hero now and by
  the Phase 1 instrumentation later.

**Out of scope for R:** lesson prose, simulators beyond the hero, exercises,
quizzes (Phase 2); interpreter (Phase 1).

### 4.1 Phase 0 — GitHub Pages deploy

- Create public repo `h9-tec/transformer-xray`, push `main`.
- `next.config.ts`: `basePath` and `assetPrefix` from
  `NEXT_PUBLIC_BASE_PATH` (empty locally, `/transformer-xray` in CI).
- `lib/basePath.ts` exports `withBase(path)`; every runtime asset URL goes
  through it: `WORKER_URL` (generated file emits a relative name and the
  loader prefixes it), `fetch('/model/...')`, `fetch('/model/corpus.txt')`.
- Root `index.html` redirect to `./en/` (static export has no root page);
  `public/.nojekyll` so `_next/` is served.
- `deploy.yml`: on push to `main` → pnpm install, test, build with the base
  path, `actions/upload-pages-artifact` + `actions/deploy-pages`. Existing
  `ci.yml` keeps running tests on PRs.
- Playwright config gains a `BASE_PATH` mode that serves `out/` under the
  prefix so the smoke suite proves the deployed layout, not just the local
  one. CI e2e runs in that mode.

### 4.2 Phase 1 — CUDA interpreter (`web/src/cuda/`)

**Language subset.** Fixed by a forcing function: all five existing `.cu`
kernels must parse and run unchanged (bench harness code after `void run_`
is stripped, as the GPU panel already does). Concretely:

- Declarations: `__global__` kernels, `__device__ inline` functions,
  `#define NAME <numeric literal>`, `#include` ignored, `__restrict__`,
  `const`, `extern __shared__ float s[]`, `__shared__ T a[N][M]`.
- Types: `int`, `unsigned`, `float`, `bool`, `float4`, `dim3`, `T*`
  (flat device arrays: pointer arithmetic `p + k` and `p[i]` only),
  fixed-size local arrays.
- Statements: declaration with init, assignment and compound assignment,
  `if/else`, `for`, `while`, `break`, `continue`, `return`, blocks,
  expression statements, `__syncthreads()`.
- Expressions: arithmetic, comparison, logical, bitwise and shifts,
  ternary, casts, increment/decrement, member access (`.x .y .z .w`),
  indexing, calls.
- Builtins: `threadIdx blockIdx blockDim gridDim warpSize`;
  `__shfl_sync __shfl_up_sync __shfl_down_sync __shfl_xor_sync`;
  `atomicAdd` (int/float) `atomicMax atomicMin atomicExch`;
  `expf logf sqrtf rsqrtf tanhf fmaxf fminf fabsf powf floorf ceilf
  make_float4`; `printf` (captured per thread, rate-limited).
- Anything else → `UnsupportedError` with line/col and a one-line
  "not in the course subset" message.

**Execution model.** `launch(program, kernelName, grid, block, args, opts)`
returns a `LaunchResult`. Blocks run sequentially (order is unspecified in
CUDA, so sequential is legal). Inside a block, each warp of 32 lanes is a
generator that executes compiled statements in lockstep with an active
mask: `if/else` runs the then-arm under the taken mask and the else-arm
under its complement, then reconverges; loops iterate while any lane is
active; `return`/`break`/`continue` clear lanes from the mask. A scheduler
round-robins the block's warps and parks each at `__syncthreads()` until
all warps in the block arrive; a warp that exits while others wait, or a
different barrier count per warp, raises `BarrierDivergenceError`.

**Compilation.** The AST is compiled to closures over a per-warp
environment where each scalar local is a 32-wide typed array (SoA), so a
statement executes as a tight loop over active lanes rather than a tree
walk per lane. Float math is done in `Math.fround` semantics to match fp32.
An instruction budget per launch (default 50M lane-ops) turns infinite
loops into a friendly timeout.

**Instrumentation** (the whole point). Every memory access is one event per
warp-instruction, recorded by `instrument.ts`:

- Global: addresses touched per warp → 32-byte sectors → `sectors/request`
  (1 = perfectly coalesced, up to 32 = fully scattered) and total bytes.
- Shared: bank = `(byteAddr / 4) % 32`; conflict degree = max lanes per bank
  after collapsing same-address broadcasts.
- Divergence: fraction of warp-instructions executed with a partial mask.
- Races: per address, record `(lastWriterThread, epoch)` where epoch
  increments at each block-wide barrier; a write or read by a different
  thread in the same epoch after a write is a `DataRace` (reported, not
  fatal, with both line numbers). Atomics are exempt.
- Out-of-bounds: every indexed access is checked against the buffer length
  → `OutOfBoundsError` with the index, the buffer, and the thread.

Instrumentation can be switched off for speed when only the result matters.
Everything runs in a dedicated worker (`cuda/worker/`) so the UI never
freezes; the worker is bundled by the same esbuild script as the engine.

**Tests.** `tests/cuda/`: lexer/parser golden tests, semantics (divergence
reconvergence, barrier parking, shuffles, atomics), instrumentation with
known answers (row-major read of 32 floats = 4 sectors; stride-32 shared
access = 32-way conflict; stride-33 = none), race detection positives and
negatives, and `fixtures.test.ts`: each real `.cu` kernel runs on small
shapes and matches the TS engine op to `< 1e-5`.

### 4.3 Phase 2 — the `/learn` track

**Routes.** `/[locale]/(app)/learn/` (course map) and
`/[locale]/(app)/learn/[slug]/` (lesson), statically generated from
`learn/lessons.ts`. The shell nav and the landing page gain a fourth door,
"Learn CUDA".

**Lesson page** (`LessonShell`), top to bottom:

1. **Concept** — MDX with the existing `<Junior>`/`<Senior>` toggle and
   KaTeX; diagrams inline as SVG components.
2. **Simulator** — a lesson-specific canvas component driven by the
   interpreter running the lesson's reference kernel with sliders (block
   size, shape, stride). Shared sims: `ThreadGrid` (grid→block→thread with
   live indices), `WarpTimeline` (lockstep + divergence), `MemoryLanes`
   (32 lanes → sectors), `BankGrid` (32 banks, conflicts lit),
   `TileAnimator` (shared-memory staging per k-step), `ReductionTree`
   (shuffle steps), `Roofline` (measured kernels plotted).
3. **Exercise** — CodeMirror 6 editor (C++ mode, dark theme matching the
   tokens) seeded with starter code containing `// TODO`. "Run" sends
   source + the exercise's inputs to the CUDA worker; the result is checked
   by `learn/exercises/<slug>.ts` against a TS reference (`maxAbsDiff`
   threshold) *and* against performance targets expressed in
   instrumentation terms ("≤ 1.5 sectors per request", "0 bank conflicts",
   "no data races"). Output: pass/fail per target, first mismatching
   element, the instrumentation card, and an inline error with the line
   highlighted. "Show solution" reveals the reference kernel after the first
   run.
4. **Quiz** — three multiple-choice questions from `learn/quizzes.ts`,
   graded locally, explanation shown after answering.
5. **In the real model** — links to the inspector + GPU panel that uses this
   kernel, with its measured time.

**Lessons** (slug · exercise · kernel it builds toward):

| # | lesson | exercise | real kernel |
|---|---|---|---|
| 1 | Hello, thread: grid, block, index | vector add with the index formula | — |
| 2 | Bounds checks & grid-stride loops | `residual_add` for any n | `05_elementwise.cu` |
| 3 | Warps: the real unit of execution | remove divergence from a branchy kernel | — |
| 4 | The memory hierarchy, with numbers | predict traffic; copy kernel | — |
| 5 | Coalescing | `embedding_gather`; fix a column-major read | `01_embedding.cu` |
| 6 | Shared memory & `__syncthreads` | stage a tile; find the missing barrier | `03_matmul.cu` |
| 7 | Bank conflicts | matrix transpose; pad to kill conflicts | — |
| 8 | Tiled matmul: the idea behind every GEMM | `matmul_tiled` from `matmul_naive` | `03_matmul.cu` |
| 9 | Reductions & warp shuffles | `layernorm_warp` from `layernorm_naive` | `02_layernorm.cu` |
| 10 | Atomics | `embedding_scatter_add`; see the race without them | `01_embedding.cu` |
| 11 | Fusing a pipeline: attention in one kernel | `attention_row`: scores→softmax→PV | `04_attention.cu` |
| 12 | Roofline & vectorization; what's next | `gelu_forward_f4`; read the measured table | `05_elementwise.cu` |

Exercise shapes are small enough to run in well under a second in the
interpreter (e.g. n=4096, 64×64×64 matmul, T=16·D=64 layernorm). Scaling to
the model's real shapes is shown with the existing traffic arithmetic, not
by running them.

**Progress.** `stores/learnStore.ts` (Zustand + `persist` to
`localStorage`): per lesson `{ exercisePassed, bestTargets, quizScore }`.
The course map renders a 12-node rail with completion state; the lesson
page shows "Lesson k of 12 · next: …". Reset button in the map.

**Design.** Keep the lab-instrument aesthetic; amber is already "interactive",
so editor chrome, run button and passing targets use it; failures use
`--color-danger`. Canvas sims animate on a `requestAnimationFrame` clock with
a play/step/pause control so a learner can single-step warps. Mobile:
concept, sims and quiz are usable at phone width; the editor is usable but
desktop-first.

**Tests.** Every exercise's reference solution passes its own checker and
the starter fails it (`tests/learn/exercises.test.ts`). Playwright: open a
lesson, run starter (fails), paste solution (passes), reload (progress
persisted); course map reflects it.

### 4.4 Phase 3 — WebGPU backend

**Interface.** `engine/backend/types.ts` defines `Backend` with the ops the
engine already factors out (`matmul/matmulTB/matmulTA`, `layernorm` ±
backward, `gelu` ± backward, causal `softmax`, `attention`, `embedding`
gather/scatter, `crossEntropy`, `adamwStep`). `cpu.ts` wraps the existing
`engine/ops/*`. `gpu.ts` implements the same on `GPUDevice` with buffers
kept resident; results come back as `Float32Array` only where the UI or
the tape asks for them.

**Shaders.** One `.wgsl` per CUDA kernel in `engine/gpu/shaders/`, same
names, same tiling (`TILE = 32`, warp-per-row layernorm using subgroup ops
where available with a workgroup-shared fallback, grid-stride elementwise,
one-workgroup-per-row attention). The GPU panel gains a "WGSL twin" tab
beside the CUDA source so the two can be read side by side.

**Where it runs.** Inside the existing engine worker (`navigator.gpu` is
available in dedicated workers in Chromium; Firefox and Safari support is
checked at runtime). The playground and trainer get a "Run on: CPU · your
GPU" switch; when no adapter is available the switch is disabled with a
one-line reason, and the CPU path is unchanged. Sub-phases: 3a forward +
generation (playground), 3b backward + AdamW (trainer).

**Honesty.** GPU timings shown in the app are labeled "your GPU via WebGPU"
and never mixed with the CUDA timings measured by `make bench`. Per-op
timings use `timestamp-query` when the adapter grants it; otherwise only the
whole-step wall time is shown.

**Parity.** `e2e/gpu-parity.spec.ts` compares GPU forward logits against
the CPU engine (`< 1e-3`, accumulation order differs) and one training step
(`< 5e-3`). Runs in Chromium with WebGPU enabled; in CI, if no adapter is
available the test is marked skipped and the job reports it visibly. The
authoritative run is local on the RTX 5080 before each release.

## 5. Error handling

- Interpreter errors (`Unsupported`, `OutOfBounds`, `BarrierDivergence`,
  `Timeout`, parse errors) are typed, carry `{line, col, thread?}` and a
  plain-English message; the editor highlights the line.
- Worker crashes restart the worker and surface "the kernel runner
  restarted" rather than a hung Run button.
- WebGPU device loss → fall back to CPU and show a banner; the model state
  lives in the worker's CPU copy so nothing is lost.
- Pages: a broken base path is caught by the prefixed e2e run before deploy.

## 6. Testing summary

| layer | tool | gate |
|---|---|---|
| interpreter semantics + instrumentation | Vitest | PR CI |
| the five `.cu` fixtures match the TS engine | Vitest | PR CI |
| every exercise: solution passes, starter fails | Vitest | PR CI |
| lesson flow, progress persistence, base path | Playwright | PR CI |
| WebGPU parity | Playwright (Chromium) | PR CI when adapter present; local RTX 5080 always |
| existing 457 parity tests + lint + tsc | Vitest / oxlint / tsc | PR CI |

## 7. Risks and mitigations

- **Interpreter too slow** → closure compilation + SoA lanes, worker, small
  exercise shapes, instruction budget. Measure in Phase 1 before building
  sims on top.
- **Subset gaps frustrate learners** → fixtures define the subset; exercises
  stay inside it; unsupported constructs fail fast with a clear message and
  a link to the subset reference page.
- **WebGPU availability** → runtime detection, CPU fallback, skipped-not-
  failed CI.
- **Base-path bugs only on Pages** → prefixed Playwright mode in CI.
- **Scope creep** → three phases, each a separate plan; no phase starts
  before the previous one is on the live site.

## 8. Phases and rough effort

| phase | deliverable | effort |
|---|---|---|
| 0 | repo + Pages live with today's app | half a day (done 2026-10-18) |
| R | new visual identity, landing, lesson map, lesson shell, restyled labs | ~3 days |
| 1 | interpreter + fixtures green + perf measured | ~5 days |
| 2 | 12 lessons live with editor, sims, quizzes, progress | ~6 days |
| 3a / 3b | WebGPU forward (playground) / backward (trainer) | ~3 + ~2 days |

Each phase gets its own implementation plan via the writing-plans skill.

# Phase 2 — The /learn Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 12 lesson shells into real lessons: concept prose, a simulator, a write-the-kernel exercise graded by the in-browser GPU on correctness and on instrumentation targets, a quiz, and persisted progress.

**Architecture:** Lesson content is data: `learn/exercises/<slug>.ts` (starter, solution, launch, inputs, reference, targets), `learn/quizzes.ts`, `content/en/learn/<slug>.mdx`. The exercise runner is one component (`Editor` + `RunPanel`) that sends the editor text to `CudaClient`, checks the returned buffers against the TS reference and the returned stats against the exercise's targets, highlights the error line, and calls `learnStore.markPassed`. Simulators are small canvas components fed by pure models (`learn/sims/*.ts`) that reuse `lib/memory.ts` and, where cheap, the interpreter itself.

**Tech Stack:** Next 16 static export, MDX, CodeMirror 6 (`codemirror`, `@codemirror/lang-cpp`), Canvas 2D, Zustand, Vitest, Playwright. Interpreter API from Phase 1: `compileProgram`, `launch`, `CudaClient`, `LaunchStats`, `RaceReport`.

**Spec:** `docs/superpowers/specs/2026-10-18-cuda-course-webgpu-pages-design.md` §4.3 (lesson table, page anatomy, progress) and §4.0 (visual rules).

## Global Constraints

- Exercise shapes run in well under a second instrumented (Phase 1 measured tiled matmul 64³ at 337 ms). No exercise exceeds a 64³ matmul in work.
- Every exercise's `solution` passes its own checker and its `starter` fails it (unit-tested, all 12).
- Instrumentation targets are expressed on `LaunchStats`/`RaceReport` only, so they are deterministic.
- All UI strings in `messages/en.json` and `messages/ar.json`; lesson prose is English MDX (spec §2) wrapped `lang="en" dir="ltr"`.
- Visual rules from §4.0: glow only on hero/tiles/primary buttons; amber (`accent`) for interactive/pass, `danger` for fail; no uppercase eyebrows, no `·`, no `→` on links.
- Gates: `tsc`, `oxlint`, vitest, build, Playwright root + prefix. Commands in `web/` with `npx pnpm`. Work on `main`; push at the end.

## Review Focus

1. A learner pastes code with a syntax error: the editor highlights the exact line and the message reads in plain English; nothing else on the page breaks. Pinned in Task 4 (e2e: parse error → `.cm-error-line` on the line, message visible).
2. A learner's kernel loops forever: Run returns within the budget with a timeout message and the Run button is usable again. Pinned in Task 4 (e2e with a `while(true)` paste).
3. A learner passes the exercise, reloads, and the tile is lit and the rail count is updated. Pinned in Task 6 (e2e).
4. The quiz does not reveal answers before the learner commits, and a wrong answer shows the explanation without changing the stored "passed" state of the exercise. Pinned in Task 6 (unit + e2e).
5. A phone user can read the concept and use the quiz; the editor is usable (scrolls inside its box, no horizontal page scroll). Pinned in Task 4 (e2e at 360px on a lesson page).

---

### Task 1: Exercise framework and the first exercise

**Files:**
- Create: `web/src/learn/exercise.ts` (types + `gradeExercise`), `web/src/learn/exercises/hello-thread.ts`, `web/src/learn/exercises/index.ts` (`EXERCISES: Record<slug, Exercise>`)
- Test: `web/tests/learn/exercises.test.ts` (generic over `EXERCISES`)

**Interfaces:**
- Produces: `export interface Target { id: string; label: string; pass: (ctx: { stats: LaunchStats; races: RaceReport[]; maxDiff: number }) => boolean; detail: (ctx) => string }`; `export interface Exercise { slug: string; kernel: string; starter: string; solution: string; launch: { grid: [n,n,n]; block: [n,n,n]; sharedBytes?: number }; build: (seed: number) => { args: ArgValue[]; outputs: number[] /* indices into args that are outputs */; reference: Float32Array[] /* one per output */ ; tolerance: number }; targets: Target[]; hint?: (ctx) => string | null }`; `export interface GradeResult { ok: boolean; maxDiff: number; firstMismatch: { index: number; got: number; want: number } | null; targets: { id; label; pass: boolean; detail: string }[]; stats: LaunchStats; races: RaceReport[]; printf: string[]; ms: number }`; `export function gradeExercise(ex: Exercise, source: string, seed?: number): GradeResult` (synchronous, uses `compileProgram`/`launch` directly; throws `CudaError` for compile/runtime errors so the UI can highlight them); `export function gradeFromWire(ex, built, buffers, result): GradeResult` for the worker path (same logic on returned buffers).
- The first target of every exercise is `correct` (`maxDiff <= tolerance`). Standard targets exported from `exercise.ts`: `coalesced(maxSectorsPerRequest)`, `noRaces()`, `noBankConflicts()`, `noDivergence(maxDivergentInstructions)`, `trafficAtMost(bytes)`, `requestsAtMost(n)`.
- `hello-thread`: vector add, n=4096, grid 16 × block 256, starter has `int i = 0; // TODO: which element does this thread own?` and no bounds check; solution the real formula with bounds check; targets: `correct`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/learn/exercises.test.ts
import { EXERCISES } from '@/learn/exercises';
import { gradeExercise } from '@/learn/exercise';
import { LESSONS } from '@/learn/lessons';
describe.each(Object.values(EXERCISES))('exercise $slug', (ex) => {
  it('solution passes every target', () => { const g = gradeExercise(ex, ex.solution); expect(g.targets.filter(t => !t.pass)).toEqual([]); expect(g.ok).toBe(true); });
  it('starter does not pass', () => { let ok = true; try { ok = gradeExercise(ex, ex.starter).ok; } catch { ok = false; } expect(ok).toBe(false); });
  it('runs in under 1.5 s instrumented', () => { const t0 = performance.now(); gradeExercise(ex, ex.solution); expect(performance.now() - t0).toBeLessThan(1500); });
});
it('every lesson has an exercise', () => { for (const l of LESSONS) expect(EXERCISES[l.slug], l.slug).toBeDefined(); });
```
(The last test fails until Task 2; keep it and let Task 2 turn it green.)

- [ ] **Step 2: Run** → FAIL (modules missing).
- [ ] **Step 3: Implement** `exercise.ts`, `exercises/hello-thread.ts`, `exercises/index.ts`.
- [ ] **Step 4: Run** → hello-thread's three tests pass; "every lesson has an exercise" fails (expected until Task 2).
- [ ] **Step 5: Commit** — `feat(learn): exercise framework + lesson 1 exercise`

---

### Task 2: Exercises 2 to 12

**Files:**
- Create: `web/src/learn/exercises/{bounds-grid-stride,warps,memory-hierarchy,coalescing,shared-memory-barriers,bank-conflicts,tiled-matmul,reductions-shuffles,atomics,fusing-attention,roofline-float4}.ts`
- Modify: `web/src/learn/exercises/index.ts`

**Interfaces:** consumes Task 1. Each file's `solution` is the real kernel from `kernels/` where one exists (copied verbatim; the fixture tests already prove they run).

| slug | task for the learner | starter defect | targets beyond `correct` |
|---|---|---|---|
| bounds-grid-stride | `residual_add` for any n (n=5000, grid 4, block 256) | one element per thread, no loop, no bounds check | none (correctness alone fails the starter: OOB) |
| warps | `y[i] = odd ? a[i]*2 : a[i]+1` without lane-level divergence (n=1024) | `if (i % 2)` / `else` | `noDivergence(0)` (solution: select arithmetic or `?:`) |
| memory-hierarchy | `copy` kernel with float4 (n=4096) | scalar copy | `requestsAtMost(2 * 4096/4/32)` (one f4 request per warp per direction) |
| coalescing | `embedding_gather` (T=16, D=64, V=65) | index split as `t = idx % T; i = idx / T` (column-major) | `coalesced(4)` (sectors per request) |
| shared-memory-barriers | reverse each 64-block through shared memory | barrier missing | `noRaces()` |
| bank-conflicts | 32×32 transpose through shared memory | `s[32][32]`, column read | `noBankConflicts()` (solution pads to 33) |
| tiled-matmul | `matmul_tiled` from `matmul_naive` (64³, grid 2×2, block 32×32) | the naive kernel | `trafficAtMost(naiveBytes / 16)`, `noRaces()` |
| reductions-shuffles | `layernorm_warp` (T=16, D=64, grid T, block 32) | lane 0 does the whole row serially | `noDivergence(4)` |
| atomics | `embedding_scatter_add` with repeated ids (T=16, D=64, V=65, ids mod 5) | plain `+=` | `noRaces()` |
| fusing-attention | `attention_row` (T=16, hd=32, block 128, sharedBytes T*4) | softmax step skipped (scores used raw) and first barrier missing | `noRaces()` |
| roofline-float4 | `gelu_forward_f4` (n=4096, n4=1024, grid 256, block 256) | scalar `gelu_forward` as the body | `requestsAtMost(scalarRequests / 4 + 8)` |

- [ ] **Step 1:** The test from Task 1 already fails for these. **Step 2: Implement** the 11 files. **Step 3: Run** `npx pnpm vitest run tests/learn/exercises.test.ts` → 37 passed. **Step 4: Commit** — `feat(learn): exercises for lessons 2–12`

---

### Task 3: Concept prose and quizzes

**Files:**
- Create: `web/src/content/en/learn/<slug>.mdx` ×12 (uses `<Junior>`/`<Senior>` from `@/components/mdx`, KaTeX, fenced ```cuda blocks rendered by `CudaCode` via `mdx-components.tsx`)
- Create: `web/src/learn/quizzes.ts` (`QUIZZES: Record<slug, Quiz>`; `Quiz = { questions: { q: string; choices: string[]; answer: number; why: string }[] }`, exactly 3 questions each)
- Modify: `web/src/mdx-components.tsx` (map `pre > code.language-cuda` to `CudaCode`)
- Test: `web/tests/learn/content.test.ts`

**Interfaces:** `export function lessonDoc(slug): Promise<MDXContent>` via a static `import.meta.glob`-free map (`content/en/learn/index.ts` exporting `DOCS: Record<slug, () => Promise<{ default: MDXContent }>>` so the lesson page can `dynamic()` import) — or a plain `switch` of static imports; both are fine for static export.

- [ ] **Step 1: Write failing tests** — `every lesson has an MDX file` (fs check), `every lesson has exactly 3 quiz questions with a valid answer index`, `no MDX contains '·' or '→' in prose`, `each MDX mentions the exercise's kernel name at least once when the lesson has a kernelFile`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Write** the 12 MDX files (300–600 words each; Intuition first; the Senior block carries the arithmetic: traffic formulas, bank math, shuffle steps; one ```cuda fragment each) and the 36 questions. **Step 4: Run** → pass; `pnpm build` compiles the MDX.
- [ ] **Step 5: Commit** — `feat(learn): concept prose and quizzes for all 12 lessons`

---

### Task 4: Editor and RunPanel

**Files:**
- Install: `codemirror`, `@codemirror/lang-cpp` (exact versions pinned by pnpm)
- Create: `web/src/components/learn/Editor.tsx` (client; CodeMirror 6; props `{ value, onChange, errorLine: number | null, id }`; dark theme from tokens; `cpp()` language; `errorLine` renders a line decoration `.cm-error-line`)
- Create: `web/src/components/learn/RunPanel.tsx` (client; props `{ exercise }`; owns editor text in state seeded from `exercise.starter`, persisted per slug in `localStorage` key `xray-ex-<slug>`; "Run on the GPU" → `CudaClient.run` with `instrument: true`, then `gradeFromWire`; shows: target list with pass/fail, first mismatch, instrumentation card (sectors/request, worst bank conflict, divergent %, races), printf output, error card with line/thread; "Reset to starter"; "Show solution" after the first run; on all targets pass → `useLearnStore.markPassed(slug)`)
- Create: `web/src/components/learn/ExerciseSection.tsx` (server wrapper that picks `EXERCISES[slug]` and renders `RunPanel`)
- Modify: `web/src/components/learn/LessonShell.tsx` (replace the Exercise `Slot` with `ExerciseSection`)
- Modify: `web/messages/*.json` (`learn.run`, `learn.running`, `learn.reset`, `learn.showSolution`, `learn.passed`, `learn.failed`, `learn.firstMismatch`, `learn.errorAt`, `learn.printf`, `learn.instr*`)
- Test: `web/e2e/exercise.spec.ts`

- [ ] **Step 1: Write failing e2e tests** (use `page.getByTestId('editor')` and set text via `page.evaluate` on the CodeMirror view exposed as `window.__cm[slug]`):

```ts
test('starter fails, solution passes and lights the tile', ...)      // goto en/learn/hello-thread/; click Run; expect target 'correct' fail; set editor to solution (from window.__exercise.solution); Run; all targets pass; goto en/learn/; first tile data-status=passed
test('a syntax error highlights the line', ...)                        // set text with 'y[0] = ;' on line 3; Run; expect .cm-error-line count 1; error card contains 'expected'
test('an infinite loop times out and Run is usable again', ...)        // 'while (true) {}' ; Run; error card matches /budget|terminating/; Run button enabled
test('lesson page at 360px has no horizontal scroll and the editor scrolls inside', ...)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → pass; full suite green. **Step 5: Commit** — `feat(learn): CodeMirror editor and GPU run panel with graded targets`

---

### Task 5: Simulators

**Files:**
- Create: `web/src/learn/sims/models.ts` (pure: `threadGrid(n, block)`, `gridStrideCoverage(n, grid, block)`, `divergenceTimeline(pattern, lanes)`, `tileTraffic(M,N,K,tile)`, `bankMap(stride, pad)`, `reductionSteps()`, `rooflinePoints(results)`), tests in `web/tests/learn/sims.test.ts`
- Create: `web/src/components/learn/sims/{ThreadGrid,WarpTimeline,MemoryLanes,BankGrid,TileAnimator,ReductionTree,AtomicsRace,AttentionRows,Roofline,Hierarchy}.tsx` (client canvas/SVG components with sliders; play/step/pause where animated; `prefers-reduced-motion` → static frame)
- Create: `web/src/components/learn/Simulator.tsx` (maps slug → component: 1 ThreadGrid, 2 ThreadGrid grid-stride mode, 3 WarpTimeline, 4 Hierarchy, 5 MemoryLanes, 6 TileAnimator, 7 BankGrid, 8 TileAnimator, 9 ReductionTree, 10 AtomicsRace (runs both scatter kernels on the interpreter in a worker and shows lost updates), 11 AttentionRows, 12 Roofline from `content/gpu-results.json`)
- Modify: `LessonShell.tsx` (replace the Simulator `Slot`)

- [ ] **Step 1: Write failing unit tests** for `models.ts` with known values (`tileTraffic(128,192,576,32)` ≈ naive/32; `bankMap(32,0)` worst 32, `bankMap(32,1)` worst 1; `gridStrideCoverage(5000,4,256)` covers all indices with max 5 iterations) and an e2e `every lesson page renders a simulator canvas or svg` (loop over 12 slugs, expect `[data-testid="sim"]` visible).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → pass. **Step 5: Commit** — `feat(learn): simulators for all 12 lessons`

---

### Task 6: Quiz, progress polish, subset page, ship

**Files:**
- Create: `web/src/components/learn/Quiz.tsx` (client; radio per question; "Check answers" reveals per-question correct/incorrect + `why`; stores `quizScore` in `learnStore` (add `quiz: Record<slug, number>` + `setQuiz`))
- Modify: `LessonShell.tsx` (replace the Quiz `Slot`; add link to `/learn/subset`), `LessonMap.tsx` ("Start here" instead of "Continue" when `passed` is empty), `learnStore.ts`
- Create: `web/src/app/[locale]/(app)/learn/subset/page.tsx` rendering `content/en/cuda-subset.mdx`
- Test: `web/tests/learn/learnStore.test.ts` (quiz score), `web/e2e/learn.spec.ts` additions

- [ ] **Step 1: Write failing tests** — unit: `setQuiz('hello-thread', 2)` persists and does not alter `passed`; e2e: answer the quiz on lesson 1 (choose answers), check, see 'why' text; the exercise's passed state unchanged; `/en/learn/subset/` renders the heading; fresh visitor sees 'Start here'.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** full gates: tsc, lint, vitest, build, Playwright root + prefix.
- [ ] **Step 5: Commit** `feat(learn): quizzes, progress polish, subset page`; push `main`; poll deploy + CI to success; `curl` `/en/learn/hello-thread/` → 200; open it in Chrome, run the starter, see the failing target, paste the solution, see it pass; screenshot for the user.

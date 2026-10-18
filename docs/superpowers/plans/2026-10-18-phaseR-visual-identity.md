# Phase R — Visual Identity, Landing, Lesson Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rejected look with the approved premium-dark identity: new tokens and type, a landing page with a live warp instrument and a 12-lesson map, a `/learn` index and lesson shells, and labs restyled so the first screen is the story.

**Architecture:** Token values and fonts change in one place (`globals.css` `@theme` + `layout.tsx`), so every existing component re-skins for free. New surface: `learn/lessons.ts` (the course table), `stores/learnStore.ts` (progress), `lib/memory.ts` (sector/bank math shared with Phase 1), and new components under `components/landing/` and `components/learn/`. Lesson pages are statically generated from the lesson table. The GPU panel moves into a collapsible section on inspector pages.

**Tech Stack:** Next 16 static export, next-intl, Tailwind 4 `@theme`, `next/font/google` (Syne, Instrument Sans, JetBrains Mono, IBM Plex Sans Arabic), Zustand + `persist`, Canvas 2D, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-18-cuda-course-webgpu-pages-design.md` §4.0. Mockup: https://claude.ai/artifact/JWqk8DHzW4trQKDv6gL2r4 (file `/tmp/claude-1000/-home-hesham/f8d036ea-4ac1-438c-852a-19de29bbb6e6/scratchpad/xray-redesign.html` holds the exact CSS and canvas code to port).

## Global Constraints

- Token names stay (`bg surface raised line ink muted faint accent fwd bwd update danger`); add `ok`. Values per spec §4.0 table. Fonts: display Syne, body Instrument Sans, mono JetBrains Mono, Arabic IBM Plex Sans Arabic.
- Glow (box/text shadow) only on: hero instrument, lit lesson tiles, primary buttons.
- No uppercase eyebrow labels, no `·` separators, no `→` appended to link text, no numbered markers except the lesson numbers (a real sequence).
- Every new UI string goes in `messages/en.json` **and** `messages/ar.json` (next-intl fails loudly on a missing key).
- Existing gates stay green: `tsc`, `oxlint`, 460 vitest tests, build, Playwright 6/6 at root and under `BASE_PATH=/transformer-xray`.
- Commands run in `/home/hesham/llm/web` with `npx pnpm`. Commit messages without trailers. Work on `main`; push only at the last task.

## Review Focus

1. A viewer on a phone opens `/`: the warp canvas, lesson map and labs must stack to one column with no horizontal scroll. Pinned in Task 4 (Playwright viewport 390×844 assertion `document.documentElement.scrollWidth <= innerWidth`).
2. A viewer with `prefers-reduced-motion` must still see the instrument in a sensible resting state, not a blank canvas. Pinned in Task 4 (Playwright `reducedMotion: 'reduce'` run asserts the readout is populated).
3. Progress saved in `localStorage` must survive a reload and must not crash in private mode. Pinned in Task 2 (store test with a throwing storage mock) and Task 5 (e2e reload).
4. The `/ar/` chrome must not show raw message keys after new strings are added. Pinned in Task 6 (e2e: `/ar/` body text contains no `home.` / `nav.` / `learn.` substrings).
5. `/learn/<slug>/` for a lesson without a kernel file (lessons 1, 3, 4, 7) must render without the kernel block rather than throw at build time. Pinned in Task 5 (build generates all 12 pages; e2e opens `hello-thread`).

---

### Task 1: Memory math shared by the hero and Phase 1

**Files:**
- Create: `web/src/lib/memory.ts`
- Test: `web/tests/lib/memory.test.ts`

**Interfaces:**
- Produces: `sectorsTouched(byteAddrs: number[], sectorBytes = 32): number` — count of distinct `floor(addr / sectorBytes)`. `bankConflicts(byteAddrs: number[], banks = 32, wordBytes = 4): number` — max lanes mapped to one bank after collapsing identical addresses (broadcast); 1 means conflict-free. `laneAddresses(pattern: 'row' | 'col' | 'scatter', opts: { lanes?: number; elemBytes?: number; rowStride?: number; seed?: number }): number[]` — byte addresses for one warp-wide load: row = `lane*elemBytes`; col = `lane*rowStride*elemBytes`; scatter = deterministic mulberry32 picks in `[0, 480*elemBytes)`.

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, it } from 'vitest';
import { bankConflicts, laneAddresses, sectorsTouched } from '@/lib/memory';

describe('sectorsTouched', () => {
  it('32 contiguous floats touch 4 sectors', () => {
    expect(sectorsTouched(laneAddresses('row', {}))).toBe(4);
  });
  it('stride-192 column read touches 32 sectors', () => {
    expect(sectorsTouched(laneAddresses('col', { rowStride: 192 }))).toBe(32);
  });
  it('scatter is deterministic for a seed', () => {
    expect(laneAddresses('scatter', { seed: 7 })).toEqual(laneAddresses('scatter', { seed: 7 }));
  });
});

describe('bankConflicts', () => {
  it('contiguous words are conflict-free', () => {
    expect(bankConflicts(laneAddresses('row', {}))).toBe(1);
  });
  it('stride-32 words is a 32-way conflict', () => {
    expect(bankConflicts(laneAddresses('col', { rowStride: 32 }))).toBe(32);
  });
  it('stride-33 words is conflict-free', () => {
    expect(bankConflicts(laneAddresses('col', { rowStride: 33 }))).toBe(1);
  });
  it('same address for all lanes is a broadcast, not a conflict', () => {
    expect(bankConflicts(new Array(32).fill(64))).toBe(1);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `npx pnpm vitest run tests/lib/memory.test.ts` → FAIL, cannot resolve `@/lib/memory`.
- [ ] **Step 3: Implement `web/src/lib/memory.ts`** per Interfaces (pure, no imports).
- [ ] **Step 4: Run to verify pass** — same command → 7 passed.
- [ ] **Step 5: Commit** — `git add web/src/lib/memory.ts web/tests/lib/memory.test.ts && git commit -m "feat(lib): sector and bank-conflict math for memory-access visuals"`

---

### Task 2: Lesson table and progress store

**Files:**
- Create: `web/src/learn/lessons.ts`
- Create: `web/src/stores/learnStore.ts`
- Test: `web/tests/learn/lessons.test.ts`, `web/tests/learn/learnStore.test.ts`

**Interfaces:**
- Produces: `export interface Lesson { n: number; slug: string; title: string; blurb: string; minutes: number; kernelFile?: string; inspectorHref?: string; goals: string[]; builds: string }` and `export const LESSONS: Lesson[]` (12 entries, slugs: `hello-thread, bounds-grid-stride, warps, memory-hierarchy, coalescing, shared-memory-barriers, bank-conflicts, tiled-matmul, reductions-shuffles, atomics, fusing-attention, roofline-float4`; titles/blurbs/minutes exactly as the mockup's die tiles; `kernelFile` per spec §4.3 table (`05_elementwise.cu` for 2 and 12, `01_embedding.cu` for 5 and 10, `03_matmul.cu` for 6 and 8, `02_layernorm.cu` for 9, `04_attention.cu` for 11, none for 1, 3, 4, 7); `inspectorHref` = `/inspect/residual` (2, 12), `/inspect/embeddings` (5, 10), `/inspect/mlp` (6, 8), `/inspect/layernorm` (9), `/inspect/attention` (11); `goals` 3–4 short strings each; `builds` one sentence "You will write …"). Also `lessonBySlug(slug): Lesson | undefined`.
- Produces: `useLearnStore` (Zustand, `persist` name `xray-learn`, `createJSONStorage(() => localStorage)` wrapped so a throwing `localStorage` degrades to in-memory) with `{ passed: string[]; current: string; markPassed(slug): void; setCurrent(slug): void; reset(): void }`. `current` defaults to `'hello-thread'`; `markPassed` also advances `current` to the next lesson if the passed one was current. Export `lessonStatus(slug, state): 'passed' | 'current' | 'todo'`.

- [ ] **Step 1: Write failing tests**

```ts
// tests/learn/lessons.test.ts
it('has 12 lessons numbered 1..12 with unique slugs', ...)   // LESSONS.map(l=>l.n) equals [1..12]; new Set(slugs).size === 12
it('every kernelFile names a real file in kernels/', ...)     // fs.existsSync(path.join(process.cwd(),'..','kernels',l.kernelFile))
it('every inspectorHref points at an existing inspector route', ...) // fs.existsSync(`src/app/[locale]/(app)${href}/page.tsx`)

// tests/learn/learnStore.test.ts  (use `// @vitest-environment jsdom`? — no jsdom installed: instead stub globalThis.localStorage with a Map-backed object before importing the store)
it('starts at hello-thread with nothing passed', ...)
it('markPassed records the slug and advances current', ...)  // markPassed('hello-thread') → passed includes it, current === 'bounds-grid-stride'
it('markPassed on a non-current lesson leaves current alone', ...)
it('survives a localStorage that throws', ...)                // stub getItem/setItem to throw; store still works in memory
it('lessonStatus reports passed/current/todo', ...)
```

- [ ] **Step 2: Run** — `npx pnpm vitest run tests/learn` → FAIL (modules missing).
- [ ] **Step 3: Implement both files** per Interfaces.
- [ ] **Step 4: Run** — → all pass. Also `npx pnpm vitest run` → 460 + new, all green.
- [ ] **Step 5: Commit** — `feat(learn): lesson table and persisted progress store`

---

### Task 3: Design tokens and type

**Files:**
- Modify: `web/src/app/globals.css` (`@theme` block, body background, new utilities `.glow-accent`, `.glow-fwd`, `.panel`, `.btn-primary`)
- Modify: `web/src/app/[locale]/layout.tsx` (fonts: `Syne` → `--font-display`, `Instrument_Sans` → `--font-body`, keep JetBrains + Plex Arabic; update `@theme inline` font vars accordingly)
- Modify: `web/e2e/smoke.spec.ts` (new test)

**Interfaces:**
- Produces: CSS tokens per spec §4.0 (names unchanged, plus `--color-ok`), font vars `--font-display`, `--font-body`, `--font-mono`, `--font-arabic`. Utility classes: `.glow-accent { box-shadow: 0 0 24px rgba(255,154,77,.35) }`, `.glow-fwd` (ice equivalent), `.panel { border:1px solid var(--color-line); background: var(--color-surface); border-radius: 18px }`, `.btn-primary` (copper gradient, dark text, glow).

- [ ] **Step 1: Write the failing e2e test** (append to `smoke.spec.ts`):

```ts
test('new identity: display type and copper accent are live', async ({ page }) => {
  await page.goto('en/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveCSS('font-family', /Syne/);
  const accent = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim());
  expect(accent).toBe('#ff9a4d');
});
```

- [ ] **Step 2: Run** — `npx pnpm build && npx pnpm exec playwright test -g "new identity"` → FAIL (font is Bricolage, accent `#ffb454`).
- [ ] **Step 3: Implement** tokens, fonts, utilities. Replace the body radial gradients with the mockup's hero-free ground (plain `var(--color-bg)`; hero adds its own glow). Keep `:lang(ar) body` font rule.
- [ ] **Step 4: Run** — rebuild, full `playwright test` → 7 passed; `npx pnpm lint && npx pnpm exec tsc --noEmit` clean.
- [ ] **Step 5: Commit** — `feat(design): premium-dark tokens, Syne/Instrument Sans type, glow utilities`

---

### Task 4: Shell and landing page

**Files:**
- Modify: `web/src/components/Shell.tsx` (logo dot with glow; nav: Learn, Playground, Training; CTA pill "Start learning" → `/learn`; depth + locale stay)
- Create: `web/src/components/landing/WarpInstrument.tsx` (client; canvas; chips row/col/scatter; readout: requests, sectors, bytes-moved-per-useful-byte; uses `laneAddresses` + `sectorsTouched`; `prefers-reduced-motion` → static frame at phase 1)
- Create: `web/src/components/learn/LessonMap.tsx` (client; the die grid; reads `useLearnStore`; tile = `Link` to `/learn/${slug}`; classes `done`/`now` from `lessonStatus`; legend)
- Create: `web/src/components/landing/HowItWorks.tsx`, `web/src/components/landing/Labs.tsx` (three cards; thumbnails: bars, loss curve SVG, heat grid as in mockup)
- Rewrite: `web/src/app/[locale]/page.tsx`
- Modify: `web/messages/en.json`, `web/messages/ar.json` (`nav.learn`, `nav.start`, new `home.*` keys: `kicker`, `title`, `lede`, `ctaStart`, `ctaPlay`, `proof1..3`, `pathTitle`, `pathSub`, `legendPassed/Current/Todo/Total`, `howTitle`, `howSub`, `how1Title/Body` … `how3`, `labsTitle`, `labsSub`, `labPlay*`, `labTrain*`, `labInspect*`, `footer1`, `footer2`)
- Test: `web/e2e/landing.spec.ts`

**Interfaces:**
- Consumes: `laneAddresses`, `sectorsTouched` (Task 1); `LESSONS`, `useLearnStore`, `lessonStatus` (Task 2); tokens/utilities (Task 3).
- Produces: `LessonMap` reused by Task 5. Hero copy exactly as the mockup: h1 "Learn CUDA by x-raying a GPT that thinks in your browser" with "x-raying" in `text-accent` glow.

- [ ] **Step 1: Write failing e2e tests**

```ts
// e2e/landing.spec.ts
test('landing shows the 12-lesson map linking to /learn pages', async ({ page }) => {
  await page.goto('en/');
  const tiles = page.getByRole('list', { name: 'Lesson map' }).getByRole('link');
  await expect(tiles).toHaveCount(12);
  await expect(tiles.first()).toHaveAttribute('href', /\/en\/learn\/hello-thread\/$/);
});
test('warp instrument recomputes sectors per access pattern', async ({ page }) => {
  await page.goto('en/');
  const sectors = page.getByTestId('sectors');
  await expect(sectors).toHaveText('4');
  await page.getByRole('button', { name: /column/ }).click();
  await expect(sectors).toHaveText('32');
});
test('landing has no horizontal scroll at phone width', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('en/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
test('reduced motion still renders a populated instrument', async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: 'reduce', baseURL: process.env.BASE_URL });
  const page = await ctx.newPage();
  await page.goto('en/');
  await expect(page.getByTestId('sectors')).toHaveText('4');
  await ctx.close();
});
```
(For the last test, read `baseURL` from the Playwright config via `test.use`/`testInfo.project.use.baseURL` rather than env; pick whichever compiles.)

- [ ] **Step 2: Run** — `npx pnpm exec playwright test e2e/landing.spec.ts` → FAIL (no list named "Lesson map").
- [ ] **Step 3: Implement** Shell, components, page, messages. Port canvas code from the mockup's `drawWarp` (sector count must come from `sectorsTouched`, not a hand-coded number). `LessonMap` root: `<ul aria-label="Lesson map">`. Sector readout: `data-testid="sectors"`. Chips: `aria-pressed`. Arabic strings: translate meaningfully (not transliterate); numbers stay Latin.
- [ ] **Step 4: Run** — rebuild; `playwright test` → all pass (incl. existing smoke); `lint`, `tsc`.
- [ ] **Step 5: Commit** — `feat(landing): hero warp instrument, lesson map, how-it-works, labs; new shell`

---

### Task 5: `/learn` index and lesson shells

**Files:**
- Create: `web/src/app/[locale]/(app)/learn/page.tsx` (intro + `<LessonMap/>`)
- Create: `web/src/app/[locale]/(app)/learn/[slug]/page.tsx` (`generateStaticParams` from `LESSONS` × locales; `notFound()` for unknown slug)
- Create: `web/src/components/learn/LessonRail.tsx` (client; sticky list of 12 with status dots + "k of 12 passed" bar)
- Create: `web/src/components/learn/LessonShell.tsx` (server; header: "Lesson n of 12", minutes, "Builds toward kernels/<file>"; goals chips; "What you will build" paragraph; `KernelSource` block when `kernelFile` (reuse `readKernel` + `CudaCode`); three slots `Simulator`, `Exercise`, `Quiz` rendered as quiet panels whose body is the i18n string `learn.opensWithInterpreter`; "In the model" link to `inspectorHref` when present; next-lesson card)
- Modify: `web/src/components/inspect/NavRail.tsx` — no change; but add `learn.*` keys to both message files (`indexTitle`, `indexSub`, `lessonOf`, `minutes`, `buildsToward`, `goals`, `youWillBuild`, `simulator`, `exercise`, `quiz`, `opensWithInterpreter`, `inModel`, `next`, `continue`, `passedOf`)
- Test: `web/e2e/learn.spec.ts`

**Interfaces:**
- Consumes: `LESSONS`, `lessonBySlug`, `useLearnStore` (Task 2); `LessonMap` (Task 4); `readKernel` (`@/lib/kernels`), `CudaCode` (`@/components/gpu/CudaCode`).

- [ ] **Step 1: Write failing e2e tests**

```ts
test('/learn lists all lessons', ...)                       // goto 'en/learn/'; 12 links in list "Lesson map"
test('a lesson with a kernel shows its real source', ...)   // goto 'en/learn/tiled-matmul/'; expect text 'matmul_tiled' visible; link to /en/inspect/mlp/
test('a lesson without a kernel renders without the kernel block', ...) // goto 'en/learn/hello-thread/'; h1 contains 'Hello, thread'; no element [data-testid="kernel-source"]
test('progress persists across reload', ...)                // goto 'en/learn/'; page.evaluate(() => window.__stores.learn.getState().markPassed('hello-thread')); reload; expect first tile to have class /done/ ; rail text '1 of 12 passed' on a lesson page
```
Expose the learn store on `window.__stores.learn` the same way `StoreBridge` exposes the others (read `StoreBridge.tsx` first).

- [ ] **Step 2: Run** → FAIL (routes 404).
- [ ] **Step 3: Implement** routes, components, messages. Build must emit 24 lesson pages (`out/en/learn/*/index.html` ×12, `out/ar/...` ×12).
- [ ] **Step 4: Run** — rebuild; `ls out/en/learn | wc -l` → 13 (12 + index.html); `playwright test` all pass; `lint`, `tsc`, `vitest`.
- [ ] **Step 5: Commit** — `feat(learn): course index and 12 lesson shells with real kernel sources`

---

### Task 6: Labs restyle: story first

**Files:**
- Modify: `web/src/components/inspect/Panels.tsx` — wrap the `gpu` slot in `<details className="panel">` with `<summary>` text `inspect.underTheHood` (default closed)
- Modify: `web/src/components/inspect/InspectBar.tsx`, `NavRail.tsx`, `PipelineMap.tsx`, `playground/PlaygroundView.tsx`, `training/TrainView.tsx` — pass through replacing ad-hoc borders/backgrounds with `.panel`, primary actions with `.btn-primary`, and removing any uppercase `tracking-[…]` eyebrow labels; no behavior changes
- Modify: `web/messages/en.json`, `ar.json` (`inspect.underTheHood`)
- Test: `web/e2e/smoke.spec.ts` (new test)

- [ ] **Step 1: Write failing e2e tests**

```ts
test('GPU panel is collapsed under the hood by default', async ({ page }) => {
  await page.goto('en/inspect/mlp/');
  const details = page.locator('details', { hasText: 'Under the hood' });
  await expect(details).not.toHaveAttribute('open', '');
  await details.locator('summary').click();
  await expect(page.getByText('kernels/03_matmul.cu')).toBeVisible();
});
test('/ar renders no raw message keys', async ({ page }) => {
  await page.goto('ar/');
  const text = await page.locator('body').innerText();
  expect(text).not.toMatch(/\b(home|nav|learn|inspect)\.[a-zA-Z]+/);
});
```

- [ ] **Step 2: Run** → first FAILS (no details element); second may pass already — if it passes, keep it (it guards Task 4/5 strings) and note that in the ledger.
- [ ] **Step 3: Implement.** Visual pass only; keep all `aria-*`, test ids and the `.junior-only/.senior-only` gating intact.
- [ ] **Step 4: Run** — rebuild; full `playwright test` (all specs) → pass; `BASE_PATH=/transformer-xray` run after a prefixed build → pass; `vitest`, `lint`, `tsc`.
- [ ] **Step 5: Commit** — `feat(labs): story-first inspectors, GPU panel under the hood, token pass`

---

### Task 7: Ship

- [ ] **Step 1:** `git push origin main`; poll `https://api.github.com/repos/h9-tec/transformer-xray/actions/runs` until `deploy` and `ci` complete with `success`.
- [ ] **Step 2:** Verify live: `curl -s -o /dev/null -w '%{http_code}' https://h9-tec.github.io/transformer-xray/en/learn/tiled-matmul/` → 200; open `/en/` in Chrome, click the "column" chip, confirm the sector readout changes to 32, screenshot for the user.
- [ ] **Step 3:** Report: live URL, what changed, deferred items.

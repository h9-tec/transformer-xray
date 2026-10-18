# Phase 0 — GitHub Pages Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put today's Transformer X-Ray on `https://h9-tec.github.io/transformer-xray/`, built by GitHub Actions on every push to `main`, with the base path proven by the e2e suite before deploy.

**Architecture:** Next static export gains a `basePath`/`assetPrefix` read from `NEXT_PUBLIC_BASE_PATH`; the three runtime asset URLs that bypass Next (engine worker script, model files, training corpus) go through one `withBase()` helper. A dependency-free static server script serves `out/` under an optional prefix so Playwright can run the existing smoke suite in the exact layout Pages will serve. A `deploy.yml` workflow builds with the prefix and publishes with `actions/deploy-pages`.

**Tech Stack:** Next 16 (`output: 'export'`), pnpm 11, Node 24, Vitest, Playwright, GitHub Actions (`actions/configure-pages`, `actions/upload-pages-artifact`, `actions/deploy-pages`), `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-10-18-cuda-course-webgpu-pages-design.md` §4.1

## Global Constraints

- Repo: public, `h9-tec/transformer-xray`; site URL `https://h9-tec.github.io/transformer-xray/`; base path `/transformer-xray`.
- `NEXT_PUBLIC_BASE_PATH` is empty locally and `/transformer-xray` in CI/deploy. Nothing may hardcode the prefix outside the workflows.
- Static export stays: no middleware, `localePrefix: 'always'`, `trailingSlash: true`.
- Existing gates must stay green: `pnpm exec tsc --noEmit`, `pnpm lint`, `pnpm test` (457 parity tests), `pnpm build`, Playwright smoke.
- Commits: no Co-Authored-By trailers. Work on `main` (no remote exists yet; Task 4 creates it).
- All `web/` commands run from `/home/hesham/llm/web` with `npx pnpm` (pnpm is not on PATH).

## Review Focus

1. A learner opens the bare site root `https://h9-tec.github.io/transformer-xray/` → must land on `/transformer-xray/en/`, not a 404 (static export has no root page). Pinned in Task 2 (root redirect e2e).
2. The engine worker script is requested from `/transformer-xray/engine.worker.<hash>.js`, not `/engine.worker...` → playground generates text on the deployed site. Pinned in Task 2 (smoke suite run under prefix; generation test fails if the worker 404s).
3. `/model/weights.bin` and `/model/corpus.txt` fetched under the prefix → inspectors trace and training runs on Pages. Pinned in Task 2 (existing smoke tests under prefix).
4. Local dev with empty `NEXT_PUBLIC_BASE_PATH` must be unchanged → `withBase('/x') === '/x'`. Pinned in Task 1 (unit test).
5. `_next/` directory must not be swallowed by Jekyll → `.nojekyll` present in `out/`. Pinned in Task 3 (build-output assertion in the deploy workflow).

---

### Task 1: `withBase()` helper, Next base path, and the three runtime URLs

**Files:**
- Create: `web/src/lib/basePath.ts`
- Create: `web/tests/lib/basePath.test.ts`
- Modify: `web/next.config.ts`
- Modify: `web/src/engine/worker/client.ts:83`
- Modify: `web/src/lib/loadModel.ts:32`
- Modify: `web/src/stores/trainStore.ts:31-32`
- Modify: `web/public/index.html`
- Create: `web/public/.nojekyll`

**Interfaces:**
- Produces: `export const BASE_PATH: string` and `export function withBase(path: string): string` in `@/lib/basePath`. `BASE_PATH` is `process.env.NEXT_PUBLIC_BASE_PATH ?? ''` with any trailing slash removed. `withBase('/model')` returns `BASE_PATH + '/model'`; a path without a leading slash gets one added.

- [ ] **Step 1: Write the failing unit test**

```ts
// web/tests/lib/basePath.test.ts
import { describe, expect, it } from 'vitest';
import { withBase } from '@/lib/basePath';

describe('withBase', () => {
  it('returns the path unchanged when no base path is configured', () => {
    expect(withBase('/model/config.json')).toBe('/model/config.json');
  });
  it('adds a missing leading slash', () => {
    expect(withBase('engine.worker.abc.js')).toBe('/engine.worker.abc.js');
  });
});
```

Also add a second file `web/tests/lib/basePath.prefixed.test.ts` that sets
`process.env.NEXT_PUBLIC_BASE_PATH = '/transformer-xray/'` via `vi.stubEnv`
*before* a dynamic `await import('@/lib/basePath')` and asserts
`withBase('/model')` is `'/transformer-xray/model'` (trailing slash stripped,
no double slash).

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx pnpm vitest run tests/lib`
Expected: FAIL, cannot resolve `@/lib/basePath`.

- [ ] **Step 3: Implement `web/src/lib/basePath.ts`**

`BASE_PATH` and `withBase` as in Interfaces. Pure module, no imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx pnpm vitest run tests/lib`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire Next config and the three runtime URLs**

- `web/next.config.ts`: `const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/$/, '');` then set `basePath: basePath || undefined` and `assetPrefix: basePath || undefined` on `nextConfig`.
- `client.ts:83`: `new Worker(withBase(WORKER_URL))`.
- `loadModel.ts:32`: default parameter becomes `base = withBase('/model')`.
- `trainStore.ts:31-32`: fetch `withBase('/model/corpus.txt')` and name that URL in the error message.
- `web/public/index.html`: make the redirect relative so it works under any prefix: `content="0;url=en/"`, drop the `canonical` link, `href="en/"`.
- `web/public/.nojekyll`: empty file.

- [ ] **Step 6: Verify the no-prefix build and the existing gates**

Run (in `web/`): `npx pnpm exec tsc --noEmit && npx pnpm lint && npx pnpm test && npx pnpm build`
Expected: all pass; `out/.nojekyll` exists; `grep -c transformer-xray out/en/index.html` prints `0`.

- [ ] **Step 7: Verify the prefixed build emits prefixed URLs**

Run (in `web/`): `NEXT_PUBLIC_BASE_PATH=/transformer-xray npx pnpm build && grep -o '/transformer-xray/_next/static' out/en/index.html | head -1 && grep -l '"/transformer-xray"' out/_next/static/chunks/*.js | head -1`
Expected: the first grep prints the prefixed asset path; the second prints a chunk file (Next inlines `NEXT_PUBLIC_BASE_PATH`, which `withBase` prepends at runtime). Then rebuild without the env var so `out/` is back to the local layout.

- [ ] **Step 8: Commit**

```bash
git add web/src/lib/basePath.ts web/tests/lib web/next.config.ts web/src/engine/worker/client.ts web/src/lib/loadModel.ts web/src/stores/trainStore.ts web/public/index.html web/public/.nojekyll
git commit -m "feat(web): base-path aware asset URLs for GitHub Pages"
```

---

### Task 2: Prefix-aware static server and Playwright mode

**Files:**
- Create: `web/scripts/serve-static.mjs`
- Modify: `web/playwright.config.ts`
- Modify: `web/e2e/smoke.spec.ts` (every `page.goto('/…')`)
- Create: `web/e2e/root-redirect.spec.ts`

**Interfaces:**
- Produces: `node scripts/serve-static.mjs --dir out --port 4930 --prefix /transformer-xray` serves `out/` so that `GET /transformer-xray/en/` returns `out/en/index.html`, `GET /transformer-xray/` returns `out/index.html`, any path outside the prefix returns 404, and a prefix of `''` (default) serves at root. Directory requests without trailing slash 301 to the slash form. Content types: html, js, css, json, txt, bin (`application/octet-stream`), svg, ico, woff2.
- Playwright reads `BASE_PATH` (default `''`): `baseURL` is `http://localhost:4930${BASE_PATH}/` and `webServer.command` passes `--prefix ${BASE_PATH}`.
- Convention for all e2e specs from now on: `page.goto('en/playground/')` — **relative, no leading slash** — so the prefix in `baseURL` is kept.

- [ ] **Step 1: Write the failing root-redirect e2e test**

```ts
// web/e2e/root-redirect.spec.ts
import { expect, test } from '@playwright/test';

test('site root redirects to the English home', async ({ page }) => {
  await page.goto('./');
  await page.waitForURL(/\/en\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Transformer');
});
```

- [ ] **Step 2: Rewrite `playwright.config.ts` and `serve-static.mjs`**

Config per Interfaces; `webServer.command` is `node scripts/serve-static.mjs --dir out --port 4930 --prefix "${BASE_PATH ?? ''}"`, `reuseExistingServer: false` (a stale un-prefixed server would mask prefix bugs). Server: plain `node:http` + `node:fs`, parse the three flags from `process.argv`, decode the URL path, strip the prefix (404 if absent when prefix is non-empty), resolve inside `--dir` only (reject `..`), map `/` → `index.html` and `dir/` → `dir/index.html`.

- [ ] **Step 3: Convert `smoke.spec.ts` gotos to relative paths**

`'/en/playground/'` → `'en/playground/'`, and likewise for `en/inspect/attention/`, `en/train/`, `ar/`.

- [ ] **Step 4: Run both e2e modes against a fresh build**

Run (in `web/`): `npx pnpm build && npx pnpm exec playwright test`
Expected: all smoke tests + root redirect PASS at root.

Run: `NEXT_PUBLIC_BASE_PATH=/transformer-xray npx pnpm build && BASE_PATH=/transformer-xray npx pnpm exec playwright test`
Expected: all PASS under the prefix. (If the playground test fails here with a worker or fetch 404, Task 1's wiring missed a URL — fix there, not by loosening the test.)

Rebuild once more without the env var afterwards.

- [ ] **Step 5: Commit**

```bash
git add web/scripts/serve-static.mjs web/playwright.config.ts web/e2e
git commit -m "test(e2e): serve the export under an optional prefix; root redirect spec"
```

---

### Task 3: Deploy workflow, CI prefix mode, README

**Files:**
- Create: `.github/workflows/deploy.yml`
- Modify: `.github/workflows/ci.yml` (e2e job)
- Modify: `README.md` (top: live link; "Run it": deploy note)

**Interfaces:**
- Consumes: `BASE_PATH` Playwright mode from Task 2; `NEXT_PUBLIC_BASE_PATH` from Task 1.

- [ ] **Step 1: Write `deploy.yml`**

Trigger `push: branches: [main]` + `workflow_dispatch`. `permissions: { contents: read, pages: write, id-token: write }`. `concurrency: { group: pages, cancel-in-progress: true }`. Job `build` (same pnpm/Node setup as `ci.yml`, `working-directory: web`): install → `pnpm test` → `pnpm build` with `env: NEXT_PUBLIC_BASE_PATH: /transformer-xray` → assertion step `test -f out/.nojekyll && test -f out/index.html && grep -q '/transformer-xray/_next/static' out/en/index.html` → `actions/configure-pages@v5` → `actions/upload-pages-artifact@v3` with `path: web/out`. Job `deploy` (`needs: build`, `environment: { name: github-pages, url: ${{ steps.deployment.outputs.page_url }} }`) runs `actions/deploy-pages@v4` as step id `deployment`.

- [ ] **Step 2: Make the CI e2e job run in prefix mode**

In `ci.yml` e2e job: `pnpm build` gets `env: NEXT_PUBLIC_BASE_PATH: /transformer-xray`; `pnpm exec playwright test` gets `env: BASE_PATH: /transformer-xray`.

- [ ] **Step 3: README**

Under the title add: `**Live:** https://h9-tec.github.io/transformer-xray/`. In "Run it" add a 4th block: "Deploy — push to `main`; `.github/workflows/deploy.yml` builds with `NEXT_PUBLIC_BASE_PATH=/transformer-xray` and publishes to GitHub Pages. To test the Pages layout locally: `BASE_PATH=/transformer-xray npx pnpm exec playwright test` after a prefixed build."

- [ ] **Step 4: Validate workflow syntax locally**

Run: `node -e "const y=require('/home/hesham/llm/web/node_modules/yaml'); for (const f of ['ci','deploy']) y.parse(require('fs').readFileSync('.github/workflows/'+f+'.yml','utf8')); console.log('ok')"` from the repo root. If `yaml` is not present in `web/node_modules`, use `python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('ok')" .github/workflows/*.yml`.
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/deploy.yml .github/workflows/ci.yml README.md
git commit -m "ci: GitHub Pages deploy workflow; e2e runs under the Pages prefix"
```

---

### Task 4: Create the repo, push, enable Pages, verify live

**Files:** none in-repo.

**Interfaces:**
- Consumes: everything above on local `main`.

- [ ] **Step 1: Authenticate `gh`**

Run: `gh auth status`. If not logged in, the user must run `! gh auth login -p ssh -w` in this session (interactive). SSH to GitHub already works (`ssh -T git@github.com` greets `h9-tec`), so git pushes need no token; `gh` is only for repo creation, Pages enablement and run watching.

- [ ] **Step 2: Create the public repo and push**

Run (repo root):
```bash
gh repo create h9-tec/transformer-xray --public --source=. --remote=origin --description "LLM math + CUDA, live in your browser" --push
```
Expected: `origin` set to `git@github.com:h9-tec/transformer-xray.git` (or https), `main` pushed. Do **not** push `.playwright-mcp/` or `.serena/` — add both to `.gitignore` first and commit that (`chore: ignore local tool dirs`).

- [ ] **Step 3: Enable Pages with the Actions source**

Run: `gh api -X POST repos/h9-tec/transformer-xray/pages -f build_type=workflow`
Expected: HTTP 201 JSON with `"build_type": "workflow"`. (409 means already enabled: fine.)

- [ ] **Step 4: Watch the deploy run**

Run: `gh run list --workflow deploy.yml --limit 1` then `gh run watch <id> --exit-status`.
Expected: exit 0. If the `build` job fails on `pnpm test`, that's a real regression — fix locally, push, rerun.

- [ ] **Step 5: Verify the live site**

Run:
```bash
B=https://h9-tec.github.io/transformer-xray
curl -s -o /dev/null -w '%{http_code}\n' $B/            # 200
curl -s -o /dev/null -w '%{http_code}\n' $B/en/          # 200
curl -s -o /dev/null -w '%{http_code}\n' $B/model/config.json   # 200
curl -s $B/en/ | grep -o '/transformer-xray/engine.worker[^"]*' | head -1   # a hashed worker URL
curl -s -o /dev/null -w '%{http_code}\n' "$B$(curl -s $B/en/ | grep -o '/transformer-xray/engine.worker[^"]*' | head -1 | sed 's|^/transformer-xray||')"   # 200
```
Expected: four `200`s and one worker URL. Then open `$B/en/playground/` in a browser, click generate, see text.

- [ ] **Step 6: Record**

Note the live URL and repo in the project README (already done in Task 3) and report back with: live URL, run URL, and the e2e results from Task 2.

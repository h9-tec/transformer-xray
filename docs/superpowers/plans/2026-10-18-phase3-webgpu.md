# Phase 3 — WebGPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The model's forward pass, generation and training step run on the viewer's GPU through WebGPU shaders that mirror the CUDA kernels, selectable in the playground and the trainer, parity-tested against the TypeScript engine, with honest CPU fallback.

**Architecture:** `web/src/engine/gpu/` holds a small WebGPU runtime (`device.ts`: adapter/device, buffers, pipeline cache, readback), WGSL sources as TypeScript strings (`shaders.ts`, one per CUDA kernel plus the glue the op-by-op engine needs), `forward.ts` (`gpuForward`, op by op in the same order as `engine/gpt.ts`), `model.ts` (weights resident on the GPU, `gpuGenerate`), and `train.ts` (backward + AdamW on the GPU). The engine worker gains a `setBackend` message; generation and training dispatch to the GPU path when it is active. Traces (the inspectors) stay on the CPU engine. Because Node has no WebGPU, parity is tested in the browser: a static page `/debug/gpu/` runs the self-test and prints a JSON report that a dedicated Playwright project reads, skipping visibly when no adapter exists.

**Tech Stack:** WebGPU (WGSL), Next 16 static export, Zustand, Playwright (a second project with Chromium GPU flags). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-18-cuda-course-webgpu-pages-design.md` §4.4.

## Global Constraints

- Shader names and tiling mirror the CUDA files: `matmul_tiled` (TILE 16 in WGSL, 16×16 workgroups), `layernorm_warp` (one 64-thread workgroup per row with a workgroup reduction, since WGSL subgroups are not universally available), `gelu_forward`, `residual_add`, `embedding_gather`, `attention` as scores → causal softmax → PV with batched matmuls over heads.
- fp32 everywhere; parity: forward logits `< 1e-3` max abs diff vs CPU; one training step post-update params `< 5e-3`.
- GPU timings are labeled "your GPU via WebGPU" and never mixed with the CUDA `make bench` numbers.
- No adapter → the switch is disabled with a one-line reason; the CPU path is byte-for-byte unchanged.
- Playwright: the default project ignores `e2e/gpu/**`; the `gpu` project runs only `e2e/gpu/**` with Chromium flags `--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan`; its tests call `test.skip()` when the report says `adapter: null`. CI keeps running the default project; the deploy workflow is unchanged.
- Gates: `tsc`, `oxlint`, vitest, build, Playwright default project at root and prefix; the `gpu` project green locally on the RTX 5080 before the push. Commands in `web/` with `npx pnpm`. Work on `main`; push at the end.

## Review Focus

1. A viewer without WebGPU (Firefox on Linux, older Safari) opens the playground: the switch shows "CPU" selected, "your GPU" disabled with the reason, generation works exactly as before. Pinned in Task 3 (default-project e2e in the headless shell, which has no adapter).
2. A viewer whose GPU device is lost mid-generation (tab backgrounded, driver reset): the UI falls back to CPU with a banner and the model state is intact. Pinned in Task 3 (worker test: simulate `device.lost` resolution → `backend` message `{active:'cpu', reason}`).
3. A viewer switches to GPU, generates, switches back: both paths give identical greedy text. Pinned in Task 2/3 (`gpu` project e2e).
4. A viewer trains on the GPU, clicks "x-ray the current weights": the inspectors show the GPU-trained weights, not the stale CPU copy. Pinned in Task 4 (`gpu` project e2e: loss after N GPU steps < initial, and inspectNow trace loss matches the last reported loss within 1e-3).
5. The debug page must not break the static export or leak into navigation. Pinned in Task 1 (build emits `out/en/debug/gpu/index.html`; Shell nav has no link to it).

---

### Task 1: WebGPU runtime, forward kernels, in-browser kernel parity

**Files:**
- Create: `web/src/engine/gpu/device.ts`, `web/src/engine/gpu/shaders.ts`, `web/src/engine/gpu/ops.ts`, `web/src/engine/gpu/selftest.ts`
- Create: `web/src/app/[locale]/debug/gpu/page.tsx` (client page: runs `runGpuSelfTest()` on mount, renders `<pre data-testid="gpu-report">{JSON}</pre>`)
- Create: `web/e2e/gpu/kernels.spec.ts`
- Modify: `web/playwright.config.ts` (two projects), `.github/workflows/ci.yml` (`--project default` on the e2e step)

**Interfaces:**
- `device.ts`: `getGpu(): Promise<Gpu | null>` cached; `Gpu { device: GPUDevice; adapterInfo: string; hasTimestamps: boolean; lost: Promise<GPUDeviceLostInfo> }`; `class Buf { constructor(gpu, byteLength, usage); static from(gpu, data: Float32Array | Int32Array): Buf; read(): Promise<Float32Array>; readI32(): Promise<Int32Array>; destroy() }`; `pipeline(gpu, name: keyof typeof SHADERS, entry = 'main'): GPUComputePipeline` cached.
- `shaders.ts`: `export const SHADERS: Record<'embedding_gather'|'layernorm_warp'|'matmul_tiled'|'split_heads'|'merge_heads'|'causal_softmax'|'gelu_forward'|'residual_add', string>`. Uniform/params layout per shader documented in a header comment inside the WGSL. `matmul_tiled` params: `{M,K,N,batch, strideA, strideB, strideC, transA: 0|1, transB: 0|1, hasBias: 0|1, accumulate: 0|1}` so one shader serves `matmul`, `matmulTB`, `matmulTA`, batched per-head products, bias add, and residual accumulation.
- `ops.ts`: typed async ops over `Buf`s mirroring `engine/ops/*` signatures but on the GPU: `matmul(gpu, a, b, dims, opts)`, `layernorm(gpu, x, g, b, T, D, eps) → {out, mean, rstd}`, `gelu`, `add`, `gather(wte, ids, T, D)`, `splitHeads`, `mergeHeads`, `causalSoftmax(scores, H, T, scale)`. Each records into one `GPUCommandEncoder` passed in; `submit(gpu, enc)`.
- `selftest.ts`: `runGpuSelfTest(level: 'kernels' | 'forward' | 'train'): Promise<GpuReport>` where `GpuReport = { adapter: string | null; reason?: string; kernels?: Record<string, { maxDiff: number; ms: number }>; forwardMaxDiff?: number; trainMaxDiff?: number; lossGpu?: number; lossCpu?: number }`. Kernel level compares each op on random inputs (seeded `mulberry32`) against `engine/ops/*` at the model's shapes (T=128, d=192, h=6, hd=32, V=65).

- [ ] **Step 1: Write the failing e2e** (`e2e/gpu/kernels.spec.ts`): goto `en/debug/gpu/?level=kernels`, wait for `[data-testid=gpu-report][data-done=true]`, parse JSON; `test.skip(report.adapter === null, report.reason)`; expect every `kernels[*].maxDiff < 1e-4`. Add the Playwright projects. Run `npx pnpm exec playwright test --project gpu` → FAIL (page 404).
- [ ] **Step 2: Implement** device, shaders, ops, selftest, the debug page. Verify locally: `--project gpu` → PASS with a real adapter string; `npx pnpm exec playwright test --project default` still 46 passed; `ls out/en/debug/gpu/index.html`.
- [ ] **Step 3: Commit** — `feat(gpu): WebGPU runtime, forward kernels, in-browser kernel parity`

---

### Task 2: `gpuForward`, resident weights, `gpuGenerate`

**Files:**
- Create: `web/src/engine/gpu/model.ts` (`uploadModel(gpu, weights, cfg): GpuModel`, `GpuModel.download(): Promise<Weights>`, `destroy()`), `web/src/engine/gpu/forward.ts` (`gpuForward(gpu, model, cfg, ids, opts?: { keep?: boolean }) → Promise<{ logits: Float32Array; ms: number; acts?: GpuActivations }>` in the exact op order of `engine/gpt.ts forward()`), `gpuGenerate(gpu, model, cfg, prompt, maxNewTokens, sampleOpts, onToken)` (full forward per step, sampling on the CPU with `sampleFromLogits`, same RNG contract as `generateKV`)
- Modify: `selftest.ts` (`level: 'forward'`: loads the real model files, compares `gpuForward` logits to CPU `forward` on the e2e smoke prompt; `forwardMaxDiff`)
- Create: `web/e2e/gpu/forward.spec.ts`

- [ ] **Step 1: Failing e2e**: `?level=forward` → `forwardMaxDiff < 1e-3` and greedy top token per position identical to CPU (report field `argmaxMatch: true`).
- [ ] **Step 2: Implement.** **Step 3:** `--project gpu` PASS. **Step 4: Commit** — `feat(gpu): forward pass and generation on the GPU`

---

### Task 3: Backend switch in the worker and the playground; WGSL twin

**Files:**
- Modify: `web/src/engine/worker/protocol.ts` (`{type:'setBackend', backend:'cpu'|'gpu'}` request; `{type:'backend', active:'cpu'|'gpu', adapter?: string, reason?: string}` response; `token` messages gain `ms?: number`), `engine.worker.ts` (holds `GpuModel` when active; `handleGenerate` dispatches; on `device.lost` → post `backend` with `active:'cpu'` and reason; `trainInit/trainRun` unchanged in this task), `engine/worker/client.ts` (`setBackend()`), `stores/modelStore.ts` (`backend`, `backendInfo`, `setBackend`), `stores/uiStore.ts` (nothing), `components/playground/PlaygroundView.tsx` + new `components/BackendSwitch.tsx` ("Run on: CPU · your GPU" as two `aria-pressed` buttons, the GPU one disabled with `title`/text reason when unavailable; shows adapter name and last-step ms "your GPU via WebGPU"), `components/gpu/GpuPanel.tsx` (a "WGSL twin" `<details>` under the CUDA source showing `SHADERS[...]` for the stage), `messages/*.json`
- Create: `web/e2e/gpu/playground.spec.ts`; Modify `web/e2e/smoke.spec.ts` (default project: switch renders, GPU option disabled with a reason in the headless shell)

- [ ] **Step 1: Failing tests**: default-project test (switch present; `button[name=/your GPU/]` disabled; CPU generation unchanged and deterministic); gpu-project test (click "your GPU", wait for `backend` active, generate with seed 1337 → same text as CPU generation with the same seed; a "ms" readout appears). Worker unit test in vitest for the `backend` message on lost device: factor the dispatch logic into a pure `chooseBackend(request, hasGpu)` helper tested in Node.
- [ ] **Step 2: Implement.** **Step 3:** default 47+ passed; gpu project PASS. **Step 4: Commit** — `feat(gpu): CPU/GPU backend switch in the playground; WGSL twin in the GPU panel`

---

### Task 4: Training on the GPU

**Files:**
- Modify: `shaders.ts` (`layernorm_bwd`, `gelu_bwd`, `causal_softmax_bwd`, `cross_entropy` (loss + dlogits), `embedding_scatter_add` as a gather-by-row kernel (one thread per (v, i) scanning T, since WGSL has no f32 atomics), `adamw_step`, `reduce_sum` for the loss)
- Create: `web/src/engine/gpu/train.ts` (`GpuTrainer { constructor(gpu, model, cfg, hp); step(x: Int32Array, y: Int32Array): Promise<{ loss: number; ms: number }>; download(): Promise<Weights> }` mirroring `engine/gpt.ts lossBackward` + `adamwStep` op for op)
- Modify: `selftest.ts` (`level: 'train'`: identical init, one step on both engines, `trainMaxDiff` over all params `< 5e-3`, `lossGpu`/`lossCpu` within 1e-4), `engine.worker.ts` (`trainInit`/`trainRun`/`trainReset`/`inspectNow` use the GPU trainer when active; `inspectNow` downloads weights first), `components/training/TrainView.tsx` (BackendSwitch + "your GPU" step time)
- Create: `web/e2e/gpu/train.spec.ts`

- [ ] **Step 1: Failing tests**: `?level=train` → `trainMaxDiff < 5e-3`; gpu-project e2e: train 20 steps on GPU from scratch → loss decreases; click "x-ray the current weights" → trace loss equals the last reported loss within 1e-3.
- [ ] **Step 2: Implement.** **Step 3:** both projects green. **Step 4: Commit** — `feat(gpu): backward pass and AdamW on the GPU; trainer switch`

---

### Task 5: Ship

- [ ] Full gates (tsc, lint, vitest, build, default e2e root + prefix, gpu project locally). README: "Run on your GPU" paragraph + how to run the gpu Playwright project. `docs/superpowers/specs/...` status line updated. Push `main`; poll deploy + CI; open the live playground in Chrome, switch to "your GPU", generate; screenshot for the user.

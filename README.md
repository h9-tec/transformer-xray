# Transformer X-Ray

**LLM math + CUDA, live in your browser — for juniors and seniors.**

**Live:** https://h9-tec.github.io/transformer-xray/

A real 1.8M-parameter GPT (GPT-2 architecture, char-level Tiny Shakespeare)
runs its forward pass, its backward pass, and AdamW steps **inside your
browser**, hand-written in TypeScript. Every tensor, gradient, attention
matrix, and optimizer update on screen is a real fp32 number you can hover.
Each operation ships with the real CUDA kernel that would run it on a GPU —
source that compiles, verified against CPU references, with launch geometry
and memory-traffic arithmetic computed from the live shapes.

Nothing is canned. That claim is enforced by tests.

## The credibility chain

| claim | enforcement |
|---|---|
| the TS forward pass is PyTorch's math | 457 parity tests vs golden traces (`< 1e-4` per tensor) |
| the backward pass is PyTorch's math | activation + parameter grads vs autograd (`< 5e-4`) |
| the AdamW step is PyTorch's step | post-step params vs `torch.optim.AdamW` (`< 5e-6` on identical grads) |
| generation matches | greedy decode reproduces PyTorch token-for-token |
| the CUDA kernels are real | `kernels/` compiles; `make bench` verifies vs CPU at the model's shapes |
| the in-browser GPU runs real CUDA | the same five `.cu` files run on the interpreter and match the engine (`< 1e-5`) |
| the WebGPU path is the same model | forward `< 1e-3`, one AdamW step `< 5e-3`, seeded text identical (`web/e2e/gpu/`) |

## Layout

```
model/     Python: train the checkpoint, export weights + golden traces (the parity source of truth)
web/       Next.js static export: landing, /learn course, playground, watch-it-learn, 9 inspectors, GPU panels
web/src/cuda/  the in-browser CUDA interpreter (lexer → parser → checker → lockstep warp runtime + instrumentation)
kernels/   CUDA teaching kernels (naive + optimized) + bench harness
```

## Run it

```bash
# 1. Python pipeline (one-time; a trained checkpoint is committed)
python3 -m venv model/.venv && model/.venv/bin/pip install -r model/requirements.txt
make train          # ~10 min on a consumer GPU (or use the committed ckpt)
make artifacts      # export weights + golden traces into web/

# 2. Web app
cd web && npx pnpm install
npx pnpm test       # the parity suite
npx pnpm dev        # http://localhost:3000

# 3. CUDA kernels (optional, needs nvcc)
make bench
```

Deploy: push to `main`. `.github/workflows/deploy.yml` builds with
`NEXT_PUBLIC_BASE_PATH=/transformer-xray` and publishes to GitHub Pages.
To test the Pages layout locally:

```bash
cd web && NEXT_PUBLIC_BASE_PATH=/transformer-xray npx pnpm build
BASE_PATH=/transformer-xray npx pnpm exec playwright test --project default
```

WebGPU parity runs only locally, in full Chromium with a real adapter (CI's headless
shell has none, so CI runs `--project default`):

```bash
cd web && npx pnpm build && npx pnpm exec playwright test --project=gpu
```

## What's inside

- **Playground** — prompt → live generation (~300 tok/s, KV-cached), click any
  character to see the exact distribution it was drawn from, all 65 bars.
- **Watch it learn** — a real training loop (~3 steps/s of fwd+bwd+AdamW) from
  random noise or the checkpoint; loss curve, live samples, and an
  "x-ray the current weights" button that feeds the live params to every inspector.
- **Nine inspectors** — tokens, embeddings, layernorm, attention (the four-step
  score→scale→mask→softmax story), MLP, residual stream, logits, loss, AdamW —
  each with a junior/senior depth toggle and a **forward/backward phase switch**
  that mirrors every view into its gradient twin.
- **An in-browser GPU** — a TypeScript interpreter for a CUDA subset
  (`docs/cuda-subset.md`) that runs kernels warp by warp in lockstep with
  barriers, shuffles, atomics and shared memory, and reports sectors per
  request, bank conflicts, divergence and data races. All five `kernels/*.cu`
  files run on it unchanged and match the TS engine (`web/tests/cuda/fixtures.test.ts`).
- **Run on your GPU** — the playground and the trainer have a "Run on: CPU · your GPU"
  switch. The GPU path is a WebGPU twin of the engine (`web/src/engine/gpu/`): one WGSL
  shader per CUDA kernel, the same op order as `engine/gpt.ts`, parity-tested in the
  browser (`/en/debug/gpu/`, Playwright project `gpu`) — forward logits `< 1e-3`, one
  AdamW step `< 5e-3` on every parameter, seeded text identical to the CPU path. No
  adapter → the switch is disabled with the reason and nothing else changes.
- **GPU panels** — per-op CUDA source, thread-block tiling diagrams driven by the
  live shapes, naive-vs-tiled traffic arithmetic, and timings measured on real
  hardware (labeled), never simulated.
- **i18n** — English now; the UI chrome ships in Arabic (`/ar`, full RTL) with
  content translation prepared for.

## Numbers

Model: 4 layers · d=192 · 6 heads (head_dim 32 = warp size, on purpose) ·
context 256 · vocab 65 chars · 1,841,472 params · val loss 1.543
(~“Shakespeare-flavored”; the point is transparency, not fluency — this model
is ~3,000× smaller than GPT-2).

# Transformer X-Ray

**LLM math + CUDA, live in your browser — for juniors and seniors.**

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

## Layout

```
model/     Python: train the checkpoint, export weights + golden traces (the parity source of truth)
web/       Next.js static export: playground, watch-it-learn, 9 inspectors, GPU panels
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

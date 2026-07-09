# kernels/

The CUDA half of Transformer X-Ray: every kernel the web app's GPU panels
display lives here, compiles, and is verified against a CPU reference at the
model's real shapes (d=192, h=6, hd=32, V=65, T=128).

```
make bench                     # compile + verify + time, prints JSON
make bench ARCH=compute_86     # default; PTX-only so newer GPUs JIT via the driver
```

Requires the CUDA toolkit (nvcc). The default `compute_86` PTX target runs
on Ampere and everything after it (including Blackwell laptops via driver
JIT, which is how the committed `results/rtx-5060-laptop.json` was produced).

| file | kernels | the lesson |
|---|---|---|
| `01_embedding.cu` | gather, scatter-add | memory-bound ops, coalescing, why backward needs atomics |
| `02_layernorm.cu` | naive, warp-shuffle | parallelizing reductions; measured ~10× from using the warp |
| `03_matmul.cu` | naive, 32×32 tiled | shared-memory tiling — the idea behind every fast GEMM |
| `04_attention.cu` | fused row softmax | causal masking as loop bounds; the load-imbalance it creates |
| `05_elementwise.cu` | gelu, gelu float4, residual add | grid-stride loops, vectorized loads, the roofline |

What the web app claims about these files is enforced by construction: the
GPU panels read this exact source at build time, and the launch-geometry
diagrams use the same `⌈N/TILE⌉` arithmetic as the launch code above.

To record timings for your GPU: `make bench > results/<your-gpu>.json`, then
copy it to `web/src/content/gpu-results.json` and rebuild the site.

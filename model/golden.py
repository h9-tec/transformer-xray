"""Golden traces: the parity source of truth for the TypeScript engine.

Runs the trained model on fixed prompts in strict fp32 on CPU (TF32 off,
deterministic algorithms) and exports:

  golden-t4:  activations + activation grads + ALL param grads
              + params after one AdamW step (full optimizer coverage,
              tiny activations so the file stays small)
  golden-t64: activations + activation grads + loss (coverage at depth)

plus a greedy-generation transcript the TS engine must reproduce
token-for-token. Tensor names match TS TraceNode ids exactly; "grad:<id>"
for activation grads, "grad:param:<name>" / "step:param:<name>" for
parameter grads and post-step values (in the same [in,out] layout as
weights.bin).
"""

import json
import os
from dataclasses import asdict

import numpy as np
import torch

from blobio import write_blob
from config import ARTIFACTS_DIR, DATA_DIR, GoldenAdamW, ModelConfig
from export import canonical_tensors
from model import GPT, Trace

GOLDEN_DIR = os.path.join(ARTIFACTS_DIR, "golden")

# Fixed prompts. x/y are corpus-style char sequences, y = x shifted by one.
T4 = {"x": "ROME", "y": "OMEO"}
T64_TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\nAll:\n"  # 67 chars
GREEDY = {"prompt": "O God, O God!", "tokens": 50}


def encode(s: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([[stoi[c] for c in s]], dtype=torch.long)


def trace_case(model: GPT, x: str, y: str, stoi: dict[str, int]):
    model.zero_grad(set_to_none=True)
    trace = Trace()
    _, loss = model(encode(x, stoi), encode(y, stoi), trace=trace)
    loss.backward()
    return trace, float(loss.item())


def squeeze_batch(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy().astype(np.float32)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def activation_tensors(trace: Trace) -> list[tuple[str, np.ndarray]]:
    out = []
    for name, t in trace.tensors.items():
        out.append((name, squeeze_batch(t)))
        if t.grad is not None:
            out.append((f"grad:{name}", squeeze_batch(t.grad)))
    return out


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    cfg = ModelConfig()
    ckpt = torch.load(os.path.join(ARTIFACTS_DIR, "ckpt.pt"), map_location="cpu", weights_only=True)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])

    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        chars = json.load(f)["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    # 1. Greedy generation transcript (before any param mutation).
    idx = encode(GREEDY["prompt"], stoi)
    out = model.generate(idx, GREEDY["tokens"], greedy=True)
    greedy_output = "".join(chars[i] for i in out[0].tolist())

    # 2. T=64 trace: activations + activation grads.
    x64, y64 = T64_TEXT[:64], T64_TEXT[1:65]
    trace64, loss64 = trace_case(model, x64, y64, stoi)
    manifest64 = write_blob(os.path.join(GOLDEN_DIR, "golden-t64.bin"), activation_tensors(trace64))
    with open(os.path.join(GOLDEN_DIR, "golden-t64.json"), "w") as f:
        json.dump({"x": x64, "y": y64, "loss": loss64, "tensors": manifest64}, f)

    # 3. T=4 trace: activations + activation grads + param grads + one AdamW step.
    trace4, loss4 = trace_case(model, T4["x"], T4["y"], stoi)
    tensors4 = activation_tensors(trace4)

    grads = {k: p.grad for k, p in model.named_parameters()}
    tensors4 += [(f"grad:param:{n}", a) for n, a in canonical_tensors(grads, cfg.n_layer)]

    g = GoldenAdamW()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=g.lr,
        betas=(g.beta1, g.beta2),
        eps=g.eps,
        weight_decay=g.weight_decay,
    )
    opt.step()  # exactly one step on the grads from the T=4 backward
    stepped = dict(model.named_parameters())
    tensors4 += [(f"step:param:{n}", a) for n, a in canonical_tensors(stepped, cfg.n_layer)]

    manifest4 = write_blob(os.path.join(GOLDEN_DIR, "golden-t4.bin"), tensors4)
    with open(os.path.join(GOLDEN_DIR, "golden-t4.json"), "w") as f:
        json.dump(
            {
                "x": T4["x"],
                "y": T4["y"],
                "loss": loss4,
                "adamw": asdict(g),
                "greedy": {**GREEDY, "output": greedy_output},
                "tensors": manifest4,
            },
            f,
        )

    n64 = sum(t["size"] for t in manifest64)
    n4 = sum(t["size"] for t in manifest4)
    print(f"golden-t64: {len(manifest64)} tensors, {n64 * 4 / 1e6:.1f} MB | loss {loss64:.4f}")
    print(f"golden-t4:  {len(manifest4)} tensors, {n4 * 4 / 1e6:.1f} MB | loss {loss4:.4f}")
    print(f"greedy: {greedy_output!r}")


if __name__ == "__main__":
    main()

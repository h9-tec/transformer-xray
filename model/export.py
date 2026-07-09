"""Export the trained checkpoint for the TypeScript engine.

Weight layout contract: PyTorch Linear.weight is [out, in]; we export it
TRANSPOSED as [in, out] row-major, so the TS matmul  x[T,in] @ W[in,out]
never transposes at runtime. Gradients and stepped params in golden.py go
through the same canonicalization so layouts always agree.

The tied lm_head.weight is not exported — TS derives logits from wte.
"""

import json
import os

import numpy as np
import torch

from blobio import read_blob, write_blob
from config import ARTIFACTS_DIR, DATA_DIR, ModelConfig


def canonical_tensors(source: dict, n_layer: int) -> list[tuple[str, np.ndarray]]:
    """Map PyTorch state-dict-keyed tensors to canonical export names/layouts.

    `source` maps state_dict keys to tensors (params, grads, or stepped params).
    2-D Linear weights are transposed to [in, out]; everything else passes through.
    """

    def t(key: str, transpose: bool = False) -> np.ndarray:
        arr = source[key].detach().cpu().numpy()
        return arr.T if transpose else arr

    items: list[tuple[str, np.ndarray]] = [
        ("wte", t("transformer.wte.weight")),
        ("wpe", t("transformer.wpe.weight")),
    ]
    for b in range(n_layer):
        p = f"transformer.h.{b}"
        items += [
            (f"block{b}.ln1.g", t(f"{p}.ln_1.g")),
            (f"block{b}.ln1.b", t(f"{p}.ln_1.b")),
            (f"block{b}.attn.qkv.w", t(f"{p}.attn.c_attn.weight", transpose=True)),
            (f"block{b}.attn.qkv.b", t(f"{p}.attn.c_attn.bias")),
            (f"block{b}.attn.proj.w", t(f"{p}.attn.c_proj.weight", transpose=True)),
            (f"block{b}.attn.proj.b", t(f"{p}.attn.c_proj.bias")),
            (f"block{b}.ln2.g", t(f"{p}.ln_2.g")),
            (f"block{b}.ln2.b", t(f"{p}.ln_2.b")),
            (f"block{b}.mlp.fc.w", t(f"{p}.mlp.c_fc.weight", transpose=True)),
            (f"block{b}.mlp.fc.b", t(f"{p}.mlp.c_fc.bias")),
            (f"block{b}.mlp.proj.w", t(f"{p}.mlp.c_proj.weight", transpose=True)),
            (f"block{b}.mlp.proj.b", t(f"{p}.mlp.c_proj.bias")),
        ]
    items += [
        ("final_ln.g", t("transformer.ln_f.g")),
        ("final_ln.b", t("transformer.ln_f.b")),
    ]
    return items


def main() -> None:
    cfg = ModelConfig()
    ckpt = torch.load(os.path.join(ARTIFACTS_DIR, "ckpt.pt"), map_location="cpu", weights_only=True)
    state = ckpt["model_state"]

    tensors = canonical_tensors(state, cfg.n_layer)
    bin_path = os.path.join(ARTIFACTS_DIR, "weights.bin")
    manifest = write_blob(bin_path, tensors)

    n_params = sum(t["size"] for t in manifest)
    assert n_params == 1_841_472, f"unexpected param count {n_params:,}"

    with open(os.path.join(ARTIFACTS_DIR, "config.json"), "w") as f:
        json.dump(
            {
                "model": cfg.to_dict(),
                "val_loss": ckpt["val_loss"],
                "trained_iters": ckpt["iter"],
                "n_params": n_params,
                "tensors": manifest,
            },
            f,
            indent=1,
        )

    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.json"), "w") as f:
        json.dump({"chars": meta["chars"]}, f)

    # Round-trip verification: what TS will read must equal what PyTorch has.
    back = read_blob(bin_path, manifest)
    for name, arr in tensors:
        assert np.array_equal(back[name], arr.astype(np.float32)), f"round-trip mismatch: {name}"

    print(
        f"exported {n_params:,} params ({os.path.getsize(bin_path) / 1e6:.2f} MB), "
        f"val loss {ckpt['val_loss']:.4f}, round-trip OK"
    )


if __name__ == "__main__":
    main()

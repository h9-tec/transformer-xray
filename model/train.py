"""Train the tiny GPT on Tiny Shakespeare. Minutes on a consumer GPU."""

import argparse
import math
import os
import time

import numpy as np
import torch

from config import ARTIFACTS_DIR, DATA_DIR, ModelConfig, TrainConfig
from model import GPT


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


def lr_at(it: int, t: TrainConfig) -> float:
    if it < t.warmup_iters:
        return t.lr * (it + 1) / t.warmup_iters
    progress = (it - t.warmup_iters) / max(1, t.max_iters - t.warmup_iters)
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, splits, m: ModelConfig, t: TrainConfig, device: str) -> dict:
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(t.eval_iters)
        for i in range(t.eval_iters):
            x, y = get_batch(data, m.block_size, t.batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    m, t = ModelConfig(), TrainConfig()
    if args.max_iters is not None:
        t = TrainConfig(max_iters=args.max_iters)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    torch.set_float32_matmul_precision(
        "high"
    )  # TF32 fine for training; golden.py is strict fp32 CPU

    splits = {
        "train": np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16),
        "val": np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16),
    }

    model = GPT(m).to(args.device)
    print(f"params: {model.num_params():,} | device: {args.device}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=t.lr, betas=(t.beta1, t.beta2), weight_decay=t.weight_decay
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for it in range(t.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, t)

        if it % t.eval_interval == 0 or it == t.max_iters:
            losses = estimate_loss(model, splits, m, t, args.device)
            elapsed = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f} | {elapsed:6.1f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": m.to_dict(),
                        "iter": it,
                        "val_loss": best_val,
                    },
                    os.path.join(ARTIFACTS_DIR, "ckpt.pt"),
                )
        if it == t.max_iters:
            break

        x, y = get_batch(splits["train"], m.block_size, t.batch_size, args.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        opt.step()

    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()

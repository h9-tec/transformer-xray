"""Sanity check: generate text from the trained checkpoint."""

import argparse
import json
import os

import torch

from config import ARTIFACTS_DIR, DATA_DIR, ModelConfig
from model import GPT


def load_model(device: str = "cpu") -> tuple[GPT, str, float]:
    ckpt = torch.load(
        os.path.join(ARTIFACTS_DIR, "ckpt.pt"), map_location=device, weights_only=True
    )
    model = GPT(ModelConfig()).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        chars = json.load(f)["chars"]
    return model, chars, ckpt["val_loss"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, chars, val_loss = load_model()
    stoi = {ch: i for i, ch in enumerate(chars)}

    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(f"--- val loss {val_loss:.4f} ---")
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()

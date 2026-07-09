"""Tiny Shakespeare, character-level.

Downloads the corpus, builds the 65-char vocabulary, and writes
train.bin / val.bin (uint16 token ids) plus meta.json (the charset).
"""

import json
import os
import urllib.request

import numpy as np

from config import DATA_DIR

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text() -> str:
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def build() -> None:
    text = load_text()
    chars = sorted(set(text))
    assert len(chars) == 65, f"expected 65 chars, got {len(chars)}"
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(len(data) * 0.9)
    data[:n].tofile(os.path.join(DATA_DIR, "train.bin"))
    data[n:].tofile(os.path.join(DATA_DIR, "val.bin"))

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"chars": "".join(chars), "vocab_size": len(chars)}, f)

    print(f"corpus: {len(data):,} chars | train {n:,} | val {len(data) - n:,} | vocab {len(chars)}")


if __name__ == "__main__":
    build()

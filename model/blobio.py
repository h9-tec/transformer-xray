"""Shared binary tensor blob writer.

Format: raw little-endian float32, tensors concatenated; a JSON manifest
lists {name, shape, offset(bytes), size(floats)} per tensor. Both
weights.bin and the golden traces use this — the TS loader has exactly
one format to parse.
"""

import numpy as np


def write_blob(path_bin: str, tensors: list[tuple[str, np.ndarray]]) -> list[dict]:
    manifest = []
    offset = 0
    with open(path_bin, "wb") as f:
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            f.write(arr.tobytes())
            manifest.append(
                {"name": name, "shape": list(arr.shape), "offset": offset, "size": int(arr.size)}
            )
            offset += arr.nbytes
    return manifest


def read_blob(path_bin: str, manifest: list[dict]) -> dict[str, np.ndarray]:
    raw = np.fromfile(path_bin, dtype=np.float32)
    out = {}
    for t in manifest:
        start = t["offset"] // 4
        out[t["name"]] = raw[start : start + t["size"]].reshape(t["shape"])
    return out

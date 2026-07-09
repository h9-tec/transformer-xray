"""Single source of truth for model + training hyperparameters.

The TypeScript engine reads the exported config.json, never this file,
but every number originates here.
"""

from dataclasses import asdict, dataclass

ARTIFACTS_DIR = "model/artifacts"
DATA_DIR = "model/data"


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 4
    n_embd: int = 192
    n_head: int = 6  # head_dim = 32 = warp size, on purpose (the CUDA story maps 1:1)
    block_size: int = 256
    vocab_size: int = 65  # asserted against the actual Tiny Shakespeare charset
    ln_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        return d


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 5000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    seed: int = 1337


@dataclass(frozen=True)
class GoldenAdamW:
    """Hyperparameters for the single visualized/parity-tested AdamW step.

    These are the numbers the optimizer inspector teaches; golden.py and the
    TS engine must both use exactly these.
    """

    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1

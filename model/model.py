"""GPT-2 style tiny transformer, written for transparency.

The math is deliberately explicit — manual LayerNorm (biased variance,
eps inside the sqrt), explicit attention (no F.scaled_dot_product_attention),
tanh-approximated GELU — so this reference and the TypeScript engine are
line-for-line comparable. Every intermediate tensor can be recorded through
the `Trace` hook; golden.py uses that to export activations and gradients
for the parity tests.

No dropout anywhere: determinism and parity beat regularization at 1.8M params
(best-val checkpointing handles overfitting).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelConfig


class Trace:
    """Records named intermediate tensors and keeps their grads after backward."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}

    def rec(self, name: str, t: torch.Tensor) -> torch.Tensor:
        self.tensors[name] = t
        if t.requires_grad:
            t.retain_grad()
        return t


def _norec(name: str, t: torch.Tensor) -> torch.Tensor:
    return t


class LayerNorm(nn.Module):
    """Explicit layernorm exposing mean/rstd (the backward pass needs them,
    and the inspector shows them)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, rec=_norec, name: str = "") -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)  # biased, like nn.LayerNorm
        rstd = torch.rsqrt(var + self.eps)
        rec(f"{name}.mean", mean.squeeze(-1))
        rec(f"{name}.rstd", rstd.squeeze(-1))
        return (x - mean) * rstd * self.g + self.b


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer(
            "mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False
        )

    def forward(self, x: torch.Tensor, rec=_norec, p: str = "attn") -> torch.Tensor:
        B, T, C = x.shape
        qkv = rec(f"{p}.qkv", self.c_attn(x))  # [B,T,3C], columns ordered q|k|v
        q, k, v = qkv.split(C, dim=2)
        q = rec(f"{p}.q", q.view(B, T, self.n_head, self.head_dim).transpose(1, 2))  # [B,h,T,hd]
        k = rec(f"{p}.k", k.view(B, T, self.n_head, self.head_dim).transpose(1, 2))
        v = rec(f"{p}.v", v.view(B, T, self.n_head, self.head_dim).transpose(1, 2))

        raw = rec(f"{p}.scores_raw", q @ k.transpose(-2, -1))  # [B,h,T,T]
        scaled = rec(f"{p}.scores", raw / math.sqrt(self.head_dim))
        masked = rec(f"{p}.masked", scaled.masked_fill(~self.mask[:, :, :T, :T], float("-inf")))
        att = rec(f"{p}.softmax", F.softmax(masked, dim=-1))
        wv = rec(f"{p}.wv", att @ v)  # [B,h,T,hd]

        y = wv.transpose(1, 2).contiguous().view(B, T, C)  # merge heads (a view in the TS engine)
        return rec(f"{p}.proj", self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor, rec=_norec, p: str = "mlp") -> torch.Tensor:
        x = rec(f"{p}.fc", self.c_fc(x))
        x = rec(f"{p}.gelu", F.gelu(x, approximate="tanh"))
        return rec(f"{p}.proj", self.c_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.ln_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.ln_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, rec=_norec, p: str = "block") -> torch.Tensor:
        h = rec(f"{p}.ln1", self.ln_1(x, rec, f"{p}.ln1"))
        x = rec(f"{p}.res1", x + self.attn(h, rec, f"{p}.attn"))
        h = rec(f"{p}.ln2", self.ln_2(x, rec, f"{p}.ln2"))
        x = rec(f"{p}.res2", x + self.mlp(h, rec, f"{p}.mlp"))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg.n_embd, cfg.ln_eps),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):  # GPT-2 scaled init for residual projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for _, p in self.named_parameters())  # tied weight counted once

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, trace: Trace | None = None
    ):
        rec = trace.rec if trace is not None else _norec
        B, T = idx.shape
        assert T <= self.cfg.block_size

        pos = torch.arange(T, device=idx.device)
        tok = rec("emb.token", self.transformer.wte(idx))  # [B,T,d]
        pos_emb = rec("emb.pos", self.transformer.wpe(pos))  # [T,d], broadcast over batch
        x = rec("emb.sum", tok + pos_emb)

        for b, block in enumerate(self.transformer.h):
            x = block(x, rec, f"block{b}")

        x = rec("final.ln", self.transformer.ln_f(x, rec, "final.ln"))
        logits = rec("logits", self.lm_head(x))  # [B,T,V]

        loss = None
        if targets is not None:
            per_token = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), targets.reshape(-1), reduction="none"
            )
            rec("loss.per_token", per_token.view(B, T))
            loss = rec("loss", per_token.mean())
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

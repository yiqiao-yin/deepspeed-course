#!/usr/bin/env python3
"""
Multi-head Latent Attention, from scratch, against the alternatives it replaces.

    uv run mla.py            # the whole comparison, on CPU, in seconds

Three attention variants behind one interface — MHA, GQA and MLA — so that
swapping them is a controlled experiment rather than three unrelated models.
Everything here is plain PyTorch on plain tensors: no GPU, no download, no
transformers.

What MLA is for
---------------
Autoregressive generation caches keys and values for every token it has already
seen. That cache is what makes long context expensive, and it grows linearly
with sequence length:

    MHA   cache = 2 * n_heads    * head_dim   per token per layer
    GQA   cache = 2 * n_kv_heads * head_dim   per token per layer
    MLA   cache = kv_lora_rank + qk_rope_head_dim

GQA shrinks it by sharing K/V across query heads — fewer heads, less cache, and
some quality lost with them. **MLA takes a different route: it caches a
low-rank LATENT and reconstructs K and V from it on the fly.** The cache stops
depending on the head count at all.

That last point is the one worth internalising, and the property test asserts
it: doubling `n_heads` doubles MHA's and GQA's cache and leaves MLA's
*completely unchanged*.

The decoupled RoPE detail
-------------------------
There is a catch that the DeepSeek-V2 paper spends real space on. RoPE is
position-dependent and does not commute with the low-rank reconstruction — if
you rotate the reconstructed keys, you can no longer fold the up-projection
into the query (see `absorbed=True` below), and MLA loses its speed advantage.

The fix is to split the key in two: a compressed part carrying content, and a
small **decoupled** part carrying position, which is cached separately and
shared across heads. So the cache is `kv_lora_rank + qk_rope_head_dim`, and
that second term is why.

Matrix absorption
-----------------
At inference `W_UK` can be folded into `W_UQ` once, ahead of time, so keys are
never reconstructed at all — you attend directly in the latent space. This is
what makes MLA fast rather than merely small. `forward(absorbed=True)` does
that, and `test_mla.py` asserts the two paths agree to float tolerance, because
"an optimisation that changes the answer" is the failure mode that matters.

Reference: DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model* (arXiv:2405.04434), §2.1.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttnConfig:
    """Dimensions shared by all three variants, so comparisons are honest."""

    hidden_size: int = 512
    n_heads: int = 8
    head_dim: int = 64
    # GQA only: how many key/value heads the query heads share.
    n_kv_heads: int = 2
    # MLA only.
    kv_lora_rank: int = 64
    q_lora_rank: int = 192
    qk_rope_head_dim: int = 16
    max_seq_len: int = 512

    @property
    def qk_nope_head_dim(self) -> int:
        """The content half of the MLA query/key, i.e. everything but RoPE."""
        return self.head_dim - self.qk_rope_head_dim


def rope_cache(seq_len: int, dim: int, base: float = 10000.0,
               device=None, dtype=None):
    """
    Precomputed cos/sin for rotary embeddings. Standard, not DeepSpeed-specific.

    `device` is a parameter rather than a default because building these on CPU
    and then using them against CUDA activations raises
    "Expected all tensors to be on the same device" from inside the rotation --
    a long way from the line that actually made the choice.
    """
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    cos = torch.cos(freqs).repeat_interleave(2, -1)
    sin = torch.sin(freqs).repeat_interleave(2, -1)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels. x is (batch, heads, seq, dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos[None, None, :, :] + rot * sin[None, None, :, :]


def _causal_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   scale: float) -> torch.Tensor:
    """Shared attention core, so no variant wins on a different masking rule."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    t = scores.shape[-2]
    mask = torch.triu(torch.ones(t, scores.shape[-1], dtype=torch.bool,
                                 device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(F.softmax(scores, dim=-1), v)


class MultiHeadAttention(nn.Module):
    """Vanilla MHA. The baseline every cache figure is quoted against."""

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, d, bias=False)
        self.o_proj = nn.Linear(d, cfg.hidden_size, bias=False)

    def cache_per_token(self) -> int:
        """Values cached per token per layer. Grows with the head count."""
        return 2 * self.cfg.n_heads * self.cfg.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.cfg
        shape = (b, t, c.n_heads, c.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = _causal_attend(q, k, v, c.head_dim ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class GroupedQueryAttention(nn.Module):
    """
    GQA: fewer K/V heads, shared across query heads.

    The cache shrinks by exactly n_heads / n_kv_heads, and that ratio is also
    how much key/value capacity is given up. It is a direct trade; MLA's whole
    argument is that you do not have to make it.
    """

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def cache_per_token(self) -> int:
        return 2 * self.cfg.n_kv_heads * self.cfg.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        rep = c.n_heads // c.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = _causal_attend(q, k, v, c.head_dim ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MultiHeadLatentAttention(nn.Module):
    """
    MLA: cache a low-rank latent, reconstruct K and V from it.

    The cache holds `kv_lora_rank` values of compressed content plus
    `qk_rope_head_dim` values of decoupled position, and **neither term
    mentions the head count**. That is the whole idea.

    Query compression (`q_lora_rank`) does NOT affect the cache — queries are
    not cached, only the current token's is needed. It exists to cut parameters
    and activation memory during training, and is included because leaving it
    out would misrepresent the architecture.
    """

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        nope, rope = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim

        # Query: down-project, then up-project into content + position halves.
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(cfg.q_lora_rank)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.head_dim,
                                  bias=False)

        # Key/value: ONE down-projection whose output is the cache, plus the
        # decoupled RoPE key, which is shared across heads.
        self.kv_a_proj = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank + rope,
                                   bias=False)
        self.kv_a_norm = nn.RMSNorm(cfg.kv_lora_rank)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank,
                                   cfg.n_heads * (nope + cfg.head_dim),
                                   bias=False)

        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden_size,
                                bias=False)

    def cache_per_token(self) -> int:
        """
        Independent of n_heads. Doubling the head count leaves this untouched,
        which is the property `test_mla.py` pins.
        """
        return self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim

    def forward(self, x: torch.Tensor, absorbed: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.cfg
        nope, rope = c.qk_nope_head_dim, c.qk_rope_head_dim
        cos, sin = rope_cache(t, rope, device=x.device, dtype=x.dtype)

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(b, t, c.n_heads, c.head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([nope, rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        # THE CACHE: everything downstream is rebuilt from these two tensors.
        kv = self.kv_a_proj(x)
        c_kv, k_rope = kv.split([c.kv_lora_rank, rope], dim=-1)
        c_kv = self.kv_a_norm(c_kv)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)   # shared by all heads

        kv_up = self.kv_b_proj(c_kv).view(b, t, c.n_heads, nope + c.head_dim)
        kv_up = kv_up.transpose(1, 2)
        k_nope, v = kv_up.split([nope, c.head_dim], dim=-1)

        scale = c.head_dim ** -0.5
        if absorbed:
            # Fold W_UK into the query so keys are never reconstructed: attend
            # in the LATENT space instead. Mathematically identical, and the
            # reason MLA is fast rather than merely small.
            w = self.kv_b_proj.weight.view(c.n_heads, nope + c.head_dim,
                                           c.kv_lora_rank)
            w_uk = w[:, :nope, :]                       # (heads, nope, rank)
            q_latent = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
            scores = torch.einsum("bhtr,bsr->bhts", q_latent, c_kv)
        else:
            scores = torch.matmul(q_nope, k_nope.transpose(-1, -2))

        scores = scores + torch.matmul(q_rope, k_rope.transpose(-1, -2))
        scores = scores * scale
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device),
                          diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


VARIANTS = {
    "mha": (MultiHeadAttention, "Vanilla. Cache grows with every head."),
    "gqa": (GroupedQueryAttention, "Shares K/V across query heads."),
    "mla": (MultiHeadLatentAttention, "Caches a latent; reconstructs K and V."),
}


def build(name: str, cfg: AttnConfig) -> nn.Module:
    if name not in VARIANTS:
        raise SystemExit(f"Unknown variant {name!r}. Choose from: {list(VARIANTS)}")
    return VARIANTS[name][0](cfg)


def cache_table(cfg: AttnConfig, seq_len: int, layers: int,
                dtype_bytes: int = 2) -> dict:
    """Bytes of KV cache for one sequence, per variant. Pure arithmetic."""
    out = {}
    for name in VARIANTS:
        per_token = build(name, cfg).cache_per_token()
        out[name] = per_token * seq_len * layers * dtype_bytes
    return out


def _demo() -> None:
    bar = "=" * 78
    torch.manual_seed(0)
    cfg = AttnConfig()

    print(bar)
    print("  Multi-head Latent Attention — what the cache actually costs")
    print(bar)
    print(f"  hidden {cfg.hidden_size}   heads {cfg.n_heads}   head_dim {cfg.head_dim}")
    print(f"  GQA kv_heads {cfg.n_kv_heads}   MLA kv_lora_rank {cfg.kv_lora_rank}"
          f" + rope {cfg.qk_rope_head_dim}")
    print(bar)
    print(f"  {'variant':<8} {'params':>10} {'cache/token':>13} {'vs MHA':>8}")
    base = None
    for name in VARIANTS:
        m = build(name, cfg)
        p = sum(x.numel() for x in m.parameters())
        c = m.cache_per_token()
        base = base or c
        print(f"  {name:<8} {p:>10,} {c:>13,} {base / c:>7.1f}x")

    print(bar)
    print("  The property that matters: cache vs head count")
    print(bar)
    # GQA is shown at a FIXED RATIO (kv_heads = heads/4), which is how it is
    # actually used -- Llama-3 70B is 64/8, Mistral 32/8. Holding n_kv_heads
    # constant while heads grow would flatten this column, but only by pushing
    # the sharing ratio up, and the quality goes with it. The point is that
    # GQA's cache is a knob you trade against capacity; MLA's does not exist.
    print(f"  {'n_heads':>8} {'kv_heads':>9} {'mha':>10} {'gqa':>10} {'mla':>10}")
    for h in (8, 16, 32, 64):
        kv = max(1, h // 4)
        c2 = AttnConfig(n_heads=h, n_kv_heads=kv)
        row = [build(n, c2).cache_per_token() for n in VARIANTS]
        print(f"  {h:>8} {kv:>9} {row[0]:>10,} {row[1]:>10,} {row[2]:>10,}")
    print()
    print("  MHA and GQA both scale with the head count. MLA does not move at")
    print("  all — its cache is kv_lora_rank + qk_rope_head_dim, and neither")
    print("  term mentions heads.")
    print()
    print("  You CAN flatten the GQA column by pinning n_kv_heads while heads")
    print("  grow. That is the trade being made explicit: the cache shrinks")
    print("  because fewer distinct keys and values exist, not because they")
    print("  were stored more cleverly.")

    print(bar)
    print("  Matrix absorption: the fast path must give the SAME answer")
    print(bar)
    m = build("mla", cfg).eval()
    x = torch.randn(2, 16, cfg.hidden_size)
    with torch.no_grad():
        naive = m(x, absorbed=False)
        fast = m(x, absorbed=True)
    delta = (naive - fast).abs().max().item()
    print(f"  max |naive - absorbed| = {delta:.2e}")
    print("  Folding W_UK into the query means keys are never reconstructed.")
    print("  An optimisation that changed the answer would be a bug, so the")
    print("  test suite pins this identity rather than trusting it.")
    print(bar)
    print("  Training these on a GPU with DeepSpeed: train_deepseek_from_scratch.py")
    print(bar)


if __name__ == "__main__":
    _demo()

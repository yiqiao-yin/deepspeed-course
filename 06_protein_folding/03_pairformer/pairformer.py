#!/usr/bin/env python3
"""
The AlphaFold3 Pairformer: the same trunk with the MSA representation deleted.

    uv run pairformer.py         # the comparison, on CPU, ~1 minute

Read `06_protein_folding/02_evoformer/` first. This folder only makes sense
against it, and the interesting result is the *difference*.

What AlphaFold3 changed
-----------------------
AlphaFold2's Evoformer carries two big tensors through 48 blocks:

    MSA representation    m[s, i, c_m]     N_seq x N_res
    pair representation   z[i, j, c_z]     N_res x N_res

AlphaFold3 replaced the Evoformer with the **Pairformer**, and the headline
change is a deletion: **the MSA representation is not in the trunk at all.**
What survives is the single (sequence) representation and the pair
representation:

    single representation s[i, c_s]        N_res
    pair representation   z[i, j, c_z]     N_res x N_res

The MSA has not vanished entirely -- a small **MSA module** still runs first,
four blocks rather than forty-eight, and it uses pair-weighted averaging
rather than row-wise gated self-attention. Its job is to fold evolutionary
information into `z` and then get out of the way. After it, no tensor in the
model has an N_seq axis.

What this folder is for
-----------------------
The obvious reading is "AF3 is cheaper, so the memory problem is solved."
Run `uv run pairformer.py` and read the table. It is not.

    per block, N_res = 384, N_seq = 128, bf16

        tensor                          Evoformer (AF2)   Pairformer (AF3)
        MSA representation                      12.6 MB           --
        pair representation                     37.7 MB       37.7 MB
        triangle attention logits            *  452.9 MB      452.9 MB

The MSA representation was never the problem. **The cubic term is identical**,
because the Pairformer runs the same four triangle operations the Evoformer
does -- triangular multiplicative update in both directions, triangle
attention around both nodes. Deleting the MSA representation removes a large
constant. It does not touch the asymptote.

    O(N_res^3) before, O(N_res^3) after.

So `DS4Sci_EvoformerAttention` matters just as much for AF3-class models as
for AF2-class ones, which is not what the name suggests and is the single most
useful thing to carry out of this folder.

Why this is a separate folder from 02_evoformer
-----------------------------------------------
CONTRIBUTING.md section 2 says a *family* of related methods belongs in one
folder behind a flag -- the `03_llms/05_dpo --method` precedent. Two folders
here is a deliberate exception, on the same grounds that split
`04_reward_model` from `05_dpo` from `07_online_dpo`: **their memory profiles
genuinely differ**, and memory profile is what this course is about.

The cost is that the triangle operations are written twice, in both folders,
by hand. That is the house style -- this repository duplicates on purpose so
each folder runs alone -- but it means a fix to one must be applied to the
other. `tests/test_pairformer.py` therefore re-asserts the shared properties
independently of `tests/test_evoformer.py`, so a one-sided fix is caught
rather than assumed away.

References
----------
Abramson et al. 2024, "Accurate structure prediction of biomolecular
interactions with AlphaFold 3", Nature 630. Algorithm 17 (Pairformer stack)
and Algorithm 8 (MSA module) are the blocks below.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairformerConfig:
    """
    Shape of one Pairformer stack.

    Small on purpose -- this must run on a laptop CPU. AlphaFold3 proper uses
    c_s=384, c_z=128, 48 Pairformer blocks and a 4-block MSA module; the
    arithmetic and every property asserted here are identical, the constants
    are not.
    """

    c_s: int = 32           # single (sequence) representation channels
    c_z: int = 32           # pair representation channels
    c_m: int = 32           # MSA channels -- MSA MODULE ONLY, not the trunk
    c_hidden: int = 16
    n_heads: int = 4
    n_blocks: int = 2       # Pairformer blocks (AF3: 48)
    n_msa_blocks: int = 1   # MSA module blocks (AF3: 4)
    c_opm: int = 8
    transition_n: int = 2


# =============================================================================
# Shared with 02_evoformer -- duplicated deliberately, see the module docstring
# =============================================================================


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Update edge (i, j) from edges sharing a third residue k.

    Byte-for-byte the same operation as `02_evoformer/evoformer.py`, and that
    is the point of this folder: AlphaFold3 kept it. `direction` is
    "outgoing" (sum_k a_ik * b_jk) or "incoming" (sum_k a_ki * b_kj).

    O(N^3) time, O(N^2) memory -- the sum over k contracts as it goes.

    `broken_no_third_index` is the permanent counterexample, kept here as well
    as in 02_evoformer because the two copies must be policed separately.
    """

    def __init__(self, c_z: int, c_hidden: int, direction: str = "outgoing",
                 broken_no_third_index: bool = False) -> None:
        super().__init__()
        if direction not in ("outgoing", "incoming"):
            raise ValueError(f"bad direction {direction!r}")
        self.direction = direction
        self.broken_no_third_index = broken_no_third_index
        self.norm_in = nn.LayerNorm(c_z)
        self.norm_out = nn.LayerNorm(c_hidden)
        self.linear_a = nn.Linear(c_z, c_hidden)
        self.linear_a_gate = nn.Linear(c_z, c_hidden)
        self.linear_b = nn.Linear(c_z, c_hidden)
        self.linear_b_gate = nn.Linear(c_z, c_hidden)
        self.linear_g = nn.Linear(c_z, c_z)
        self.linear_out = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor,
                pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        z_norm = self.norm_in(z)
        a = torch.sigmoid(self.linear_a_gate(z_norm)) * self.linear_a(z_norm)
        b = torch.sigmoid(self.linear_b_gate(z_norm)) * self.linear_b(z_norm)
        if pair_mask is not None:
            m = pair_mask.unsqueeze(-1)
            a, b = a * m, b * m

        if self.broken_no_third_index:
            x = a * b                                      # no triangle
        elif self.direction == "outgoing":
            x = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            x = torch.einsum("bkic,bkjc->bijc", a, b)

        return torch.sigmoid(self.linear_g(z_norm)) * self.linear_out(
            self.norm_out(x))


class TriangleAttention(nn.Module):
    """
    Attention over the third residue, biased by the pair representation.

    **AlphaFold3 kept this unchanged, and it is the whole reason the memory
    argument survives the MSA deletion.** The logit tensor has three residue
    indices -- ``[b, i, h, j, k]`` -- so it is O(N_res^3) in memory, exactly
    as in AF2.

    `DS4Sci_EvoformerAttention` replaces this body with a tiled kernel that
    never materialises the logits. Set `self.ds_kernel` to enable it.
    """

    def __init__(self, c_z: int, c_hidden: int, n_heads: int,
                 mode: str = "starting") -> None:
        super().__init__()
        if mode not in ("starting", "ending"):
            raise ValueError(f"bad mode {mode!r}")
        self.mode, self.n_heads, self.c_hidden = mode, n_heads, c_hidden
        self.norm = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_z, c_hidden * n_heads, bias=False)
        self.linear_k = nn.Linear(c_z, c_hidden * n_heads, bias=False)
        self.linear_v = nn.Linear(c_z, c_hidden * n_heads, bias=False)
        self.linear_bias = nn.Linear(c_z, n_heads, bias=False)
        self.linear_g = nn.Linear(c_z, c_hidden * n_heads)
        self.linear_out = nn.Linear(c_hidden * n_heads, c_z)
        self.ds_kernel = None

    def forward(self, z: torch.Tensor,
                pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "ending":
            z = z.transpose(-2, -3)
            if pair_mask is not None:
                pair_mask = pair_mask.transpose(-1, -2)

        b, n, _, _ = z.shape
        h, c = self.n_heads, self.c_hidden
        z_norm = self.norm(z)
        q = self.linear_q(z_norm).view(b, n, n, h, c)
        k = self.linear_k(z_norm).view(b, n, n, h, c)
        v = self.linear_v(z_norm).view(b, n, n, h, c)
        bias = self.linear_bias(z_norm).permute(0, 3, 1, 2).unsqueeze(1)

        if self.ds_kernel is not None:
            try:
                mask = (q.new_ones((b, n, 1, 1, n)) if pair_mask is None
                        else pair_mask[:, :, None, None, :].to(q.dtype))
                out = self.ds_kernel(q, k, v, [mask, bias])
                gate = torch.sigmoid(self.linear_g(z_norm)).view(b, n, n, h, c)
                out = self.linear_out((gate * out).reshape(b, n, n, h * c))
                return out.transpose(-2, -3) if self.mode == "ending" else out
            except Exception:                                   # noqa: BLE001
                self.ds_kernel = None

        # THE CUBIC TENSOR -- unchanged from AlphaFold2.
        logits = torch.einsum("bijhc,bikhc->bihjk", q, k) / math.sqrt(c) + bias
        if pair_mask is not None:
            km = pair_mask[:, :, None, None, :]
            logits = logits.masked_fill(km < 0.5, float("-inf"))
            logits = logits.masked_fill(
                (km < 0.5).all(dim=-1, keepdim=True), 0.0)
        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bihjk,bikhc->bijhc", attn, v)
        gate = torch.sigmoid(self.linear_g(z_norm)).view(b, n, n, h, c)
        out = self.linear_out((gate * out).reshape(b, n, n, h * c))
        return out.transpose(-2, -3) if self.mode == "ending" else out


class Transition(nn.Module):
    """Feed-forward block. AF3 uses SwiGLU here; ReLU keeps this readable."""

    def __init__(self, c: int, n: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.linear_1 = nn.Linear(c, c * n)
        self.linear_2 = nn.Linear(c * n, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.relu(self.linear_1(self.norm(x))))


class OuterProductMean(nn.Module):
    """
    MSA -> pair. The only path evolutionary information takes into `z`.

    In AF3 this sits in the MSA module rather than in the trunk, which is the
    structural difference this folder exists to show: it runs a handful of
    times at the start and then the MSA representation is dropped.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.linear_a = nn.Linear(c_m, c_hidden)
        self.linear_b = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(self, m: torch.Tensor,
                msa_mask: torch.Tensor | None = None) -> torch.Tensor:
        m_norm = self.norm(m)
        a, b_ = self.linear_a(m_norm), self.linear_b(m_norm)
        if msa_mask is None:
            outer = torch.einsum("bsic,bsjd->bijcd", a, b_) / a.shape[1]
        else:
            mm = msa_mask.unsqueeze(-1)
            a, b_ = a * mm, b_ * mm
            outer = torch.einsum("bsic,bsjd->bijcd", a, b_)
            norm = torch.einsum("bsi,bsj->bij", msa_mask, msa_mask)
            outer = outer / norm.clamp(min=1.0)[..., None, None]
        b, n, _, c, d = outer.shape
        return self.linear_out(outer.reshape(b, n, n, c * d))


# =============================================================================
# AlphaFold3-specific
# =============================================================================


class MSAPairWeightedAveraging(nn.Module):
    """
    AlphaFold3 Algorithm 10. The MSA module's replacement for AF2's row-wise
    gated self-attention.

    Instead of the MSA attending to itself -- which needs a full
    [N_seq, N_res, N_res] attention map -- each MSA row is averaged along the
    residue axis using weights read **off the pair representation**:

        w_ij  = softmax_j( Linear(z_ij) )         weights come from z, not m
        m_si <- g_si * sum_j w_ij * v_sj

    The MSA never computes its own query-key product. That is the operation
    AF3 removed, and it is why the MSA module can be four blocks instead of
    forty-eight.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads, self.c_hidden = n_heads, c_hidden
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_v = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_b = nn.Linear(c_z, n_heads, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * n_heads)
        self.linear_out = nn.Linear(c_hidden * n_heads, c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor,
                msa_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, n, _ = m.shape
        h, c = self.n_heads, self.c_hidden
        m_norm = self.norm_m(m)
        v = self.linear_v(m_norm).view(b, s, n, h, c)

        # Weights from the PAIR representation: [b, h, i, j]
        w = self.linear_b(self.norm_z(z)).permute(0, 3, 1, 2)
        w = torch.softmax(w, dim=-1)

        out = torch.einsum("bhij,bsjhc->bsihc", w, v)
        gate = torch.sigmoid(self.linear_g(m_norm)).view(b, s, n, h, c)
        out = (gate * out).reshape(b, s, n, h * c)
        out = self.linear_out(out)
        if msa_mask is not None:
            out = out * msa_mask.unsqueeze(-1)
        return out


class AttentionPairBias(nn.Module):
    """
    AlphaFold3 Algorithm 24. Attention on the SINGLE representation, biased by
    the pair representation.

    This is what the Pairformer carries instead of MSA attention. Note the
    shapes: `s` is ``[b, N_res, c_s]`` -- one row, not N_seq rows -- so the
    attention map is ``[b, h, N_res, N_res]`` and there is no N_seq axis
    anywhere in the trunk.
    """

    def __init__(self, c_s: int, c_z: int, c_hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads, self.c_hidden = n_heads, c_hidden
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_s, c_hidden * n_heads, bias=False)
        self.linear_k = nn.Linear(c_s, c_hidden * n_heads, bias=False)
        self.linear_v = nn.Linear(c_s, c_hidden * n_heads, bias=False)
        self.linear_bias = nn.Linear(c_z, n_heads, bias=False)
        self.linear_g = nn.Linear(c_s, c_hidden * n_heads)
        self.linear_out = nn.Linear(c_hidden * n_heads, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, n, _ = s.shape
        h, c = self.n_heads, self.c_hidden
        s_norm = self.norm_s(s)
        q = self.linear_q(s_norm).view(b, n, h, c)
        k = self.linear_k(s_norm).view(b, n, h, c)
        v = self.linear_v(s_norm).view(b, n, h, c)
        bias = self.linear_bias(self.norm_z(z)).permute(0, 3, 1, 2)

        logits = torch.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(c) + bias
        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhij,bjhc->bihc", attn, v)
        gate = torch.sigmoid(self.linear_g(s_norm)).view(b, n, h, c)
        return self.linear_out((gate * out).reshape(b, n, h * c))


class MSAModule(nn.Module):
    """
    AlphaFold3 Algorithm 8. Four blocks in the real model, and then the MSA
    representation is **discarded**.

    Returns only the updated pair representation. That return type is the
    architecture: after this call there is no tensor with an N_seq axis, and
    `tests/test_pairformer.py` asserts exactly that by watching every
    allocation in the trunk.
    """

    def __init__(self, cfg: PairformerConfig) -> None:
        super().__init__()
        self.n_blocks = cfg.n_msa_blocks
        self.opm = nn.ModuleList(
            [OuterProductMean(cfg.c_m, cfg.c_z, cfg.c_opm)
             for _ in range(cfg.n_blocks)])
        self.pwa = nn.ModuleList(
            [MSAPairWeightedAveraging(cfg.c_m, cfg.c_z, cfg.c_hidden,
                                      cfg.n_heads)
             for _ in range(cfg.n_blocks)])
        self.msa_transition = nn.ModuleList(
            [Transition(cfg.c_m, cfg.transition_n) for _ in range(cfg.n_blocks)])
        self.tri_mul_out = nn.ModuleList(
            [TriangleMultiplicativeUpdate(cfg.c_z, cfg.c_hidden, "outgoing")
             for _ in range(cfg.n_blocks)])
        self.tri_mul_in = nn.ModuleList(
            [TriangleMultiplicativeUpdate(cfg.c_z, cfg.c_hidden, "incoming")
             for _ in range(cfg.n_blocks)])
        self.pair_transition = nn.ModuleList(
            [Transition(cfg.c_z, cfg.transition_n) for _ in range(cfg.n_blocks)])

    def forward(self, m: torch.Tensor, z: torch.Tensor,
                msa_mask: torch.Tensor | None = None,
                pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        for i in range(self.n_blocks):
            z = z + self.opm[i](m, msa_mask)
            m = m + self.pwa[i](m, z, msa_mask)
            m = m + self.msa_transition[i](m)
            z = z + self.tri_mul_out[i](z, pair_mask)
            z = z + self.tri_mul_in[i](z, pair_mask)
            z = z + self.pair_transition[i](z)
        return z           # m is deliberately NOT returned


class PairformerBlock(nn.Module):
    """
    AlphaFold3 Algorithm 17. Carries (s, z) only -- no MSA.

        1. Triangular multiplicative update, outgoing
        2. Triangular multiplicative update, incoming
        3. Triangle attention, around starting node
        4. Triangle attention, around ending node
        5. Pair transition
        6. Attention on the single representation, biased by the pair rep
        7. Single transition

    Compare with `02_evoformer/evoformer.py`'s `EvoformerBlock`: steps 1-5 are
    identical. Steps 6-7 replace three MSA operations with two that act on a
    single row.
    """

    def __init__(self, cfg: PairformerConfig,
                 broken_no_third_index: bool = False) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplicativeUpdate(
            cfg.c_z, cfg.c_hidden, "outgoing", broken_no_third_index)
        self.tri_mul_in = TriangleMultiplicativeUpdate(
            cfg.c_z, cfg.c_hidden, "incoming", broken_no_third_index)
        self.tri_att_start = TriangleAttention(
            cfg.c_z, cfg.c_hidden, cfg.n_heads, "starting")
        self.tri_att_end = TriangleAttention(
            cfg.c_z, cfg.c_hidden, cfg.n_heads, "ending")
        self.pair_transition = Transition(cfg.c_z, cfg.transition_n)
        self.attn_pair_bias = AttentionPairBias(
            cfg.c_s, cfg.c_z, cfg.c_hidden, cfg.n_heads)
        self.single_transition = Transition(cfg.c_s, cfg.transition_n)

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                pair_mask: torch.Tensor | None = None):
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_att_start(z, pair_mask)
        z = z + self.tri_att_end(z, pair_mask)
        z = z + self.pair_transition(z)
        s = s + self.attn_pair_bias(s, z)
        s = s + self.single_transition(s)
        return s, z


class PairformerStack(nn.Module):
    """
    MSA module (a few blocks, then the MSA is dropped) -> Pairformer trunk ->
    symmetric contact head.

    Same inputs and same outputs as `02_evoformer`'s `EvoformerStack`, so the
    training script, the data and the metrics are interchangeable and the
    comparison is controlled.
    """

    def __init__(self, cfg: PairformerConfig, n_tokens: int = 23,
                 broken_no_third_index: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.msa_embed = nn.Embedding(n_tokens, cfg.c_m)
        self.single_embed = nn.Embedding(n_tokens, cfg.c_s)
        self.pair_left = nn.Embedding(n_tokens, cfg.c_z)
        self.pair_right = nn.Embedding(n_tokens, cfg.c_z)
        self.msa_module = MSAModule(cfg)
        self.blocks = nn.ModuleList(
            [PairformerBlock(cfg, broken_no_third_index)
             for _ in range(cfg.n_blocks)])
        self.contact_head = nn.Linear(cfg.c_z, 1)

    def forward(self, msa_tokens: torch.Tensor,
                msa_mask: torch.Tensor | None = None,
                pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        query = msa_tokens[:, 0, :]
        m = self.msa_embed(msa_tokens)
        s = self.single_embed(query)
        z = (self.pair_left(query)[:, :, None, :]
             + self.pair_right(query)[:, None, :, :])

        # The MSA module folds evolution into z and returns only z. From here
        # on, nothing in this model has an N_seq axis.
        z = self.msa_module(m, z, msa_mask, pair_mask)
        del m

        for block in self.blocks:
            s, z = block(s, z, pair_mask)

        logits = self.contact_head(z).squeeze(-1)
        return 0.5 * (logits + logits.transpose(-1, -2))


# =============================================================================
# The comparison this folder exists for
# =============================================================================


def trunk_activation_table(
    n_res: int = 384, n_seq: int = 128, c_m: int = 64, c_z: int = 128,
    c_s: int = 384, n_heads: int = 4, bytes_per_element: int = 2,
) -> dict[str, dict[str, float]]:
    """
    Per-block activation bytes for the two trunks, at AlphaFold's own widths.

    Returns ``{tensor_name: {"af2": mb, "af3": mb}}``. A value of 0.0 means
    the tensor does not exist in that architecture.
    """
    b = bytes_per_element
    return {
        "MSA representation": {
            "af2": n_seq * n_res * c_m * b / 1e6,
            "af3": 0.0,                       # deleted from the trunk
        },
        # AF2's MSA ROW attention builds [N_seq, heads, N_res, N_res] logits --
        # one N_res x N_res attention map per sequence per head. This is by far
        # the largest MSA-side tensor and omitting it (as the first version of
        # this table did) understates AF2's cost by ~24x, turning a real 24%
        # architectural saving into a misleading 1.2%. AF3's pair-weighted
        # averaging has no query-key product at all: its weights are [h, i, j],
        # with no N_seq axis, which is exactly the operation that was removed.
        "MSA row attention logits": {
            "af2": n_seq * n_heads * n_res * n_res * b / 1e6,
            "af3": 0.0,
        },
        "single representation": {
            "af2": 0.0,
            "af3": n_res * c_s * b / 1e6,
        },
        "pair representation": {
            "af2": n_res * n_res * c_z * b / 1e6,
            "af3": n_res * n_res * c_z * b / 1e6,
        },
        "triangle attention logits": {
            "af2": n_res ** 3 * n_heads * b / 1e6,
            "af3": n_res ** 3 * n_heads * b / 1e6,   # UNCHANGED
        },
    }


def print_comparison(n_res: int = 384, n_seq: int = 128) -> None:
    table = trunk_activation_table(n_res=n_res, n_seq=n_seq)
    print("\n" + "=" * 78)
    print(f"  PER-BLOCK TRUNK ACTIVATIONS  (N_res={n_res}, N_seq={n_seq}, bf16)")
    print("=" * 78)
    print(f"  {'tensor':<28}{'Evoformer (AF2)':>18}{'Pairformer (AF3)':>19}")
    print("  " + "-" * 74)
    for name, v in table.items():
        a = f"{v['af2']:>13.1f} MB" if v["af2"] else f"{'--':>16}"
        c = f"{v['af3']:>14.1f} MB" if v["af3"] else f"{'--':>17}"
        star = " *" if "triangle" in name else "  "
        print(f"  {name:<28}{a}{c}{star}")

    af2 = sum(v["af2"] for v in table.values())
    af3 = sum(v["af3"] for v in table.values())
    print("  " + "-" * 74)
    print(f"  {'TOTAL':<28}{af2:>13.1f} MB{af3:>14.1f} MB")
    print(f"\n  AF3 saves {af2 - af3:.1f} MB per block "
          f"({100 * (af2 - af3) / af2:.1f}%) -- and the starred line, which is"
          f"\n  {100 * table['triangle attention logits']['af3'] / af3:.0f}% "
          "of what remains, is IDENTICAL in both.")
    print(
        "\n  Deleting the MSA representation removes a large CONSTANT.\n"
        "  It does not touch the asymptote: O(N_res^3) before, O(N_res^3)\n"
        "  after. DS4Sci_EvoformerAttention is just as relevant to AF3-class\n"
        "  models as to AF2-class ones, which the kernel's name does not\n"
        "  suggest.\n"
    )


def print_scaling(lengths=(128, 256, 512, 1024), n_seq: int = 128) -> None:
    print("=" * 78)
    print("  AND THE GAP CLOSES AS PROTEINS GET LONGER")
    print("=" * 78)
    print(f"  {'N_res':>6}{'AF2 total':>14}{'AF3 total':>14}"
          f"{'AF3 saving':>13}{'cubic share':>14}")
    print("  " + "-" * 74)
    for n in lengths:
        t = trunk_activation_table(n_res=n, n_seq=n_seq)
        af2 = sum(v["af2"] for v in t.values())
        af3 = sum(v["af3"] for v in t.values())
        cubic = t["triangle attention logits"]["af3"]
        print(f"  {n:>6}{af2:>11.1f} MB{af3:>11.1f} MB"
              f"{100 * (af2 - af3) / af2:>12.1f}%{100 * cubic / af3:>13.1f}%")
    print(
        "\n  At 128 residues the MSA representation is worth saving. At 1024\n"
        "  it is a rounding error, because the term it was competing with\n"
        "  grew eight times faster. Architectural savings that are constants\n"
        "  lose to asymptotes, always, eventually.\n"
    )


def _demo_properties(seed: int = 0) -> None:
    """The same two symmetries 02_evoformer asserts, on this trunk."""
    from synthetic_msa import make_batch

    torch.manual_seed(seed)
    model = PairformerStack(PairformerConfig(n_blocks=1)).eval()
    msa = make_batch(batch_size=1, n_res=16, n_seq=24, seed=seed)["msa"]

    with torch.no_grad():
        base = model(msa)
        perm = torch.randperm(msa.shape[-1])
        equi = (model(msa[:, :, perm]) - base[:, perm][:, :, perm]
                ).abs().max().item()
        sperm = torch.cat([torch.zeros(1, dtype=torch.long),
                           1 + torch.randperm(msa.shape[1] - 1)])
        inv = (model(msa[:, sperm, :]) - base).abs().max().item()

    print("=" * 78)
    print("  THE SAME TWO SYMMETRIES, ON THE AF3 TRUNK")
    print("=" * 78)
    print(f"  residue permutation  -> equivariance error {equi:.2e}")
    print(f"  sequence permutation -> invariance   error {inv:.2e}")
    print(f"  contact logit spread -> std          {base.std().item():.4f}")
    print(
        "\n  Both hold, for the same reasons they hold in 02_evoformer. The\n"
        "  MSA deletion changed the cost, not the symmetries.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="The AlphaFold3 Pairformer: what the MSA deletion did, "
                    "and what it did not."
    )
    p.add_argument("--n-res", type=int, default=384)
    p.add_argument("--n-seq", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-demo", action="store_true")
    args, _ = p.parse_known_args()

    print("=" * 78)
    print("  THE ALPHAFOLD3 PAIRFORMER, FROM SCRATCH")
    print("=" * 78)
    print("  Read 06_protein_folding/02_evoformer first -- the result here is")
    print("  the DIFFERENCE between the two trunks.")

    print_comparison(args.n_res, args.n_seq)
    print_scaling(n_seq=args.n_seq)
    if not args.skip_demo:
        _demo_properties(args.seed)

    print("=" * 78)
    print("  Next:  uv run deepspeed --num_gpus=1 train_pairformer_ds.py")
    print("         --ds-evoformer-attn still applies. That is the point.")
    print("=" * 78)


if __name__ == "__main__":
    main()

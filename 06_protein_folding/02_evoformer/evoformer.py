#!/usr/bin/env python3
"""
The AlphaFold2 Evoformer, from scratch, and the memory wall it walks into.

    uv run evoformer.py          # the whole demonstration, on CPU, ~1 minute

Plain PyTorch on plain tensors: no GPU, no download, no transformers, no
OpenFold. Every block below is written out so you can read the arithmetic
rather than trust it.

What the Evoformer is
---------------------
AlphaFold2's trunk carries two tensors and spends 48 blocks passing information
between them:

    MSA representation    m[s, i, c_m]     s sequences x i residues
    pair representation   z[i, j, c_z]     every residue PAIR

The pair representation is the interesting one. It is the model's running
hypothesis about *which residues touch which*, and it is where the folding
actually happens. The MSA is how that hypothesis gets evidence: positions that
mutate together across evolution are positions that touch in space.

Why this topic is in a DeepSpeed course
---------------------------------------
Because of where the memory goes, and it is not where the rest of this course
has trained you to look.

Every other example here has a memory problem in the *parameters* -- weights,
gradients, optimizer state -- which is exactly what ZeRO shards. The Evoformer
does not. This module is small. Its problem is the **activations**, and they
scale with sequence length, not parameter count:

    pair representation        O(N_res^2 * c_z)          quadratic
    triangle attention logits  O(N_res^3 * n_heads)      CUBIC

That second line is the wall. Triangle attention computes, for every residue i,
an attention matrix over every pair (j, k) -- so the logit tensor has three
residue indices. Materialise it and memory grows as the cube of the protein
length. Printed by `uv run evoformer.py`, at AlphaFold2's real c_z=128 with
4 heads in bf16:

    N_res     pair rep      triangle logits     ratio
      128       4.2 MB              16.8 MB      4.0x
      256      16.8 MB             134.2 MB      8.0x
      512      67.1 MB            1073.7 MB     16.0x
     1024     268.4 MB            8589.9 MB     32.0x

Two of those logit tensors per block, 48 blocks in AlphaFold2 proper.

**ZeRO cannot help you here.** ZeRO-1/2/3 shard optimizer state, gradients and
parameters. A cubic activation tensor is none of those three -- every rank
materialises it in full, every step. Sharding a 3 MB parameter tensor across
eight GPUs does nothing about an 8 GB intermediate.

That is why DeepSpeed ships a *kernel* for this model family rather than a
sharding strategy. `DS4Sci_EvoformerAttention`, from the DeepSpeed4Science and
OpenFold collaboration, computes the attention in tiles and never materialises
the cubic logit tensor at all. `train_evoformer_ds.py --ds-evoformer-attn`
turns it on and reports peak memory both ways.

    The lesson generalises past proteins: when the activation is the problem,
    sharding the model is answering a question nobody asked.

The three operations worth understanding
----------------------------------------
**1. Triangular multiplicative update.** The cheap one, and the one that gives
the trunk its name. To update edge (i, j), look at every third residue k and
combine edges (i, k) and (j, k):

    z_ij <- sum_k  a_ik * b_jk          ("outgoing")
    z_ij <- sum_k  a_ki * b_kj          ("incoming")

This is a triangle: i, j and k. It is how a constraint on i-k and a constraint
on j-k become a constraint on i-j, which is the geometric reasoning the whole
architecture exists to do. O(N^3) time, but only O(N^2) memory -- the sum over
k collapses as it goes.

**2. Triangle attention.** The expensive one. Same triangle, but the third
residue is chosen by attention rather than summed uniformly. That requires the
logits for all (i, j, k) at once: O(N^3) memory as well as time.

**3. Outer product mean.** The bridge from evolution to geometry. For residues
i and j, average the outer product of their MSA columns over all sequences. If
positions i and j covary across homologs, that average is large -- which is
precisely the coevolution signal that makes MSAs worth their cost.

Two properties that are load-bearing, and one deliberate bug
------------------------------------------------------------
`tests/test_evoformer.py` asserts these, and they are not decoration:

- **The pair representation is EQUIVARIANT to residue permutation.** Relabel
  the residues and the pair representation must permute identically along both
  axes. A model that reads residue order instead of residue content passes
  every shape check and is silently wrong -- this is the exact bug
  `02_intermediate/04_groupwise_ranking` shipped and caught.
- **The pair representation is INVARIANT to permuting the non-query MSA rows.**
  The order homologs happen to arrive in carries no information.
  `OuterProductMean` averages over sequences, so this holds by construction --
  and asserting it catches anyone who "optimises" that mean into something
  order-dependent.

  Note *non-query*. Row 0 is the sequence actually being folded, and the pair
  representation is initialised from it, so moving row 0 changes the problem
  rather than its presentation. Stating the invariance over all rows would be
  a false claim, and the first version of the demonstration below made exactly
  that mistake and measured a 1.51 "failure" that was really a bad claim.

`TriangleMultiplicativeUpdate(broken_no_third_index=True)` is kept here on
purpose. It replaces the sum over k with an elementwise product of edges
(i, j) -- so it still runs, still trains, still produces the right shapes, and
has no triangle in it at all. The test suite asserts that the correct version
propagates a constraint through k and that this one measurably does not,
because a closure test that never sees a non-closing implementation would pass
while proving nothing.

References
----------
Jumper et al. 2021, "Highly accurate protein structure prediction with
AlphaFold", Nature 596. Algorithms 7, 8, 10, 11, 12, 13, 14 in the
Supplementary Information are the blocks implemented below, named to match.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvoformerConfig:
    """
    Shape of one Evoformer stack.

    The defaults are deliberately small -- this module must run on a laptop CPU
    in about a minute. AlphaFold2 proper uses c_m=256, c_z=128, 48 blocks and
    8 heads; the arithmetic and every property asserted here are identical, the
    constants are not.
    """

    c_m: int = 32          # MSA representation channels
    c_z: int = 32          # pair representation channels
    c_hidden: int = 16     # hidden width inside the triangle ops
    n_heads: int = 4       # attention heads
    n_blocks: int = 2      # Evoformer blocks in the stack
    c_opm: int = 8         # outer-product-mean projection width
    transition_n: int = 2  # transition layer expansion factor


# =============================================================================
# Algorithm 11 / 12 -- Triangular multiplicative update
# =============================================================================


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Update edge (i, j) from edges that share a third residue k.

    AlphaFold2 Supplementary Algorithms 11 (outgoing) and 12 (incoming).

        outgoing:  z_ij <- g_ij * Linear(LayerNorm( sum_k a_ik * b_jk ))
        incoming:  z_ij <- g_ij * Linear(LayerNorm( sum_k a_ki * b_kj ))

    where a and b are gated projections of z. The difference between the two
    directions is only which index of a and b the sum runs over, and both are
    used in every block because they propagate information in opposite
    directions around the triangle.

    Cost: O(N^3) time, O(N^2) memory. The sum over k is contracted by the
    einsum rather than materialised, which is what keeps memory quadratic here
    and makes triangle ATTENTION -- where the logits cannot be contracted away
    -- the expensive sibling.

    Parameters
    ----------
    direction
        "outgoing" or "incoming".
    broken_no_third_index
        **Deliberately wrong, kept permanently.** Replaces the sum over k with
        an elementwise product a_ij * b_ij, removing the third residue -- and
        with it the entire point of the operation. Used by
        `tests/test_evoformer.py` as the counterexample that makes the triangle
        closure assertion mean something. Never set this in real training.
    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        direction: str = "outgoing",
        broken_no_third_index: bool = False,
    ) -> None:
        super().__init__()
        if direction not in ("outgoing", "incoming"):
            raise ValueError(
                f"direction must be 'outgoing' or 'incoming', got {direction!r}"
            )
        self.direction = direction
        self.broken_no_third_index = broken_no_third_index

        self.norm_in = nn.LayerNorm(c_z)
        self.norm_out = nn.LayerNorm(c_hidden)

        # a and b each get a projection and a sigmoid gate (AF2 lines 2-3).
        self.linear_a = nn.Linear(c_z, c_hidden)
        self.linear_a_gate = nn.Linear(c_z, c_hidden)
        self.linear_b = nn.Linear(c_z, c_hidden)
        self.linear_b_gate = nn.Linear(c_z, c_hidden)

        # Output gate and projection back to c_z (AF2 lines 4-5).
        self.linear_g = nn.Linear(c_z, c_z)
        self.linear_out = nn.Linear(c_hidden, c_z)

    def forward(
        self, z: torch.Tensor, pair_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z
            Pair representation, ``[batch, N_res, N_res, c_z]``.
        pair_mask
            Optional ``[batch, N_res, N_res]`` mask; masked pairs contribute
            nothing to the sum over k.

        Returns
        -------
        The update to add to ``z`` -- this module returns the *delta*, not the
        updated tensor, so the caller owns the residual connection.
        """
        z_norm = self.norm_in(z)

        a = torch.sigmoid(self.linear_a_gate(z_norm)) * self.linear_a(z_norm)
        b = torch.sigmoid(self.linear_b_gate(z_norm)) * self.linear_b(z_norm)

        if pair_mask is not None:
            m = pair_mask.unsqueeze(-1)
            a = a * m
            b = b * m

        if self.broken_no_third_index:
            # THE DELIBERATE BUG. No sum over k, so no triangle: edge (i, j) is
            # updated only from itself. Runs fine, trains fine, and cannot
            # propagate a constraint from (i, k) and (j, k) to (i, j) -- which
            # is the only thing this operation is for.
            x = a * b
        elif self.direction == "outgoing":
            # z_ij <- sum_k a_ik * b_jk
            x = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            # z_ij <- sum_k a_ki * b_kj
            x = torch.einsum("bkic,bkjc->bijc", a, b)

        gate = torch.sigmoid(self.linear_g(z_norm))
        return gate * self.linear_out(self.norm_out(x))


# =============================================================================
# Algorithm 13 / 14 -- Triangle attention
# =============================================================================


class TriangleAttention(nn.Module):
    """
    Attention over the third residue, biased by the pair representation.

    AlphaFold2 Supplementary Algorithms 13 ("around starting node") and 14
    ("around ending node"). For each row i, attend from pair (i, j) over all
    pairs (i, k), with a bias read off the pair representation at (j, k):

        logits[i, j, k] = q_ij . k_ik / sqrt(c) + b_jk
        z_ij <- g_ij * sum_k softmax_k(logits)[i, j, k] * v_ik

    **This is the cubic one.** `logits` has three residue indices, so it cannot
    be contracted away the way the multiplicative update's sum over k can. At
    N_res = 512 with 4 heads in fp16 the logit tensor alone is ~1 GB, per
    operation, per block -- and there are two of these per block.

    `DS4Sci_EvoformerAttention` exists precisely to avoid materialising it: the
    kernel tiles the logits and computes the softmax in pieces, so peak memory
    stops depending on N_res^3. `train_evoformer_ds.py --ds-evoformer-attn`
    swaps this module's maths for that kernel and measures the difference.

    The "ending node" variant is the "starting node" computation applied to the
    transpose of the pair representation, which is how AF2 defines it and why
    only one attention body is written here.
    """

    def __init__(
        self, c_z: int, c_hidden: int, n_heads: int, mode: str = "starting"
    ) -> None:
        super().__init__()
        if mode not in ("starting", "ending"):
            raise ValueError(
                f"mode must be 'starting' or 'ending', got {mode!r}"
            )
        self.mode = mode
        self.n_heads = n_heads
        self.c_hidden = c_hidden

        self.norm = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_z, c_hidden * n_heads, bias=False)
        self.linear_k = nn.Linear(c_z, c_hidden * n_heads, bias=False)
        self.linear_v = nn.Linear(c_z, c_hidden * n_heads, bias=False)
        # The pair bias: one scalar per head, read from z_jk.
        self.linear_bias = nn.Linear(c_z, n_heads, bias=False)
        self.linear_g = nn.Linear(c_z, c_hidden * n_heads)
        self.linear_out = nn.Linear(c_hidden * n_heads, c_z)

        # Set by `train_evoformer_ds.py --ds-evoformer-attn` to DeepSpeed's
        # DS4Sci_EvoformerAttention. Left None here so this module stays
        # importable, runnable and testable on a CPU-only box -- the kernel
        # requires CUDA >= 11.3, compute capability >= 7.0 and fp16/bf16.
        self.ds_kernel = None

    def _use_ds_kernel(self, q, k, v, bias, pair_mask):
        """
        Hand the attention to DS4Sci_EvoformerAttention.

        The kernel's triangle-self-attention signature takes Q/K/V shaped
        ``[B, N_res, N_res, H, C]`` -- which is exactly the layout built
        above, so no permute is needed -- plus a residue mask
        ``[B, N_res, 1, 1, N_res]`` and an edge bias ``[B, 1, H, N_res, N_res]``.

        It never materialises the ``[B, i, H, j, k]`` logit tensor; it tiles
        the computation and accumulates the softmax per tile. That is the
        whole difference, and it is why peak memory stops tracking N_res^3.
        """
        b, n = q.shape[0], q.shape[1]
        if pair_mask is None:
            mask = q.new_ones((b, n, 1, 1, n))
        else:
            mask = pair_mask[:, :, None, None, :].to(q.dtype)
        return self.ds_kernel(q, k, v, [mask, bias])

    def forward(
        self, z: torch.Tensor, pair_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the update to add to ``z`` (``[batch, N_res, N_res, c_z]``)."""
        # "Ending node" is "starting node" on the transposed pair rep. Transpose
        # in, transpose out -- AF2 Algorithm 14 is defined exactly this way.
        if self.mode == "ending":
            z = z.transpose(-2, -3)
            if pair_mask is not None:
                pair_mask = pair_mask.transpose(-1, -2)

        b, n, _, _ = z.shape
        h, c = self.n_heads, self.c_hidden
        z_norm = self.norm(z)

        # [b, n, n, h, c]
        q = self.linear_q(z_norm).view(b, n, n, h, c)
        k = self.linear_k(z_norm).view(b, n, n, h, c)
        v = self.linear_v(z_norm).view(b, n, n, h, c)

        # Pair bias b_jk -> [b, 1, h, n, n]; broadcast over the row index i.
        bias = self.linear_bias(z_norm).permute(0, 3, 1, 2).unsqueeze(1)

        if self.ds_kernel is not None:
            # The tiled path. Same maths, no cubic tensor. Falls through to
            # the explicit implementation below if anything about this call is
            # unsupported -- an optional accelerator must never be able to
            # change the ANSWER, only the memory it takes to get there.
            try:
                out = self._use_ds_kernel(q, k, v, bias, pair_mask)
                gate = torch.sigmoid(self.linear_g(z_norm)).view(b, n, n, h, c)
                out = self.linear_out((gate * out).reshape(b, n, n, h * c))
                return out.transpose(-2, -3) if self.mode == "ending" else out
            except Exception:                                   # noqa: BLE001
                self.ds_kernel = None                # do not retry every step

        # THE CUBIC TENSOR: [b, i, h, j, k].
        logits = torch.einsum("bijhc,bikhc->bihjk", q, k) / math.sqrt(c)
        logits = logits + bias

        if pair_mask is not None:
            # Mask over k; -inf before softmax so masked pairs get zero weight.
            km = pair_mask[:, :, None, None, :]
            logits = logits.masked_fill(km < 0.5, float("-inf"))
            # A row that is entirely masked would softmax to NaN; keep it finite.
            all_masked = (km < 0.5).all(dim=-1, keepdim=True)
            logits = logits.masked_fill(all_masked, 0.0)

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bihjk,bikhc->bijhc", attn, v)

        gate = torch.sigmoid(self.linear_g(z_norm)).view(b, n, n, h, c)
        out = (gate * out).reshape(b, n, n, h * c)
        out = self.linear_out(out)

        if self.mode == "ending":
            out = out.transpose(-2, -3)
        return out


# =============================================================================
# Algorithm 7 / 8 -- MSA attention
# =============================================================================


class MSARowAttentionWithPairBias(nn.Module):
    """
    Attention along the residue axis, within each sequence, biased by the pair
    representation. AlphaFold2 Supplementary Algorithm 7.

    This is the direction the pair representation talks *back* to the MSA: the
    model's current geometric hypothesis (z) biases which residues each
    sequence attends to. Without this bias the two representations would only
    communicate one way.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_hidden = c_hidden
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_k = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_v = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_bias = nn.Linear(c_z, n_heads, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * n_heads)
        self.linear_out = nn.Linear(c_hidden * n_heads, c_m)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        m
            MSA representation, ``[batch, N_seq, N_res, c_m]``.
        z
            Pair representation, ``[batch, N_res, N_res, c_z]``.
        msa_mask
            Optional ``[batch, N_seq, N_res]``.

        Returns
        -------
        The update to add to ``m``.
        """
        b, s, n, _ = m.shape
        h, c = self.n_heads, self.c_hidden
        m_norm = self.norm_m(m)

        q = self.linear_q(m_norm).view(b, s, n, h, c)
        k = self.linear_k(m_norm).view(b, s, n, h, c)
        v = self.linear_v(m_norm).view(b, s, n, h, c)

        # Pair bias is shared across sequences: [b, 1, h, n, n].
        bias = self.linear_bias(self.norm_z(z)).permute(0, 3, 1, 2).unsqueeze(1)

        logits = torch.einsum("bsihc,bsjhc->bshij", q, k) / math.sqrt(c)
        logits = logits + bias

        if msa_mask is not None:
            jm = msa_mask[:, :, None, None, :]
            logits = logits.masked_fill(jm < 0.5, float("-inf"))
            all_masked = (jm < 0.5).all(dim=-1, keepdim=True)
            logits = logits.masked_fill(all_masked, 0.0)

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bshij,bsjhc->bsihc", attn, v)

        gate = torch.sigmoid(self.linear_g(m_norm)).view(b, s, n, h, c)
        out = (gate * out).reshape(b, s, n, h * c)
        return self.linear_out(out)


class MSAColumnAttention(nn.Module):
    """
    Attention along the sequence axis, within each residue column. AlphaFold2
    Supplementary Algorithm 8. No pair bias -- this direction is about which
    homologs matter, and the pair representation has nothing to say about that.

    Note the shape of the claim this supports: attention is permutation
    EQUIVARIANT over the axis it attends along, so permuting the MSA rows
    permutes this module's output identically. The pair representation only
    becomes permutation *invariant* after `OuterProductMean` averages the
    sequence axis away.
    """

    def __init__(self, c_m: int, c_hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_m)
        self.linear_q = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_k = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_v = nn.Linear(c_m, c_hidden * n_heads, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * n_heads)
        self.linear_out = nn.Linear(c_hidden * n_heads, c_m)

    def forward(
        self, m: torch.Tensor, msa_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the update to add to ``m`` (``[batch, N_seq, N_res, c_m]``)."""
        b, s, n, _ = m.shape
        h, c = self.n_heads, self.c_hidden
        m_norm = self.norm(m)

        q = self.linear_q(m_norm).view(b, s, n, h, c)
        k = self.linear_k(m_norm).view(b, s, n, h, c)
        v = self.linear_v(m_norm).view(b, s, n, h, c)

        # Attend over the sequence axis (s, t) independently per residue i.
        logits = torch.einsum("bsihc,btihc->bihst", q, k) / math.sqrt(c)

        if msa_mask is not None:
            tm = msa_mask.permute(0, 2, 1)[:, :, None, None, :]
            logits = logits.masked_fill(tm < 0.5, float("-inf"))
            all_masked = (tm < 0.5).all(dim=-1, keepdim=True)
            logits = logits.masked_fill(all_masked, 0.0)

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bihst,btihc->bsihc", attn, v)

        gate = torch.sigmoid(self.linear_g(m_norm)).view(b, s, n, h, c)
        out = (gate * out).reshape(b, s, n, h * c)
        return self.linear_out(out)


# =============================================================================
# Algorithm 10 -- Outer product mean
# =============================================================================


class OuterProductMean(nn.Module):
    """
    Turn coevolution into geometry. AlphaFold2 Supplementary Algorithm 10.

        o_ij = mean_s ( a_si (x) b_sj )
        z_ij <- z_ij + Linear(flatten(o_ij))

    For residues i and j, average the outer product of their MSA columns across
    all sequences. If i and j mutate together across homologs, that average
    carries structure; if they vary independently, it averages towards the
    product of their marginals and carries almost nothing.

    **This is the only place the MSA reaches the pair representation**, and the
    mean over sequences is what makes the pair representation invariant to the
    order the homologs arrive in. `tests/test_evoformer.py` asserts that
    invariance -- an "optimisation" that replaced this mean with anything
    order-dependent would break a real symmetry of the problem while still
    training.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_m)
        self.linear_a = nn.Linear(c_m, c_hidden)
        self.linear_b = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(
        self, m: torch.Tensor, msa_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the update to add to ``z`` (``[batch, N_res, N_res, c_z]``)."""
        m_norm = self.norm(m)
        a = self.linear_a(m_norm)          # [b, s, i, c]
        b_ = self.linear_b(m_norm)         # [b, s, j, c]

        if msa_mask is None:
            outer = torch.einsum("bsic,bsjd->bijcd", a, b_) / a.shape[1]
        else:
            mm = msa_mask.unsqueeze(-1)
            a = a * mm
            b_ = b_ * mm
            outer = torch.einsum("bsic,bsjd->bijcd", a, b_)
            # Normalise by the number of sequences observed at BOTH i and j.
            norm = torch.einsum("bsi,bsj->bij", msa_mask, msa_mask)
            outer = outer / norm.clamp(min=1.0)[..., None, None]

        b, n, _, c, d = outer.shape
        return self.linear_out(outer.reshape(b, n, n, c * d))


# =============================================================================
# Algorithm 9 / 15 -- Transition
# =============================================================================


class Transition(nn.Module):
    """
    The feed-forward block, used on both representations. AlphaFold2
    Supplementary Algorithms 9 (MSA) and 15 (pair). Expand, ReLU, project back.
    """

    def __init__(self, c: int, n: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.linear_1 = nn.Linear(c, c * n)
        self.linear_2 = nn.Linear(c * n, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the update to add to ``x``."""
        return self.linear_2(F.relu(self.linear_1(self.norm(x))))


# =============================================================================
# The block and the stack
# =============================================================================


class EvoformerBlock(nn.Module):
    """
    One Evoformer block: AlphaFold2 Supplementary Algorithm 6.

    Order matters and is not arbitrary. The MSA is updated first, then feeds
    the pair representation through the outer product mean, then the pair
    representation is refined by four triangle operations, and only then does
    the next block let the refined pair representation bias the MSA again.

        1. MSA row attention, biased by the pair representation
        2. MSA column attention
        3. MSA transition
        4. Outer product mean                     MSA  -> pair
        5. Triangular multiplicative update, outgoing
        6. Triangular multiplicative update, incoming
        7. Triangle attention, around starting node
        8. Triangle attention, around ending node
        9. Pair transition

    Every sub-block is residual, which is why each module above returns a delta
    rather than a new tensor.
    """

    def __init__(self, cfg: EvoformerConfig, broken_no_third_index: bool = False):
        super().__init__()
        self.msa_row = MSARowAttentionWithPairBias(
            cfg.c_m, cfg.c_z, cfg.c_hidden, cfg.n_heads
        )
        self.msa_col = MSAColumnAttention(cfg.c_m, cfg.c_hidden, cfg.n_heads)
        self.msa_transition = Transition(cfg.c_m, cfg.transition_n)
        self.opm = OuterProductMean(cfg.c_m, cfg.c_z, cfg.c_opm)
        self.tri_mul_out = TriangleMultiplicativeUpdate(
            cfg.c_z, cfg.c_hidden, "outgoing", broken_no_third_index
        )
        self.tri_mul_in = TriangleMultiplicativeUpdate(
            cfg.c_z, cfg.c_hidden, "incoming", broken_no_third_index
        )
        self.tri_att_start = TriangleAttention(
            cfg.c_z, cfg.c_hidden, cfg.n_heads, "starting"
        )
        self.tri_att_end = TriangleAttention(
            cfg.c_z, cfg.c_hidden, cfg.n_heads, "ending"
        )
        self.pair_transition = Transition(cfg.c_z, cfg.transition_n)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_row(m, z, msa_mask)
        m = m + self.msa_col(m, msa_mask)
        m = m + self.msa_transition(m)

        z = z + self.opm(m, msa_mask)
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_att_start(z, pair_mask)
        z = z + self.tri_att_end(z, pair_mask)
        z = z + self.pair_transition(z)
        return m, z


class EvoformerStack(nn.Module):
    """
    An Evoformer trunk plus a contact head.

    The contact head is a symmetric linear read-out of the pair representation:
    contacts are a property of the unordered pair {i, j}, so the logit for
    (i, j) and (j, i) must agree by construction rather than by training.
    """

    def __init__(
        self,
        cfg: EvoformerConfig,
        n_tokens: int = 23,
        broken_no_third_index: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.msa_embed = nn.Embedding(n_tokens, cfg.c_m)
        # Pair init from the query sequence: an outer sum of two projections,
        # the standard AF2 "left/right" single-sequence pair initialisation.
        self.pair_left = nn.Embedding(n_tokens, cfg.c_z)
        self.pair_right = nn.Embedding(n_tokens, cfg.c_z)
        self.blocks = nn.ModuleList(
            [
                EvoformerBlock(cfg, broken_no_third_index)
                for _ in range(cfg.n_blocks)
            ]
        )
        self.contact_head = nn.Linear(cfg.c_z, 1)

    def forward(
        self,
        msa_tokens: torch.Tensor,
        msa_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        msa_tokens
            Integer MSA, ``[batch, N_seq, N_res]``. Row 0 is the query.

        Returns
        -------
        Contact logits, ``[batch, N_res, N_res]``, symmetric in the last two
        dimensions.
        """
        m = self.msa_embed(msa_tokens)
        query = msa_tokens[:, 0, :]
        z = self.pair_left(query)[:, :, None, :] + self.pair_right(query)[:, None, :, :]

        for block in self.blocks:
            m, z = block(m, z, msa_mask, pair_mask)

        logits = self.contact_head(z).squeeze(-1)
        # Symmetrise: a contact is a property of the unordered pair.
        return 0.5 * (logits + logits.transpose(-1, -2))


# =============================================================================
# The memory table -- the reason this topic is in a DeepSpeed course
# =============================================================================


def memory_table(
    lengths: tuple[int, ...] = (128, 256, 512, 1024),
    n_heads: int = 4,
    c_z: int = 128,
    bytes_per_element: int = 2,
) -> list[dict[str, float]]:
    """
    Bytes for the two activation tensors that matter, as a function of protein
    length. Analytic, not measured -- the point is the exponent, and the
    exponent does not need a GPU to be true.

    Defaults use AlphaFold2's real c_z=128 and bf16, not this module's toy
    config, because the wall is a property of the published architecture.

    Returns one dict per length with keys ``n_res``, ``pair_mb``,
    ``logits_mb`` and ``ratio``.
    """
    rows = []
    for n in lengths:
        pair = n * n * c_z * bytes_per_element
        logits = n * n * n * n_heads * bytes_per_element
        rows.append(
            {
                "n_res": n,
                "pair_mb": pair / 1e6,
                "logits_mb": logits / 1e6,
                "ratio": logits / pair,
            }
        )
    return rows


def print_memory_table() -> None:
    """Print the table, and the sentence it exists to support."""
    print("\n" + "=" * 78)
    print("  WHERE THE MEMORY GOES  (AlphaFold2 c_z=128, 4 heads, bf16)")
    print("=" * 78)
    print(f"  {'N_res':>6}  {'pair rep':>14}  {'triangle logits':>18}  {'ratio':>8}")
    print("  " + "-" * 74)
    for r in memory_table():
        print(
            f"  {r['n_res']:>6}  {r['pair_mb']:>11.1f} MB  "
            f"{r['logits_mb']:>15.1f} MB  {r['ratio']:>7.1f}x"
        )
    print(
        "\n  The pair representation is quadratic in length. The triangle\n"
        "  attention logits are CUBIC, and there are two of those per block.\n"
        "\n  ZeRO shards parameters, gradients and optimizer state. This is an\n"
        "  ACTIVATION -- every rank materialises it in full, every step. That\n"
        "  is why DeepSpeed ships DS4Sci_EvoformerAttention, a kernel that\n"
        "  tiles the logits instead of a strategy that shards the model.\n"
    )


# =============================================================================
# Demonstration
# =============================================================================


def _demo_properties(seed: int = 0) -> None:
    """
    Show the two symmetries the test suite asserts, and the deliberate bug.

    This is a demonstration, not the test. `tests/test_evoformer.py` is the
    thing that fails CI.
    """
    from synthetic_msa import make_batch

    torch.manual_seed(seed)
    cfg = EvoformerConfig(n_blocks=1)
    model = EvoformerStack(cfg).eval()

    batch = make_batch(batch_size=1, n_res=16, n_seq=24, seed=seed)
    msa = batch["msa"]

    with torch.no_grad():
        base = model(msa)

        # 1. Residue permutation -> the contact map must permute identically.
        perm = torch.randperm(msa.shape[-1])
        permuted = model(msa[:, :, perm])
        expected = base[:, perm][:, :, perm]
        equivariance_err = (permuted - expected).abs().max().item()

        # 2. Sequence permutation -> the contact map must not move at all.
        #
        # Permute rows 1.. only. Row 0 is the QUERY -- the sequence actually
        # being folded -- and the pair representation is initialised from it,
        # so moving it changes the problem rather than the presentation. The
        # first version of this demo permuted all rows and measured an
        # invariance error of 1.51, which looked like a broken symmetry and
        # was in fact a broken claim.
        sperm = torch.cat(
            [torch.zeros(1, dtype=torch.long), 1 + torch.randperm(msa.shape[1] - 1)]
        )
        reordered = model(msa[:, sperm, :])
        invariance_err = (reordered - base).abs().max().item()

        # 3. And the map must actually depend on the residues, or (1) is
        #    satisfied vacuously by a constant function.
        spread = base.std().item()

    print("\n" + "=" * 78)
    print("  TWO SYMMETRIES, ONE VACUITY CHECK")
    print("=" * 78)
    print(f"  residue permutation  -> equivariance error {equivariance_err:.2e}")
    print(f"  sequence permutation -> invariance   error {invariance_err:.2e}")
    print(f"  contact logit spread -> std          {spread:.4f}  (must be > 0)")
    print(
        "\n  The first says the model reads residue CONTENT, not residue ORDER.\n"
        "  The second says the order homologs arrive in carries no information.\n"
        "  The third says the model is not constant, without which the first\n"
        "  two are free.\n"
    )


def _demo_triangle_closure(seed: int = 0) -> None:
    """
    Show that the correct multiplicative update propagates a constraint through
    a third residue, and that the deliberately broken one cannot.
    """
    torch.manual_seed(seed)
    n, c_z, c_hidden = 8, 16, 8

    good = TriangleMultiplicativeUpdate(c_z, c_hidden, "outgoing").eval()
    bad = TriangleMultiplicativeUpdate(
        c_z, c_hidden, "outgoing", broken_no_third_index=True
    ).eval()
    bad.load_state_dict(good.state_dict())   # same weights, different maths

    z = torch.zeros(1, n, n, c_z)
    i, j, k = 0, 1, 5

    with torch.no_grad():
        before_good = good(z)[0, i, j].clone()
        before_bad = bad(z)[0, i, j].clone()

        # Plant evidence on the two edges that SHARE residue k, and on nothing
        # else. Edge (i, j) itself is untouched.
        #
        # The evidence must VARY ACROSS CHANNELS. `z_ev[0, i, k] = 1.0` fills
        # the channel vector with a constant, and `norm_in` is a LayerNorm over
        # channels -- which maps any constant vector to zeros. The first
        # version of this demo planted constants and measured a change of
        # exactly 0.0 for the correct implementation as well as the broken one,
        # i.e. it "proved" the two were identical by feeding both an input
        # neither could see.
        z_ev = z.clone()
        z_ev[0, i, k] = torch.randn(c_z)
        z_ev[0, j, k] = torch.randn(c_z)

        delta_good = (good(z_ev)[0, i, j] - before_good).abs().max().item()
        delta_bad = (bad(z_ev)[0, i, j] - before_bad).abs().max().item()

    print("\n" + "=" * 78)
    print("  TRIANGLE CLOSURE:  evidence on (i,k) and (j,k) must reach (i,j)")
    print("=" * 78)
    print(f"  correct  sum_k a_ik * b_jk    ->  change at (i,j) = {delta_good:.3e}")
    print(f"  broken   a_ij * b_ij          ->  change at (i,j) = {delta_bad:.3e}")
    print(
        "\n  The broken version runs, trains, and returns the right shape. It\n"
        "  simply has no triangle in it. A shape assertion cannot tell these\n"
        "  two apart, which is why the test suite keeps the broken one.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The AlphaFold2 Evoformer on CPU: the blocks, the "
                    "symmetries, and the memory wall."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="print the memory table only",
    )
    args, _ = parser.parse_known_args()

    print("=" * 78)
    print("  THE ALPHAFOLD2 EVOFORMER, FROM SCRATCH")
    print("=" * 78)
    print(
        "  No GPU, no download, no transformers. Everything below is plain\n"
        "  PyTorch on plain tensors."
    )

    print_memory_table()
    if not args.skip_demo:
        _demo_triangle_closure(args.seed)
        _demo_properties(args.seed)

    print("=" * 78)
    print("  Next:  uv run deepspeed --num_gpus=1 train_evoformer_ds.py")
    print("         and --ds-evoformer-attn to see the kernel do its work.")
    print("=" * 78)


if __name__ == "__main__":
    main()

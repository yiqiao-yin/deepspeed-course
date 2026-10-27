#!/usr/bin/env python3
"""
Pair representation to 3D: a guaranteed symmetry versus a learned one.

    uv run structure.py          # the comparison, on CPU, ~1 minute

Read `02_evoformer` and `03_pairformer` first. They produce a pair
representation -- the model's hypothesis about which residues touch. This
folder turns that into coordinates, and the interesting part is a design
decision AlphaFold3 made that AlphaFold2 did not.

The symmetry that matters
-------------------------
A protein structure has no preferred position or orientation. Rotate the whole
molecule, translate it anywhere, and it is the same structure. So a model that
predicts coordinates must satisfy, for every rotation R and translation t:

    predict(R @ x + t)  ==  R @ predict(x) + t

This is **SE(3) equivariance**, and there are exactly two ways to get it.

**Build it in.** AlphaFold2's structure module uses *Invariant Point
Attention*. Every residue carries a local coordinate frame, and IPA only ever
compares points after mapping them into those frames. Distances between points
in a shared frame do not change when the whole molecule moves, so the attention
logits are invariant *by construction* -- as a matter of arithmetic, before
any training.

**Learn it.** AlphaFold3 **removed IPA.** Its diffusion head uses ordinary
attention blocks over atom coordinates, which are not invariant to anything,
and recovers the symmetry by randomly rotating and translating every training
example. The model learns to ignore orientation because it never sees a
consistent one.

Both work. They are not equally *verifiable*, and that is the lesson.
Measured by `uv run structure.py`, error relative to the size of the molecule,
mean of three seeds, float64:

    head          equivariance error        guaranteed?
    ipa                    3.6e-16          yes, by construction
    diffusion              4.5e-02          no, must be learned
    mlp                    2.7e-01          no, and not even trying

IPA's figure is float noise. The arithmetic cannot produce anything else; it
never saw a rotation during training and does not need to.

And augmentation really does help, but only so far. Training the AF3-style
head for 200 steps with and without random SE(3) augmentation, three seeds:

    arm                           equivariance error    final FAPE
    diffusion, no augmentation             7.1e-02           0.143
    diffusion + augmentation               3.5e-02           0.603

Augmentation halves the error and never approaches zero -- it is still
fourteen orders of magnitude above IPA. It also *costs* fit at a fixed budget,
because randomly reorienting every example makes the task harder.

A guarantee holds on inputs you never tested. A learned symmetry holds on
inputs that resemble training, and degrades quietly elsewhere -- which is
exactly the failure mode a held-out split drawn from the same distribution
cannot catch.

Why AlphaFold3 made that trade anyway
--------------------------------------
Generality. IPA needs well-defined residue frames built from N, CA and C
atoms. AlphaFold3 predicts ligands, nucleic acids, ions and modified residues,
many of which have no backbone and therefore no frame to build. Dropping the
architectural guarantee is what let the model treat everything as atoms.

So this is not AF2 being careful and AF3 being sloppy. It is a real trade:
**a symmetry you can prove, against a model that can represent more things.**

What the loss has to do
-----------------------
The same symmetry applies to the objective. Frame-Aligned Point Error compares
predicted and true coordinates *after* mapping both into each residue's local
frame, so a global rotation of the whole prediction costs nothing. A plain RMSD
on raw coordinates would punish a perfectly correct structure for being
rotated, and the model would waste capacity learning an arbitrary orientation.

References
----------
Jumper et al. 2021, Nature 596, Supplementary Algorithm 22 (IPA), Algorithm 23
(backbone update) and Algorithm 28 (FAPE).
Abramson et al. 2024, Nature 630, for the diffusion module that replaced it.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StructureConfig:
    """Small on purpose -- this module must run on a laptop CPU."""

    c_s: int = 32            # single representation channels
    c_z: int = 32            # pair representation channels
    c_hidden: int = 16
    n_heads: int = 4
    n_query_points: int = 4  # IPA: 3D points per head for queries/keys
    n_value_points: int = 4  # IPA: 3D points per head for values
    n_blocks: int = 2


# =============================================================================
# Rigid frames
# =============================================================================


def frames_from_backbone(
    n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a local coordinate frame per residue from its N, CA and C atoms.

    AlphaFold2 Supplementary Algorithm 21, Gram-Schmidt. Returns
    ``(R, t)`` with ``R`` of shape ``[..., 3, 3]`` and ``t`` of shape
    ``[..., 3]``; the translation is the CA position.

    The frames are what make IPA invariant: a point expressed in a residue's
    own frame does not move when the whole protein moves, because the frame
    moves with it.
    """
    v1 = c_xyz - ca_xyz
    v2 = n_xyz - ca_xyz
    e1 = F.normalize(v1, dim=-1, eps=1e-8)
    u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)          # columns are the basis
    return R, ca_xyz


def apply_frame(R: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Local point -> global: ``R @ p + t``. ``p`` is ``[..., P, 3]``."""
    return torch.einsum("...ij,...pj->...pi", R, p) + t.unsqueeze(-2)


def invert_frame(R: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Global point -> local: ``R^T @ (p - t)``."""
    return torch.einsum("...ji,...pj->...pi", R, p - t.unsqueeze(-2))


def random_se3(batch: int, device=None, dtype=torch.float32, seed: int | None = None):
    """A random rotation and translation, for testing and for augmentation."""
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(batch, 3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Fix the sign so this is a rotation (det +1), not a reflection.
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[torch.det(q) < 0, :, 0] *= -1
    t = torch.randn(batch, 3, generator=g, dtype=torch.float64) * 10.0
    return q.to(device=device, dtype=dtype), t.to(device=device, dtype=dtype)


# =============================================================================
# AlphaFold2 -- Invariant Point Attention
# =============================================================================


class InvariantPointAttention(nn.Module):
    """
    AlphaFold2 Supplementary Algorithm 22.

    Ordinary attention has a query-key dot product. IPA adds a second term
    built from **3D points**: each head emits query points and key points in
    each residue's local frame, maps them to global coordinates, and scores
    pairs by the squared distance between them.

        logit_ij = q_i . k_j / sqrt(c)
                 + b_ij                                   (pair bias)
                 - gamma * sum_p || T_i(qp_i^p) - T_j(kp_j^p) ||^2 / 2

    **The third term is why this is invariant.** Apply a global rotation and
    translation to every frame and each point moves with its frame, so the
    distance between any two of them is unchanged. The logits -- and therefore
    the output -- do not move at all. No training required, and it holds for
    inputs the model has never seen.

    The output is an update to the single representation `s`, which is
    invariant. Equivariance of the *structure* comes from the backbone update
    below, which interprets that invariant output as a change expressed in
    each residue's own local frame.
    """

    def __init__(self, cfg: StructureConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, c = cfg.n_heads, cfg.c_hidden
        self.linear_q = nn.Linear(cfg.c_s, h * c, bias=False)
        self.linear_k = nn.Linear(cfg.c_s, h * c, bias=False)
        self.linear_v = nn.Linear(cfg.c_s, h * c, bias=False)
        self.linear_qp = nn.Linear(cfg.c_s, h * cfg.n_query_points * 3, bias=False)
        self.linear_kp = nn.Linear(cfg.c_s, h * cfg.n_query_points * 3, bias=False)
        self.linear_vp = nn.Linear(cfg.c_s, h * cfg.n_value_points * 3, bias=False)
        self.linear_b = nn.Linear(cfg.c_z, h, bias=False)
        # Per-head learned weight on the point term (softplus-ed to stay > 0).
        self.head_weight = nn.Parameter(torch.zeros(h))
        self.linear_out = nn.Linear(
            h * (c + cfg.c_z + cfg.n_value_points * 4), cfg.c_s
        )

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
        R: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = s.shape
        cfg = self.cfg
        h, c = cfg.n_heads, cfg.c_hidden
        qp, vp = cfg.n_query_points, cfg.n_value_points

        q = self.linear_q(s).view(b, n, h, c)
        k = self.linear_k(s).view(b, n, h, c)
        v = self.linear_v(s).view(b, n, h, c)
        bias = self.linear_b(z).permute(0, 3, 1, 2)                 # [b,h,i,j]

        # Points live in each residue's LOCAL frame, then move to global.
        qpts = self.linear_qp(s).view(b, n, h * qp, 3)
        kpts = self.linear_kp(s).view(b, n, h * qp, 3)
        vpts = self.linear_vp(s).view(b, n, h * vp, 3)
        qg = apply_frame(R, t, qpts).view(b, n, h, qp, 3)
        kg = apply_frame(R, t, kpts).view(b, n, h, qp, 3)
        vg = apply_frame(R, t, vpts).view(b, n, h, vp, 3)

        # Squared distance between query points of i and key points of j.
        d2 = ((qg.unsqueeze(2) - kg.unsqueeze(1)) ** 2).sum(-1).sum(-1)  # [b,i,j,h]
        d2 = d2.permute(0, 3, 1, 2)                                      # [b,h,i,j]

        w = F.softplus(self.head_weight).view(1, h, 1, 1)
        logits = (
            torch.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(c)
            + bias
            - 0.5 * w * d2 * math.sqrt(2.0 / (9.0 * max(qp, 1)))
        )
        attn = torch.softmax(logits, dim=-1)

        o = torch.einsum("bhij,bjhc->bihc", attn, v).reshape(b, n, h * c)
        o_pair = torch.einsum("bhij,bijc->bihc", attn, z).reshape(b, n, h * cfg.c_z)

        # Attend over value POINTS in global space, then map back into the
        # local frame -- the step that turns a global quantity into an
        # invariant one.
        o_pt_g = torch.einsum("bhij,bjhpc->bihpc", attn, vg)
        o_pt = invert_frame(R, t, o_pt_g.reshape(b, n, h * vp, 3))
        norms = torch.linalg.vector_norm(o_pt, dim=-1, keepdim=True)
        o_pt = torch.cat([o_pt, norms], dim=-1).reshape(b, n, h * vp * 4)

        return self.linear_out(torch.cat([o, o_pair, o_pt], dim=-1))


# =============================================================================
# AlphaFold3 -- ordinary attention on coordinates
# =============================================================================


class CoordinateAttention(nn.Module):
    """
    An AlphaFold3-style head: plain attention, coordinates as features.

    AF3 dropped IPA and uses standard attention blocks in its diffusion
    module, which is **not** SE(3)-invariant -- all the relevant symmetries
    have to be learned, which AF3 arranges by applying random rotations and
    translations to every training example.

    This module is the honest minimal version of that: concatenate the raw
    coordinates onto the single representation and run ordinary attention. It
    trains, it works, and `uv run structure.py` will show you that its
    equivariance error is about six orders of magnitude worse than IPA's --
    and that augmentation shrinks the error without ever reaching zero.
    """

    def __init__(self, cfg: StructureConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, c = cfg.n_heads, cfg.c_hidden
        self.coord_in = nn.Linear(3, cfg.c_s)
        self.norm = nn.LayerNorm(cfg.c_s)
        self.linear_q = nn.Linear(cfg.c_s, h * c, bias=False)
        self.linear_k = nn.Linear(cfg.c_s, h * c, bias=False)
        self.linear_v = nn.Linear(cfg.c_s, h * c, bias=False)
        self.linear_b = nn.Linear(cfg.c_z, h, bias=False)
        self.linear_out = nn.Linear(h * c, cfg.c_s)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
        R: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = s.shape
        h, c = self.cfg.n_heads, self.cfg.c_hidden
        # The translation IS the CA coordinate -- fed in raw, which is exactly
        # where the invariance is lost.
        x = self.norm(s + self.coord_in(t))
        q = self.linear_q(x).view(b, n, h, c)
        k = self.linear_k(x).view(b, n, h, c)
        v = self.linear_v(x).view(b, n, h, c)
        bias = self.linear_b(z).permute(0, 3, 1, 2)
        logits = torch.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(c) + bias
        attn = torch.softmax(logits, dim=-1)
        o = torch.einsum("bhij,bjhc->bihc", attn, v).reshape(b, n, h * c)
        return self.linear_out(o)


class MLPHead(nn.Module):
    """
    The strawman, kept permanently as a counterexample.

    Flattens the coordinates and pushes them through an MLP, with no notion of
    frames or distances at all. Included so the test suite can show what
    "badly broken" looks like next to "learned approximately" and "guaranteed
    exactly" -- three points make the middle one legible in a way two cannot.
    """

    def __init__(self, cfg: StructureConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.c_s + 3, cfg.c_s * 2), nn.ReLU(),
            nn.Linear(cfg.c_s * 2, cfg.c_s),
        )

    def forward(self, s, z, R, t):
        return self.net(torch.cat([s, t], dim=-1))


# =============================================================================
# Backbone update and the structure module
# =============================================================================


class BackboneUpdate(nn.Module):
    """
    AlphaFold2 Supplementary Algorithm 23.

    Predicts a rotation (as three unnormalised quaternion components) and a
    translation **in each residue's local frame**, then composes them onto the
    existing frame.

    Local is the whole point. An invariant input produces a local update, and
    composing a local update onto an equivariant frame yields an equivariant
    frame. That composition is where invariance becomes equivariance.
    """

    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.linear = nn.Linear(c_s, 6)

    def forward(self, s, R, t):
        upd = self.linear(s)
        bc, cc, dd = upd[..., 0], upd[..., 1], upd[..., 2]
        trans_local = upd[..., 3:6]

        # Quaternion (1, b, c, d), normalised, to a rotation matrix.
        norm = torch.sqrt(1.0 + bc ** 2 + cc ** 2 + dd ** 2)
        a, bq, cq, dq = 1.0 / norm, bc / norm, cc / norm, dd / norm
        R_upd = torch.stack([
            a * a + bq * bq - cq * cq - dq * dq, 2 * (bq * cq - a * dq), 2 * (bq * dq + a * cq),
            2 * (bq * cq + a * dq), a * a - bq * bq + cq * cq - dq * dq, 2 * (cq * dq - a * bq),
            2 * (bq * dq - a * cq), 2 * (cq * dq + a * bq), a * a - bq * bq - cq * cq + dq * dq,
        ], dim=-1).reshape(*s.shape[:-1], 3, 3)

        R_new = R @ R_upd
        t_new = t + torch.einsum("...ij,...j->...i", R, trans_local)
        return R_new, t_new


HEADS = {"ipa": InvariantPointAttention,
         "diffusion": CoordinateAttention,
         "mlp": MLPHead}


class StructureModule(nn.Module):
    """
    Pair representation plus initial frames -> refined coordinates.

    All three heads share this wrapper and this interface, so switching
    `--head` is a controlled experiment rather than three different programs.
    """

    def __init__(self, cfg: StructureConfig, head: str = "ipa") -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {sorted(HEADS)}, got {head!r}")
        self.cfg, self.head_name = cfg, head
        self.norm_s = nn.LayerNorm(cfg.c_s)
        self.attn = nn.ModuleList([HEADS[head](cfg) for _ in range(cfg.n_blocks)])
        self.transition = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(cfg.c_s), nn.Linear(cfg.c_s, cfg.c_s),
                          nn.ReLU(), nn.Linear(cfg.c_s, cfg.c_s))
            for _ in range(cfg.n_blocks)])
        self.bb_update = nn.ModuleList(
            [BackboneUpdate(cfg.c_s) for _ in range(cfg.n_blocks)])

    def forward(self, s, z, R, t):
        """Returns ``(R, t)`` -- refined frames. ``t`` is the CA position."""
        s = self.norm_s(s)
        for attn, trans, upd in zip(self.attn, self.transition, self.bb_update):
            s = s + attn(s, z, R, t)
            s = s + trans(s)
            R, t = upd(s, R, t)
        return R, t


# =============================================================================
# FAPE
# =============================================================================


def fape_loss(
    R_pred, t_pred, R_true, t_true, clamp: float = 10.0, eps: float = 1e-4
) -> torch.Tensor:
    """
    Frame-Aligned Point Error. AlphaFold2 Supplementary Algorithm 28.

    For every (frame i, point j) pair, express the point in frame i -- for both
    prediction and truth -- and take the distance between the two. Because both
    sides are mapped into their own frames, a global rotation or translation of
    the whole prediction cancels, so FAPE is **invariant** and the model is
    never penalised for an arbitrary overall orientation.

    A plain RMSD on raw coordinates would punish a perfectly correct structure
    for being rotated, and the model would burn capacity learning an
    orientation that carries no information.
    """
    x_pred = invert_frame(R_pred, t_pred, t_pred.unsqueeze(1).expand(
        -1, t_pred.shape[1], -1, -1).reshape(t_pred.shape[0], t_pred.shape[1], -1, 3))
    x_true = invert_frame(R_true, t_true, t_true.unsqueeze(1).expand(
        -1, t_true.shape[1], -1, -1).reshape(t_true.shape[0], t_true.shape[1], -1, 3))
    d = torch.sqrt(((x_pred - x_true) ** 2).sum(-1) + eps)
    return d.clamp(max=clamp).mean()


# =============================================================================
# The measurement
# =============================================================================


def equivariance_error(
    head: str, seed: int = 0, n_res: int = 16, model: nn.Module | None = None
) -> float:
    """
    Transform the input frames by a random SE(3); how far does the output move
    from where it should be?

    Zero means the symmetry holds exactly. This is the number the whole folder
    is about.

    Pass `model` to measure a **trained** module. Omit it and a fresh one is
    built, which is the right default for `uv run structure.py` -- IPA's
    guarantee is a property of the arithmetic and needs no training to
    demonstrate.

    That distinction is not academic. `train_structure_ds.py` originally
    called this without a model and printed the result as though it described
    the network it had just trained. It did not, so `--augment` reported the
    *identical* error with and without augmentation -- the one comparison the
    flag exists to make, silently measuring an untrained model both times.
    """
    cfg = StructureConfig(n_blocks=2)
    if model is None:
        torch.manual_seed(seed)
        model = StructureModule(cfg, head)
    model = model.double().eval()

    b = 1
    s = torch.randn(b, n_res, cfg.c_s, dtype=torch.float64)
    z = torch.randn(b, n_res, n_res, cfg.c_z, dtype=torch.float64)
    # A plausible extended backbone: CA atoms ~3.8 A apart.
    t0 = torch.cumsum(torch.randn(b, n_res, 3, dtype=torch.float64) * 0.5
                      + torch.tensor([3.8, 0.0, 0.0], dtype=torch.float64), dim=1)
    R0 = torch.eye(3, dtype=torch.float64).expand(b, n_res, 3, 3).contiguous()

    with torch.no_grad():
        _, t_pred = model(s, z, R0, t0)

        Rg, tg = random_se3(b, dtype=torch.float64, seed=seed + 1)
        R0_g = Rg.unsqueeze(1) @ R0
        t0_g = torch.einsum("bij,bnj->bni", Rg, t0) + tg.unsqueeze(1)
        _, t_pred_g = model(s, z, R0_g, t0_g)

        # Where the output SHOULD be if the model is equivariant.
        expect = torch.einsum("bij,bnj->bni", Rg, t_pred) + tg.unsqueeze(1)
        # Scale-free: compare against the size of the molecule.
        scale = (t_pred - t_pred.mean(1, keepdim=True)).norm(dim=-1).mean()
        return ((t_pred_g - expect).norm(dim=-1).mean() / scale).item()


def main() -> None:
    p = argparse.ArgumentParser(
        description="SE(3) equivariance: guaranteed, learned, or absent."
    )
    p.add_argument("--n-res", type=int, default=16)
    p.add_argument("--seeds", type=int, default=3)
    args, _ = p.parse_known_args()

    print("=" * 78)
    print("  SE(3) EQUIVARIANCE: A GUARANTEE VS A LEARNED SYMMETRY")
    print("=" * 78)
    print(
        "  Rotate and translate the input frames. A correct structure module\n"
        "  must move its prediction by exactly the same transform.\n"
        "  Error is relative to the size of the molecule, so 1.0 means the\n"
        "  prediction moved as far as the protein is wide.\n"
    )
    print(f"  {'head':<12}{'equivariance error':>22}   guaranteed?")
    print("  " + "-" * 60)
    notes = {
        "ipa": "YES -- by construction",
        "diffusion": "no  -- must be learned",
        "mlp": "no  -- and not trying",
    }
    for head in ("ipa", "diffusion", "mlp"):
        errs = [equivariance_error(head, seed=s, n_res=args.n_res)
                for s in range(args.seeds)]
        mean = sum(errs) / len(errs)
        print(f"  {head:<12}{mean:>22.3e}   {notes[head]}")

    print(
        "\n  IPA's error is float noise: the arithmetic cannot produce anything\n"
        "  else. It never saw a rotation in training and does not need to.\n"
        "\n  The AF3-style head is wrong by a fraction of the molecule's own\n"
        "  size. Random rotation augmentation during training shrinks that --\n"
        "  AlphaFold3 does exactly this -- but it is a statement about the\n"
        "  training distribution, not about the function. A guarantee holds on\n"
        "  inputs you never tested; a learned symmetry holds on inputs that\n"
        "  resemble the ones you did.\n"
    )
    print("  So why did AlphaFold3 give it up? Generality. IPA needs residue")
    print("  frames built from N, CA and C. Ligands, ions and nucleic acids")
    print("  have no backbone and no frame -- dropping the guarantee is what")
    print("  let AF3 treat everything as atoms.\n")
    print("=" * 78)


if __name__ == "__main__":
    main()

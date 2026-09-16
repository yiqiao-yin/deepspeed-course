#!/usr/bin/env python3
"""
Mixture of Experts, from scratch, and the load balancing that keeps it alive.

    uv run moe.py            # the whole comparison, on CPU, in about a minute

One MoE layer behind one interface, with three load-balancing strategies —
`none`, `aux` and `bias` — so that swapping them is a controlled experiment
rather than three unrelated models. Plain PyTorch on plain tensors: no GPU, no
download, no transformers.

What MoE is for
---------------
A dense FFN spends every parameter on every token. An MoE layer holds `N`
experts and fires only `k` of them, so the parameter count and the FLOP count
stop being the same number:

    dense   params = FLOPs-worth of weights, all of them, every token
    MoE     params = N experts;  FLOPs = k experts

GLM-5.3 activates 8 of 256 experts per token. That is ~743 B parameters of
capacity at the compute cost of ~30 B. **This is the only technique in the
course that makes a model bigger and cheaper at the same time**, and the
router that decides it is 0.12 B of those 743 B — 0.016% of the model.

What load balancing actually buys, which is not what it sounds like
------------------------------------------------------------------
The usual telling is "without load balancing the router collapses, so balance
it." That is half true and the wrong half is the interesting one. Measured
here, on CPU, 300 steps, numbers you can reproduce with `uv run moe.py`:

    experts  groups  balance   eval loss   max/min   dead   purity
         16       4  none          0.144     264:1      0    0.975
         16       4  bias          0.246       1.5      0    0.793
         64       4  none          0.310       inf     12    0.932
         64       4  bias          0.634       3.0      0    0.662

Two things fall out, and the second is the one worth carrying:

**1. Collapse is real, but it needs surplus.** At 64 experts for a 4-group
task, 12 experts receive *nothing* -- paid for, never trained, dead weight.
At 16 experts the same router leaves nothing dead but hands one expert 264x
the traffic of another. Collapse is what happens when you buy far more experts
than the task has structure to fill.

**2. Balancing is a TAX, not an improvement -- at world size 1.** Look down the
loss column: in every configuration measured here, balancing makes the model
*worse*. 0.144 to 0.246. 0.310 to 0.634. (Read the scope carefully; a 2-GPU
measurement disagrees, and the section below says so.) Specialisation drops with it -- purity 0.975 to 0.793 -- because
forcing 16 experts to share a 4-group task means splitting each group across
four experts that each learn a blurrier version of it.

That is not a bug in this implementation. It is the trade the DeepSeek-V3
authors name explicitly, and it is why they went looking for a cheaper
mechanism in the first place:

    "However, too large an auxiliary loss will impair the model performance."
                                                        -- arXiv:2412.19437 §2.1.2

An open question, and a measurement that disagrees
--------------------------------------------------
**Everything in the table above was measured single-process, at world size 1.**
That scope matters, because a 2x RTX 3090 run of `train_moe_ds.py` under
DeepSpeed reported the opposite ordering:

    world size 2, NCCL, 500 steps, one run each
        --balance bias    eval 0.024
        --balance none    eval 0.795      <- 33x WORSE, and RISING during
                                             training (0.515 -> 0.827)

A rising loss is divergence, not poor specialisation, so this is a different
phenomenon from anything above rather than a louder version of it.

What has been checked, and what has not:

  * At world size 1 the table holds across **six seeds**, with no overlap
    between the two groups (none 0.138-0.206, bias 0.246-0.315). So it is not
    seed luck.
  * Reproducing it on **two gloo ranks on CPU**, with gradients all-reduced
    exactly as data parallelism does, did **not** reverse the ordering:
    none 0.0044 against bias 0.0170. So plain data parallelism alone does not
    explain the 2-GPU result.
  * The 2-GPU observation is **one run per arm**, on hardware not available
    here.

So the honest position is that the ordering is established at world size 1 and
**unresolved above it**. Do not read the table as a claim about multi-rank
training. If the reversal holds under repetition it is a stronger version of
this topic's thesis -- balancing would not be a tax you pay for schedulability
but a requirement for convergence once experts span ranks -- and this file will
be rewritten around it. It is not yet that.

So why balance at all? **Because the unbalanced router is a better model and a
much worse program**, and that only becomes visible once the experts live on
different GPUs. Under expert parallelism each rank owns a slice of the experts,
and every rank waits at an all-to-all for the slowest one. A 264:1 token
imbalance is a rank doing 264x the work while its peers idle at a barrier --
the step time is set by the busiest expert, not the average one. You accept a
measurable loss penalty to avoid a far larger wall-clock penalty.

**This is the entire reason MoE belongs in a DeepSpeed course rather than an
architecture course.** The balancing mechanism is not there to make the model
better. It is there to make the model *schedulable*. See
`train_moe_ds.py --expert-parallel`.

Three strategies, which is the comparison this file exists to run:

    none    top-k and nothing else. Kept BECAUSE it misbehaves: it is the
            counterexample that makes the other two mean something.
    aux     an auxiliary load-balancing loss (GShard / Switch Transformer).
            A second term in the objective, fighting the first.
    bias    DeepSeek-V3's auxiliary-loss-FREE bias. A per-expert bias nudged up
            when an expert is starved and down when it is swamped, touching the
            objective not at all.

The two details this implementation gets right on purpose
---------------------------------------------------------
Both are places where a reasonable-looking implementation silently diverges
from the paper, and in both cases the model trains fine either way — which is
exactly what makes them worth stating.

**1. The bias is used for SELECTION ONLY.** It decides *which* experts fire and
never touches the weight their output is multiplied by. The paper, immediately
after Eq. 16:

    "Note that the bias term is only used for routing. The gating value, which
     will be multiplied with the FFN output, is still derived from the original
     affinity score s_{i,t}."

Fold the bias into the gate and load-balancing pressure starts perturbing the
model's actual output — reintroducing through the back door the very coupling
the aux-loss-free design exists to remove.

**2. The affinity is a SIGMOID, normalised among the selected experts.** Not a
softmax over the top-k logits. Eq. 15 is `s = Sigmoid(u @ e_i)`, and Eq. 13
normalises among selected scores. The paper is explicit that this changed:

    "Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function
     to compute the affinity scores."

The difference is not cosmetic. Softmax over top-k forces the gates to sum to 1
*by construction*, so a token routed to two bad experts and a token routed to
two good ones produce equally confident mixtures. Sigmoid-then-normalise lets
the raw affinities carry magnitude before normalisation.

Shared experts
--------------
DeepSeekMoE's own contribution on top of vanilla MoE: `n_shared` experts that
process *every* token, alongside the routed ones. The argument is that some
computation is common to all tokens, and forcing the router to rediscover it
in every expert wastes capacity on redundancy. Eq. 12 sums both paths.

References
----------
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437 §2.1.2
  (Eqs. 12-16, auxiliary-loss-free load balancing)
- Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization*,
  arXiv:2401.06066 (fine-grained + shared experts)
- Fedus et al., *Switch Transformers*, arXiv:2101.03961 (the auxiliary loss)
- Shazeer et al., *Outrageously Large Neural Networks*, arXiv:1701.06538
  (top-k gating, and the first description of routing collapse)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


@dataclass
class MoEConfig:
    """
    Deliberately small. Every number here is chosen so the whole comparison
    runs on a CPU in about a minute -- the shapes are a scale model of GLM-5.3,
    not a reproduction of it.
    """
    d_model: int = 128
    n_routed: int = 16          # GLM-5.3: 256
    n_shared: int = 1           # DeepSeekMoE's shared path
    top_k: int = 2              # GLM-5.3: 8
    expert_hidden: int = 256
    # DeepSeek-V3 calls this the "bias update speed" (gamma). Too small and the
    # router collapses before balance arrives; too large and it oscillates.
    bias_update_speed: float = 1e-2
    # Weight on the auxiliary loss for balance="aux". Switch Transformers uses
    # 1e-2; the paper's own warning is that too large a value hurts the model.
    aux_weight: float = 1e-2
    # 0.0 means unlimited. Above 0, each expert accepts at most
    # capacity_factor * (tokens * top_k / n_routed) tokens and the rest are
    # DROPPED -- see the note on order dependence in `route()`.
    capacity_factor: float = 0.0


class ExpertFFN(nn.Module):
    """
    One expert: a two-layer MLP.

    Note the hidden dimension is *smaller* than a dense FFN's would be. That is
    DeepSeekMoE's "fine-grained" idea -- many narrow experts rather than a few
    wide ones, so a token's top-k selection can combine more specialised pieces.
    """

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    """
    A DeepSeekMoE-style layer: shared experts + top-k routed experts.

    balance:
        "none"  no load balancing. Kept as the counterexample -- it is what
                makes the other two mean something.
        "aux"   auxiliary load-balancing loss, exposed as `self.aux_loss`.
        "bias"  DeepSeek-V3's auxiliary-loss-free per-expert bias.

    The gating path is IDENTICAL across all three. Only the balancing mechanism
    changes, so a difference in utilisation is attributable to the mechanism and
    not to a different router. (This is the same discipline as `mla.py`, where
    three attention variants sit behind one interface.)
    """

    def __init__(self, cfg: MoEConfig, balance: str = "bias") -> None:
        super().__init__()
        if balance not in ("none", "aux", "bias"):
            raise ValueError(f"balance must be none|aux|bias, got {balance!r}")
        if cfg.top_k > cfg.n_routed:
            raise ValueError(f"top_k {cfg.top_k} > n_routed {cfg.n_routed}")

        self.cfg = cfg
        self.balance = balance

        self.routed = nn.ModuleList(
            [ExpertFFN(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_routed)])
        self.shared = nn.ModuleList(
            [ExpertFFN(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_shared)])

        # The router. One centroid vector per expert -- this is the whole
        # "ranking" mechanism, and it is the smallest thing in the layer.
        self.centroids = nn.Parameter(torch.empty(cfg.n_routed, cfg.d_model))
        nn.init.normal_(self.centroids, std=cfg.d_model ** -0.5)

        # A BUFFER, not a Parameter, and the distinction is the point: this is
        # updated by an explicit rule after each step, never by the optimizer.
        # Registering it as a Parameter would let gradient descent fight the
        # balancing rule for control of the same tensor.
        self.register_buffer("bias", torch.zeros(cfg.n_routed))

        # Diagnostics, refreshed every forward. Not used by the model.
        self.register_buffer("last_counts", torch.zeros(cfg.n_routed))
        self.aux_loss: torch.Tensor = torch.zeros(())
        self.dropped_tokens: int = 0

    # -- routing ------------------------------------------------------------

    def route(self, x_flat: torch.Tensor):
        """
        Score, select, weight. Returns (topk_idx, gate, affinity).

        The three lines that matter, and why each is written this way:

            affinity = sigmoid(x @ centroids.T)     Eq. 15, NOT softmax
            select on affinity + bias               Eq. 16, bias for ROUTING only
            gate = affinity[selected], normalised   Eq. 13, from the ORIGINAL score
        """
        cfg = self.cfg

        # Eq. 15. Sigmoid, per-expert, independent -- deliberately not a softmax
        # over experts, so the scores are affinities rather than a distribution.
        affinity = torch.sigmoid(F.linear(x_flat, self.centroids))      # [N, E]

        # Eq. 16. The bias steers SELECTION. It is added here and nowhere else.
        select_on = affinity + self.bias if self.balance == "bias" else affinity
        topk_idx = torch.topk(select_on, cfg.top_k, dim=-1).indices     # [N, k]

        # Eq. 13. The gate is gathered from the UNBIASED affinity, then
        # normalised among the selected experts. Gathering from `select_on`
        # instead is the single most tempting error in this file.
        chosen = affinity.gather(-1, topk_idx)                          # [N, k]
        gate = chosen / (chosen.sum(dim=-1, keepdim=True) + EPS)

        return topk_idx, gate, affinity

    def _auxiliary_loss(self, affinity: torch.Tensor,
                        topk_idx: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer's load-balancing loss, adapted to sigmoid affinities.

            L = alpha * E * sum_i  f_i * P_i

        `f_i` is the HARD fraction of routed slots that went to expert i, and
        `P_i` is the SOFT mean routing probability. The product is minimised
        when both are uniform, and only `P_i` carries gradient -- `f_i` comes
        from a topk and has none. That asymmetry is the trick: the loss pushes
        down the probability of experts that are already winning.
        """
        n_tok, n_exp = affinity.shape
        counts = torch.bincount(topk_idx.flatten(),
                                minlength=n_exp).to(affinity.dtype)
        f = counts / counts.sum().clamp_min(1.0)                 # hard, no grad
        p = (affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(EPS)).mean(0)
        return self.cfg.aux_weight * n_exp * torch.sum(f * p)

    def _apply_capacity(self, topk_idx: torch.Tensor) -> torch.Tensor:
        """
        Enforce a per-expert capacity, dropping the overflow.

        Returns a [N, k] boolean mask of slots that were KEPT.

        **This is where MoE becomes order-dependent, and it is not a bug in
        this implementation -- it is a property of capacity-based routing.**
        Tokens are admitted in the order they appear, so whether a token
        reaches its preferred expert depends on where it sits in the batch. The
        same token, in a different batch, routes differently. `mla.py`'s sibling
        property in `04_groupwise_ranking` is permutation equivariance; here the
        equivalent property is deliberately FALSE, and the test asserts that it
        is false so nobody later "fixes" it by accident.
        """
        cfg = self.cfg
        n_tok = topk_idx.shape[0]
        cap = int(cfg.capacity_factor * n_tok * cfg.top_k / cfg.n_routed)
        cap = max(cap, 1)

        keep = torch.ones_like(topk_idx, dtype=torch.bool)
        used = torch.zeros(cfg.n_routed, dtype=torch.long, device=topk_idx.device)
        # An explicit loop, because the point being taught is the sequential
        # admission itself. A vectorised cumsum would hide it.
        for t in range(n_tok):
            for j in range(cfg.top_k):
                e = int(topk_idx[t, j])
                if used[e] >= cap:
                    keep[t, j] = False
                else:
                    used[e] += 1
        return keep

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Eq. 12:  h = u + sum(shared_i(u)) + sum(g_i * routed_i(u))

        Input and output are both [B, S, D].
        """
        cfg = self.cfg
        b, s, d = x.shape
        x_flat = x.reshape(-1, d)

        topk_idx, gate, affinity = self.route(x_flat)

        if cfg.capacity_factor > 0.0:
            keep = self._apply_capacity(topk_idx)
            self.dropped_tokens = int((~keep).sum())
            gate = gate * keep.to(gate.dtype)
        else:
            self.dropped_tokens = 0

        self.aux_loss = (self._auxiliary_loss(affinity, topk_idx)
                         if self.balance == "aux"
                         else torch.zeros((), device=x.device, dtype=x.dtype))

        with torch.no_grad():
            self.last_counts = torch.bincount(
                topk_idx.flatten(), minlength=cfg.n_routed).to(self.last_counts.dtype)

        # Shared path: every token, every shared expert. No routing involved.
        out = x_flat.clone()
        for expert in self.shared:
            out = out + expert(x_flat)

        # Routed path. Gather the tokens for each expert, run it once on that
        # sub-batch, scatter the weighted result back.
        #
        # NOTE ON COST: this loops over ALL experts, including ones no token
        # chose. That is O(n_routed), not O(active), and at GLM-5.3's 256
        # experts it is 256 iterations to do 8 experts' work. Real
        # implementations dispatch with a permutation and grouped GEMM, and
        # under expert parallelism the loop disappears entirely -- each RANK
        # owns a slice of experts and an all-to-all moves tokens to them. See
        # train_moe_ds.py --expert-parallel.
        for e in range(cfg.n_routed):
            rows, slot = (topk_idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            w = gate[rows, slot].unsqueeze(-1)
            out = out.index_add(0, rows, self.routed[e](x_flat[rows]) * w)

        return out.view(b, s, d)

    # -- the balancing rule, and the diagnostics --------------------------

    @torch.no_grad()
    def update_bias(self) -> None:
        """
        DeepSeek-V3's auxiliary-loss-free update. Call once per optimizer step.

            "we will decrease the bias term by gamma if its corresponding
             expert is overloaded, and increase it by gamma if its
             corresponding expert is underloaded"

        This reuses the counts recorded during `forward`, rather than running
        the router a second time over the same batch. Recomputing would double
        the router's cost for a number already in hand, and -- worse -- would
        compute it under the NEW bias while attributing it to the OLD step.
        """
        if self.balance != "bias":
            return

        counts = self.last_counts
        # Across ranks, the load must be measured on the WHOLE batch, which the
        # paper says in as many words: "we keep monitoring the expert load on
        # the whole batch of each training step".
        #
        # This matters more than it looks. `bias` is a BUFFER, not a Parameter,
        # so no optimizer and no ZeRO stage synchronises it. Without this
        # all-reduce every rank would nudge its own private copy from its own
        # local token counts, the routers would silently drift apart, and each
        # rank would be running a different model -- with nothing raising and
        # the loss curve looking entirely normal.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            counts = counts.clone()
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

        target = counts.sum() / self.cfg.n_routed
        # Sign of the violation, as the paper describes: a fixed step up or
        # down, not a proportional one. tanh keeps it smooth near balance.
        violation = (target - counts) / (target + EPS)
        self.bias += self.cfg.bias_update_speed * torch.tanh(violation)

    @torch.no_grad()
    def utilisation(self) -> torch.Tensor:
        """Fraction of routed slots each expert received in the last forward."""
        c = self.last_counts
        return c / c.sum().clamp_min(1.0)

    @torch.no_grad()
    def balance_metrics(self) -> dict:
        """
        Three numbers, because no single one of them is honest alone.

        entropy     normalised to [0, 1]. 1.0 is perfectly uniform. Forgiving:
                    stays high while a few experts starve.
        maxmin      busiest / quietest. Catches starvation that entropy hides,
                    but is infinite the moment ONE expert is idle.
        dead        how many experts received nothing at all. The blunt one,
                    and the one a reader should look at first.
        """
        u = self.utilisation()
        nz = u[u > 0]
        entropy = float(-(nz * nz.log()).sum() / torch.tensor(
            float(self.cfg.n_routed)).log())
        dead = int((u == 0).sum())
        maxmin = float("inf") if dead else float(u.max() / u.min().clamp_min(EPS))
        return {"entropy": entropy, "maxmin": maxmin, "dead": dead}


# ---------------------------------------------------------------------------
# Parameter accounting: the arithmetic that makes MoE worth the trouble.
# ---------------------------------------------------------------------------

def param_table(cfg: MoEConfig) -> dict:
    """
    Total vs ACTIVE parameters for one MoE layer, and the dense layer it
    replaces. Pure arithmetic -- no model is built.

    The active count is what a forward pass actually touches. The gap between
    the two columns is the entire argument for MoE.
    """
    per_expert = 2 * cfg.d_model * cfg.expert_hidden
    routed_total = cfg.n_routed * per_expert
    shared_total = cfg.n_shared * per_expert
    router = cfg.n_routed * cfg.d_model

    return {
        "router": router,
        "shared": shared_total,
        "routed_total": routed_total,
        "routed_active": cfg.top_k * per_expert,
        "total": router + shared_total + routed_total,
        "active": router + shared_total + cfg.top_k * per_expert,
        # A dense FFN of comparable width, for the comparison that matters.
        "dense_equivalent": 2 * cfg.d_model * (cfg.expert_hidden * cfg.n_routed),
    }


# ---------------------------------------------------------------------------
# A task where specialisation is possible, so utilisation means something.
# ---------------------------------------------------------------------------

def synthetic_groups(n_tokens: int, cfg: MoEConfig, n_groups: int = 4,
                     noise: float = 1.0, seed: int = 0, task_seed: int = 1234):
    """
    Tokens from `n_groups` latent groups. The group is READABLE FROM THE TOKEN,
    and each group has its own target transformation.

    Both halves of that sentence are load-bearing, and the first version of this
    function got the first half wrong.

    It drew `x = randn(...)` and `group = randint(...)` independently, so the
    group carried **zero mutual information** with the token. No router could
    recover it, every expert was equally right, and utilisation measured
    nothing but noise. The tell was per-group centroid norms of 0.35 on 128
    dimensions -- exactly sqrt(d/n), the sampling-noise floor. That is the same
    defect `01_basics/02_convnet` shipped with random labels, reproduced here
    in new code.

    So: each group owns a fixed prototype, and a token is that prototype plus
    noise. Now the router CAN read the group, and `routing_purity()` can ask
    whether it did.

    The target is `y = x + delta_g(x)`, a residual plus a group-specific delta,
    because `MoELayer.forward` returns `x + shared(x) + routed(x)` per Eq. 12.
    The first version targeted `y = delta_g(x)` alone, which made the layer's
    own residual connection actively harmful -- predicting `x` scored 2.02
    against 1.01 for predicting zeros, so the model began at twice worse than
    trivial and spent its capacity cancelling its own architecture.

    `task_seed` draws the prototypes and the maps; `seed` draws the samples. So
    train and eval describe the SAME task. Redrawing per call would make them
    unrelated problems -- a bug this course has shipped before, in a ranking
    generator.

    Returns (x, y, group_id).
    """
    g_task = torch.Generator().manual_seed(task_seed)
    protos = torch.randn(n_groups, cfg.d_model, generator=g_task)
    maps = torch.randn(n_groups, cfg.d_model, cfg.d_model, generator=g_task)
    maps = maps / cfg.d_model ** 0.5

    g = torch.Generator().manual_seed(seed)
    group = torch.randint(0, n_groups, (n_tokens,), generator=g)
    x = protos[group] + noise * torch.randn(n_tokens, cfg.d_model, generator=g)
    y = x + torch.einsum("nd,ndm->nm", x, maps[group])
    return x, y, group


@torch.no_grad()
def routing_purity(layer: "MoELayer", x: torch.Tensor,
                   group: torch.Tensor, n_groups: int) -> float:
    """
    Did the router discover the task's partition?

    Normalised mutual information between the group label and the top-1 expert,
    in [0, 1]. 0 means routing is independent of the structure -- which is what
    a collapsed router gives, and also what a perfectly balanced but random one
    gives. **Balance alone is not specialisation**, which is why this is
    reported next to the balance metrics rather than instead of them.
    """
    idx, _, _ = layer.route(x)
    top1 = idx[:, 0]
    n_exp = layer.cfg.n_routed

    joint = torch.zeros(n_groups, n_exp)
    for g_i, e_i in zip(group.tolist(), top1.tolist()):
        joint[g_i, e_i] += 1
    joint /= joint.sum().clamp_min(1.0)

    p_g = joint.sum(1, keepdim=True)
    p_e = joint.sum(0, keepdim=True)
    nz = joint > 0
    mi = float((joint[nz] * (joint[nz] / (p_g @ p_e)[nz]).log()).sum())
    h_g = float(-(p_g[p_g > 0] * p_g[p_g > 0].log()).sum())
    return mi / h_g if h_g > 0 else 0.0


BALANCE_STRATEGIES = {
    "none": "top-k only. The counterexample: best loss, worst balance.",
    "aux":  "auxiliary load-balancing loss (Switch Transformer).",
    "bias": "DeepSeek-V3 auxiliary-loss-free per-expert bias.",
}


def train_demo(balance: str, cfg: MoEConfig, steps: int = 300,
               batch: int = 256, lr: float = 3e-3, seed: int = 0,
               n_groups: int = 4) -> dict:
    """Train one MoE layer on the grouped task. Returns final metrics."""
    torch.manual_seed(seed)
    layer = MoELayer(cfg, balance=balance)
    opt = torch.optim.AdamW(layer.parameters(), lr=lr)

    x, y, group = synthetic_groups(batch * 8, cfg, n_groups=n_groups, seed=seed)

    for step in range(steps):
        lo = (step * batch) % (x.shape[0] - batch)
        xb, yb = x[lo:lo + batch], y[lo:lo + batch]
        pred = layer(xb.unsqueeze(0)).squeeze(0)
        loss = F.mse_loss(pred, yb) + layer.aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # The bias update is NOT part of backprop. It runs after the optimizer
        # step, from the counts the forward pass recorded.
        layer.update_bias()

    xe, ye, ge = synthetic_groups(1024, cfg, n_groups=n_groups, seed=seed + 99)
    with torch.no_grad():
        eval_loss = float(F.mse_loss(layer(xe.unsqueeze(0)).squeeze(0), ye))
    m = layer.balance_metrics()
    m["loss"] = eval_loss
    m["purity"] = routing_purity(layer, xe, ge, n_groups)
    return m


def _demo() -> None:
    bar = "=" * 78
    cfg = MoEConfig()

    print(bar)
    print("  Mixture of Experts — what the router costs, and what it breaks")
    print(bar)
    print(f"  d_model {cfg.d_model}   routed experts {cfg.n_routed}   "
          f"shared {cfg.n_shared}   top_k {cfg.top_k}")

    p = param_table(cfg)
    print(f"\n  {'component':<18} {'parameters':>12}")
    print(f"  {'-' * 18} {'-' * 12}")
    print(f"  {'router':<18} {p['router']:>12,}")
    print(f"  {'shared experts':<18} {p['shared']:>12,}")
    print(f"  {'routed experts':<18} {p['routed_total']:>12,}")
    print(f"  {'-' * 18} {'-' * 12}")
    print(f"  {'TOTAL':<18} {p['total']:>12,}")
    print(f"  {'ACTIVE / token':<18} {p['active']:>12,}"
          f"   ({100 * p['active'] / p['total']:.0f}% of total)")
    print(f"\n  The router is {100 * p['router'] / p['total']:.2f}% of the layer "
          f"and decides where the other {100 - 100 * p['router'] / p['total']:.2f}% goes.")

    print(f"\n{bar}")
    print("  Load balancing: same task, same gating, three strategies")
    print(bar)
    print("  Balancing is a TAX. Read the loss column first: it goes UP.")
    print("  You pay it so the experts can be SCHEDULED across GPUs.\n")

    hdr = (f"  {'experts':>7} {'strategy':<8} {'eval loss':>10} {'entropy':>8}"
           f" {'max/min':>9} {'dead':>5} {'purity':>7}")
    for n_exp in (16, 64):
        if n_exp == 16:
            print(hdr)
            print(f"  {'-' * 7} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 9} "
                  f"{'-' * 5} {'-' * 7}")
        cfg_n = MoEConfig(n_routed=n_exp)
        for name in BALANCE_STRATEGIES:
            m = train_demo(name, cfg_n)
            mm = "inf" if m["maxmin"] == float("inf") else f"{m['maxmin']:.1f}"
            print(f"  {n_exp:>7} {name:<8} {m['loss']:>10.4f} {m['entropy']:>8.3f}"
                  f" {mm:>9} {m['dead']:>5} {m['purity']:>7.3f}")
        if n_exp == 16:
            print()

    print(f"\n  dead    = experts that received NOTHING. Collapse needs surplus:")
    print(f"            16 experts for a 4-group task leaves none dead; 64 does.")
    print(f"  purity  = normalised MI between the task's groups and the chosen")
    print(f"            expert. Balance without purity is a router spreading")
    print(f"            tokens evenly and at random.")
    print(f"  max/min = busiest expert / quietest. THIS is what expert")
    print(f"            parallelism cares about: the step waits for the busiest.")
    print(bar)


if __name__ == "__main__":
    _demo()

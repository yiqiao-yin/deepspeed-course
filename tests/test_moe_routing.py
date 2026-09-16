# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""
Regression test: Mixture-of-Experts routing and load balancing.

Run:
    uv run tests/test_moe_routing.py

Why this suite exists
---------------------
Every claim MoE makes is quantitative, and every way of getting it wrong still
produces a model that trains. The reference implementation this topic was built
against tested its MoE layer like this, in full:

    print("New output shape:", output_new.shape)

A shape assertion passes on a router that sends every token to expert 0, on a
router whose gate is constant, and on a load-balancing rule that does nothing.
CONTRIBUTING.md section 7 is explicit about this: assert properties, not shapes.

So each property here is asserted with a counterexample where one exists. It is
not enough that the balanced router is balanced -- the UNBALANCED one must be
shown to be unbalanced, or the check would pass on a function returning a
constant.

The four properties, and what breaks if each is wrong
-----------------------------------------------------
1. **The bias steers SELECTION ONLY.** DeepSeek-V3, right after Eq. 16: "the
   bias term is only used for routing. The gating value ... is still derived
   from the original affinity score." Fold the bias into the gate and
   load-balancing pressure starts perturbing the model's output. The model
   still trains. This is the single most tempting error in the implementation,
   and the reference notebook makes it.

2. **Balancing actually balances**, and the unbalanced variant actually does
   not. Measured both ways, because "entropy is high" is satisfied by a router
   that ignores its input entirely.

3. **Routing is order-dependent under capacity**, and the test asserts that it
   IS -- the opposite polarity from `04_groupwise_ranking`'s permutation
   equivariance check. Token dropping admits tokens in batch order, so the same
   token routes differently in a different batch. That is a property of
   capacity-based MoE, not a defect here, and pinning it stops someone
   "fixing" it into silence later.

4. **Active parameters are genuinely fewer than total.** The entire argument
   for MoE is that these two numbers differ. An implementation that quietly ran
   every expert would be correct, slow, and pointless -- and no shape assertion
   would notice.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "03_llms" / "11_moe"))

from moe import (MoEConfig, MoELayer, param_table,  # noqa: E402
                 routing_purity, synthetic_groups, train_demo)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        if detail:
            for line in detail.splitlines():
                print(f"          {line}")


def main() -> None:
    torch.manual_seed(0)
    bar = "=" * 74
    print(bar)
    print("  test_moe_routing.py")
    print(bar)

    cfg = MoEConfig()
    x = torch.randn(2, 64, cfg.d_model)

    # ---- shapes, briefly, because they are necessary but not sufficient ----
    print("\n  -- the layer is a drop-in replacement for an FFN --")
    for balance in ("none", "aux", "bias"):
        out = MoELayer(cfg, balance=balance)(x)
        check(f"{balance}: maps (2,64,{cfg.d_model}) to itself",
              out.shape == x.shape, f"got {tuple(out.shape)}")

    # ---- THE property: the bias must not touch the gate -------------------
    print("\n  -- DeepSeek-V3: the bias is used for ROUTING ONLY --")
    layer = MoELayer(cfg, balance="bias")
    x_flat = torch.randn(512, cfg.d_model)

    with torch.no_grad():
        idx_before, gate_before, aff_before = layer.route(x_flat)
        # A lopsided bias of a REALISTIC magnitude. The affinity is a sigmoid
        # in (0, 1), so a +-5 bias swamps it and re-routes every token, leaving
        # no rows on which to compare gates -- which is how the first version of
        # this test failed. DeepSeek-V3's bias_update_speed is 1e-2 per step, so
        # +-0.05 is the right order: it moves marginal tokens and leaves the
        # rest, which is exactly the population this check needs.
        layer.bias.copy_(torch.linspace(-0.05, 0.05, cfg.n_routed))
        idx_after, gate_after, aff_after = layer.route(x_flat)

    check("the affinity itself is unchanged by the bias",
          torch.equal(aff_before, aff_after),
          "the bias must not enter the affinity computation at all")

    selection_moved = not torch.equal(idx_before, idx_after)
    check("a large bias DOES change which experts are selected",
          selection_moved,
          "if selection did not move, the bias is not being applied and the "
          "rest of this section proves nothing")

    # The gate values for a token whose selection did NOT change must be
    # identical, because they come from the unbiased affinity.
    same_sel = (idx_before == idx_after).all(dim=-1)
    if int(same_sel.sum()) == 0:
        check("some tokens kept the same selection (needed for the next check)",
              False, "bias was so large that every token re-routed; "
                     "reduce it so the gate comparison has rows to compare")
    else:
        delta = (gate_before[same_sel] - gate_after[same_sel]).abs().max()
        check(f"gates are BIT-IDENTICAL where selection held "
              f"(max delta {float(delta):.2e}, {int(same_sel.sum())} tokens)",
              float(delta) == 0.0,
              "The gate is being computed from the BIASED logits. DeepSeek-V3: "
              "'the bias term is only used for routing. The gating value, which "
              "will be multiplied with the FFN output, is still derived from "
              "the original affinity score.' Gather the gate from `affinity`, "
              "not from `affinity + bias`.")

    # The counterexample. Without it, a route() that ignored the bias entirely
    # would pass everything above.
    biased_gate = aff_after + layer.bias
    chosen_biased = biased_gate.gather(-1, idx_after)
    wrong_gate = chosen_biased / chosen_biased.sum(-1, keepdim=True).clamp_min(1e-9)
    check("the WRONG implementation is measurably different",
          float((wrong_gate - gate_after).abs().max()) > 1e-3,
          "if gathering from biased logits gave the same answer, the check "
          "above would be vacuous")

    # ---- balancing balances, and not-balancing does not -------------------
    print("\n  -- load balancing changes the load (both directions) --")
    m_none = train_demo("none", MoEConfig(n_routed=32), steps=150, n_groups=4)
    m_bias = train_demo("bias", MoEConfig(n_routed=32), steps=150, n_groups=4)

    check(f"bias balances: entropy {m_bias['entropy']:.3f} > 0.95",
          m_bias["entropy"] > 0.95,
          f"got {m_bias['entropy']:.3f}; the balancing rule is not working")
    check(f"bias leaves no expert dead ({m_bias['dead']} dead of 32)",
          m_bias["dead"] == 0)
    # The counterexample, and it must genuinely misbehave or the check above
    # is measuring nothing.
    check(f"UNBALANCED is measurably worse: entropy {m_none['entropy']:.3f} "
          f"< {m_bias['entropy']:.3f}",
          m_none["entropy"] < m_bias["entropy"],
          "if the unbalanced router balanced itself, this task cannot "
          "demonstrate load balancing and the suite proves nothing")

    # ---- balancing is a TAX, which is the topic's actual claim ------------
    print("\n  -- balancing costs loss, which is why it needed a cheaper form --")
    check(f"unbalanced loss {m_none['loss']:.3f} <= balanced {m_bias['loss']:.3f}",
          m_none["loss"] <= m_bias["loss"] + 1e-6,
          f"got {m_none['loss']:.3f} vs {m_bias['loss']:.3f}. The README, the "
          "module docstring and the docs page all state that balancing makes "
          "the model WORSE and is worth it for schedulability. If that ordering "
          "flips, those claims are wrong and must be rewritten -- do not "
          "loosen this check to make it pass.")

    # ---- the task must be learnable, or none of the above means anything --
    print("\n  -- the synthetic task carries a signal --")
    cfg_s = MoEConfig()
    xs, ys, gs = synthetic_groups(2048, cfg_s, n_groups=4, seed=0)
    mse_zero = float((ys ** 2).mean())
    # Within-group distance must be smaller than between-group distance, or
    # the group is not readable from the token and no router could find it.
    # (`cdist(...).mean() > 0` stood here first -- a check that cannot fail,
    # which CONTRIBUTING.md section 7 warns about by name.)
    within, between = [], []
    for gi in range(4):
        members = xs[gs == gi][:128]
        others = xs[gs != gi][:128]
        within.append(float(torch.cdist(members, members).mean()))
        between.append(float(torch.cdist(members, others).mean()))
    w, b = sum(within) / 4, sum(between) / 4
    check(f"the group is READABLE from the token "
          f"(within-group {w:.2f} < between-group {b:.2f})",
          w < b,
          "tokens cluster no more tightly within a group than across groups, "
          "so the group carries no information the router could use. This is "
          "the defect the first synthetic_groups shipped: x and group drawn "
          "INDEPENDENTLY, mutual information exactly zero, chance as the "
          "ceiling. 01_basics/02_convnet shipped the same bug with labels.")
    check(f"a trained router finds the structure "
          f"(purity {m_bias['purity']:.3f} > 0.3)",
          m_bias["purity"] > 0.3,
          f"got {m_bias['purity']:.3f}. If the group were INDEPENDENT of the "
          "token there would be nothing to find and purity would sit near 0 -- "
          "which is exactly the bug the first version of synthetic_groups had. "
          "Chance is the CEILING when labels are independent of inputs.")

    # The counterexample: a task with no group structure must NOT show purity.
    cfg_flat = MoEConfig()
    torch.manual_seed(0)
    flat_layer = MoELayer(cfg_flat, balance="bias")
    x_noise = torch.randn(1024, cfg_flat.d_model)
    g_random = torch.randint(0, 4, (1024,))
    purity_noise = routing_purity(flat_layer, x_noise, g_random, 4)
    check(f"structureless data gives near-zero purity ({purity_noise:.3f})",
          purity_noise < 0.2,
          "if random group labels scored high purity, the metric is broken "
          "and every purity number in this topic is meaningless")

    # ---- capacity makes routing order-dependent, ON PURPOSE ---------------
    print("\n  -- under capacity, routing depends on batch ORDER (asserted) --")
    cfg_cap = MoEConfig(capacity_factor=0.5)
    cap_layer = MoELayer(cfg_cap, balance="none")
    xb = torch.randn(64, cfg_cap.d_model)
    perm = torch.randperm(64)

    with torch.no_grad():
        idx_a, _, _ = cap_layer.route(xb)
        keep_a = cap_layer._apply_capacity(idx_a)
        idx_b, _, _ = cap_layer.route(xb[perm])
        keep_b = cap_layer._apply_capacity(idx_b)

    check("the ROUTER itself is permutation-equivariant (scores ignore order)",
          torch.equal(idx_a[perm], idx_b),
          "the affinity of a token must not depend on where it sits in the "
          "batch -- if this fails the router is reading position, which is the "
          "bug 04_groupwise_ranking's property test exists to catch")
    check("but CAPACITY makes the kept set order-dependent",
          not torch.equal(keep_a[perm], keep_b),
          "token dropping admits tokens in batch order, so a permuted batch "
          "must drop a different set. If this passes identically, capacity is "
          "not being enforced at all.")

    # ---- the whole point: active < total ----------------------------------
    print("\n  -- MoE decouples parameter count from FLOPs --")
    p = param_table(MoEConfig(n_routed=64, top_k=2))
    check(f"active {p['active']:,} < total {p['total']:,} "
          f"({100 * p['active'] / p['total']:.0f}%)",
          p["active"] < p["total"] / 4,
          "with 2 of 64 experts firing, active parameters must be a small "
          "fraction of the total. If they are equal the layer is running every "
          "expert and MoE is buying nothing.")
    p16 = param_table(MoEConfig(n_routed=16, top_k=2))
    p64 = param_table(MoEConfig(n_routed=64, top_k=2))
    # Quadrupling the experts must NOT change the expert compute per token.
    check("4x the experts leaves EXPERT compute per token identical",
          p64["routed_active"] == p16["routed_active"],
          f"routed_active {p16['routed_active']:,} -> {p64['routed_active']:,}; "
          "top_k experts fire regardless of how many exist, so this must not "
          "move. If it does, the layer is running experts it did not select.")
    # The router is the one active term that DOES grow -- one centroid per
    # expert. An earlier version of this test asserted `active` was constant
    # and failed, because the code was right and the assertion was wrong: the
    # 6,144-parameter delta is exactly (64-16) * d_model of new centroids.
    router_delta = p64["router"] - p16["router"]
    check(f"only the ROUTER grows with expert count ({p16['router']:,} -> "
          f"{p64['router']:,}), and it stays a rounding error",
          (p64["active"] - p16["active"]) == router_delta
          and p64["router"] / p64["active"] < 0.05,
          f"active {p16['active']:,} -> {p64['active']:,}, router "
          f"{p16['router']:,} -> {p64['router']:,}. The router carries one "
          "centroid per expert so it MUST scale; the claim is that it stays "
          "negligible. GLM-5.3: 0.12 B of 743 B.")
    check(f"total capacity DOES scale ({p16['total']:,} -> {p64['total']:,})",
          p64["total"] > 3 * p16["total"],
          "4x the experts should be ~4x the parameters; that is the capacity "
          "MoE is buying.")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""
Regression test: Multi-head Latent Attention.

Run:
    uv run tests/test_mla.py

Why this suite exists
---------------------
Every claim MLA makes is quantitative, and every way of getting it wrong still
produces a model that trains:

  * **A cache that secretly depends on the head count.** The entire argument for
    MLA is that `kv_lora_rank + qk_rope_head_dim` mentions no head count. An
    implementation that caches the *reconstructed* K and V has thrown that away
    while producing identical outputs and an identical loss curve.
  * **Matrix absorption that changes the answer.** Folding `W_UK` into the query
    is what makes MLA fast rather than merely small. An "optimisation" that
    quietly returns something different is the worst kind of bug: faster, and
    wrong.
  * **RoPE applied to the compressed key.** It does not commute with the
    low-rank reconstruction. Do it and absorption silently stops being valid —
    which is exactly why the architecture carries a separate decoupled key, and
    why that decoupled term is the reason the cache is 576 and not 512.
  * **Attending to the future.** A causal mask that is off by one leaks the next
    token, and the loss drops beautifully.

So each property is asserted with a counterexample where one exists: it is not
enough that MLA's cache is flat in the head count — MHA's and GQA's must be
shown to *grow*, or the check would pass on a function that returned a constant.
"""

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "03_huggingface" / "10_deepseek_from_scratch"))

from mla import (AttnConfig, VARIANTS, build, cache_table,  # noqa: E402
                 MultiHeadLatentAttention)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def main() -> None:
    torch.manual_seed(0)
    bar = "=" * 74
    print(bar)
    print("  test_mla.py")
    print(bar)

    cfg = AttnConfig()

    # ---- shapes and basic sanity -------------------------------------------
    print("\n  -- all three variants are drop-in replacements --")
    x = torch.randn(2, 12, cfg.hidden_size)
    for name in VARIANTS:
        out = build(name, cfg)(x)
        check(f"{name} maps (2,12,{cfg.hidden_size}) to itself",
              out.shape == x.shape, f"got {tuple(out.shape)}")

    # ---- THE property: the cache does not depend on head count -------------
    print("\n  -- MLA's cache is independent of the head count --")
    heads = [8, 16, 32, 64]
    mla_caches = [build("mla", AttnConfig(n_heads=h)).cache_per_token()
                  for h in heads]
    check(f"MLA cache is identical at {heads} heads: {mla_caches[0]}",
          len(set(mla_caches)) == 1,
          f"got {mla_caches} -- if this varies, the implementation is caching "
          "reconstructed K/V rather than the latent, which discards the entire "
          "point while still training fine")

    # The counterexample. Without it, a cache_per_token() that returned a
    # constant would pass everything above.
    mha_caches = [build("mha", AttnConfig(n_heads=h)).cache_per_token()
                  for h in heads]
    gqa_caches = [build("gqa", AttnConfig(n_heads=h, n_kv_heads=max(1, h // 4)))
                  .cache_per_token() for h in heads]
    check(f"MHA's cache DOES grow with heads {mha_caches}",
          len(set(mha_caches)) == len(heads) and
          mha_caches == sorted(mha_caches))
    check(f"GQA at a fixed ratio DOES grow too {gqa_caches}",
          len(set(gqa_caches)) == len(heads),
          "GQA's cache is set by n_kv_heads; held at a ratio it scales, which "
          "is the trade MLA avoids")

    check("MLA's cache equals kv_lora_rank + qk_rope_head_dim exactly",
          build("mla", cfg).cache_per_token()
          == cfg.kv_lora_rank + cfg.qk_rope_head_dim,
          "the decoupled RoPE key is a REAL cost and must be counted; "
          "omitting it understates the cache")

    # ---- matrix absorption must not change the answer ----------------------
    print("\n  -- matrix absorption is an optimisation, not a different model --")
    m = build("mla", cfg).eval()
    x = torch.randn(2, 16, cfg.hidden_size)
    with torch.no_grad():
        naive = m(x, absorbed=False)
        fast = m(x, absorbed=True)
    delta = (naive - fast).abs().max().item()
    check(f"absorbed path matches the naive one (max delta {delta:.2e})",
          delta < 1e-4,
          "folding W_UK into the query is supposed to be an identity; if it "
          "is not, MLA is fast and wrong")
    # And the check must be capable of noticing a difference at all.
    with torch.no_grad():
        other = m(torch.randn_like(x), absorbed=True)
    check("the comparison can detect a genuine difference",
          (naive - other).abs().max().item() > 1e-3,
          "if two unrelated inputs also compared equal, the tolerance above "
          "would be meaningless")

    # ---- causality ---------------------------------------------------------
    print("\n  -- no variant attends to the future --")
    for name in VARIANTS:
        mod = build(name, cfg).eval()
        a = torch.randn(1, 10, cfg.hidden_size)
        b = a.clone()
        b[:, 7:, :] = torch.randn_like(b[:, 7:, :])   # change only the tail
        with torch.no_grad():
            out_a, out_b = mod(a), mod(b)
        head_delta = (out_a[:, :7] - out_b[:, :7]).abs().max().item()
        check(f"{name}: changing tokens 7+ leaves 0-6 untouched "
              f"({head_delta:.2e})", head_delta < 1e-5,
              "an off-by-one in the causal mask leaks the next token, and the "
              "loss curve looks excellent while it happens")

    # ---- gradients ---------------------------------------------------------
    print("\n  -- every parameter is connected to the loss --")
    for name in VARIANTS:
        mod = build(name, cfg)
        mod(torch.randn(2, 8, cfg.hidden_size)).sum().backward()
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        live = sum(1 for g in grads if g.abs().sum() > 0)
        check(f"{name}: {live}/{len(list(mod.parameters()))} tensors got a "
              "nonzero gradient",
              live == len(list(mod.parameters())),
              "a disconnected projection trains to nothing and is invisible")

    # ---- cross-check against the repo's own GLM-5.3 arithmetic -------------
    # 03_huggingface/01_llm_finetuning/train_glm53_ds.py computes MLA's cache
    # from GLM-5.3's published config and reports a 57x saving. That figure was
    # derived independently, from a config file, by different code. If this
    # implementation is right, plugging the same dimensions in must reproduce
    # it -- two routes to one number.
    print("\n  -- agrees with the GLM-5.3 analysis elsewhere in this repo --")
    glm = AttnConfig(hidden_size=6144, n_heads=64, head_dim=256,
                     kv_lora_rank=512, qk_rope_head_dim=64)
    mla_c = build("mla", glm).cache_per_token()
    mha_c = build("mha", glm).cache_per_token()
    check(f"GLM-5.3 dims give an MLA cache of {mla_c} values/token",
          mla_c == 576, f"got {mla_c}; the published config is "
                        "kv_lora_rank 512 + qk_rope_head_dim 64")
    ratio = mha_c / mla_c
    check(f"and a {ratio:.0f}x saving over vanilla attention",
          56 <= ratio <= 58,
          f"got {ratio:.1f}x; train_glm53_ds.py reports 57x from the config "
          "alone, and the two derivations must agree")

    # ---- the byte-level table ----------------------------------------------
    print("\n  -- cache_table() reports bytes, not values --")
    t = cache_table(cfg, seq_len=1024, layers=8, dtype_bytes=2)
    check("mla < gqa < mha at equal sequence length",
          t["mla"] < t["gqa"] < t["mha"], f"got {t}")
    check("doubling the sequence doubles the cache",
          cache_table(cfg, 2048, 8)["mla"] == 2 * cache_table(cfg, 1024, 8)["mla"],
          "the cache is linear in sequence length for every variant -- MLA "
          "shrinks the constant, it does not change the growth")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

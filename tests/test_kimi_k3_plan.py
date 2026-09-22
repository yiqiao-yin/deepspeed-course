# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Regression test: `analyze_kimi_k3.py --plan` reads the architecture correctly.

Run:
    uv run tests/test_kimi_k3_plan.py

Why this suite exists
---------------------
`03_llms/01_llm_finetuning/analyze_kimi_k3.py` cannot be validated by running
it. Kimi K3 is 1,561 GB of weights and its remote code does not import on the
transformers this course pins, so there is no end-to-end run that would catch a
mistake. The script's *entire* claim to being worth reading is that every
number it prints was derived from `config.json` rather than guessed -- which
makes the derivation the only thing there is to test.

So these checks run the shipped functions (via `tests/_srcload.py`, no torch, no
download, no network) against a fixture config and assert the ARITHMETIC.

The counterexample is the point
-------------------------------
This suite was written *after* the script shipped a wrong number, and it carries
that exact input permanently.

K3's `full_attn_layers` is [4, 8, 12, ... 92, 93] -- 22 gaps of 4 and **one gap
of 1**, because 93 layers do not divide evenly. `hybrid_layer_split` took
`min(gaps)`, so `--plan` printed:

    pattern          every 1th layer is full attention  ->  2:1 linear:full

Both halves wrong, and neither detectable by a shape assertion: the field was
populated, the type was right, the value was a plausible small integer. The fix
is the MODE of the gaps, and `test_irregular_gap_does_not_fool_the_spacing`
feeds in the real index list to keep it fixed.

The ratio was wrong a second way. `linear // full` is integer division, so
69 // 24 == 2 -- it reported 2:1 for a model whose nominal design is 3:1. A
truncating ratio is a bug that gets *worse* the closer you are to correct.

What is asserted
----------------
1. the hybrid split, including the irregular gap above
2. MoE accounting -- and specifically that `active_fraction` is the ~1.8% that
   makes K3 a 104 B model wearing a 2.78 T coat
3. MLA cache is `kv_lora_rank + qk_rope_head_dim` and mentions NO head count,
   which is the property that distinguishes MLA from MHA
4. `capacity()` scales linearly with bit width and is honest about being
   weights-only
5. the alternate config key spellings really are alternates -- a nested
   `text_config` and a flat config must give identical answers
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from _srcload import load_function  # noqa: E402

SCRIPT = REPO / "03_llms/01_llm_finetuning/analyze_kimi_k3.py"

_text_config = load_function(SCRIPT, "_text_config")
_G = {"_text_config": _text_config}

hybrid_layer_split = load_function(SCRIPT, "hybrid_layer_split", extra_globals=_G)
moe_split = load_function(SCRIPT, "moe_split", extra_globals=_G)
mla_cache_per_token = load_function(SCRIPT, "mla_cache_per_token", extra_globals=_G)
capacity = load_function(SCRIPT, "capacity")


# The real published shape, trimmed to the keys these functions read.
# full_attn_layers is verbatim from the published config: 4, 8, ... 92, then
# 93 -- 23 steps of 4 and one of 1. Confirmed against the Hub, not recalled.
K3_FULL_ATTN = [i for i in range(4, 93, 4)] + [93]

K3 = {
    "text_config": {
        "num_hidden_layers": 93,
        "linear_attn_config": {"full_attn_layers": K3_FULL_ATTN},
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "moe_router_activation_func": "sigmoid",
        "moe_intermediate_size": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        # present on purpose: MLA's cache must NOT depend on these
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
    }
}

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        FAILED.append(name)


def test_hybrid_split() -> None:
    h = hybrid_layer_split(K3)
    check("93 layers, 24 full / 69 linear",
          (h["n_layers"], h["full"], h["linear"]) == (93, 24, 69),
          f"got {h['n_layers']}/{h['full']}/{h['linear']}")
    check("it is recognised as hybrid at all", h["is_hybrid"] is True)
    check("full_fraction ~ 26%", abs(h["full_fraction"] - 24 / 93) < 1e-9)


def test_irregular_gap_does_not_fool_the_spacing() -> None:
    """
    THE COUNTEREXAMPLE. This exact index list shipped 'every 1th layer'.

    22 gaps of 4 and one of 1. min() -> 1 (true of one pair, false of the
    design); mode -> 4 (the design). Asserted on the real list, not a tidied
    one, because tidying it is what makes the bug invisible.
    """
    gaps = [K3_FULL_ATTN[i + 1] - K3_FULL_ATTN[i]
            for i in range(len(K3_FULL_ATTN) - 1)]
    check("the fixture really does contain the irregular gap",
          min(gaps) == 1 and gaps.count(4) > gaps.count(1),
          f"gaps={sorted(set(gaps))} -- fixture no longer exercises the bug")

    h = hybrid_layer_split(K3)
    check("spacing is the MODE (4), not the min (1)", h["interval"] == 4,
          f"got interval={h['interval']} -- min() regression")

    # And the SHIPPED ratio must not truncate. This reads the value the
    # script returns -- recomputing it here would be a check that cannot
    # fail. 69/24 = 2.875, which prints 2.9 and must never print 2.
    ratio = h["linear_per_full"]
    check("linear:full is a real ratio, not floor division",
          abs(ratio - 2.875) < 1e-9 and f"{ratio:.1f}" == "2.9",
          f"got {ratio} -- floor division would give exactly 2")


def test_a_dense_model_is_not_reported_as_hybrid() -> None:
    """A guard that only ever sees hybrid configs proves nothing."""
    dense = {"text_config": {"num_hidden_layers": 32}}
    h = hybrid_layer_split(dense)
    check("a model with no linear_attn_config is NOT hybrid",
          h["is_hybrid"] is False and h["full"] == 0 and h["linear"] == 32,
          f"got {h}")
    check("spacing is None when there is no pattern to report",
          h["interval"] is None)


def test_moe_split() -> None:
    m = moe_split(K3)
    check("896 routed, 16 active, 2 shared",
          (m["routed"], m["active"], m["shared"]) == (896, 16, 2), f"got {m}")
    check("active_fraction is ~1.8%, not ~18% or 1.8",
          abs(m["active_fraction"] - 16 / 896) < 1e-12
          and 0.017 < m["active_fraction"] < 0.019,
          f"got {m['active_fraction']}")
    check("router is read from config, not assumed softmax",
          m["router"] == "sigmoid", f"got {m['router']}")


def test_mla_cache_mentions_no_head_count() -> None:
    c = mla_cache_per_token(K3)
    check("MLA cache = kv_lora_rank + qk_rope_head_dim = 576",
          c["supported"] and c["per_token_per_layer"] == 512 + 64,
          f"got {c}")

    # The distinguishing property: MHA's cache is heads x head_dim, so if this
    # function depended on a head count, doubling the heads would change it.
    doubled = {"text_config": dict(K3["text_config"],
                                   num_attention_heads=128,
                                   num_key_value_heads=128)}
    check("doubling the head count does NOT change the MLA cache",
          mla_cache_per_token(doubled)["per_token_per_layer"]
          == c["per_token_per_layer"],
          "the cache is reading a head count -- that is MHA, not MLA")

    check("a config without kv_lora_rank reports unsupported, not 0",
          mla_cache_per_token({"text_config": {}})["supported"] is False)


def test_capacity_is_derived_from_the_PARAMETER_COUNT() -> None:
    """
    THE SECOND COUNTEREXAMPLE, and the more expensive one.

    `capacity()` used to take the published 1,561 GB and scale it by
    `bits / 16` -- which assumes the checkpoint ships in bf16. It does not.
    2.72 T of K3's 2.78 T parameters are stored as U8, so it is already
    0.56 bytes/parameter, DENSER than fp8. The old model reported "~390 GB at
    4-bit, about 5 x H100" for a model whose weights do not fit on eight
    B200s. 390 GB is the bf16 size of a 780 B model -- a different model.

    The arithmetic was right and the premise was wrong, which is why nothing
    caught it: 1561 * 4/16 really is 390.
    """
    # The published facts, both from the Hub, both cross-checked.
    PARAMS = 2_779_931_837_184
    STORED_GB = 1561.0

    shipped_bpp = STORED_GB * 1e9 / PARAMS
    check("the checkpoint really is already quantised (< 1 byte/param)",
          shipped_bpp < 1.0, f"got {shipped_bpp:.2f} B/param")

    # capacity() must take a PARAMETER COUNT. If it still took GB, feeding it
    # a parameter count would produce a number ~1e12 too large.
    c = capacity(PARAMS, 180.0, shipped_bpp)
    check("as-shipped weights are ~1,561 GB",
          abs(c["gb"] - STORED_GB) < 1.0, f"got {c['gb']:,.0f} GB")

    check("8 x B200 (1,440 GB) does NOT hold the as-shipped weights",
          c["gb"] > 8 * 180, f"{c['gb']:,.0f} GB vs 1,440 GB")
    check("it needs more than 8 B200s", c["cards"] > 8.0,
          f"got {c['cards']:.1f} cards")

    # And the specific wrong answer must be unreachable.
    four_bit = capacity(PARAMS, 180.0, 0.5)
    check("true 4-bit is ~1,390 GB, NOT the old 390 GB",
          1380 < four_bit["gb"] < 1400, f"got {four_bit['gb']:,.0f} GB")
    check("even true 4-bit needs ~8 B200s, not ~5 H100s",
          four_bit["cards"] > 7.0, f"got {four_bit['cards']:.1f}")

    # Scaling is still linear in bytes/param -- the fix must not break that.
    check("bf16 is 4x true-4-bit",
          abs(capacity(PARAMS, 180.0, 2.0)["gb"] / four_bit["gb"] - 4.0) < 1e-9)


def test_flat_and_nested_configs_agree() -> None:
    """
    `_text_config` falls back to the top level when there is no `text_config`.
    If that fallback were broken, every number would silently come from an
    empty dict and print as 0 -- which looks like a small model, not an error.
    """
    flat = dict(K3["text_config"])
    for fn in (hybrid_layer_split, moe_split, mla_cache_per_token):
        check(f"{fn.__name__} gives the same answer flat as nested",
              fn(flat) == fn(K3), f"{fn(flat)} != {fn(K3)}")


def main() -> int:
    print("=" * 78)
    print("  Kimi K3 --plan: the architecture arithmetic")
    print("=" * 78)
    for fn in (test_hybrid_split,
               test_irregular_gap_does_not_fool_the_spacing,
               test_a_dense_model_is_not_reported_as_hybrid,
               test_moe_split,
               test_mla_cache_mentions_no_head_count,
               test_capacity_is_derived_from_the_PARAMETER_COUNT,
               test_flat_and_nested_configs_agree):
        print(f"\n{fn.__name__}")
        fn()

    print("\n" + "=" * 78)
    if FAILED:
        print(f"  {len(FAILED)} CHECK(S) FAILED")
        for f in FAILED:
            print(f"    - {f}")
        print("=" * 78)
        return 1
    print("  All checks passed")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

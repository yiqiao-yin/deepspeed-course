# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Regression test: GLM-5.3 architecture analysis and LoRA targeting.

Run:
    uv run tests/test_glm53_arch.py

Why this suite exists
---------------------
`train_glm53_ds.py` decides three things before it downloads 755 GB: where the
parameters are, which modules LoRA should adapt, and whether the hardware can
hold the model at all. Each has a failure mode that produces a confident wrong
answer rather than an error:

  * treating a sparse MoE like a dense model — count `intermediate_size` once
    per layer and a 743B model reports as ~14B, so the capacity check passes
    and the reader OOMs after an hour-long download
  * MLA confused with a parameter saving. It is not one — on GLM-5.3's
    dimensions MLA attention is slightly LARGER than vanilla would be. What it
    compresses is the KV cache, by 57x, which is the only reason a 1M-token
    context is possible. Claiming the wrong benefit means sizing the wrong
    resource
  * LoRA targets copied from a Llama example (`q_proj`/`k_proj`/`v_proj`) —
    GLM-5.3 has none of those names, so peft matches nothing. Depending on
    version that either raises or silently trains an adapter attached to
    nothing, and the loss still goes down because the base model is good
  * "LoRA makes it fit" — LoRA removes optimizer state, not the weights. A
    capacity check that applies a LoRA discount to the base model says 8xA100
    is fine when it is 1.4x short

The configs below are the REAL published ones, trimmed to the fields the code
reads. Numbers are cross-checked against measured file sizes on the Hub: the
GLM-5.3 arithmetic must land near 755.7 GB of fp8 bytes, and glm-edge-1.5b
near its advertised 1.5B parameters. That cross-check is what makes this a
test rather than a restatement of the implementation.

Uses tests/_srcload.py to pull the functions out of the shipped script without
importing torch, transformers or deepspeed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _srcload import load_function  # noqa: E402

SCRIPT = "03_huggingface/01_llm_finetuning/train_glm53_ds.py"

moe_parameter_split = load_function(SCRIPT, "moe_parameter_split")
lora_target_modules = load_function(SCRIPT, "lora_target_modules")
capacity_report = load_function(SCRIPT, "capacity_report",
                                extra_globals={"math": __import__("math")})
kv_cache_bytes_per_token = load_function(SCRIPT, "kv_cache_bytes_per_token")

# --- real published configs, trimmed to the fields the code reads -----------
GLM53 = dict(
    model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"],
    hidden_size=6144, num_hidden_layers=78, vocab_size=154880,
    n_routed_experts=256, num_experts_per_tok=8, moe_intermediate_size=2048,
    n_shared_experts=1, first_k_dense_replace=3, intermediate_size=12288,
    q_lora_rank=2048, kv_lora_rank=512, num_attention_heads=64,
    qk_head_dim=256, v_head_dim=256, qk_rope_head_dim=64, qk_nope_head_dim=192,
    max_position_embeddings=1048576, index_topk=2048,
    tie_word_embeddings=False,
)
GLM_EDGE = dict(
    model_type="glm", architectures=["GlmForCausalLM"],
    hidden_size=2048, num_hidden_layers=28, vocab_size=59264,
    intermediate_size=6144, num_attention_heads=16, num_key_value_heads=2,
    head_dim=128, max_position_embeddings=8192, tie_word_embeddings=False,
)

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
    bar = "=" * 74
    print(bar)
    print("  test_glm53_arch.py")
    print(bar)

    # ---- parameter split, cross-checked against measured bytes ------------
    print("\n  -- where the parameters are --")
    s = moe_parameter_split(GLM53)
    total_b = s["total"] / 1e9

    # GLM-5.3 ships 755.7 GB of fp8 weights. fp8 is ~1 byte per parameter plus
    # block scale tensors, so the computed count must land just under that.
    # Wide enough to tolerate the scales, tight enough that a dense-model
    # miscount (which lands near 14B) fails loudly.
    check(f"GLM-5.3 totals ~743B, consistent with 755.7 GB of fp8 "
          f"(got {total_b:.1f}B)",
          700 <= total_b <= 760,
          f"{total_b:.1f}B is not consistent with the measured file size — "
          "a dense-model miscount lands near 14B")

    check(f"experts dominate: {s['expert_fraction']*100:.1f}% of parameters",
          s["expert_fraction"] > 0.95,
          "the expert MLPs are 97% of this model; anything else means the "
          "MoE layers are being counted as dense")

    check("attention is a small minority of parameters",
          0.01 < s["attention"] / s["total"] < 0.05,
          f"attention is {100*s['attention']/s['total']:.1f}% — MLA counted "
          "as vanilla 4*h*h would be ~3x too large")

    # MLA does NOT save parameters, and asserting that it does would be
    # wrong: on GLM-5.3's dimensions (64 heads x 256 head_dim = 2.7x the
    # hidden size) the attention blocks are LARGER than vanilla would be.
    # This assertion is here because the first version of this test asserted
    # the opposite and failed against correct code.
    vanilla_params = 4 * GLM53["hidden_size"] ** 2 * GLM53["num_hidden_layers"]
    check(f"MLA does not reduce PARAMETERS "
          f"({s['attention']/1e9:.1f}B vs vanilla {vanilla_params/1e9:.1f}B)",
          s["attention"] > vanilla_params,
          "if MLA were smaller in parameters here, the head dimensions are "
          "not being read correctly")

    # What MLA actually compresses is the cache.
    kv = kv_cache_bytes_per_token(GLM53)
    check(f"MLA shrinks the KV CACHE by {kv['ratio']:.0f}x "
          f"({kv['mla']/1024:.0f} KB/token vs {kv['vanilla']/1024:.0f} KB)",
          kv["ratio"] > 20,
          "MLA caches a low-rank latent plus the RoPE key instead of full k "
          "and v per head; a ratio near 1 means the latent path is unused")

    ctx = GLM53["max_position_embeddings"]
    check(f"at {ctx:,} tokens the cache is ~{kv['mla']*ctx/1e9:.0f} GB, "
          f"not ~{kv['vanilla']*ctx/1e9:,.0f} GB",
          kv["mla"] * ctx / 1e9 < 200 and kv["vanilla"] * ctx / 1e9 > 1000,
          "this is what makes a 1M-token context possible at all")

    # A dense model with no MLA must report no saving rather than a fake one.
    kv_d = kv_cache_bytes_per_token(GLM_EDGE)
    check("a model without MLA reports ratio 1.0 (no saving invented)",
          kv_d["ratio"] == 1.0,
          f"got {kv_d['ratio']:.2f} — glm-edge has no kv_lora_rank, so there "
          "is no compression to report")

    # first_k_dense_replace=3 means exactly 3 dense layers, 75 MoE layers.
    check("the first 3 layers are dense, not MoE",
          s["dense_mlp"] > 0 and s["dense_mlp"] < s["experts"] / 100,
          f"dense_mlp={s['dense_mlp']/1e9:.2f}B — first_k_dense_replace is "
          "being ignored")

    # ---- a dense model must NOT report expert parameters ------------------
    d = moe_parameter_split(GLM_EDGE)
    check(f"glm-edge-1.5b totals ~1.5B (got {d['total']/1e9:.2f}B)",
          1.2 <= d["total"] / 1e9 <= 1.8,
          "the advertised size is 1.5B")
    check("a dense model reports ZERO expert parameters",
          d["experts"] == 0 and d["shared_experts"] == 0 and d["router"] == 0,
          f"experts={d['experts']} — dense models have no experts, and "
          "reporting some means the MoE branch ran on a dense config")
    check("a dense model's MLP is its largest component",
          d["dense_mlp"] > d["attention"] and d["dense_mlp"] > d["embedding"])

    # ---- LoRA targeting ---------------------------------------------------
    print("\n  -- LoRA target modules --")
    t53 = lora_target_modules(GLM53)
    # These names were verified against GLM-5.3's published
    # model.safetensors.index.json, which lists all 118,629 tensor names.
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
        check(f"GLM-5.3 targets {name} (verified in its safetensors index)",
              name in t53, f"got {t53}")

    check("GLM-5.3 does NOT target q_proj/k_proj/v_proj",
          not any(n in t53 for n in ("q_proj", "k_proj", "v_proj")),
          f"got {t53} — those names do not exist in this model; peft would "
          "match nothing and the loss would still go down")

    check("NO expert module is targeted",
          not any("expert" in n for n in t53),
          f"got {t53} — an adapter on expert k trains only when the router "
          "picks k, which at top-8 of 256 is ~3% of tokens")

    check("the router (gate) is NOT targeted",
          not any(n in ("gate", "mlp.gate") for n in t53),
          f"got {t53} — training the router changes WHICH experts fire, and "
          "routing collapse is how fine-tuned MoEs quietly degrade")

    tedge = lora_target_modules(GLM_EDGE)
    check("a dense GLM targets the vanilla q/k/v/o projections",
          all(n in tedge for n in ("q_proj", "k_proj", "v_proj", "o_proj")),
          f"got {tedge}")
    check("the two architectures get DIFFERENT targets",
          set(t53) != set(tedge),
          "one hardcoded list for both architectures means one of them is "
          "wrong")

    # ---- capacity ---------------------------------------------------------
    print("\n  -- capacity --")
    # The real numbers from the model card, against real hardware.
    c = capacity_report(755.7, 8, 80.0)      # 8 x A100/H100
    check("8 x 80GB does NOT hold GLM-5.3", not c["fits"],
          "640 GB cannot hold 755.7 GB of weights — this is the check that "
          "stops a reader paying for the download first")
    c = capacity_report(755.7, 8, 141.0)     # 8 x H200
    check("8 x H200 (1128 GB) DOES hold it", c["fits"])
    c = capacity_report(755.7, 1, 80.0)
    check(f"1 x 80GB is ~11x short (got {c['shortfall_x']:.1f}x)",
          c["shortfall_x"] > 10)
    check("it reports how many GPUs would be needed",
          capacity_report(755.7, 1, 80.0)["gpus_needed"] >= 12)

    # The core misconception this check exists to prevent.
    check("the requirement EXCEEDS the raw weight size",
          capacity_report(100.0, 1, 1000.0)["needed"] > 100.0,
          "LoRA frees optimizer state, not the base weights; a needed-size "
          "below the weight size means a LoRA discount was applied to the "
          "frozen model")

    check("a 3.2 GB model fits one 24 GB card",
          capacity_report(3.2, 1, 24.0)["fits"])

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

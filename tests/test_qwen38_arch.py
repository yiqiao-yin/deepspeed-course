# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Regression test: Qwen3.8-27B hybrid-attention analysis and LoRA targeting.

Run:
    uv run tests/test_qwen38_arch.py

Why this suite exists
---------------------
Qwen3.8 is a HYBRID: 48 of its 64 layers use a recurrent linear-attention
operator and only 16 use ordinary attention. Every mistake available here
produces a confident wrong answer rather than an error:

  * **The Llama-default LoRA list silently adapts a quarter of the model.**
    `q_proj`/`k_proj`/`v_proj`/`o_proj` DO exist here — on the 16
    full-attention layers only. peft matches them, attaches adapters, trains,
    and the loss falls, because three quarters of a good model is still a good
    model. Nothing raises. This is strictly nastier than GLM-5.3's version of
    the same bug, where the names match nothing and at least might raise.
  * **Sizing the KV cache over all 64 layers overstates it by 4x.** A
    linear-attention layer keeps a fixed recurrent state, not a growing cache,
    so it contributes nothing per token.
  * **Treating the linear state as per-token.** It is per *sequence* and
    constant in length — identical for 1 token and for 262,144. Getting this
    backwards inverts the entire argument for the architecture.
  * **Reading the top-level config.** Qwen3.8 is a vision-language model, so
    everything about the language model is nested under `text_config`.
    `config["num_hidden_layers"]` is simply absent, and a `.get(...)` default
    turns that into a plausible zero.

Numbers are cross-checked against facts established independently: the layer
counts against the published `layer_types` list, and the module names against
the real module tree built on the meta device (`--verify-arch`), which found
`q_proj` exactly 16 times and `in_proj_qkv` exactly 48 times.

Uses tests/_srcload.py to pull the functions out of the shipped script without
importing torch, transformers or deepspeed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _srcload import load_function  # noqa: E402

SCRIPT = "03_huggingface/01_llm_finetuning/train_qwen38_ds.py"

FULL = ["q_proj", "k_proj", "v_proj", "o_proj"]
LINEAR = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
_G = {"FULL_ATTN_TARGETS": FULL, "LINEAR_ATTN_TARGETS": LINEAR,
      "MLP_TARGETS": MLP}

_text_config = load_function(SCRIPT, "_text_config")
_G["_text_config"] = _text_config
hybrid_layer_split = load_function(SCRIPT, "hybrid_layer_split", extra_globals=_G)
_G["hybrid_layer_split"] = hybrid_layer_split
kv_cache_bytes_per_token = load_function(SCRIPT, "kv_cache_bytes_per_token",
                                         extra_globals=_G)
linear_state_bytes = load_function(SCRIPT, "linear_state_bytes", extra_globals=_G)
lora_target_modules = load_function(SCRIPT, "lora_target_modules", extra_globals=_G)
layer_coverage = load_function(SCRIPT, "layer_coverage", extra_globals=_G)
capacity_report = load_function(SCRIPT, "capacity_report",
                                extra_globals={"math": __import__("math")})

# --- the real published config, trimmed to the fields the code reads --------
# layer_types alternates 3 linear then 1 full, 64 entries, exactly as published.
_TYPES = [("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
          for i in range(64)]
QWEN38 = dict(
    model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
    vision_config=dict(depth=27, hidden_size=1152, patch_size=16,
                       out_hidden_size=5120),
    text_config=dict(
        model_type="qwen3_5_text", hidden_size=5120, num_hidden_layers=64,
        intermediate_size=17408, vocab_size=248320, layer_types=_TYPES,
        full_attention_interval=4, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_key_head_dim=128,
        linear_num_value_heads=48, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, mamba_ssm_dtype="float32",
        max_position_embeddings=262144, tie_word_embeddings=False,
    ),
)
# A uniform full-attention model, as the control.
QWEN3_DENSE = dict(
    model_type="qwen3", architectures=["Qwen3ForCausalLM"],
    hidden_size=2048, num_hidden_layers=28, intermediate_size=6144,
    vocab_size=151936, num_attention_heads=16, num_key_value_heads=8,
    head_dim=128, max_position_embeddings=32768, tie_word_embeddings=False,
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
    print("  test_qwen38_arch.py")
    print(bar)

    # ---- the hybrid split --------------------------------------------------
    print("\n  -- hybrid layer split --")
    s = hybrid_layer_split(QWEN38)
    check(f"64 layers: {s['linear']} linear + {s['full']} full",
          s["n_layers"] == 64 and s["linear"] == 48 and s["full"] == 16,
          f"got {s}")
    check("it is recognised as hybrid", s["is_hybrid"])
    check("full attention is every 4th layer", s["interval"] == 4)
    check(f"only {100*s['full_fraction']:.0f}% of layers keep a KV cache",
          abs(s["full_fraction"] - 0.25) < 1e-9)

    # The nested-config trap: reading the top level finds nothing.
    check("the analysis reads text_config, not the top level",
          _text_config(QWEN38).get("num_hidden_layers") == 64
          and QWEN38.get("num_hidden_layers") is None,
          "Qwen3.8 is a VLM; the language model lives under text_config, and "
          "a .get() default would turn that into a plausible zero")

    d = hybrid_layer_split(QWEN3_DENSE)
    check("a uniform model reports 0 linear layers and is not hybrid",
          d["linear"] == 0 and d["full"] == 28 and not d["is_hybrid"],
          f"got {d}")

    # ---- KV cache ----------------------------------------------------------
    print("\n  -- KV cache --")
    kv = kv_cache_bytes_per_token(QWEN38)
    check(f"cache counts ONLY the 16 full-attention layers "
          f"({kv['hybrid']/1024:.0f} KB/token)",
          kv["hybrid"] == 2 * 4 * 256 * 2 * 16,
          f"got {kv['hybrid']} bytes; counting all 64 layers gives "
          f"{2*4*256*2*64}")
    check(f"the all-full counterfactual is {kv['ratio']:.0f}x larger",
          abs(kv["ratio"] - 4.0) < 1e-9,
          "48 of 64 layers keep no cache, so the ratio must be 64/16")

    ctx = QWEN38["text_config"]["max_position_embeddings"]
    check(f"at {ctx:,} tokens the cache is ~{kv['hybrid']*ctx/1e9:.0f} GB, "
          f"not ~{kv['all_full']*ctx/1e9:.0f} GB",
          10 < kv["hybrid"] * ctx / 1e9 < 25
          and kv["all_full"] * ctx / 1e9 > 60)

    kvd = kv_cache_bytes_per_token(QWEN3_DENSE)
    check("a uniform model reports ratio 1.0 (no saving invented)",
          kvd["ratio"] == 1.0, f"got {kvd['ratio']}")

    # ---- the linear state is per-sequence, not per-token -------------------
    print("\n  -- linear-attention state --")
    st = linear_state_bytes(QWEN38)
    check(f"the 48 linear layers hold a fixed {st['total']/1e6:.0f} MB state",
          100e6 < st["total"] < 250e6, f"got {st['total']/1e6:.0f} MB")
    check("it is float32, per the config's mamba_ssm_dtype",
          st["recurrent"] == 48 * 128 * 128 * 4 * 48,
          "the SSM state dtype is pinned to float32 independently of the "
          "model dtype")
    check("a model with no linear layers reports zero state",
          linear_state_bytes(QWEN3_DENSE)["total"] == 0)

    # The property that makes the architecture worth having: the state does
    # not depend on sequence length at all. There is no length argument.
    import inspect
    src = inspect.getsource(linear_state_bytes)
    check("the state calculation takes no sequence length",
          "seq" not in inspect.signature(linear_state_bytes).parameters,
          "if length entered this calculation it would not be a fixed state")

    # ---- LoRA targeting: the headline failure ------------------------------
    print("\n  -- LoRA targets --")
    t = lora_target_modules(QWEN38)
    for n in FULL:
        check(f"default targets include {n} (16 full-attention layers)",
              n in t, f"got {t}")
    for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        check(f"default targets include {n} (48 linear-attention layers)",
              n in t, f"got {t}")

    cov = layer_coverage(QWEN38, t)
    check(f"the default covers ALL 64 layers ({cov['covered_layers']}/64)",
          cov["fraction"] == 1.0, f"got {cov}")

    # The counterexample. Without it, the coverage check proves nothing.
    bad = lora_target_modules(QWEN38, "attention-full")
    cov_bad = layer_coverage(QWEN38, bad)
    check(f"the Llama-default list covers only "
          f"{cov_bad['covered_layers']}/64 layers "
          f"({100*cov_bad['fraction']:.0f}%)",
          cov_bad["covered_layers"] == 16,
          "q_proj/k_proj/v_proj/o_proj exist here, but only on the 16 "
          "full-attention layers — this trains happily and adapts a quarter "
          "of the depth")
    check("the two scopes genuinely differ",
          set(t) != set(bad) and cov["fraction"] > cov_bad["fraction"])

    check("attention+mlp covers every layer too",
          layer_coverage(QWEN38, lora_target_modules(QWEN38, "attention+mlp"))
          ["fraction"] == 1.0)
    check("a uniform model gets only the full-attention names",
          set(lora_target_modules(QWEN3_DENSE)) == set(FULL),
          "there are no linear_attn modules to target in a dense model")

    # ---- capacity ----------------------------------------------------------
    print("\n  -- capacity --")
    check("1 x 48GB does NOT hold 55.6 GB of weights",
          not capacity_report(55.6, 1, 48.0)["fits"])
    check("2 x 48GB DOES", capacity_report(55.6, 2, 48.0)["fits"])
    check("2 x 24GB does not", not capacity_report(55.6, 2, 24.0)["fits"])
    check("the requirement EXCEEDS the raw weight size",
          capacity_report(100.0, 1, 1000.0)["needed"] > 100.0,
          "LoRA frees optimizer state, not the base weights")

    print("\n" + bar)
    print(f"  {PASS} passed, {FAIL} failed")
    print(bar)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

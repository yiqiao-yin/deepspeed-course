#!/usr/bin/env python3
"""
Where does ZeRO-3 actually become necessary? Answered before downloading.

    uv run verify_arch.py                      # the capacity table
    uv run verify_arch.py --model Qwen/Qwen2.5-VL-7B-Instruct
    uv run verify_arch.py --build              # real module tree, meta device

NOTE: there is deliberately no `require_gpu()` here. This script reads
a `config.json` over HTTPS and, with `--build`, instantiates the model
on torch's META device, which allocates nothing. A guard that can
never fire is decoration.

WHY THIS RUNS FIRST
-------------------
This course has now twice concluded that renting hardware was the
wrong answer, both times by measuring: lab 5 found the GPU 1.7x
SLOWER for its policy update, and lab 6 found GPU physics at 1.50x
against a 5x bar. Using ZeRO-3 here because it is a DeepSpeed course
would be the same mistake wearing a different hat.

So the question is asked numerically: at what point does this model
stop fitting on one card? The answer turns out to be sharp, and it is
not where the obvious design would have put it.

    LoRA on a 7B VLM fits on ONE card.
    Full fine-tuning the same model does not fit on EIGHT.

That is the whole argument for lab 7 using ZeRO-3, and it also says
which regime would have been cargo cult. A LoRA run under ZeRO-3
shards 15.8 GB that already fitted.

WHAT THE ARITHMETIC IS
----------------------
Per parameter, for mixed-precision Adam:

    bf16 weights          2 bytes   every parameter
    fp32 master copy      4 bytes   trainable only
    Adam m, v             8 bytes   trainable only
    bf16 gradients        2 bytes   trainable only

ZeRO-3 shards all four across ranks. What it does NOT shard is
activations, and for a vision-language model those are not small --
image tokens dominate the sequence. The table below is therefore a
LOWER BOUND on what a card must hold, and it is labelled as one. Lab
6's sibling mistake was quoting a figure without the condition it was
measured under; the condition here is "weights and optimizer only".
"""

from __future__ import annotations

import argparse
import json
import sys

GB = 1024 ** 3


def fetch_config(repo: str) -> dict:
    """The config alone, over HTTPS. Kilobytes, not gigabytes."""
    import urllib.request

    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=60) as r:      # noqa: S310
        return json.loads(r.read())


def param_count(cfg: dict) -> tuple[int, str]:
    """
    Parameters from the config, with the SOURCE of the number named.

    Derived rather than looked up, because a looked-up number cannot
    be checked against the config a reader actually downloads, and
    because `03_llms/01_llm_finetuning` found three defects in exactly
    this kind of arithmetic the first time it was tested.
    """
    tc = cfg.get("text_config", cfg)
    h = tc.get("hidden_size")
    layers = tc.get("num_hidden_layers")
    vocab = tc.get("vocab_size") or cfg.get("vocab_size")
    inter = tc.get("intermediate_size")
    if not all((h, layers, vocab, inter)):
        return 0, "config does not expose the dimensions"

    kvh = tc.get("num_key_value_heads") or tc.get("num_attention_heads")
    heads = tc.get("num_attention_heads")
    hd = h // heads
    # attention: q (h*h) + k,v (h * kvh*hd each) + o (h*h)
    attn = h * h + 2 * h * (kvh * hd) + h * h
    mlp = 3 * h * inter                      # gate, up, down
    body = layers * (attn + mlp)
    embed = vocab * h * 2                    # input + output, untied worst case
    vis = cfg.get("vision_config", {})
    vh = vis.get("hidden_size") or vis.get("embed_dim") or 0
    vl = vis.get("depth") or vis.get("num_hidden_layers") or 0
    vision = vl * (4 * vh * vh + 4 * vh * vh) if vh and vl else 0
    return body + embed + vision, (
        f"{layers} layers x (attn {attn/1e6:.0f}M + mlp {mlp/1e6:.0f}M) "
        f"+ embed {embed/1e6:.0f}M + vision {vision/1e6:.0f}M")


def footprint(p_total: int, p_train: int, shards: int) -> float:
    """GB per rank for weights + optimizer, ZeRO-3 sharded. Lower bound."""
    w = p_total * 2
    opt = p_train * 12                        # fp32 master 4 + Adam m 4 + v 4
    grad = p_train * 2
    return (w + opt + grad) / shards / GB


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--also", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--lora-frac", type=float, default=0.003,
                    help="trainable fraction under LoRA r=16 on attn+mlp")
    ap.add_argument("--cards", default="1,2,4,8")
    ap.add_argument("--vram", type=float, default=24.0,
                    help="GB per card, for the verdict column")
    ap.add_argument("--build", action="store_true",
                    help="instantiate on the META device and count for real")
    a, _ = ap.parse_known_args()

    cards = [int(c) for c in a.cards.split(",")]
    repos = [r for r in (a.model, a.also) if r]

    print("=" * 78)
    print("  CAPACITY — does this model need ZeRO-3, or is it decoration?")
    print("=" * 78)
    print(f"  weights + optimizer only; activations NOT included, and they "
          f"do not shard.\n  verdict column assumes {a.vram:.0f} GB cards.\n")

    for repo in repos:
        try:
            cfg = fetch_config(repo)
        except Exception as e:                              # noqa: BLE001
            print(f"  {repo}: config unreachable ({type(e).__name__})")
            continue
        p, how = param_count(cfg)
        if not p:
            print(f"  {repo}: {how}")
            continue
        print(f"  {repo}")
        print(f"    {p/1e9:.2f} B parameters  ({how})")
        for regime, ptrain in (("full fine-tune", p),
                               ("LoRA r=16", int(p * a.lora_frac))):
            row = []
            for n in cards:
                gb = footprint(p, ptrain, n)
                row.append(f"{gb:>8.1f}" + ("" if gb <= a.vram else "*"))
            fits = next((n for n in cards
                         if footprint(p, ptrain, n) <= a.vram), None)
            verdict = (f"fits on {fits}" if fits else
                       f"does NOT fit on {max(cards)}")
            print(f"      {regime:<16}" + "".join(row) + f"   -> {verdict}")
        print(f"      {'cards':<16}" + "".join(f"{n:>8}" for n in cards))
        print()

    print("  * = exceeds one card at that shard count\n")
    print("  Reading: LoRA fits on a single card, so running it under ZeRO-3")
    print("  would shard what already fitted. Full fine-tuning is the regime")
    print("  that needs sharding, and it is the one this lab trains.")

    if a.build:
        build_on_meta(a.model)
    return 0


def build_on_meta(repo: str) -> None:
    """
    The real module tree, no weights, no download of safetensors.

    `03_llms` uses this to catch -- for free -- the three things that
    otherwise fail only AFTER a multi-hundred-gigabyte download: an
    unsupported architecture, LoRA target names that match nothing,
    and parameter arithmetic that disagrees with the implementation.
    The third is the one that applies here: the table above is derived
    from a config, and this is what checks it against the code.
    """
    import torch
    import transformers
    from transformers import AutoConfig

    print("\n" + "=" * 78)
    print("  META DEVICE — the real module tree, no weights")
    print("=" * 78)
    print(f"  transformers {transformers.__version__}")

    # API drift, the "vanished symbol" class: fails at IMPORT, before
    # anything runs, so `compileall` never sees it. `AutoModelForVision2Seq`
    # was the vision-language auto class through transformers 4.x and is
    # GONE in 5.x, replaced by `AutoModelForImageTextToText`. This lab
    # locks 5.16.1, where only the second name exists -- but the first is
    # what every tutorial written before 2025 still says, so both are
    # tried and the one that resolved is printed.
    auto = None
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        auto = getattr(transformers, name, None)
        if auto is not None:
            print(f"  auto class: {name}")
            break
    if auto is None:
        print("  no vision-language auto class in this transformers")
        return

    cfg = AutoConfig.from_pretrained(repo)
    try:
        with torch.device("meta"):
            model = auto.from_config(cfg)
    except Exception as e:                                  # noqa: BLE001
        print(f"  could not build: {type(e).__name__}: {str(e)[:200]}")
        return
    real = sum(p.numel() for p in model.parameters())
    derived, _ = param_count(AutoConfig.from_pretrained(repo).to_dict())
    err = (derived - real) / real
    print(f"  implementation reports {real/1e9:.3f} B parameters")
    print(f"  the table above derived {derived/1e9:.3f} B  "
          f"({err:+.1%})")
    # Catching this disagreement is the POINT of building on meta.
    # The table is arithmetic over a config; the model is code. Where
    # they differ, the code wins, and the gap is reported rather than
    # quietly absorbed -- `03_llms/01_llm_finetuning` found three
    # defects of exactly this shape the first time it was tested.
    if abs(err) > 0.05:
        print("  WARNING: over 5% apart. Trust the implementation, and "
              "treat the\n           capacity table as indicative only.")
    else:
        print("  within 5%: the capacity table is sound for sizing. The "
              "residual is\n  tied embeddings and the vision tower, which "
              "the config does not fully describe.")

    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    names = [n for n, _ in model.named_modules()]
    print("\n  do the LoRA target names resolve against the real tree?")
    for t in targets:
        hits = sum(1 for n in names if n.endswith(t))
        print(f"    {t:<12} {hits:>5} modules" +
              ("" if hits else "   <-- MATCHES NOTHING"))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Qwen3-VL's architecture, on the meta device: no GPU, no weights, ~2 seconds.

    uv run verify_arch.py                    # Qwen3-VL-8B
    uv run verify_arch.py --compare          # against Qwen2.5-VL-3B

This is the `--verify-arch` technique from `03_llms/01_llm_finetuning`, pointed
at a vision-language model. torch's **meta device** builds the real module tree
without allocating a byte of weights, so three things that otherwise fail only
*after* a 17 GB download can be checked for free:

  1. whether the architecture is supported by the installed transformers,
  2. whether your LoRA target names resolve against the real module tree,
  3. what the parameter arithmetic actually is.

Why this matters more for a VL model than for a text one
--------------------------------------------------------
**A vision-language model has two sub-models with independent naming, and peft
matches by NAME.** A target list that looks like it covers "the attention
layers" may cover one sub-model, both, or -- silently -- neither, and peft will
cheerfully report a trainable-parameter count either way.

Measured here, and the reason this script exists:

    Qwen3-VL-8B     vision attention is a FUSED `qkv`
                    q_proj/k_proj/v_proj match 36 language, 0 vision
                    vision MLP is linear_fc1/linear_fc2 -- no collision

    Qwen2.5-VL-3B   vision attention is also a fused `qkv`
                    BUT its vision MLP is gate_proj/up_proj/down_proj,
                    the SAME names the language model uses. Adding those
                    to a target list unfreezes 32 vision modules you did
                    not ask for.

So on one model the MLP names are safe and on the other they are not, and
nothing in either config file says so. The only way to know is to build the
tree and look -- which costs two seconds here and a download otherwise.

DeepStack
---------
Qwen3-VL injects visual features at *several* depths of the language model
rather than only at the input embedding. `config.vision_config` carries
`deepstack_visual_indexes: [8, 16, 24]`, and the tree carries a matching
`visual.deepstack_merger_list` with one merger per index. Qwen2.5-VL has
neither. That is a real architectural difference, not a version bump.
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--compare", action="store_true",
                   help="Also build Qwen2.5-VL-3B and diff the two trees.")
    p.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated LoRA target names to resolve.")
    return p.parse_known_args()[0]


def inspect(model_id: str, targets: list) -> dict:
    """Build on meta and report. Returns a dict so --compare can diff."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    arch = (cfg.architectures or ["?"])[0]

    # Resolve the class transformers itself would pick, so this script does not
    # hardcode a model family and silently go stale.
    import transformers
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise SystemExit(
            f"\n{arch} is not available in transformers "
            f"{transformers.__version__}.\nThis is exactly the failure this "
            f"script exists to surface BEFORE a multi-gigabyte download.\n")

    with torch.device("meta"):
        model = cls(cfg)

    lin = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    total = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for n, p in model.named_parameters() if "visual" in n)

    res = {"id": model_id, "arch": arch, "total": total, "vision": vision,
           "targets": {}, "deepstack": []}
    for t in targets:
        hit = [n for n in lin if n.split(".")[-1] == t]
        res["targets"][t] = (len([n for n in hit if "visual" not in n]),
                             len([n for n in hit if "visual" in n]))
    res["deepstack"] = [n for n, _ in model.named_modules()
                        if "deepstack" in n.lower()]
    import collections
    res["vision_leaves"] = collections.Counter(
        n.split(".")[-1] for n in lin if "visual" in n)
    return res


def report(r: dict) -> None:
    bar = "-" * 74
    print(f"\n  {r['id']}")
    print(f"  {bar}")
    print(f"  class            {r['arch']}")
    print(f"  parameters       {r['total']/1e9:.2f} B")
    print(f"  vision tower     {r['vision']/1e9:.2f} B  "
          f"({100*r['vision']/r['total']:.1f}%)")
    print(f"  bf16 weights     {r['total']*2/1e9:.1f} GB   <- the memory floor")

    print(f"\n  vision tower Linear leaf names:")
    print(f"    {', '.join(f'{k} x{v}' for k, v in r['vision_leaves'].most_common())}")

    print(f"\n  {'LoRA target':<14} {'language':>9} {'VISION':>8}   verdict")
    for t, (lang, vis) in r["targets"].items():
        if vis:
            verdict = "ALSO adapts the vision tower"
        elif lang:
            verdict = "language model only"
        else:
            verdict = "MATCHES NOTHING -- check the name"
        print(f"    {t:<12} {lang:>9} {vis:>8}   {verdict}")

    ds = r["deepstack"]
    print(f"\n  DeepStack        {'yes, ' + str(len(ds)) + ' modules' if ds else 'no'}")
    for n in ds[:4]:
        print(f"    {n}")


def main() -> None:
    args = parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    bar = "=" * 74
    print(bar)
    print("  Qwen3-VL architecture check — meta device, no weights downloaded")
    print(bar)

    import time
    t0 = time.time()
    a = inspect(args.model, targets)
    report(a)

    if args.compare:
        b = inspect("Qwen/Qwen2.5-VL-3B-Instruct", targets)
        report(b)
        print(f"\n{bar}")
        print("  What changed between the generations")
        print(bar)
        print(f"  parameters       {b['total']/1e9:.2f} B  ->  {a['total']/1e9:.2f} B"
              f"   ({a['total']/b['total']:.1f}x)")
        print(f"  DeepStack        {'yes' if b['deepstack'] else 'no':<4}"
              f"       ->  {'yes' if a['deepstack'] else 'no'}")
        # The trap, resolved for real rather than described: the same three
        # MLP names are safe on one generation and not on the other.
        mlp = ["gate_proj", "up_proj", "down_proj"]
        a2 = inspect(args.model, mlp)
        b2 = inspect("Qwen/Qwen2.5-VL-3B-Instruct", mlp)
        print(f"\n  The MLP-name trap -- same names, different consequence:")
        print(f"    {'target':<12} {'Qwen2.5-VL':>22} {'Qwen3-VL':>22}")
        for t in mlp:
            lb, vb = b2["targets"][t]
            la, va = a2["targets"][t]
            fb = f"{lb} lang + {vb} VISION" if vb else f"{lb} lang only"
            fa = f"{la} lang + {va} VISION" if va else f"{la} lang only"
            print(f"    {t:<12} {fb:>22} {fa:>22}")
        print(f"\n    Adding these to a Qwen2.5-VL target list unfreezes the")
        print(f"    vision tower. On Qwen3-VL it does not, because that tower's")
        print(f"    MLP is linear_fc1/linear_fc2. Nothing in either config says so.")

    print(f"\n{bar}")
    print(f"  done in {time.time()-t0:.1f}s, zero weights downloaded")
    print(bar)


if __name__ == "__main__":
    main()

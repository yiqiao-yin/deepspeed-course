#!/usr/bin/env python3
"""
Kimi K3: reading a 2.8-trillion-parameter model you cannot run.

    # everything below, from config.json alone. No GPU, no download:
    uv run analyze_kimi_k3.py --plan

    # build the real module tree on the meta device (needs the pins below)
    uv run analyze_kimi_k3.py --verify-arch

This is the third frontier-analysis script in this folder, beside
`train_glm53_ds.py` (755 GB sparse MoE) and `train_qwen38_ds.py` (hybrid
linear/full attention). Kimi K3 is both of those at once, and larger than
either by a wide margin.

--------------------------------------------------------------------------
WHY THIS SCRIPT DOES NOT TRAIN
--------------------------------------------------------------------------
`train_qwen38_ds.py` analyses *and* trains, because 27 B fits on two 48 GB
cards. This one only analyses, and the reason is arithmetic rather than
laziness:

    parameters        2.78 T total, 104 B activated per token
    weights on Hub    1,561 GB  (96 safetensors shards, index total_size)
    as shipped        0.56 bytes/param -- ALREADY quantised (2.72 T in U8),
                      so there is no 4x saving left to take
    realistic floor   >1,561 GB, i.e. 9+ B200-180GB. 8 x B200 is 1,440 GB and
                      falls 121 GB short before any activations

There are two further blockers that no amount of hardware fixes:

**The remote code does not import on the pinned transformers.** K3 ships
custom modelling code (`auto_map` -> `modeling_kimi_k3.py`) that imports
`OutputRecorder` from `transformers.utils.generic`. That symbol survives in
transformers 5.1.0 and is **gone in 5.2.0** (verified by installing each). Every lab in this course
pins **5.16.1**, and `tests/test_config_kwargs.py` fails CI if any lock
disagrees. `--verify-arch` therefore needs its own pinned environment; see
the note on that flag below.

**It needs `fla-core`** (flash-linear-attention) for Kimi Delta Attention,
which is a CUDA-compiled dependency.

So this script is a reading exercise, and `--plan` is the part that always
works: it needs nothing but `config.json`.

--------------------------------------------------------------------------
WHAT KIMI K3 IS
--------------------------------------------------------------------------
Every number here was read from the published `config.json` and confirmed by
building the model on the meta device -- no weights downloaded.

    architecture      KimiK3ForConditionalGeneration (model_type: kimi_k3)
    text backbone     KimiLinearForCausalLM (model_type: kimi_linear)
    parameters        2.78 T total, of which 2.72 T (97.9%) are MoE experts
    vision tower      0.4 B -- 0.01% of the model
    layers            93, of which 24 are FULL attention and 69 are LINEAR
    layer pattern     every 4th layer is full attention, i.e. a nominal 3:1
                      linear:full -- measured 2.9:1, because the 93 layers do
                      not divide evenly and one step is irregular
    experts           896 routed, 16 activated per token, 2 shared
    router            sigmoid affinity -- the DeepSeek-V3 convention, and the
                      same one 03_llms/11_moe implements from the paper
    context           1,048,576 tokens

**97.9% of the parameters are experts.** That single number explains the
model: K3 is not a dense 2.8 T model, it is a 104 B model with an enormous
lookup table of specialists attached. The router that picks 16 of 896 is a
rounding error in the parameter count and decides where almost everything
goes -- exactly the point `03_llms/11_moe` makes at toy scale.

--------------------------------------------------------------------------
WHAT THIS COURSE ALREADY TEACHES THAT K3 USES
--------------------------------------------------------------------------
K3 is a good capstone read precisely because it composes ideas that already
have their own folders:

    MLA-style compressed KV     kv_lora_rank 512, q_lora_rank 1536
                                -> 03_llms/10_deepseek_from_scratch
    fine-grained + shared MoE   896 routed + 2 shared, sigmoid router
                                -> 03_llms/11_moe
    hybrid linear/full layers   24 full of 93
                                -> 03_llms/01_llm_finetuning (Qwen3.8)
    expert parallelism          the paper's own training strategy
                                -> 03_llms/11_moe --expert-parallel

--------------------------------------------------------------------------
A NOTE ON A WIDELY-COPIED TRAINING SNIPPET
--------------------------------------------------------------------------
A fine-tuning snippet for "kimi-k3" circulates on tutorial sites. Run against
the versions this course pins, it fails four ways, and `--plan --audit-snippet`
reproduces the checks:

    model_id = "kimi/kimi-k3"       -> 401. The id is moonshotai/Kimi-K3
    SFTTrainer(tokenizer=...)       -> removed in transformers 5.x
    SFTTrainer(max_seq_length=...)  -> removed in trl 1.x
    train_dataset="train.jsonl"     -> a str, not a Dataset

Its LoRA targets, though, are **fine**: `q_proj/k_proj/v_proj/o_proj` resolve
to 89 modules in the full-attention layers and 211 in the linear ones. That is
worth stating because the obvious guess -- that a linear-attention model uses
different names -- is wrong here, and only building the tree shows it.

Reference: *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653.
"""

import argparse
import json
import os
import sys
import urllib.request

MODEL = "moonshotai/Kimi-K3"

# Measured from the safetensors index, not estimated: sum of shard sizes.
WEIGHTS_GB = 1561.0

# Total parameters, from the Hub's own safetensors census -- NOT derived from
# the byte count, because the two do not have the constant ratio you would
# expect. 2.72 T of these are stored as U8.
PARAMS = 2_779_931_837_184


# NOTE: there is deliberately no `require_gpu()` here.
#
# Every other training script in this course calls one before importing torch,
# because a CPU-only reader would otherwise get a CUDA traceback. This script
# never touches a GPU -- `--plan` reads a JSON file over HTTPS and `--verify-arch`
# builds on torch's META device, which allocates nothing. A guard that can never
# fire is not a safety net, it is decoration, and the repo's own rule against
# using a distributed launcher where there is nothing to distribute applies
# equally to guarding a resource that is never used.


def fetch_config(model: str = MODEL, token: str = None) -> dict:
    """
    Read a model's config.json without cloning it.

    Uses huggingface_hub when present and a plain HTTPS GET otherwise, so
    `--plan` works in a bare environment with nothing installed.
    """
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model, "config.json", token=token)
        return json.load(open(p))
    except Exception:                                    # noqa: BLE001
        url = f"https://huggingface.co/{model}/resolve/main/config.json"
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)


def _text_config(config: dict) -> dict:
    """K3 nests the language model under text_config; be tolerant either way."""
    return config.get("text_config", config)


def hybrid_layer_split(config: dict) -> dict:
    """
    Count full- versus linear-attention layers, from config alone.

    K3 names the full-attention layers explicitly rather than giving a
    `layer_types` list the way Qwen3.8 does, so the arithmetic differs even
    though the idea is identical: most layers are cheap linear attention and
    a minority are full attention that actually carries a KV cache.
    """
    t = _text_config(config)
    n = t.get("num_hidden_layers", 0)
    la = t.get("linear_attn_config") or {}
    full_idx = la.get("full_attn_layers") or []
    full = len(full_idx)
    # The MODE of the gaps, not the min. K3's full-attention indices are
    # 4, 8, 12, ... with one irregular step, so min() reports "every 1th
    # layer" -- true of exactly one pair and false of the design.
    spacing = None
    if len(full_idx) > 1:
        from collections import Counter
        gaps = Counter(full_idx[i + 1] - full_idx[i]
                       for i in range(len(full_idx) - 1))
        spacing = gaps.most_common(1)[0][0]
    # A FLOAT on purpose. `linear // full` is 69 // 24 == 2 for K3, which
    # reports 2:1 for a model whose design is 3:1 -- a truncating ratio gets
    # worse the closer you are to correct.
    return dict(n_layers=n, full=full, linear=n - full,
                full_indices=full_idx, interval=spacing,
                linear_per_full=(n - full) / full if full else 0.0,
                full_fraction=(full / n) if n else 0.0,
                # BOTH kinds must be present. `(n - full) > 0` alone calls
                # every dense model hybrid, because a model with no
                # full-attention list has n linear layers by subtraction.
                is_hybrid=full > 0 and (n - full) > 0)


def moe_split(config: dict) -> dict:
    """
    Routed / shared / active expert accounting, from config alone.

    The number worth carrying away is `active_fraction`: what proportion of
    the model a single token actually touches.
    """
    t = _text_config(config)
    routed = t.get("num_experts") or t.get("n_routed_experts") or 0
    per_tok = t.get("num_experts_per_token") or t.get("num_experts_per_tok") or 0
    shared = t.get("num_shared_experts") or t.get("n_shared_experts") or 0
    return dict(routed=routed, active=per_tok, shared=shared,
                expert_hidden=t.get("routed_expert_hidden_size")
                or t.get("moe_intermediate_size"),
                router=t.get("moe_router_activation_func"),
                active_fraction=(per_tok / routed) if routed else 0.0)


def mla_cache_per_token(config: dict) -> dict:
    """
    MLA's cache is `kv_lora_rank + qk_rope_head_dim` VALUES per token per
    FULL-ATTENTION layer -- and it mentions no head count, which is the whole
    point of MLA. 03_llms/10_deepseek_from_scratch derives this from scratch.

    The linear layers carry no KV cache at all, so on a hybrid model the cache
    scales with the FULL layers only. That is the compounding win: fewer
    cached layers, and each one cheaper.
    """
    t = _text_config(config)
    kv = t.get("kv_lora_rank")
    rope = t.get("qk_rope_head_dim") or 0
    if kv is None:
        return dict(supported=False)
    return dict(supported=True, kv_lora_rank=kv, qk_rope_head_dim=rope,
                per_token_per_layer=kv + rope)


def capacity(params: int, vram_gb: float, bytes_per_param: float) -> dict:
    """
    How many cards the WEIGHTS alone need, at a given bytes-per-parameter.

    Takes a PARAMETER COUNT, not a byte size, and that is the whole point.

    The first version of this function took `weights_gb` and scaled it by
    `bits / 16`, i.e. it assumed the published 1,561 GB was a bf16 footprint
    and that 4-bit would quarter it to 390 GB. **K3 does not ship in bf16.**
    2.72 T of its 2.78 T parameters are stored as U8, so the checkpoint is
    already 0.56 bytes/parameter -- denser than fp8 -- and there is no 4x
    left to take. 390 GB is what you would get if 1,561 GB were the bf16 size
    of a 780 B model, which is a different model entirely.

    Rescaling an ALREADY-QUANTISED checkpoint by a bit width is the error, and
    it is invisible: the arithmetic is right, the premise is not.

    Deliberately weights-only and deliberately optimistic -- CLAUDE.md's rule
    is `weights/N + overhead that does not shard`, and activations, gather
    buffers and fragmentation are all excluded.
    """
    g = params * bytes_per_param / 1e9
    return dict(bytes_per_param=bytes_per_param, gb=g, cards=g / vram_gb)


def audit_snippet() -> list:
    """
    Check the widely-copied 'kimi-k3' fine-tuning snippet against the
    INSTALLED libraries.

    Imports live, so it reports what your environment actually accepts rather
    than a remembered snapshot. Returns a list of (claim, ok, detail).
    """
    out = []
    try:
        import inspect

        import transformers
        import trl
    except ImportError as exc:
        return [("libraries importable", False, f"{exc}")]

    sft = inspect.signature(trl.SFTTrainer.__init__).parameters
    cfg = inspect.signature(trl.SFTConfig.__init__).parameters
    out.append(("SFTTrainer accepts tokenizer=", "tokenizer" in sft,
                "removed in transformers 5.x; use processing_class="))
    out.append(("SFTTrainer accepts max_seq_length=", "max_seq_length" in sft,
                "moved out of the trainer in trl 1.x"))
    out.append(("SFTConfig accepts max_seq_length=", "max_seq_length" in cfg,
                "not here either -- check the installed SFTConfig signature"))
    return out


def print_plan(config: dict, args) -> None:
    bar = "=" * 78
    t = _text_config(config)
    h = hybrid_layer_split(config)
    m = moe_split(config)
    c = mla_cache_per_token(config)

    print(bar)
    print(f"  {MODEL} — what the config implies")
    print(bar)
    print(f"  architecture     {(config.get('architectures') or ['?'])[0]}")
    print(f"  text backbone    {(t.get('architectures') or ['?'])[0]}"
          f"  (model_type: {t.get('model_type')})")
    print(f"  hidden size      {t.get('hidden_size'):,}")
    print(f"  vocab            {t.get('vocab_size'):,}")
    print(f"  context          {t.get('max_position_embeddings'):,} tokens")

    print(f"\n{bar}")
    print("  1. Attention: a hybrid, and most layers are the cheap kind")
    print(bar)
    print(f"  layers           {h['n_layers']}")
    print(f"  FULL attention   {h['full']:>3}   ({100*h['full_fraction']:.0f}%)")
    print(f"  LINEAR (KDA)     {h['linear']:>3}   ({100*(1-h['full_fraction']):.0f}%)")
    if h["interval"]:
        print(f"  pattern          every {h['interval']}th layer is full "
              f"attention  ->  {h['linear_per_full']:.1f}:1 linear:full")
    if c.get("supported"):
        print(f"\n  The {h['full']} full layers use MLA-style compressed KV:")
        print(f"    kv_lora_rank + qk_rope_head_dim = {c['kv_lora_rank']} + "
              f"{c['qk_rope_head_dim']} = {c['per_token_per_layer']} values"
              f"/token/layer")
        print(f"    Note what is ABSENT from that sum: any head count. That is")
        print(f"    MLA's whole claim — see 03_llms/10_deepseek_from_scratch.")
        print(f"  The {h['linear']} linear layers carry NO KV cache at all, so the")
        print(f"  cache scales with {h['full']} layers, not {h['n_layers']}.")

    print(f"\n{bar}")
    print("  2. MoE: the model is almost entirely experts")
    print(bar)
    print(f"  routed experts   {m['routed']}")
    print(f"  active / token   {m['active']}   "
          f"({100*m['active_fraction']:.1f}% of the routed pool)")
    print(f"  shared experts   {m['shared']}   (every token, always)")
    print(f"  expert hidden    {m['expert_hidden']}")
    print(f"  router           {m['router']} affinity")
    if m["router"] == "sigmoid":
        print(f"                   ^ the DeepSeek-V3 convention, which "
              f"03_llms/11_moe")
        print(f"                     implements from the paper (Eq. 15)")

    print(f"\n{bar}")
    print("  3. Can you run it? No, and here is the arithmetic")
    print(bar)
    print(f"  weights on the Hub   {WEIGHTS_GB:,.0f} GB  (safetensors index "
          f"total_size)")
    print(f"  total parameters     {PARAMS/1e12:,.2f} T")
    print(f"  => as shipped        {WEIGHTS_GB*1e9/PARAMS:.2f} bytes/parameter "
          f"-- ALREADY QUANTISED, denser than fp8")
    print(f"     2.72 T of the 2.78 T parameters are stored as U8, so there is")
    print(f"     no 4x saving left to take. Quantisation is not the way out.")
    print(f"\n  {'stored as':<20} {'weights':>10}   {'B200-180':>9} "
          f"{'H200-141':>9}")
    print(f"  {'-'*20} {'-'*10}   {'-'*9} {'-'*9}")
    rows = [("AS SHIPPED", WEIGHTS_GB * 1e9 / PARAMS),
            ("if true 4-bit", 0.5),
            ("if fp8", 1.0),
            ("if bf16", 2.0)]
    for name, bpp in rows:
        a = capacity(PARAMS, 180, bpp)
        b = capacity(PARAMS, 141, bpp)
        print(f"  {name:<20} {a['gb']:>7,.0f} GB   {a['cards']:>8.1f}x "
              f"{b['cards']:>8.1f}x")
    print(f"\n  8 x B200 is {8*180:,} GB -- {WEIGHTS_GB - 8*180:,.0f} GB SHORT of")
    print(f"  the as-shipped weights, before a single activation.")
    print(f"\n  Those columns are WEIGHTS ONLY. Activations, gather buffers and")
    print(f"  fragmentation do not shard -- budget per GPU as")
    print(f"  weights/N + overhead, never in aggregate (CLAUDE.md).")

    print(f"\n{bar}")
    print("  4. Two blockers hardware does not fix")
    print(bar)
    print("  * The remote code imports OutputRecorder from")
    print("    transformers.utils.generic. That existed up to 5.1 and was")
    print("    REMOVED IN 5.2.0. This course pins 5.16.1 everywhere, and")
    print("    tests/test_config_kwargs.py fails CI if a lock disagrees.")
    print("  * It needs fla-core (flash-linear-attention) for Kimi Delta")
    print("    Attention — a CUDA-compiled dependency.")
    print("\n  So --verify-arch needs its own pinned environment. See the")
    print("  header of this file for the exact pins that work.")

    if args.audit_snippet:
        print(f"\n{bar}")
        print("  5. The circulating 'kimi-k3' snippet, checked live")
        print(bar)
        for claim, ok, detail in audit_snippet():
            print(f"  {'OK  ' if ok else 'FAIL'}  {claim:<38} {'' if ok else detail}")
        print("\n  Also: model_id 'kimi/kimi-k3' is a 404/401 — the real id is")
        print(f"  {MODEL}. And train_dataset='train.jsonl' passes a str where")
        print("  a Dataset is expected.")
        print("\n  Its LoRA targets ARE fine, which is the surprising part:")
        print("  q_proj/k_proj/v_proj/o_proj resolve to 89 modules in the full")
        print("  layers and 211 in the linear ones. Verify with --verify-arch")
        print("  rather than trusting either guess.")

    print(f"\n{bar}")
    print("  What to read next, in this repo")
    print(bar)
    print("  03_llms/10_deepseek_from_scratch   MLA, built from the paper")
    print("  03_llms/11_moe                     routing + load balancing")
    print("  03_llms/01_llm_finetuning          train_qwen38_ds.py — the same")
    print("                                     hybrid idea at a size you CAN run")
    print(bar)


def verify_architecture(model: str = MODEL) -> bool:
    """
    Build the real module tree on torch's meta device: no weights, no memory.

    This is the technique from train_glm53_ds.py and train_qwen38_ds.py,
    pointed at a model whose remote code needs a DIFFERENT transformers than
    this course pins. It is kept because it is the only thing that settles
    what the LoRA targets actually resolve to -- reading the modelling file is
    not enough, and I got that wrong before building the tree: grepping
    `modeling_kimi_k3.py` shows no q_proj at all, because that file is the
    VISION tower. The text backbone lives in `modeling_kimi_linear.py`.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        print(f"\n  --verify-arch needs torch + transformers: {exc}\n")
        return False

    print(f"  building {model} on the meta device (no weights) ...")
    try:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    except ImportError as exc:
        print(f"\n  FAILED: {exc}")
        print("\n  This is the documented blocker, not a bug in this script.")
        print("  K3's remote code needs transformers <=5.1 and fla-core;")
        print("  this course pins transformers 5.16.1. Reproduce with:")
        print("\n      uv run --no-project \\")
        print("        --with 'transformers==5.0.0' --with torch --with accelerate \\")
        print("        --with einops --with fla-core \\")
        print("        python analyze_kimi_k3.py --verify-arch\n")
        return False
    except Exception as exc:                              # noqa: BLE001
        print(f"\n  FAILED: {type(exc).__name__}: {str(exc)[:200]}\n")
        return False

    import re
    tot = sum(p.numel() for p in m.parameters())
    exp = sum(p.numel() for n, p in m.named_parameters()
              if re.search(r"\.(w1|w2|w3)\.", n))
    full = set((_text_config(cfg.to_dict()).get("linear_attn_config") or {})
               .get("full_attn_layers", []))
    lin = [n for n, mo in m.named_modules() if isinstance(mo, torch.nn.Linear)]
    tgt = {"q_proj", "k_proj", "v_proj", "o_proj"}

    def layer_of(n):
        mm = re.search(r"layers\.(\d+)\.", n)
        return int(mm.group(1)) if mm else -1

    hit_full = sum(1 for n in lin if n.split(".")[-1] in tgt
                   and layer_of(n) in full)
    hit_lin = sum(1 for n in lin if n.split(".")[-1] in tgt
                  and layer_of(n) >= 0 and layer_of(n) not in full)

    print(f"\n  parameters        {tot/1e12:.2f} T")
    print(f"  MoE experts       {exp/1e12:.2f} T  ({100*exp/tot:.1f}%)")
    print(f"  full-attn layers  {len(full)}")
    print(f"\n  target_modules ['q_proj','k_proj','v_proj','o_proj'] resolve to:")
    print(f"    {hit_full:>4} modules in the FULL-attention layers")
    print(f"    {hit_lin:>4} modules in the LINEAR layers")
    print(f"\n  Both are non-zero, so the conventional target list works here.")
    print(f"  That is worth checking rather than assuming: a linear-attention")
    print(f"  model could easily have used different names, and grepping the")
    print(f"  wrong modelling file suggests it does.")
    return True


def parse_args() -> argparse.Namespace:
    """parse_known_args: the launcher injects --local_rank into argv."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--plan", action="store_true",
                   help="Capacity + architecture analysis from config.json. "
                        "No GPU, no download, no torch.")
    p.add_argument("--verify-arch", action="store_true",
                   help="Build the module tree on the meta device. Needs "
                        "transformers <=5.1 + fla-core; see --plan step 4.")
    p.add_argument("--audit-snippet", action="store_true",
                   help="With --plan: check the circulating 'kimi-k3' "
                        "fine-tuning snippet against the INSTALLED libraries.")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    if args.verify_arch:
        sys.exit(0 if verify_architecture(args.model) else 1)

    if args.plan:
        token = os.environ.get("HF_TOKEN")
        print_plan(fetch_config(args.model, token), args)
        return

    # No flag: this script does not train, and saying so beats letting a
    # reader wait for a 1,561 GB download that ends in an ImportError.
    bar = "=" * 78
    print(bar)
    print("  This script ANALYSES Kimi K3. It does not train it.")
    print(bar)
    print(f"\n  {MODEL} is 2.78 T parameters / 1,561 GB of weights, and its")
    print("  remote code does not import on the transformers this course pins.")
    print("\n  What you can do:")
    print("      uv run analyze_kimi_k3.py --plan")
    print("      uv run analyze_kimi_k3.py --plan --audit-snippet")
    print("\n  To fine-tune something in this folder that DOES run:")
    print("      deepspeed --num_gpus=2 train_ds.py --use-lora --max-steps 20")
    print("\n  That one needs 2 x 24 GB. If you have no GPU, rent one and let")
    print("  it terminate itself:")
    print("      uv run runpod/runpod_ctl.py run 03_llms/01_llm_finetuning \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print(f"\n{bar}")
    sys.exit(1)


if __name__ == "__main__":
    main()

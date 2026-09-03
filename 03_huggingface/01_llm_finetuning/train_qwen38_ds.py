#!/usr/bin/env python3
"""
Qwen3.8-27B: LoRA fine-tuning a HYBRID linear/full-attention model.

One script, four stages: download data -> download model -> fine-tune ->
generate.

    # what the architecture implies, from config alone. No GPU, no download:
    uv run train_qwen38_ds.py --plan
    uv run train_qwen38_ds.py --verify-arch     # build it on the meta device

    # the real thing (2 x 48 GB is enough)
    deepspeed --num_gpus=2 train_qwen38_ds.py

CoreWeave / SLURM:      sbatch run_qwen38.sh --max-steps 20
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 03_huggingface/01_llm_finetuning \\
                            --dry-run --collect --wait --terminate --yes

--------------------------------------------------------------------------
WHAT Qwen3.8-27B IS
--------------------------------------------------------------------------
Every number below was read from the published `config.json` and confirmed by
building the model on the meta device -- no weights downloaded.

    architecture      Qwen3_5ForConditionalGeneration (model_type: qwen3_5)
    parameters        27.36 B total   (26.90 B without the vision tower)
    weights           55.6 GB bf16
    layers            64, of which 48 are LINEAR attention and 16 are FULL
    layer pattern     full attention every 4th layer (full_attention_interval)
    full attention    GQA, 24 q-heads / 4 kv-heads, head_dim 256
    linear attention  gated-delta style: 16 k-heads x 128, 48 v-heads x 128,
                      causal conv kernel 4, SSM state in float32
    vision tower      27 layers, hidden 1152, patch 16  (0.46 B)
    context           262,144 tokens
    transformers      config saved with 5.8.0.dev0; verified working on
                      5.16.1, which is what this folder's uv.lock pins

This is a different bargain from `train_glm53_ds.py` in the same folder.
GLM-5.3 is a SPARSE MoE: it buys capacity by holding 743 B parameters and
using 39 B of them per token. Qwen3.8 is DENSE -- every parameter runs on
every token -- and instead attacks the *sequence* dimension: three quarters of
its layers replaced attention with a recurrent linear operator whose state
does not grow with context.

Both are memory techniques. GLM-5.3 shrinks what the model *is*; Qwen3.8
shrinks what the model must *remember*.

--------------------------------------------------------------------------
THE THING MOST LIKELY TO GO WRONG
--------------------------------------------------------------------------
The usual LoRA target list -- q_proj / k_proj / v_proj / o_proj, copied from
every Llama recipe -- exists in this model, on the **16 full-attention layers
only**. The other 48 layers have no q_proj at all; they use
`linear_attn.in_proj_{qkv,z,b,a}` and `linear_attn.out_proj`.

So the Llama default does not error. It does not warn. It attaches adapters to
25% of the model's depth, trains happily, and shows a perfectly healthy falling
loss. This is worse than GLM-5.3's failure mode, where the same list matches
nothing and at least has a chance of raising.

`lora_target_modules()` covers both families. `--verify-arch` proves it against
the real module tree, and `--lora-scope attention-full` reproduces the
mistake on purpose so you can see the coverage number.
"""

import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Sizes are MEASURED from the Hub API (sum of sibling blob sizes), not
# estimated, so the preflight can refuse before downloading rather than after.
# ---------------------------------------------------------------------------
KNOWN_MODELS = {
    "Qwen/Qwen3.8-27B": dict(gb=55.6, note="the headline model, hybrid attention"),
    "Qwen/Qwen3-8B":    dict(gb=16.4, note="dense, all full attention"),
    "Qwen/Qwen3-1.7B":  dict(gb=4.1,  note="small proxy"),
}

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

# Verified against the real module tree (see verify_architecture).
FULL_ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LINEAR_ATTN_TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                       "out_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, the run gets as far as loading weights and then dies deep
    inside the training stack -- here, after a 55.6 GB download. The reader has
    already waited, and the error says nothing about what went wrong.

    ALLOW_CPU=1 bypasses it, which is only sensible with a tiny --model.
    """
    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. From this folder:")
        print("            uv sync\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing on CPU.")
        print("            Only sane with a tiny --model. Disable bf16 in the")
        print("            DeepSpeed config or the trainer raises anyway.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before the run fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  This example LoRA-fine-tunes Qwen3.8-27B with DeepSpeed ZeRO-3.")
    print("  It downloads 55.6 GB of weights and needs real GPU memory.")
    print("\n  What you CAN do right now, with no GPU and no download:")
    print("      uv run train_qwen38_ds.py --plan")
    print("          hybrid-layer, KV-cache and capacity analysis from config.json")
    print("      uv run train_qwen38_ds.py --verify-arch")
    print("          build the real module tree on the meta device")
    print("      uv run ../../tests/test_qwen38_arch.py")
    print("      ./tests/run_all.sh          # the whole logic suite")
    print("      https://yiqiao-yin.github.io/deepspeed-course/")
    print("\n  Examples 01_basics and 02_intermediate teach the same DeepSpeed")
    print("  mechanics and run end to end on a CPU.")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 03_huggingface/01_llm_finetuning")
    print("      uv run runpod/runpod_ctl.py run 03_huggingface/01_llm_finetuning \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def _text_config(config: dict) -> dict:
    """
    Return the language-model half of a possibly-multimodal config.

    Qwen3.8 nests everything interesting under `text_config` because it is a
    vision-language model. Reaching for `config["num_hidden_layers"]` directly
    silently gets nothing, so every reader goes through here.
    """
    return config.get("text_config", config)


def hybrid_layer_split(config: dict) -> dict:
    """
    Count linear- versus full-attention layers, from config alone.

    Pure arithmetic on the published config: no weights, no torch, no network.
    The distinction matters for two decisions that are easy to get wrong --
    which modules LoRA must target, and how the KV cache scales.

    Args:
        config: the model's config.json as a dict

    Returns:
        dict with n_layers, full, linear, interval, and the per-family
        module-name lists.
    """
    t = _text_config(config)
    types = t.get("layer_types") or []
    n = t.get("num_hidden_layers", len(types))

    if types:
        full = sum(1 for x in types if x == "full_attention")
        linear = sum(1 for x in types if x != "full_attention")
    else:
        # A model with no layer_types is uniform full attention.
        full, linear = n, 0

    return dict(n_layers=n, full=full, linear=linear,
                interval=t.get("full_attention_interval"),
                full_fraction=(full / n) if n else 0.0,
                is_hybrid=linear > 0)


def kv_cache_bytes_per_token(config: dict, dtype_bytes: int = 2) -> dict:
    """
    KV cache per token -- counting ONLY the layers that actually keep one.

    This is the whole point of a hybrid model and the easiest thing to get
    wrong when sizing a deployment. A linear-attention layer keeps a fixed
    recurrent state instead of a growing cache, so it contributes **nothing**
    per token. Sizing all 64 layers as though they cached would overstate the
    requirement by 4x here.

    Args:
        config: the model's config.json as a dict
        dtype_bytes: 2 for bf16/fp16 cache, 1 for fp8

    Returns:
        dict with hybrid, all_full (the counterfactual), ratio
    """
    t = _text_config(config)
    split = hybrid_layer_split(config)
    kv_heads = t.get("num_key_value_heads", t.get("num_attention_heads", 0))
    head_dim = t.get("head_dim", 0)
    if not head_dim and t.get("num_attention_heads"):
        head_dim = t.get("hidden_size", 0) // t["num_attention_heads"]

    per_layer = 2 * kv_heads * head_dim * dtype_bytes   # k and v
    hybrid = per_layer * split["full"]
    all_full = per_layer * split["n_layers"]
    return dict(hybrid=hybrid, all_full=all_full,
                ratio=(all_full / hybrid) if hybrid else 1.0)


def linear_state_bytes(config: dict) -> dict:
    """
    The fixed cost of the linear-attention layers, per SEQUENCE not per token.

    A gated-delta layer carries a recurrent state and a short causal
    convolution window. Both are constant in sequence length: identical for one
    token and for 262,144. That constant is what makes a very long context
    affordable, and it is invisible if you only ever measure per-token cost.

    Args:
        config: the model's config.json as a dict

    Returns:
        dict with recurrent, conv, total (bytes per sequence)
    """
    t = _text_config(config)
    split = hybrid_layer_split(config)
    n_lin = split["linear"]
    if not n_lin:
        return dict(recurrent=0, conv=0, total=0)

    v_heads = t.get("linear_num_value_heads", 0)
    k_heads = t.get("linear_num_key_heads", 0)
    k_dim = t.get("linear_key_head_dim", 0)
    v_dim = t.get("linear_value_head_dim", 0)
    kernel = t.get("linear_conv_kernel_dim", 0)

    # SSM state is kept in float32 (mamba_ssm_dtype), not the model dtype.
    recurrent = v_heads * k_dim * v_dim * 4 * n_lin
    conv = kernel * (k_heads * k_dim * 2 + v_heads * v_dim) * 4 * n_lin
    return dict(recurrent=recurrent, conv=conv, total=recurrent + conv)


def lora_target_modules(config: dict, scope: str = "all-attention") -> list:
    """
    Choose LoRA targets that cover BOTH attention families.

    Verified against the real module tree (see verify_architecture): the 16
    full-attention layers expose q_proj/k_proj/v_proj/o_proj, and the 48
    linear-attention layers expose in_proj_qkv/in_proj_z/in_proj_b/in_proj_a
    and out_proj. They share no names.

    The failure this guards against is quiet. Targeting only the Llama-style
    names matches 16 layers of 64 -- it does not raise, it trains, and the loss
    falls, because a quarter of a good model is still a good model. You simply
    fine-tuned 25% of the depth and never found out.

    Args:
        config: the model's config.json as a dict
        scope: 'all-attention' (default, both families), 'attention-full'
               (the Llama-style mistake, kept so it can be measured),
               'attention+mlp' (adds the MLPs, ~3x the adapter)

    Returns:
        list of module-name suffixes for peft's `target_modules`
    """
    split = hybrid_layer_split(config)

    if scope == "attention-full":
        return list(FULL_ATTN_TARGETS)
    if scope == "attention+mlp":
        return FULL_ATTN_TARGETS + LINEAR_ATTN_TARGETS + MLP_TARGETS
    if not split["is_hybrid"]:
        # A uniform full-attention model has no linear_attn modules at all.
        return list(FULL_ATTN_TARGETS)
    return FULL_ATTN_TARGETS + LINEAR_ATTN_TARGETS


def layer_coverage(config: dict, targets: list) -> dict:
    """
    What fraction of the model's DEPTH would these targets actually adapt?

    The number the Llama-default mistake makes visible. Reported in --plan and
    asserted in the tests.

    Args:
        config: the model's config.json as a dict
        targets: the LoRA target suffixes

    Returns:
        dict with covered_layers, n_layers, fraction
    """
    split = hybrid_layer_split(config)
    hits_full = any(t in FULL_ATTN_TARGETS for t in targets)
    hits_linear = any(t in LINEAR_ATTN_TARGETS for t in targets)
    hits_mlp = any(t in MLP_TARGETS for t in targets)

    covered = 0
    if hits_full:
        covered += split["full"]
    if hits_linear:
        covered += split["linear"]
    if hits_mlp:
        covered = split["n_layers"]        # every layer has an MLP
    covered = min(covered, split["n_layers"])
    return dict(covered_layers=covered, n_layers=split["n_layers"],
                fraction=(covered / split["n_layers"]) if split["n_layers"] else 0.0)


def capacity_report(model_gb: float, num_gpus: int, vram_gb: float) -> dict:
    """
    Can this hardware hold this model, with LoRA?

    LoRA freezes the base weights, which removes optimizer state (~12 bytes per
    trainable parameter for Adam in mixed precision) but does NOT reduce what
    it costs to HOLD the model. The 1.2 factor covers activations, the adapter,
    gradients and fragmentation. A refuse/proceed gate, not a planner.

    Args:
        model_gb: weights on disk, in GB
        num_gpus: number of GPUs
        vram_gb: VRAM per GPU, in GB

    Returns:
        dict with total_vram, needed, fits, shortfall_x, gpus_needed
    """
    total = num_gpus * vram_gb
    needed = model_gb * 1.2
    return dict(total_vram=total, needed=needed, fits=total >= needed,
                shortfall_x=(needed / total) if total else float("inf"),
                gpus_needed=math.ceil(needed / vram_gb) if vram_gb else 0)


def fetch_config(model: str, token: str = None) -> dict:
    """Fetch config.json from the Hub without downloading any weights."""
    try:
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(model, "config.json", token=token)) as f:
            return json.load(f)
    except Exception:
        import urllib.request
        req = urllib.request.Request(
            f"https://huggingface.co/{model}/raw/main/config.json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)


def model_size_gb(model: str, token: str = None) -> float:
    """
    Total weight size in GB, measured from the Hub API rather than estimated.

    Returns 0.0 when it cannot be determined. Callers must treat that as
    "unknown", never as "small".
    """
    if model in KNOWN_MODELS:
        return KNOWN_MODELS[model]["gb"]
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{model}?blobs=true")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        return sum(s.get("size") or 0 for s in data.get("siblings", [])) / 1e9
    except Exception:
        return 0.0


def verify_architecture(model: str, targets: list, token: str = None) -> bool:
    """
    Build the model on the META device and check the LoRA targets resolve.

    The meta device allocates no memory and downloads no weights: transformers
    constructs the module tree from config.json alone. It proves the installed
    transformers implements this architecture, and that every target name
    exists in the real tree -- both of which otherwise fail only after a 55.6
    GB download.

    Args:
        model: HuggingFace model id
        targets: LoRA target module suffixes
        token: optional HF token

    Returns:
        True when every target resolves.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"  building {model} on the meta device (no weights, no memory)...")
    cfg = AutoConfig.from_pretrained(model, token=token, trust_remote_code=True)
    with torch.device("meta"):
        net = AutoModelForCausalLM.from_config(cfg)

    names = [n for n, _ in net.named_modules()]
    print(f"  built {type(net).__name__}: {len(names):,} modules")

    ok = True
    for t in targets:
        hits = sum(1 for n in names if n.endswith("." + t))
        family = ("full-attn" if t in FULL_ATTN_TARGETS else
                  "linear-attn" if t in LINEAR_ATTN_TARGETS else "mlp")
        print(f"    {t:<16} {family:<12} {'FOUND' if hits else 'MISSING':<8} {hits:>4}")
        if not hits:
            ok = False

    n_par = sum(p.numel() for p in net.parameters())
    print(f"  parameters: {n_par/1e9:,.2f} B")

    if not ok:
        print("\n  A target did not resolve. peft would attach LoRA to fewer")
        print("  modules than intended, and training would still run and still")
        print("  show a falling loss.")
    return ok


def print_plan(args, config: dict, size_gb: float, num_gpus: int,
               vram_gb: float) -> bool:
    """Print the architecture, LoRA and capacity analysis. True if it fits."""
    bar = "=" * 78
    t = _text_config(config)
    split = hybrid_layer_split(config)
    kv = kv_cache_bytes_per_token(config)
    state = linear_state_bytes(config)
    targets = lora_target_modules(config, args.lora_scope)
    cov = layer_coverage(config, targets)
    cap = capacity_report(size_gb, num_gpus, vram_gb) if size_gb else None

    print(bar)
    print(f"  {args.model}")
    print(bar)
    print(f"  model_type        {config.get('model_type')}")
    print(f"  architecture      {(config.get('architectures') or ['?'])[0]}")
    print(f"  layers            {split['n_layers']}")
    print(f"  hidden size       {t.get('hidden_size')}")
    print(f"  context           {t.get('max_position_embeddings'):,}")
    if config.get("vision_config"):
        v = config["vision_config"]
        print(f"  vision tower      {v.get('depth')} layers, hidden "
              f"{v.get('hidden_size')}, patch {v.get('patch_size')}")

    if split["is_hybrid"]:
        print(bar)
        print("  Hybrid attention: three quarters of the layers keep no cache")
        print(bar)
        print(f"    linear attention  {split['linear']:>3} layers "
              f"({100*split['linear']/split['n_layers']:.0f}%)   "
              "recurrent state, fixed size")
        print(f"    full attention    {split['full']:>3} layers "
              f"({100*split['full']/split['n_layers']:.0f}%)   "
              "keeps a growing KV cache")
        if split["interval"]:
            print(f"    pattern           full attention every "
                  f"{split['interval']}th layer")
        print()
        print(f"    KV cache/token    {kv['hybrid']/1024:>6.0f} KB   "
              f"(only the {split['full']} full-attention layers)")
        print(f"    if all {split['n_layers']} were full  "
              f"{kv['all_full']/1024:>6.0f} KB   -> {kv['ratio']:.0f}x more")
        ctx = t.get("max_position_embeddings", 0)
        if ctx:
            print(f"    at {ctx:,} tokens: {kv['hybrid']*ctx/1e9:.1f} GB "
                  f"vs {kv['all_full']*ctx/1e9:.1f} GB per request")
        print()
        print(f"    the {split['linear']} linear layers hold "
              f"{state['total']/1e6:.0f} MB per SEQUENCE,")
        print("    independent of length -- the same for 1 token or the full")
        print("    context. That constant is what makes the context affordable.")

    print(bar)
    print("  LoRA plan")
    print(bar)
    print(f"    scope             {args.lora_scope}")
    print(f"    target modules    {', '.join(targets)}")
    print(f"    rank / alpha      {args.lora_rank} / {args.lora_alpha}")
    print(f"    layer coverage    {cov['covered_layers']}/{cov['n_layers']} "
          f"({100*cov['fraction']:.0f}% of depth)")
    if cov["fraction"] < 0.99:
        print()
        print(f"    WARNING: {cov['n_layers'] - cov['covered_layers']} layers get NO adapter.")
        print("    This is exactly the Llama-default mistake: q_proj/k_proj/")
        print("    v_proj/o_proj exist here, but only on the full-attention")
        print("    layers. Training will run and the loss will fall anyway.")
        print("    Use --lora-scope all-attention unless you meant this.")

    if cap:
        print(bar)
        print("  Capacity")
        print(bar)
        print(f"    weights on disk   {size_gb:.1f} GB")
        print(f"    needed (x1.2)     {cap['needed']:.1f} GB   "
              "(LoRA frees optimizer state, NOT the base weights)")
        print(f"    you have          {num_gpus} x {vram_gb:.0f} GB = "
              f"{cap['total_vram']:.0f} GB")
        if cap["fits"]:
            print("    verdict           FITS")
        else:
            print(f"    verdict           DOES NOT FIT — short by "
                  f"{cap['shortfall_x']:.1f}x")
            print(f"    would need        ~{cap['gpus_needed']} x {vram_gb:.0f} GB")
        print(bar)
        return cap["fits"]

    print(bar)
    print("    weights on disk   UNKNOWN — could not reach the Hub API.")
    print("    Treating unknown as 'do not proceed'. Pass --force to override.")
    print(bar)
    return False


def parse_args() -> argparse.Namespace:
    """
    parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    --local_rank=N into every worker's argv, and a strict parser exits 2 with
    "unrecognized arguments" before training starts. CONTRIBUTING.md §3.2.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HuggingFace model id. Default {DEFAULT_MODEL} "
                        "(55.6 GB — see --plan).")
    p.add_argument("--plan", action="store_true",
                   help="Print the hybrid-layer, LoRA and capacity analysis, "
                        "then exit. Needs NO GPU and downloads no weights.")
    p.add_argument("--verify-arch", action="store_true",
                   help="Build the model on the meta device and check the LoRA "
                        "targets resolve. No GPU, no weight download. Do this "
                        "before renting.")
    p.add_argument("--lora-scope", default="all-attention",
                   choices=["all-attention", "attention-full", "attention+mlp"],
                   help="Which module families LoRA adapts. 'attention-full' "
                        "reproduces the Llama-default mistake on purpose so "
                        "the coverage number is visible.")
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--max-samples", type=int, default=512,
                   help="Cap the dataset. This example is about the mechanics "
                        "of fine-tuning a hybrid model, not about producing a "
                        "good one.")
    p.add_argument("--max-length", type=int, default=512,
                   help="Token cap per example. The model supports 262,144; "
                        "using it would need sequence parallelism this script "
                        "does not implement.")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after this many optimizer steps (-1 = use "
                        "epochs). The dry-run path.")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LoRA tolerates a far higher LR than full fine-tuning.")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-lora", action="store_true",
                   help="Full fine-tune. Needs roughly 12x the memory for "
                        "optimizer state alone — about 670 GB here.")
    p.add_argument("--vram-gb", type=float, default=0.0,
                   help="Override detected VRAM per GPU, for --plan.")
    p.add_argument("--num-gpus", type=int, default=0,
                   help="Override detected GPU count, for --plan.")
    p.add_argument("--force", action="store_true",
                   help="Proceed even when the capacity check says it will not "
                        "fit. You will OOM; this exists so the refusal is not "
                        "a wall.")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--prompt", default="In two sentences, explain why a model "
                                       "might replace attention with a "
                                       "recurrent layer.")
    p.add_argument("--output", default="./qwen38-lora-out")
    p.add_argument("--deepspeed", default="ds_config_qwen38.json")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    bar = "=" * 78
    hf_token = os.environ.get("HF_TOKEN")

    config = fetch_config(args.model, hf_token)
    size_gb = model_size_gb(args.model, hf_token)

    # ---- stage 0: analysis, which needs neither GPU nor weights -----------
    if args.plan:
        # Deliberately BEFORE require_gpu(): the point is that a reader with a
        # laptop can see what this architecture implies.
        print_plan(args, config, size_gb, args.num_gpus or 2,
                   args.vram_gb or 48.0)
        print("  (assumed hardware; override with --num-gpus / --vram-gb)")
        return

    if args.verify_arch:
        print(bar)
        print(f"  Architecture verification: {args.model}")
        print(bar)
        ok = verify_architecture(args.model,
                                 lora_target_modules(config, args.lora_scope),
                                 hf_token)
        print(bar)
        sys.exit(0 if ok else 1)

    require_gpu()

    import torch

    num_gpus = args.num_gpus or (torch.cuda.device_count() or 1)
    if args.vram_gb:
        vram = args.vram_gb
    elif torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        vram = 0.0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    if is_main:
        fits = print_plan(args, config, size_gb, max(num_gpus, world_size), vram)
        if not fits and not args.force:
            print()
            print("  Refusing to download weights that cannot fit.")
            print("  This check exists because the alternative is discovering")
            print("  it AFTER a 55.6 GB download.")
            print()
            print("  Your options:")
            print("    more GPUs                            2 x 48 GB is enough")
            print("    --model Qwen/Qwen3-1.7B              same code, 4.1 GB")
            print("    --force                              proceed and OOM")
            print(bar)
            sys.exit(1)

    # ---- heavy imports, only now ------------------------------------------
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # W&B stays optional and soft: no key, no tracking, no crash.
    use_wandb = False
    try:
        import wandb
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
            use_wandb = True
    except ImportError:
        wandb = None

    # ---- stage 1: data -----------------------------------------------------
    if is_main:
        print(bar)
        print(f"  [1/4] dataset: {args.dataset}")
        print(bar)
    ds = load_dataset(args.dataset, split="train")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def to_text(row: dict) -> dict:
        """Flatten an instruction row into one training string."""
        instr = row.get("instruction") or row.get("question") or ""
        inp = row.get("input") or ""
        out = row.get("output") or row.get("answer") or row.get("response") or ""
        prompt = f"{instr}\n\n{inp}".strip() if inp else instr
        return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{out}"}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    # Fail loudly: a formatter whose column names do not match the dataset
    # produces rows that are pure boilerplate, and training proceeds perfectly
    # happily on nothing at all.
    sample = ds["text"][:64]
    hollow = sum(1 for x in sample
                 if len(x.replace("### Instruction:", "")
                         .replace("### Response:", "").strip()) < 8)
    if hollow > len(sample) // 2:
        raise RuntimeError(
            f"{args.dataset}: {hollow}/{len(sample)} formatted rows are empty "
            "apart from the template. The column names in to_text() do not "
            "match this dataset — fix them rather than training on boilerplate.")
    if is_main:
        print(f"  {len(ds)} examples")
        print(f"  sample: {ds['text'][0][:160]!r}")

    # ---- stage 2: model ----------------------------------------------------
    if is_main:
        print(bar)
        print(f"  [2/4] model: {args.model}  ({size_gb:.1f} GB)")
        print(bar)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        # A tokenizer with no pad token dies on the first padded batch.
        tokenizer.pad_token = tokenizer.eos_token

    # AutoModelForCausalLM gives Qwen3_5ForCausalLM -- the language half, no
    # vision tower. That is deliberate: this folder is about LLM fine-tuning,
    # and text-only SFT of the language model is a first-class supported path.
    # Use AutoModelForImageTextToText if you want the full VLM.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, trust_remote_code=True,
        dtype=torch.bfloat16)
    model.config.use_cache = False        # incompatible with grad checkpointing

    # ---- stage 3: LoRA + train --------------------------------------------
    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig
        targets = lora_target_modules(config, args.lora_scope)
        cov = layer_coverage(config, targets)
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets)
        if is_main:
            print(f"  LoRA targets: {', '.join(targets)}")
            print(f"  layer coverage: {cov['covered_layers']}/{cov['n_layers']}"
                  f" ({100*cov['fraction']:.0f}% of depth)")

    if use_wandb and is_main:
        wandb.init(project="deepspeed-course-qwen38",
                   name=f"{args.model.split('/')[-1]}-lora",
                   config=dict(model=args.model, dataset=args.dataset,
                               lora_rank=args.lora_rank, lr=args.lr,
                               lora_scope=args.lora_scope))

    sft_config = SFTConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        max_length=args.max_length,
        report_to="wandb" if use_wandb else "none",
        deepspeed=args.deepspeed,
    )

    if is_main:
        print(bar)
        print("  [3/4] fine-tuning")
        print(bar)
    # processing_class is passed EXPLICITLY, and that is not optional here.
    # Left to itself, SFTTrainer calls AutoProcessor.from_pretrained(...), and
    # because this repo is a vision-language model that resolves to the full
    # multimodal processor -- which imports Qwen2VLImageProcessor and dies with
    #     ImportError: Qwen2VLImageProcessor requires the PIL library
    # even though we are doing text-only SFT and never touch an image. Handing
    # it the tokenizer we already built skips the processor path entirely.
    # (Observed on a real 2xL40S pod run before this line existed.)
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds,
                         processing_class=tokenizer,
                         peft_config=peft_config)

    # Assert the adapter actually attached to something. peft silently
    # produces a model with zero trainable parameters when no target matches,
    # and such a run trains, logs a loss, and changes nothing.
    trainable = sum(p.numel() for p in trainer.model.parameters()
                    if p.requires_grad)
    if not args.no_lora and trainable == 0:
        raise RuntimeError(
            "LoRA attached to nothing: 0 trainable parameters. The target "
            f"modules {lora_target_modules(config, args.lora_scope)} matched "
            "no module in this model.")
    if is_main:
        total = sum(p.numel() for p in trainer.model.parameters())
        print(f"  trainable: {trainable/1e6:.1f} M of {total/1e9:.2f} B "
              f"({100*trainable/total:.3f}%)")

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    if is_main:
        print(f"  adapter written to {args.output}")

    # ---- stage 4: inference ------------------------------------------------
    if args.skip_inference:
        return
    if is_main:
        print(bar)
        print("  [4/4] inference")
        print(bar)
    if world_size > 1:
        # Under ZeRO-3 the weights are sharded across ranks, so a plain
        # generate() on rank 0 alone would read partial tensors.
        if is_main:
            print("  Skipped: ZeRO-3 shards the weights across ranks.")
            print("  Generate from the saved adapter in a single process:")
            print(f"      deepspeed --num_gpus=1 {os.path.basename(sys.argv[0])}"
                  " --skip-inference ...")
        return

    model.config.use_cache = True
    m = trainer.model.eval()
    text = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    ids = tokenizer(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=96, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                           skip_special_tokens=True)
    print(f"  prompt:   {args.prompt}")
    print(f"  response: {gen.strip()!r}")
    if not gen.strip():
        # An empty generation after a successful train is exactly the kind of
        # quiet failure this course exists to catch.
        raise RuntimeError(
            "The model generated nothing. Something is wrong with the "
            "tokenizer, the adapter, or the prompt format.")
    print(bar)

    if use_wandb and is_main:
        wandb.finish()


if __name__ == "__main__":
    main()

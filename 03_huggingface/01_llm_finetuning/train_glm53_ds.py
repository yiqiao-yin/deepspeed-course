#!/usr/bin/env python3
"""
GLM-5.3: LoRA fine-tuning a 755 GB sparse MoE with DeepSpeed ZeRO-3.

One script, four stages: download data -> download model -> fine-tune ->
generate.

    # what the architecture implies, from config alone. No GPU, no download:
    uv run train_glm53_ds.py --plan
    uv run train_glm53_ds.py --plan --model zai-org/glm-edge-1.5b-chat

    # the real thing (see the hardware table below before you try)
    deepspeed --num_gpus=8 train_glm53_ds.py

    # the same code path on a model you can actually rent
    deepspeed --num_gpus=1 train_glm53_ds.py --model zai-org/glm-edge-1.5b-chat

CoreWeave / SLURM:      sbatch run_glm53.sh --max-steps 20
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 03_huggingface/01_llm_finetuning \\
                            --dry-run --collect --wait --terminate --yes

--------------------------------------------------------------------------
WHAT GLM-5.3 IS, AND WHY IT DOES NOT FIT
--------------------------------------------------------------------------
Released 2026-08-31 by zai-org. Every number below was read from the model's
published `config.json` and `model.safetensors.index.json` -- the index lists
all 118,629 tensor names and the total byte count without downloading any
weights, which is how this file was written without renting a frontier node.

    architecture      GlmMoeDsaForCausalLM  (model_type: glm_moe_dsa)
    parameters        ~755B total, 8 of 256 experts active per token
    weights on disk   755.7 GB fp8   /   1,506.7 GB bf16
    layers            78 (+1 multi-token-prediction layer)
    attention         MLA - compressed q/kv (q_lora_rank 2048, kv_lora_rank 512)
    sparse attention  DSA indexer, on 22 of 78 layers only
    context           1,048,576 tokens
    needs             transformers >= 5.15

The parameter split is the entire story, and it drives every decision here:

    component                       tensors    parameters
    256 experts x 76 MoE layers      58,368    ~734B   (97%)
    MLA attention x 79 layers            395    ~13B   ( 2%)
    router (gate) x 76 layers            152    tiny
    DSA indexer x 22 layers               88    tiny

A "755B model" is 97% expert MLPs, of which 8/256 fire per token. That is why
LoRA here targets **attention, not the experts** -- see `lora_target_modules`.

--------------------------------------------------------------------------
HARDWARE: THIS DOES NOT RUN ON ONE GPU, OR EIGHT SMALL ONES
--------------------------------------------------------------------------
LoRA does not reduce the memory needed to HOLD the base model; it only removes
optimizer state for the frozen weights. All 755 GB must still be resident.

    configuration              total VRAM   holds fp8 GLM-5.3?
    1 x A100 80GB                   80 GB   no  (9.4x short)
    8 x A100 80GB                  640 GB   no
    8 x H100 80GB                  640 GB   no
    8 x H200 141GB               1,128 GB   yes, with room for LoRA + activations
    8 x B200 180GB               1,440 GB   yes, comfortably

RunPod does not reliably offer 8xH200 single nodes, so **this script has never
been run against GLM-5.3 itself.** Everything below the model download is
verified against smaller GLMs on real hardware -- see the README. Nothing here
is stubbed: the same code path runs both. Only the weights differ.

`--plan` does the whole capacity calculation from `config.json` and refuses,
with arithmetic, before anything is downloaded.
"""

import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Known models. Sizes are MEASURED from the Hub API (sum of sibling blob
# sizes), not estimated, so the preflight can refuse before downloading.
# Anything not listed is looked up at runtime, and falls back to "unknown"
# rather than to a guess that would let a reader OOM after a 700 GB download.
# ---------------------------------------------------------------------------
KNOWN_MODELS = {
    "zai-org/GLM-5.3":            dict(gb=755.7, kind="moe", note="the headline model"),
    "zai-org/GLM-5.3-BF16":       dict(gb=1506.7, kind="moe", note="bf16, twice the fp8 size"),
    "zai-org/GLM-5.3-Flash":      dict(gb=328.4, kind="moe", note="multimodal (glm5_next)"),
    "zai-org/GLM-5.3-Flash-BF16": dict(gb=642.7, kind="moe", note="multimodal, bf16"),
    "zai-org/GLM-4.5-Air":        dict(gb=221.0, kind="moe", note="smaller GLM MoE"),
    "zai-org/GLM-4.7-Flash":      dict(gb=62.5, kind="moe", note="fits 1xH200 for inference"),
    "zai-org/GLM-4-9B-0414":      dict(gb=18.8, kind="dense", note="dense, 1x24GB LoRA"),
    "zai-org/glm-edge-4b-chat":   dict(gb=8.7, kind="dense", note="dense, 1x24GB comfortably"),
    "zai-org/glm-edge-1.5b-chat": dict(gb=3.2, kind="dense", note="the tested proxy"),
}

DEFAULT_MODEL = "zai-org/GLM-5.3"


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, the run gets as far as loading weights and then dies deep
    inside the training stack -- for this script, after a download that may be
    hundreds of gigabytes. The reader has already waited, and the error says
    nothing about what went wrong.

    ALLOW_CPU=1 bypasses it. Unlike the small examples in 01_basics, that is
    NOT useful here for the real model -- there is no CPU on which 755 GB of
    weights is a good idea. It exists so a reader can step through the code
    with a tiny model.
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
    print("\n  This example LoRA-fine-tunes a GLM model with DeepSpeed ZeRO-3.")
    print("  It downloads real weights and needs real GPU memory.")
    print("\n  What you CAN do right now, with no GPU and no download:")
    print("      uv run train_glm53_ds.py --plan")
    print("          the full capacity + LoRA analysis, from config.json alone")
    print("      uv run ../../tests/test_glm53_arch.py")
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


def moe_parameter_split(config: dict) -> dict:
    """
    Work out where a MoE model's parameters actually live, from config alone.

    Pure arithmetic on the published config -- no weights, no torch, no
    network. This exists because the headline parameter count of a sparse MoE
    is deeply misleading about what fine-tuning it involves: GLM-5.3's "755B"
    is 97% expert MLPs, only 8/256 of which fire for any given token.

    Returns a dict of parameter counts. Dense models return zeros for the
    expert fields, which is the correct answer rather than a special case.

    Args:
        config: the model's config.json as a dict

    Returns:
        dict with keys: experts, shared_experts, dense_mlp, attention, router,
        embedding, total, expert_fraction
    """
    h = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)

    n_experts = config.get("n_routed_experts", 0) or 0
    moe_inter = config.get("moe_intermediate_size", 0) or 0
    n_shared = config.get("n_shared_experts", 0) or 0
    dense_first = config.get("first_k_dense_replace", 0) or 0
    inter = config.get("intermediate_size", 0) or 0

    moe_layers = max(0, layers - dense_first) if n_experts else 0
    dense_layers = layers - moe_layers

    # An SwiGLU MLP is three matrices: gate, up, down.
    per_expert = 3 * h * moe_inter
    experts = per_expert * n_experts * moe_layers
    shared = per_expert * n_shared * moe_layers
    dense_mlp = 3 * h * inter * dense_layers

    # Attention. MLA (DeepSeek-style, which GLM-5.3 uses) compresses q and kv
    # through low-rank projections, so it is NOT 4 * h * h like vanilla MHA.
    q_lora = config.get("q_lora_rank") or 0
    kv_lora = config.get("kv_lora_rank") or 0
    heads = config.get("num_attention_heads", 0)
    qk_head = config.get("qk_head_dim") or config.get("head_dim", 0)
    v_head = config.get("v_head_dim") or config.get("head_dim", 0)
    rope = config.get("qk_rope_head_dim", 0)
    nope = config.get("qk_nope_head_dim", 0)

    if q_lora and kv_lora:
        q_a = h * q_lora
        q_b = q_lora * heads * qk_head
        kv_a = h * (kv_lora + rope)
        kv_b = kv_lora * heads * (nope + v_head)
        o = heads * v_head * h
        per_attn = q_a + q_b + kv_a + kv_b + o
    else:
        # vanilla multi-head / GQA
        kv_heads = config.get("num_key_value_heads", heads) or heads
        hd = qk_head or (h // heads if heads else 0)
        per_attn = h * heads * hd + 2 * h * kv_heads * hd + heads * hd * h

    attention = per_attn * layers
    router = h * n_experts * moe_layers
    embedding = vocab * h * (1 if config.get("tie_word_embeddings") else 2)

    total = experts + shared + dense_mlp + attention + router + embedding
    return dict(experts=experts, shared_experts=shared, dense_mlp=dense_mlp,
                attention=attention, router=router, embedding=embedding,
                total=total,
                expert_fraction=(experts / total) if total else 0.0)


def kv_cache_bytes_per_token(config: dict, dtype_bytes: int = 2) -> dict:
    """
    KV cache cost per token, for MLA versus what vanilla attention would cost.

    This is where MLA actually pays, and it is worth being precise because the
    obvious guess is wrong: MLA does NOT reduce parameter count. On GLM-5.3 the
    attention blocks are slightly LARGER than vanilla attention on the same
    dimensions would be (12.9B vs 11.8B), because 64 heads x 256 head_dim is
    2.7x the hidden size.

    What MLA compresses is the CACHE. Vanilla attention caches k and v per head
    per layer. MLA caches only the low-rank kv latent plus the RoPE part, and
    reconstructs k and v on the fly. At a 1M-token context that is the
    difference between a cache that fits on a GPU and one that does not.

    Args:
        config: the model's config.json as a dict
        dtype_bytes: 2 for bf16/fp16 cache, 1 for fp8

    Returns:
        dict with mla, vanilla, ratio (bytes per token, summed over layers)
    """
    layers = config.get("num_hidden_layers", 0)
    heads = config.get("num_attention_heads", 0)
    kv_lora = config.get("kv_lora_rank") or 0
    rope = config.get("qk_rope_head_dim", 0)
    qk_head = config.get("qk_head_dim") or config.get("head_dim", 0)
    v_head = config.get("v_head_dim") or config.get("head_dim", 0)

    # What vanilla MHA would cache: k and v, per head, per layer.
    vanilla = layers * (heads * qk_head + heads * v_head) * dtype_bytes

    if not (kv_lora and rope):
        return dict(mla=vanilla, vanilla=vanilla, ratio=1.0)

    # What MLA caches: the compressed latent, plus the decoupled RoPE key.
    mla = layers * (kv_lora + rope) * dtype_bytes
    return dict(mla=mla, vanilla=vanilla,
                ratio=(vanilla / mla) if mla else 1.0)


def lora_target_modules(config: dict) -> list:
    """
    Choose LoRA target modules from the architecture, not from a hardcoded list.

    The names below were verified against GLM-5.3's published
    `model.safetensors.index.json`, which lists every tensor name without
    downloading weights.

    Why attention and NOT the experts, which are 97% of the parameters:

      1. An adapter on expert k only receives gradient when the router sends a
         token to expert k. With 256 experts and top-8 routing, any one expert
         sees roughly 3% of tokens, so 256 adapters each train on a sliver of
         the data and most stay near their initialisation. You would add
         hundreds of thousands of adapter matrices to fine-tune badly.
      2. Attention is shared by every token on every layer. One adapter there
         sees the whole dataset.
      3. It is 2% of the parameters, so the adapter is small.

    The router (`mlp.gate`) is deliberately left frozen. Training it changes
    which experts fire, which is a far more destructive edit than changing how
    they are read -- routing collapse is the classic way a fine-tuned MoE
    quietly degrades.

    Args:
        config: the model's config.json as a dict

    Returns:
        list of module-name suffixes for peft's `target_modules`
    """
    mt = config.get("model_type", "")

    # MLA-style attention (GLM-5.3's glm_moe_dsa, DeepSeek-V3 lineage).
    if config.get("q_lora_rank") and config.get("kv_lora_rank"):
        return ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                "o_proj"]

    # Vanilla attention: GLM-4 / glm-edge and most dense decoders.
    if mt.startswith("glm") or mt in ("llama", "mistral", "qwen2", "qwen3"):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Unknown architecture. Return the common denominator rather than
    # guessing wide -- peft raises a clear error if none match, which is a
    # better outcome than silently adapting nothing.
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def verify_architecture(model: str, targets: list, token: str = None) -> bool:
    """
    Build the model on the META device and check the LoRA targets resolve.

    This is the strongest check available without renting the hardware. The
    meta device allocates NO memory and downloads NO weights: transformers
    constructs the full module tree from config.json alone, so a 743B model
    materialises in a second or two. What it proves:

      * your installed transformers actually implements this architecture
        (GLM-5.3 needs >= 5.15; an older one raises KeyError on model_type)
      * the LoRA target names exist in the REAL module tree, not just in the
        checkpoint's tensor names -- these differ, see below
      * the parameter count agrees with the arithmetic in moe_parameter_split

    Worth doing before renting an 8xH200 node, because all three failures
    otherwise surface AFTER a 755 GB download.

    A finding this check surfaced: GLM-5.3's 256 experts are FUSED into 3D
    parameter tensors at runtime -- `mlp.experts.gate_up_proj` has shape
    (256, 4096, 6144) -- even though the checkpoint stores them per expert as
    `mlp.experts.{k}.gate_proj`. They are therefore not nn.Linear modules, and
    stock peft cannot attach LoRA to them at all. Freezing the experts is not
    merely the better choice here; with standard tooling it is the only
    expressible one.

    Args:
        model: HuggingFace model id
        targets: LoRA target module suffixes to look for
        token: optional HF token

    Returns:
        True when every target resolves and the architecture builds.
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
        print(f"    {t:<22} {'FOUND' if hits else 'MISSING':<8} {hits:>4} instances")
        if not hits:
            ok = False

    n_par = sum(p.numel() for p in net.parameters())
    print(f"  parameters: {n_par/1e9:,.2f} B (transformers) vs "
          f"{moe_parameter_split(cfg.to_dict())['total']/1e9:,.2f} B (config arithmetic)")

    if not ok:
        print("\n  A target did not resolve. peft would attach LoRA to NOTHING,")
        print("  and training would still run and still show a falling loss.")
    return ok


def capacity_report(model_gb: float, num_gpus: int, vram_gb: float) -> dict:
    """
    Can this hardware hold this model, with LoRA?

    LoRA freezes the base weights, which removes optimizer state (~12 bytes per
    trainable parameter for Adam in mixed precision) but does NOT reduce what
    it costs to HOLD the model. That distinction is the one people get wrong
    when they assume LoRA makes any model fit any GPU.

    The 1.2 factor covers activations, the adapter, gradients and fragmentation.
    It is deliberately crude: this is a refuse/proceed gate, not a planner.

    Args:
        model_gb: weights on disk, in GB
        num_gpus: number of GPUs
        vram_gb: VRAM per GPU, in GB

    Returns:
        dict with total_vram, needed, fits, shortfall_x, gpus_needed
    """
    total = num_gpus * vram_gb
    needed = model_gb * 1.2
    return dict(
        total_vram=total,
        needed=needed,
        fits=total >= needed,
        shortfall_x=(needed / total) if total else float("inf"),
        gpus_needed=math.ceil(needed / vram_gb) if vram_gb else 0,
    )


def fetch_config(model: str, token: str = None) -> dict:
    """
    Fetch a model's config.json from the Hub without downloading weights.

    Uses huggingface_hub if present, else a plain HTTPS GET, so `--plan` works
    in a bare environment.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, "config.json", token=token)
        with open(path) as f:
            return json.load(f)
    except Exception:
        import urllib.request
        url = f"https://huggingface.co/{model}/raw/main/config.json"
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)


def model_size_gb(model: str, token: str = None) -> float:
    """
    Total weight size in GB. Measured from the Hub API, not estimated.

    Returns 0.0 when it cannot be determined -- callers must treat that as
    "unknown", never as "small".
    """
    if model in KNOWN_MODELS:
        return KNOWN_MODELS[model]["gb"]
    try:
        import urllib.request
        url = f"https://huggingface.co/api/models/{model}?blobs=true"
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        total = sum(s.get("size") or 0 for s in data.get("siblings", []))
        return total / 1e9
    except Exception:
        return 0.0


def print_plan(args, config: dict, size_gb: float, num_gpus: int,
               vram_gb: float) -> bool:
    """
    Print the architecture and capacity analysis. Returns True if it fits.

    This is the part that needs no GPU and no download, and it is where the
    interesting content of this example lives.
    """
    bar = "=" * 78
    split = moe_parameter_split(config)
    targets = lora_target_modules(config)
    cap = capacity_report(size_gb, num_gpus, vram_gb) if size_gb else None

    print(bar)
    print(f"  {args.model}")
    print(bar)
    print(f"  model_type        {config.get('model_type')}")
    print(f"  architecture      {(config.get('architectures') or ['?'])[0]}")
    print(f"  layers            {config.get('num_hidden_layers')}")
    print(f"  hidden size       {config.get('hidden_size')}")
    print(f"  context           {config.get('max_position_embeddings'):,}")
    if config.get("n_routed_experts"):
        print(f"  experts           {config['n_routed_experts']} routed, "
              f"top-{config.get('num_experts_per_tok')} per token"
              f"  ({config.get('n_shared_experts', 0)} shared, always on)")
    if config.get("q_lora_rank"):
        print(f"  attention         MLA  q_lora_rank={config['q_lora_rank']}  "
              f"kv_lora_rank={config['kv_lora_rank']}")
    if config.get("index_topk"):
        print(f"  sparse attention  DSA indexer, top-{config['index_topk']} keys")
    q = (config.get("quantization_config") or {}).get("quant_method")
    if q:
        print(f"  quantization      {q}")

    print(bar)
    print("  Where the parameters actually are")
    print(bar)
    order = [("experts", "routed experts"), ("shared_experts", "shared experts"),
             ("dense_mlp", "dense MLP layers"), ("attention", "attention"),
             ("router", "router (gate)"), ("embedding", "embeddings")]
    for key, label in order:
        v = split[key]
        if v <= 0:
            continue
        pct = 100.0 * v / split["total"] if split["total"] else 0
        print(f"    {label:<20} {v/1e9:>9.2f} B   {pct:>5.1f}%")
    print(f"    {'TOTAL':<20} {split['total']/1e9:>9.2f} B")

    kv = kv_cache_bytes_per_token(config)
    if kv["ratio"] > 1.5:
        ctx = config.get("max_position_embeddings", 0)
        print()
        print(f"    KV cache/token       {kv['mla']/1024:.1f} KB  (MLA)")
        print(f"    vanilla would be     {kv['vanilla']/1024:.1f} KB  "
              f"-> {kv['ratio']:.0f}x larger")
        if ctx:
            print(f"    at the full {ctx:,}-token context: "
                  f"{kv['mla']*ctx/1e9:.0f} GB vs {kv['vanilla']*ctx/1e9:,.0f} GB")
        print("    MLA does NOT save parameters -- attention here is slightly")
        print("    LARGER than vanilla would be. It saves the CACHE, which is")
        print("    what makes a 1M-token context possible at all.")

    if split["expert_fraction"] > 0.5:
        print()
        print(f"  {split['expert_fraction']*100:.0f}% of this model is expert MLPs, and only "
              f"{config.get('num_experts_per_tok')}/{config.get('n_routed_experts')}")
        print("  of them fire per token. The headline parameter count says very")
        print("  little about what a forward pass costs -- or about what is")
        print("  worth adapting.")

    print(bar)
    print("  LoRA plan")
    print(bar)
    print(f"    target modules    {', '.join(targets)}")
    print(f"    rank / alpha      {args.lora_rank} / {args.lora_alpha}")
    if split["experts"] > 0:
        print("    experts           FROZEN, on purpose")
        print("      An adapter on expert k only trains when the router picks k.")
        print(f"      At top-{config.get('num_experts_per_tok')} of "
              f"{config.get('n_routed_experts')}, each expert sees ~"
              f"{100.0*config.get('num_experts_per_tok',0)/max(1,config.get('n_routed_experts',1)):.0f}% of tokens.")
        print("    router            FROZEN, on purpose")
        print("      Training it changes WHICH experts fire; routing collapse is")
        print("      the classic way a fine-tuned MoE quietly degrades.")

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
            print(f"    verdict           FITS")
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
                        "(755 GB — see --plan). Use "
                        "zai-org/glm-edge-1.5b-chat to exercise the same code "
                        "path on hardware you can rent.")
    p.add_argument("--verify-arch", action="store_true",
                   help="Build the model on the meta device and check the LoRA "
                        "targets resolve against the real module tree. No GPU, "
                        "no weight download. Do this before renting.")
    p.add_argument("--plan", action="store_true",
                   help="Print the architecture, LoRA and capacity analysis, "
                        "then exit. Needs NO GPU and downloads no weights.")
    p.add_argument("--dataset", default="tatsu-lab/alpaca",
                   help="Instruction dataset for the SFT stage.")
    p.add_argument("--max-samples", type=int, default=512,
                   help="Cap the dataset. This example is about the mechanics "
                        "of fine-tuning a sparse MoE, not about producing a "
                        "good model.")
    p.add_argument("--max-length", type=int, default=512,
                   help="Token cap per example. GLM-5.3 supports 1M context; "
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
                   help="Full fine-tune. For GLM-5.3 this needs roughly 12x "
                        "the memory for optimizer state alone — about 9 TB. "
                        "Exists to make the LoRA argument concrete.")
    p.add_argument("--vram-gb", type=float, default=0.0,
                   help="Override detected VRAM per GPU, for --plan on a "
                        "machine that is not the target.")
    p.add_argument("--num-gpus", type=int, default=0,
                   help="Override detected GPU count, for --plan.")
    p.add_argument("--force", action="store_true",
                   help="Proceed even when the capacity check says it will not "
                        "fit. You will OOM; this exists so the refusal is not "
                        "a wall.")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--prompt", default="Explain what a mixture-of-experts "
                                       "layer does, in two sentences.")
    p.add_argument("--output", default="./glm53-lora-out")
    p.add_argument("--deepspeed", default="ds_config_glm53.json")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Set by the deepspeed launcher; accepted and ignored.")
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    bar = "=" * 78
    hf_token = os.environ.get("HF_TOKEN")

    # ---- stage 0: the analysis, which needs neither GPU nor weights --------
    config = fetch_config(args.model, hf_token)
    size_gb = model_size_gb(args.model, hf_token)

    if args.plan:
        # Deliberately BEFORE require_gpu(): the whole point is that a reader
        # with a laptop can see what this model implies.
        n = args.num_gpus or 8
        v = args.vram_gb or 141.0
        print_plan(args, config, size_gb, n, v)
        print("  (assumed hardware; override with --num-gpus / --vram-gb)")
        return

    if args.verify_arch:
        # Also deliberately before require_gpu(): the entire point is that it
        # needs no GPU.
        print(bar)
        print(f"  Architecture verification: {args.model}")
        print(bar)
        ok = verify_architecture(args.model, lora_target_modules(config), hf_token)
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
            print("  it AFTER a multi-hundred-gigabyte download.")
            print()
            print("  Your options:")
            print("    --model zai-org/glm-edge-1.5b-chat   same code, 3.2 GB")
            print("    --model zai-org/GLM-4-9B-0414        dense GLM, 18.8 GB")
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
    # produces rows that are pure boilerplate with no content, and training
    # proceeds perfectly happily on nothing at all.
    sample = ds["text"][:64]
    hollow = sum(1 for t in sample
                 if len(t.replace("### Instruction:", "")
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
        # A tokenizer with no pad token dies on the first padded batch. GLM
        # models set pad_token_id in config; many others do not.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, trust_remote_code=True,
        dtype=torch.bfloat16)
    model.config.use_cache = False          # incompatible with grad checkpointing

    # ---- stage 3: LoRA + train --------------------------------------------
    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig
        targets = lora_target_modules(config)
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets)
        if is_main:
            print(f"  LoRA targets: {', '.join(targets)}")

    if use_wandb and is_main:
        wandb.init(project="deepspeed-course-glm53",
                   name=f"{args.model.split('/')[-1]}-lora",
                   config=dict(model=args.model, dataset=args.dataset,
                               lora_rank=args.lora_rank, lr=args.lr))

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
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds,
                         peft_config=peft_config)
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
        # generate() on rank 0 alone would read partial tensors. Keeping
        # generation to a single-process run is the honest simple answer;
        # doing it properly needs the ZeRO-3 gather context.
        if is_main:
            print("  Skipped: ZeRO-3 shards the weights across ranks.")
            print("  Generate from the saved adapter in a single process:")
            print(f"      deepspeed --num_gpus=1 {sys.argv[0]} --skip-inference ...")
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
            "tokenizer, the adapter merge, or the prompt format.")
    print(bar)

    if use_wandb and is_main:
        wandb.finish()


if __name__ == "__main__":
    main()

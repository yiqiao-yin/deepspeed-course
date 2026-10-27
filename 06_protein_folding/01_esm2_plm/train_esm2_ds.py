#!/usr/bin/env python3
"""
Fine-tune ESM-2: the LLM stack, applied to a 20-letter alphabet.

    deepspeed --num_gpus=1 train_esm2_ds.py --task ss
    deepspeed --num_gpus=1 train_esm2_ds.py --task ss --model 650M --use-lora
    deepspeed --num_gpus=1 train_esm2_ds.py --task mlm --model 35M

The on-ramp to `06_protein_folding/`. Everything you know about fine-tuning
BERT applies here unchanged -- ESM-2 *is* BERT, trained on UniRef instead of
Wikipedia. `uv run plm.py` covers the objective and the labels on CPU first.

The model ladder, and the question it raised
---------------------------------------------
    --model    checkpoint                  params       format
    8M         esm2_t6_8M_UR50D         7,512,474       safetensors
    35M        esm2_t12_35M_UR50D      33,995,044       safetensors
    150M       esm2_t30_150M_UR50D    148,798,300       safetensors
    650M       esm2_t33_650M_UR50D    652,358,616       safetensors
    3B         esm2_t36_3B_UR50D               3B       .bin ONLY, 2 shards

The 3B checkpoint predates safetensors. That was an open question when this
lab was designed, because transformers 5.x writes safetensors exclusively and
ignores `safe_serialization=False`. **Reading legacy checkpoints still works**:
verified on transformers 5.16.1, the version this lab LOCKS, for both the
single-file and the sharded-plus-index layouts, so the 3B rung is offered. If a future
transformers drops the reader, the ladder stops at 650M and this is the
docstring to change.

What the task is
----------------
`--task ss` predicts **3-state secondary structure per residue** -- helix,
strand, or neither -- from sequence alone. The labels are DERIVED from
backbone dihedral angles rather than downloaded, which keeps one CC-BY-4.0
data spine across the whole section and means the reader can see where a
label comes from. `plm.py` has the derivation and its validation.

`--task mlm` is continued masked-language pretraining, 80/10/10, as ESM-2 was
trained.

The two numbers that matter
---------------------------
**Compare against the majority class, never against 1/3.** Secondary
structure is unbalanced, so a model that always predicts the most common
class scores about 0.50 here while learning nothing. Every run prints the
majority-class baseline beside the accuracy and the margin between them.

**Watch the label/token alignment.** ESM-2 prepends `<cls>`, so residue i is
at token i+1. An off-by-one there still trains, still converges, and costs a
few points of accuracy -- there is no error anywhere.
`tests/test_ss_derivation.py` asserts the alignment for exactly that reason.

Hardware
--------
Declared 24 GB, 1 GPU. 150M fits comfortably; 650M wants `--use-lora`; the
3B rung needs LoRA and is the one that will OOM if you ask for full
fine-tuning.
"""

import argparse
import os
import sys


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    # Imported locally so this helper stays self-contained and can be copied
    # between example scripts unchanged.
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            ds_config.json also needs \"torch_adam\": true and "
              "bf16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.")
        print("            --ds-evoformer-attn cannot work on CPU and will be "
              "ignored.\n")
        return

    print("\n" + "=" * 78)
    print("  NO GPU DETECTED -- this training run needs CUDA")
    print("=" * 78)
    print("""
  Why it stopped
      Fine-tuning a protein language model is a GPU job, and DeepSpeed's
      optimizer needs CUDA to build at all.

  What you CAN do right now, on this machine, with no GPU
      uv run plm.py              the objective and the labels: 80/10/10
                                 masking, and secondary structure derived
                                 from backbone dihedrals with its own
                                 Ramachandran validation

      Both run on CPU in about a minute and are where most of the teaching is.
      From the repository root, ./tests/run_all.sh runs every logic test.

  How to get a GPU
      uv run runpod/runpod_ctl.py run 06_protein_folding/01_esm2_plm \\
          --collect --wait --terminate --yes

      24 GB is enough for the defaults. Confirm the pod is gone afterwards
      with `uv run runpod/runpod_ctl.py pods`.

  To step through this script on CPU anyway
      ALLOW_CPU=1 python train_esm2_ds.py --max-steps 2
""")
    print("=" * 78 + "\n")
    sys.exit(1)




MODELS = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B":   "facebook/esm2_t36_3B_UR50D",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ESM-2 on masked LM or derived secondary "
                    "structure."
    )
    p.add_argument("--model", choices=tuple(MODELS), default="150M",
                   help="3B ships .bin-only checkpoints; see the module "
                        "docstring")
    p.add_argument("--task", choices=("mlm", "ss"), default="ss",
                   help="mlm = continued pretraining; ss = per-residue "
                        "3-state secondary structure, labels DERIVED from "
                        "backbone geometry")
    p.add_argument("--use-lora", action="store_true",
                   help="LoRA on the attention projections -- the only way "
                        "the 3B rung fits a 24 GB card")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--train-chains", type=int, default=2048)
    p.add_argument("--eval-chains", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total optimizer steps (for a cheap dry run)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    p.add_argument("--local_rank", type=int, default=-1)
    # parse_known_args because the deepspeed launcher injects its own flags.
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    require_gpu()                      # FIRST -- before torch or deepspeed

    # Heavy imports after the preflight.
    import json
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    import deepspeed
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

    from plm import SS_LABELS, backbone_dihedrals, secondary_structure

    try:
        import wandb
        HAVE_WANDB = True
    except ImportError:
        HAVE_WANDB = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", max(args.local_rank, 0)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    def log(msg: str = "") -> None:
        if is_main:
            print(msg, flush=True)

    checkpoint = MODELS[args.model]
    log("=" * 78)
    log("  ESM-2: A PROTEIN IS A SEQUENCE")
    log("=" * 78)
    log(f"  checkpoint     : {checkpoint}")
    log(f"  task           : {args.task}")
    log(f"  LoRA           : {'on' if args.use_lora else 'off'}")
    log(f"  max length     : {args.max_length}")
    log("=" * 78)

    # ---------------------------------------------------------------- data
    from cath_sequences import load_split

    if is_main:
        rows = load_split("train")
    if world_size > 1:
        # No timeout= : torch 2.11 (what this lab locks) rejects it.
        torch.distributed.barrier(device_ids=[local_rank])
    if not is_main:
        rows = load_split("train")
    eval_rows = load_split("validation")[: args.eval_chains]
    rows = rows[: args.train_chains]

    tok = AutoTokenizer.from_pretrained(checkpoint)

    def build(split_rows):
        """Tokenise, and derive per-residue labels aligned to the tokens."""
        seqs, labels = [], []
        for row in split_rows:
            L = int(row["length"])
            seq = row["seq"][:args.max_length - 2]
            coords = np.asarray(row["coords"], dtype=np.float64).reshape(L, 4, 3)
            mask = np.asarray(row["mask"], dtype=bool)
            phi, psi = backbone_dihedrals(coords[:, 0], coords[:, 1],
                                          coords[:, 2])
            phi[~mask], psi[~mask] = np.nan, np.nan
            ss = secondary_structure(phi, psi)[: len(seq)]
            seqs.append(seq)
            labels.append(ss)

        enc = tok(seqs, padding="max_length", truncation=True,
                  max_length=args.max_length, return_tensors="pt")

        # Align labels to tokens. ESM-2 prepends <cls> and appends <eos>, so
        # residue i is at token i+1. Everything that is not a real residue
        # gets -100 so the loss skips it -- special tokens AND padding.
        #
        # Getting this off by one is the classic per-residue bug: it still
        # trains, the accuracy is merely a few points worse, and nothing
        # anywhere raises. test_ss_derivation.py asserts the alignment.
        y = torch.full_like(enc["input_ids"], -100)
        for i, ss in enumerate(labels):
            n = min(len(ss), args.max_length - 2)
            y[i, 1:1 + n] = torch.from_numpy(ss[:n])
        return torch.utils.data.TensorDataset(
            enc["input_ids"], enc["attention_mask"], y)

    train_ds, eval_ds = build(rows), build(eval_rows)
    log(f"  train chains   : {len(train_ds)}")
    log(f"  held-out chains: {len(eval_ds)}")

    # Majority-class baseline. A per-residue classifier beats 0.333 by
    # predicting the most common class and learning nothing; this is the
    # number that actually has to be beaten.
    all_y = torch.cat([eval_ds[i][2] for i in range(len(eval_ds))])
    real = all_y[all_y != -100]
    counts = torch.bincount(real, minlength=len(SS_LABELS)).float()
    majority = (counts.max() / counts.sum()).item()
    log(f"  class balance  : " + "  ".join(
        f"{s} {c / counts.sum():.3f}" for s, c in zip(SS_LABELS, counts)))
    log(f"  majority class : {majority:.3f}  <- the number to beat")

    # --------------------------------------------------------------- model
    if args.task == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    else:
        model = _make_ss_classifier(AutoModel.from_pretrained(checkpoint),
                                    n_classes=len(SS_LABELS))

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        target = ["query", "key", "value"]
        base = model.encoder if args.task == "ss" else model
        peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                              target_modules=target, bias="none")
        wrapped = get_peft_model(base, peft_cfg)
        if args.task == "ss":
            model.encoder = wrapped
        else:
            model = wrapped
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  LoRA trainable : {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)")
    else:
        log(f"  parameters     : "
            f"{sum(p.numel() for p in model.parameters()):,}")

    with open(args.deepspeed_config) as fh:
        ds_config = json.load(fh)
    _check_cuda_toolkit(ds_config, log)

    engine, _, train_loader, _ = deepspeed.initialize(
        model=model, model_parameters=[p for p in model.parameters()
                                       if p.requires_grad],
        training_data=train_ds, config=ds_config)
    device = engine.device

    # ------------------------------------------------------------ training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step, capped = 0, False
    t0 = time.time()
    log("\n" + "-" * 78)
    for _ in range(args.epochs):
        for ids, attn, y in train_loader:
            if args.max_steps is not None and step >= args.max_steps:
                capped = True
                break
            ids, attn, y = ids.to(device), attn.to(device), y.to(device)
            if args.task == "mlm":
                ids_in, labels = _mask_batch(ids, attn, tok, device)
                loss = engine(input_ids=ids_in, attention_mask=attn,
                              labels=labels).loss
            else:
                logits = engine(input_ids=ids, attention_mask=attn)
                loss = F.cross_entropy(logits.float().view(-1, len(SS_LABELS)),
                                       y.view(-1), ignore_index=-100)
            engine.backward(loss)
            engine.step()
            step += 1
            if is_main and step % 50 == 0:
                log(f"  step {step:>5}   loss {loss.item():.4f}")
        if capped:
            break
    elapsed = time.time() - t0

    # ---------------------------------------------------------- evaluation
    engine.eval()
    correct = total_n = 0
    losses = []
    with torch.no_grad():
        for ids, attn, y in torch.utils.data.DataLoader(eval_ds, batch_size=8):
            ids, attn, y = ids.to(device), attn.to(device), y.to(device)
            if args.task == "mlm":
                ids_in, labels = _mask_batch(ids, attn, tok, device)
                out = engine(input_ids=ids_in, attention_mask=attn,
                             labels=labels)
                losses.append(out.loss.item())
                pred = out.logits.argmax(-1)
                keep = labels != -100
            else:
                logits = engine(input_ids=ids, attention_mask=attn).float()
                losses.append(F.cross_entropy(
                    logits.view(-1, len(SS_LABELS)), y.view(-1),
                    ignore_index=-100).item())
                pred = logits.argmax(-1)
                keep = y != -100
                labels = y
            correct += (pred[keep] == labels[keep]).sum().item()
            total_n += int(keep.sum().item())

    acc = correct / max(total_n, 1)
    mean_loss = float(np.mean(losses))
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9
               if torch.cuda.is_available() else 0.0)

    log("\n" + "=" * 78)
    log("  RESULTS  (held-out chains)")
    log("=" * 78)
    log(f"  loss           : {mean_loss:.4f}")
    log(f"  accuracy       : {acc:.4f}")
    if args.task == "ss":
        log(f"  majority class : {majority:.4f}   <- the number to beat")
        log(f"  margin         : {acc - majority:+.4f}")
    else:
        log(f"  random baseline: {1 / 20:.4f}   (20 amino acids)")
    log(f"  steps          : {step}")
    log(f"  wall clock     : {elapsed:.1f}s")
    if torch.cuda.is_available():
        log(f"  peak GPU memory: {peak_gb:.2f} GB")

    if capped:
        log("\n  THIS WAS A CAPPED RUN.")
        log(f"  --max-steps stopped it after {step} optimizer steps, which is")
        log("  a smoke test of the plumbing, not a trained model. Do not read")
        log("  the accuracy above as a result.")
    elif args.task == "ss" and acc <= majority:
        log("\n  The model did NOT beat predicting the majority class. It has")
        log("  learned nothing useful. Before tuning: check the label/token")
        log("  alignment -- residue i sits at token i+1 because of <cls>, and")
        log("  an off-by-one there trains fine and scores like this.")
    elif args.task == "ss":
        log(f"\n  Clear of the majority-class baseline by "
            f"{100 * (acc - majority):.1f} points.")

    if HAVE_WANDB and os.environ.get("WANDB_API_KEY") and is_main:
        wandb.init(project="deepspeed-course-esm2", config=vars(args))
        wandb.log({"accuracy": acc, "loss": mean_loss,
                   "majority_baseline": majority, "peak_gb": peak_gb})
        wandb.finish()

    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


def _make_ss_classifier(encoder, n_classes: int):
    """
    An ESM-2 encoder with a per-residue linear head.

    Built by a factory rather than declared at module scope because
    `nn.Module` needs torch, and torch must not be imported before
    `require_gpu()` runs -- otherwise a CPU-only reader gets a CUDA traceback
    instead of the preflight message.
    """
    import torch.nn as nn

    class SSClassifier(nn.Module):
        def __init__(self, encoder, n_classes):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(0.1)
            self.head = nn.Linear(encoder.config.hidden_size, n_classes)

        def forward(self, input_ids, attention_mask=None):
            h = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
            return self.head(self.dropout(h))

    return SSClassifier(encoder, n_classes)


def _mask_batch(ids, attn, tok, device):
    """BERT 80/10/10 masking on a batch, skipping specials and padding."""
    import torch

    labels = ids.clone()
    special = torch.zeros_like(ids, dtype=torch.bool)
    for sid in tok.all_special_ids:
        special |= ids == sid
    sel = (torch.rand(ids.shape, device=device) < 0.15) & ~special & attn.bool()
    labels[~sel] = -100

    out = ids.clone()
    r = torch.rand(ids.shape, device=device)
    out[sel & (r < 0.8)] = tok.mask_token_id
    rand = sel & (r >= 0.8) & (r < 0.9)
    out[rand] = torch.randint(4, tok.vocab_size, (int(rand.sum()),),
                              device=device)
    return out, labels


def _check_cuda_toolkit(ds_config: dict, log) -> None:
    """
    Fail fast, and usefully, when there is a GPU but no CUDA toolkit.

    Skipped when the config already asks for torch's own Adam, because that
    path compiles nothing and works fine without nvcc.
    """
    import shutil
    import sys

    wants_torch_adam = (
        ds_config.get("optimizer", {}).get("params", {}).get("torch_adam")
        is True
    )
    if wants_torch_adam or shutil.which("nvcc"):
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:                                       # noqa: BLE001
        CUDA_HOME = None
    if CUDA_HOME:
        return

    log("\n" + "=" * 78)
    log("  GPU FOUND, BUT NO CUDA TOOLKIT -- DeepSpeed cannot build FusedAdam")
    log("=" * 78)
    log("""
  Why this is not caught by the GPU check above
      torch.cuda.is_available() is True: the PyTorch wheels ship their own
      CUDA runtime, so tensors and training work fine. DeepSpeed's FusedAdam
      is different -- it JIT-COMPILES a CUDA extension, which needs `nvcc`
      and CUDA_HOME. Neither is present here.

      Left alone, this run would die inside deepspeed.initialize() with
      `OSError: CUDA_HOME environment variable is not set`, raised from
      torch/utils/cpp_extension.py, after the data was built and the model
      constructed.

  Two ways forward

      1. Use torch's optimizer instead of the fused one. Nothing is compiled,
         and for a ~100k-parameter trunk the speed difference is noise:

             "optimizer": { "type": "AdamW",
                            "params": { ..., "torch_adam": true } }

         Everything this lab teaches works on that path, including the
         memory comparisons.

      2. Install a CUDA toolkit matching your driver, then set CUDA_HOME.
         Required for --ds-evoformer-attn, which compiles CUTLASS and has
         no pure-PyTorch fallback worth the name.
""")
    log("=" * 78 + "\n")
    sys.exit(1)



if __name__ == "__main__":
    main()

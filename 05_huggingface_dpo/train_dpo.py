"""
Offline preference optimization with TRL + DeepSpeed: DPO, IPO, CPO, ORPO, SimPO, KTO.

WHAT THIS TOPIC IS
------------------
`05_huggingface_trl` does supervised fine-tuning: maximise the likelihood of a
reference answer. That can only ever say *"this output was good"*. It has no way
to say *"this one was better than that one"*, and no way at all to push
probability DOWN -- a maximum-likelihood objective has no mechanism for it.

This folder adds the missing downward force, using preference pairs and no
reinforcement learning. `06_huggingface_grpo` next door is the RL answer; this
is the cheaper one, and for most alignment work it is the right one.

THE FAMILY, BY WHAT EACH ONE DELETES
------------------------------------
Full RLHF holds four models: policy, critic, reward model, reference. Every
method here is an argument about which you can drop.

    DPO    May 2023   deletes the REWARD MODEL   (and the rollouts)
    IPO    Oct 2023   -- bounds DPO's objective so it stops overfitting
    CPO    Jan 2024   deletes the REFERENCE MODEL
    KTO    Feb 2024   deletes the need for PAIRED data
    ORPO   Mar 2024   deletes the REFERENCE MODEL and the separate SFT stage
    SimPO  May 2024   deletes the REFERENCE MODEL and the length bias

Note that GRPO deletes the CRITIC, which is a different component again.
"DPO removes the reward model" and "GRPO removes the critic" are two different
sentences, and conflating them is the most common confusion in this area.

WHY THE `--method` FLAG COVERS ALL SIX
---------------------------------------
TRL routes them through three trainers, not six, because most of these are the
same optimisation with a different scalar function:

    DPOTrainer   loss_type = sigmoid | ipo | hinge | robust | sigmoid_norm ...
    CPOTrainer   loss_type = sigmoid | ipo | simpo | alphapo   (+ cpo_alpha)
    ORPOTrainer  its own objective (NLL + odds-ratio)
    KTOTrainer   unpaired data

Two API details that are easy to get wrong and are handled below:

  * SimPO is **not** a `DPOTrainer` loss_type. It is
    `CPOConfig(loss_type="simpo", cpo_alpha=0.0)` plus `simpo_gamma`. Setting
    `cpo_alpha` to anything but 0 gives you CPO-SimPO, a different method.
  * `CPOTrainer` and `BCOTrainer` live under `trl.experimental` in current TRL.
    The import below tries both locations rather than pinning one.

MEMORY
------
    Qwen3-0.6B  + LoRA  ~12 GB   one card, comfortably
    Qwen2.5-7B  + LoRA  ~24 GB   with a reference model
    Qwen2.5-7B  + LoRA  ~18 GB   reference-free (CPO/ORPO/SimPO)

The reference model is the swing factor: a second frozen copy of the weights,
about 14 GB at 7B in bf16. Two ways to avoid paying it, in order of preference:

  1. **LoRA.** The reference is the base weights with the adapter disabled --
     no second copy at all. This is usually a better move than switching
     objectives to save memory.
  2. **Pick a reference-free method** (CPO, ORPO, SimPO).

VERIFY THE OBJECTIVES ON CPU FIRST
-----------------------------------
`preference_losses.py` in this folder implements all six as plain tensor maths,
no GPU and no download. Run that before renting anything:

    uv run preference_losses.py
    uv run ../tests/test_preference_losses.py     # 58 checks

RUNNING IT
----------
CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 05_huggingface_dpo \\
                            --dry-run --collect --wait --terminate --yes

    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    uv pip install deepspeed transformers trl peft accelerate datasets
"""

import argparse
import os
import sys

# Optional experiment tracking. Absent is fine; the run just skips it.
try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Which TRL trainer each method routes through, and the config it needs.
# Kept as data rather than an if-chain so the mapping is readable and so
# `--list-methods` can print it without importing torch.
METHODS = {
    "dpo":   dict(trainer="dpo",  loss_type="sigmoid",     needs_ref=True,
                  note="The founding paper. Start here."),
    "ipo":   dict(trainer="dpo",  loss_type="ipo",         needs_ref=True,
                  note="Bounded objective; use when DPO margins run away."),
    "robust": dict(trainer="dpo", loss_type="robust",      needs_ref=True,
                   note="DPO with label noise; set --label-smoothing."),
    "cpo":   dict(trainer="cpo",  loss_type="sigmoid",     needs_ref=False,
                  note="Reference-free DPO. Keeps an SFT term via cpo_alpha."),
    "simpo": dict(trainer="cpo",  loss_type="simpo",       needs_ref=False,
                  note="Reference-free AND length-normalised. cpo_alpha=0."),
    "orpo":  dict(trainer="orpo", loss_type=None,          needs_ref=False,
                  note="One stage: folds SFT and alignment into one objective."),
    "kto":   dict(trainer="kto",  loss_type=None,          needs_ref=True,
                  note="UNPAIRED data — needs a `label` column, not chosen/rejected."),
}


def require_gpu() -> None:
    """
    Stop with a clear message when no CUDA device is available.

    Without this, DeepSpeed gets as far as building its fused Adam kernel and
    dies with `OSError: CUDA_HOME environment variable is not set` raised from
    deep inside torch's C++ extension loader -- which tells a newcomer nothing
    about what went wrong or what to do next.

    Set ALLOW_CPU=1 to bypass.
    """
    import os   # noqa: F811
    import sys  # noqa: F811

    try:
        import torch
    except ImportError:
        print("\n[preflight] PyTorch is not installed. Install it with:")
        print("            uv pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121\n")
        sys.exit(1)

    if torch.cuda.is_available():
        return

    if os.environ.get("ALLOW_CPU") == "1":
        print("\n[preflight] No GPU detected; ALLOW_CPU=1 set, continuing.")
        print("            ds_config.json also needs \"torch_adam\": true and "
              "bf16 disabled,")
        print("            or DeepSpeed will still fail building its CUDA ops.\n")
        return

    bar = "=" * 72
    print("\n" + bar)
    print("  NO GPU DETECTED - stopping before DeepSpeed fails obscurely")
    print(bar)
    print("\n  torch.cuda.is_available() returned False.")
    print("\n  Preference optimization downloads a model and needs real GPU")
    print("  memory. The smallest configuration here (Qwen3-0.6B + LoRA) wants")
    print("  about 12 GB.")
    print("\n  The OBJECTIVES themselves run fine on CPU, and they are the part")
    print("  worth understanding — all six as plain tensor maths:")
    print("      uv run 05_huggingface_dpo/preference_losses.py")
    print("      uv run tests/test_preference_losses.py    # 58 checks, no GPU")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 05_huggingface_dpo")
    print("      uv run runpod/runpod_ctl.py run 05_huggingface_dpo \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def _import_cpo():
    """
    Import CPOTrainer from wherever this TRL version keeps it.

    It moved to `trl.experimental.cpo` in recent releases. Trying both rather
    than pinning one means this script survives the next reshuffle, and fails
    with a *useful* message rather than an ImportError traceback if it moves
    again.
    """
    try:
        from trl.experimental.cpo import CPOConfig, CPOTrainer
        return CPOConfig, CPOTrainer
    except ImportError:
        pass
    try:
        from trl import CPOConfig, CPOTrainer
        return CPOConfig, CPOTrainer
    except ImportError as exc:
        raise ImportError(
            "CPOTrainer not found in trl.experimental.cpo or trl. It has moved "
            "between releases; check `python -c \"import trl; print(dir(trl))\"` "
            "and pin a version. --method cpo and --method simpo both need it."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", default="dpo", choices=sorted(METHODS),
                        help="Which objective. See --list-methods.")
    parser.add_argument("--list-methods", action="store_true",
                        help="Print the method table and exit. Needs no GPU.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="0.6B fits ~12GB with LoRA; 7B wants ~24GB.")
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized",
                        help="Preference dataset with prompt/chosen/rejected.")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL strength. NOTE: for IPO, TRL reuses this as "
                             "the regularisation parameter tau, where SMALLER "
                             "targets a LARGER margin — the opposite direction "
                             "from DPO's beta.")
    parser.add_argument("--simpo-gamma", type=float, default=0.5,
                        help="SimPO target margin. Only used by --method simpo.")
    parser.add_argument("--cpo-alpha", type=float, default=1.0,
                        help="Weight of CPO's SFT/BC term. Forced to 0.0 for "
                             "--method simpo; anything else gives CPO-SimPO, "
                             "which is a different method.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label-flip probability for --method robust. "
                             "Must be in [0, 0.5).")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-lora", action="store_true",
                        help="Full fine-tune. Needs far more memory AND a real "
                             "second copy of the weights for the reference.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap steps; the RunPod --dry-run path uses this.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--output", default=None)
    parser.add_argument("--deepspeed", default="ds_config.json")
    # parse_known_args, NOT parse_args: the DeepSpeed launcher injects
    # --local_rank=N into every worker's argv, and a strict parser exits 2
    # with "unrecognized arguments" before training starts -- breaking the
    # exact command this example documents. CONTRIBUTING.md section 3.2.
    args = parser.parse_known_args()[0]
    if args.list_methods:
        bar = "=" * 78
        print(bar)
        print("  Offline preference optimization — the family")
        print(bar)
        print(f"  {'method':<8} {'TRL trainer':<14} {'reference model?':<18} note")
        print("  " + "-" * 74)
        for name, spec in METHODS.items():
            print(f"  {name:<8} {spec['trainer'] + 'Trainer':<14} "
                  f"{'yes' if spec['needs_ref'] else 'NO':<18} {spec['note']}")
        print(bar)
        print("  Verify the objectives on CPU first — no GPU, no download:")
        print("      uv run preference_losses.py")
        return

    spec = METHODS[args.method]
    output = args.output or f"./{args.method}-{args.model.split('/')[-1]}"

    require_gpu()

    # Imported AFTER the preflight so a missing GPU produces our message
    # rather than a CUDA error from inside transformers' import chain.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig

    bar = "=" * 78
    print(bar)
    print(f"  {args.method.upper()} — offline preference optimization")
    print(bar)
    print(f"  model            {args.model}")
    print(f"  device           {torch.cuda.get_device_name(0)}")
    print(f"  TRL trainer      {spec['trainer']}Trainer")
    print(f"  reference model  {'yes' if spec['needs_ref'] else 'NO — saves ~2 bytes/param'}")
    print(f"  LoRA             {'disabled (full fine-tune)' if args.no_lora else f'rank {args.lora_rank}'}")
    if not args.no_lora and spec["needs_ref"]:
        print("                   (reference = base weights with the adapter"
              " disabled,")
        print("                    so no second copy of the model is held)")
    print(f"  dataset          {args.dataset}")
    print(bar)

    dataset = load_dataset(args.dataset, split="train")

    peft_config = None if args.no_lora else LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Arguments shared by every trainer below. Kept in one dict so a change to
    # the DeepSpeed or logging setup cannot drift between methods.
    common = dict(
        output_dir=output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        deepspeed=args.deepspeed if os.path.exists(args.deepspeed) else None,
        report_to="wandb" if (WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"))
                  else "none",
    )

    if spec["trainer"] == "dpo":
        from trl import DPOConfig, DPOTrainer
        cfg = DPOConfig(
            beta=args.beta,
            # loss_type is a LIST in current TRL — it supports combining
            # objectives with loss_weights (that is how MPO is expressed).
            # A bare string still works, but the list form is what the config
            # documents, so use it.
            loss_type=[spec["loss_type"]],
            label_smoothing=args.label_smoothing,
            **common,
        )
        trainer = DPOTrainer(
            model=args.model,
            # None means "use the initial policy as the reference", which with
            # LoRA is the base weights with the adapter disabled. Never pass a
            # reference-free config to a method that needs one.
            ref_model=None,
            args=cfg,
            train_dataset=dataset,
            peft_config=peft_config,
        )

    elif spec["trainer"] == "cpo":
        CPOConfig, CPOTrainer = _import_cpo()
        # SimPO is CPO's objective with the BC/SFT regulariser switched off.
        # Leaving cpo_alpha at 1.0 silently trains CPO-SimPO instead, which is
        # a different method with different behaviour — so force it here rather
        # than trusting the caller to remember.
        cpo_alpha = 0.0 if args.method == "simpo" else args.cpo_alpha
        if args.method == "simpo" and args.cpo_alpha != 1.0:
            print(f"  [note] --cpo-alpha {args.cpo_alpha} ignored: SimPO "
                  "requires cpo_alpha=0.0")
        cfg = CPOConfig(
            beta=args.beta,
            loss_type=spec["loss_type"],
            cpo_alpha=cpo_alpha,
            simpo_gamma=args.simpo_gamma,
            **common,
        )
        trainer = CPOTrainer(
            model=args.model, args=cfg, train_dataset=dataset,
            peft_config=peft_config,
        )

    elif spec["trainer"] == "orpo":
        from trl import ORPOConfig, ORPOTrainer
        cfg = ORPOConfig(beta=args.beta, **common)
        trainer = ORPOTrainer(
            model=args.model, args=cfg, train_dataset=dataset,
            peft_config=peft_config,
        )

    else:  # kto
        from trl import KTOConfig, KTOTrainer
        # KTO consumes UNPAIRED data: one completion plus a boolean `label`.
        # A preference dataset will not load, and the error from deep inside
        # the collator is unhelpful, so check here and say what is wrong.
        cols = set(dataset.column_names)
        if "label" not in cols:
            raise ValueError(
                f"--method kto needs an UNPAIRED dataset with a `label` column "
                f"(True = desirable). Got columns {sorted(cols)}, which looks "
                f"like a preference dataset. Try trl-lib/kto-mix-14k, or use "
                f"--method dpo with this data."
            )
        cfg = KTOConfig(beta=args.beta, **common)
        trainer = KTOTrainer(
            model=args.model, args=cfg, train_dataset=dataset,
            peft_config=peft_config,
        )

    trainer.train()
    trainer.save_model(output)

    print(f"\n  saved to {output}")
    print("\n  The metric to read is rewards/margins, not loss. Loss falls for")
    print("  every method here regardless of whether it is learning the right")
    print("  thing; the margin between chosen and rejected is what tells you")
    print("  the preference was actually absorbed.")
    print("\n  Next: ../06_huggingface_grpo/ — when you have a verifier instead")
    print("  of preference pairs.")


if __name__ == "__main__":
    main()

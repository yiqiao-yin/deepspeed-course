"""
Train a Bradley-Terry reward model with TRL + DeepSpeed.

WHERE THIS SITS
---------------
Stage 2 of the classical RLHF pipeline: SFT -> REWARD MODEL -> PPO. The reward
model is the artefact that lets stage 3 score responses it has never seen.

Most alignment work now skips this entirely -- `03_huggingface/05_dpo` derives an
implicit reward from the policy itself and needs no separate model. So why build
one?

Three reasons it is still the right call:

  1. **A reward model is REUSABLE.** It scores anything, including outputs from
     a model you have not trained yet. A DPO run leaves behind no such artefact.
  2. **Best-of-n sampling** needs a scorer at inference time. Only this route
     gives you one.
  3. **Online methods need a judge.** `03_huggingface/07_online_dpo` and GRPO with
     a learned reward both consume what this folder produces.

THE OBJECTIVE
-------------
    L = -log sigmoid( r(x, y_chosen) - r(x, y_rejected) )

Logistic regression on score differences. See `reward_modeling.py` in this
folder for the objective on plain tensors, including the two properties that
matter: only differences are identified, and the loss falls while accuracy
stays flat.

WHAT TO WATCH
-------------
**Accuracy, not loss.** The loss keeps dropping as the model separates
already-correct pairs further, so it improves while the ranking does not.
Pairwise accuracy on a held-out split is the number that means something, and
anything above ~0.75 is respectable -- human annotators do not agree with each
other much more often than that.

MEMORY
------
    Qwen3-0.6B + scalar head, LoRA   ~12 GB
    Qwen2.5-7B + scalar head, LoRA   ~24 GB

Each example is a PAIR, so a micro-batch of N does 2N forward passes. Size it
by half what you would use for SFT.

RUNNING IT
----------
CoreWeave / SLURM:      sbatch run_deepspeed.sh
RunPod (auto-shutdown): uv run runpod/runpod_ctl.py run 03_huggingface/04_reward_model \
                            --dry-run --collect --wait --terminate --yes

    uv venv && source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install deepspeed transformers trl peft accelerate datasets
"""

import argparse
import os
import sys

try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
              "https://download.pytorch.org/whl/cu128\n")
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
    print("\n  Reward-model training downloads a model and needs real GPU")
    print("  memory. Qwen3-0.6B with a scalar head wants about 12 GB.")
    print("\n  The Bradley-Terry OBJECTIVE runs on CPU, and it is the part")
    print("  worth understanding:")
    print("      uv run 03_huggingface/04_reward_model/reward_modeling.py")
    print("      uv run tests/test_reward_model.py      # no GPU, no download")
    print("\n  Check your setup:")
    print("      nvidia-smi")
    print("      ds_report")
    print("\n  Rent a GPU (needs RUNPOD_API_KEY):")
    print("      uv run runpod/runpod_ctl.py recommend 03_huggingface/04_reward_model")
    print("      uv run runpod/runpod_ctl.py run 03_huggingface/04_reward_model \\")
    print("          --dry-run --collect --wait --terminate --yes")
    print("\n" + bar + "\n")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized",
                        help="Preference dataset with chosen/rejected columns.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Train on only the first N preference pairs. "
                             "--max-steps caps the OPTIMIZER, but the trainer "
                             "still tokenises the whole split first, which for "
                             "ultrafeedback_binarized is ~62k pairs and minutes "
                             "of wall clock before step 1. Use both for a "
                             "genuinely cheap smoke test.")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="LR warmup steps. Must be >0: ds_config.json "
                             "leaves warmup_num_steps 'auto', HuggingFace "
                             "fills it from this, and DeepSpeed rejects 0 "
                             "with 'warmup_num_steps must be a positive "
                             "integer'. Clamped for short runs.")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap steps; the RunPod --dry-run path uses this.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Each example is a PAIR, so this is 2x forward "
                             "passes. Half what you would use for SFT.")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", default="./reward-model")
    parser.add_argument("--deepspeed", default="ds_config.json")
    # parse_known_args, NOT parse_args. The DeepSpeed launcher injects
    # --local_rank=N into every worker's argv, and a strict parser exits 2
    # with "unrecognized arguments: --local_rank=0" before training starts.
    # That made `deepspeed --num_gpus=N train_reward_model.py` -- the only
    # command this example's README and run_deepspeed.sh document -- fail
    # every time. CONTRIBUTING.md section 3.2 states the rule.
    args = parser.parse_known_args()[0]

    # DeepSpeed's WarmupDecayLR rejects warmup_num_steps=0. ds_config.json
    # leaves it "auto"; HuggingFace substitutes TrainingArguments.warmup_steps,
    # which defaults to 0 -- so without setting it, EVERY DeepSpeed run of this
    # example dies before step one with
    #   ValueError: warmup_num_steps must be a positive integer, got 0
    # Clamped so a short --max-steps smoke test does not request more warmup
    # than it has steps to give.
    warmup_steps = args.warmup_steps
    if args.max_steps > 0:
        warmup_steps = max(1, min(warmup_steps, max(1, args.max_steps // 2)))
    # Did a launcher start us? deepspeed and torchrun both export LOCAL_RANK
    # and WORLD_SIZE; the deepspeed launcher also passes --local_rank.
    launched_distributed = (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
        or getattr(args, "local_rank", -1) >= 0
    )

    # Plain `python train_reward_model.py` on a machine with several GPUs makes
    # HuggingFace Trainer wrap the model in nn.DataParallel. DataParallel is
    # both slower than the launcher path and, with bf16 + LoRA + gradient
    # checkpointing, simply broken. Pin to one device and say so, rather than
    # letting the reader meet a device-mismatch traceback.
    #
    # Set before require_gpu() imports torch: CUDA_VISIBLE_DEVICES is read when
    # the CUDA context is first created, so it has to be in place beforehand.
    if not launched_distributed and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        _pinned_to_one_gpu = True
    else:
        _pinned_to_one_gpu = False

    require_gpu()

    # Imported AFTER the preflight so a missing GPU produces our message
    # rather than a CUDA error from inside transformers' import chain.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    bar = "=" * 76
    print(bar)
    print("  Reward model — Bradley-Terry on preference pairs")
    print(bar)
    print(f"  model      {args.model}")
    print(f"  device     {torch.cuda.get_device_name(0)}")
    print(f"  dataset    {args.dataset}")
    print(f"  LoRA       {'disabled' if args.no_lora else f'rank {args.lora_rank}'}")
    print(f"  launch     {'distributed (' + str(os.environ.get('WORLD_SIZE', '?')) + ' ranks)' if launched_distributed else 'single process'}")
    if _pinned_to_one_gpu:
        print("             pinned to GPU 0 — for multi-GPU use:")
        print("               deepspeed --num_gpus=N train_reward_model.py")
    print(bar)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # num_labels=1 replaces the LM head with a SCALAR head. That single number
    # is the reward; nothing else about the architecture changes.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(args.dataset, split="train")
    if args.max_samples > 0:
        n = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(n))
        print(f"  --max-samples: using {n} of the split's pairs")

    peft_config = None if args.no_lora else LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        # SEQ_CLS, not CAUSAL_LM. Using CAUSAL_LM here silently fails to train
        # the scalar head, and the run completes with a model that scores noise.
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = RewardTrainer(
        model=model,
        args=RewardConfig(
            warmup_steps=warmup_steps,
            output_dir=args.output,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            # Only when a launcher actually started us. A config file
            # existing is not evidence of a distributed run, and half-enabling
            # DeepSpeed in a single process leaves HuggingFace Trainer to fall
            # back to nn.DataParallel on a multi-GPU box -- which fails with
            # "module must have its parameters and buffers on device cuda:0
            # but found one of them on device: cpu".
            deepspeed=(args.deepspeed
                       if launched_distributed and os.path.exists(args.deepspeed)
                       else None),
            report_to="wandb" if (WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"))
                      else "none",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output)

    print(f"\n  saved to {args.output}")
    print("\n  Read ACCURACY, not loss. The loss keeps falling as the model")
    print("  separates already-correct pairs further — it can improve while")
    print("  the ranking does not change at all. Above ~0.75 pairwise accuracy")
    print("  is respectable; humans do not agree with each other much more.")
    print("\n  Next: ../07_online_dpo/ — use this model as a judge,")
    print("  or ../05_dpo/ — skip the reward model entirely.")


if __name__ == "__main__":
    main()

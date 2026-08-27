# /// script
# requires-python = ">=3.9"
# ///
"""
Validate every ds_config.json in the repository.

Run:
    uv run tests/test_ds_configs.py

Checks each config for the failure modes that abort a run at startup or, worse,
succeed while doing something other than intended:

  * valid JSON
  * the batch-size invariant is satisfiable for the GPU count its launcher
    requests  --  train_batch_size = micro * accum * num_gpus
  * fp16 and bf16 are not both enabled (mutually exclusive)
  * "auto" values only appear where a HuggingFace Trainer can resolve them
  * offload_param is only used with ZeRO stage 3
  * stage 3 saves a loadable checkpoint

The batch check is the one that matters most in practice: it is the single most
common first-run failure in this course, and it is trivially detectable
statically. `02_basic_convnet_cifar10_examples` shipped a config hard-coded to
1 GPU alongside a launcher requesting 2.

Pure stdlib — no dependencies.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import REPO_ROOT, Results  # noqa: E402


def launcher_gpu_counts(config_path: Path):
    """
    GPU counts requested by launcher scripts next to this config.

    Returns:
        Sorted list of distinct --num_gpus values found, or [] if none.
    """
    counts = set()
    for script in sorted(config_path.parent.glob("*.sh")):
        text = script.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"--num_gpus[= ](\d+)", text):
            counts.add(int(match.group(1)))
    return sorted(counts)


def main() -> int:
    r = Results("DeepSpeed configs — static validation")

    configs = sorted(
        p for p in REPO_ROOT.rglob("*.json")
        if ("ds_config" in p.name or "deepspeed" in p.name.lower()
            or p.name.endswith("_config.json"))
        and "node_modules" not in p.parts
        and "docusaurus-docs" not in p.parts
    )
    r.check(len(configs) > 0, f"found {len(configs)} config files to validate")

    for path in configs:
        rel = path.relative_to(REPO_ROOT)

        # ---- valid JSON ------------------------------------------------
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            r.check(False, f"{rel}: valid JSON", str(exc))
            continue
        r.check(True, f"{rel}: valid JSON")

        tbs = cfg.get("train_batch_size")
        micro = cfg.get("train_micro_batch_size_per_gpu")
        accum = cfg.get("gradient_accumulation_steps")
        is_auto = lambda v: v == "auto"  # noqa: E731

        # ---- batch invariant -------------------------------------------
        if all(v is not None and not is_auto(v) for v in (tbs, micro, accum)):
            gpus = launcher_gpu_counts(path)
            if not gpus:
                ok = tbs % (micro * accum) == 0
                r.check(
                    ok,
                    f"{rel}: batch invariant is satisfiable for some GPU count",
                    f"train_batch_size={tbs} is not divisible by "
                    f"micro({micro}) * accum({accum}) = {micro * accum}",
                )
            else:
                for n in gpus:
                    expected = micro * accum * n
                    r.check(
                        tbs == expected,
                        f"{rel}: batch invariant holds for the launcher's "
                        f"--num_gpus={n}",
                        f"train_batch_size={tbs} but micro({micro}) * "
                        f"accum({accum}) * gpus({n}) = {expected}. "
                        f"The run aborts at startup.",
                    )
        elif tbs is None and micro is not None and not is_auto(micro):
            # Omitting train_batch_size is the PORTABLE form: DeepSpeed derives
            # it, so the config works at any --num_gpus.
            r.check(True, f"{rel}: portable — train_batch_size derived from micro x accum")

        # ---- precision --------------------------------------------------
        # "auto" resolves from TrainingArguments at runtime, so it is neither
        # definitely on nor definitely off statically. Treat the two cases
        # separately: literal true/true is a guaranteed crash, while
        # true/"auto" is a latent conflict that fires only for some
        # TrainingArguments.
        fp16_raw = cfg.get("fp16", {}).get("enabled", False)
        bf16_raw = cfg.get("bf16", {}).get("enabled", False)
        fp16_true = fp16_raw is True
        bf16_true = bf16_raw is True

        r.check(
            not (fp16_true and bf16_true),
            f"{rel}: fp16 and bf16 not both enabled",
            "They are mutually exclusive; DeepSpeed raises at initialization.",
        )
        r.check(
            not ((fp16_true and bf16_raw == "auto") or (bf16_true and fp16_raw == "auto")),
            f"{rel}: no latent fp16/bf16 conflict via \"auto\"",
            f'fp16={fp16_raw!r}, bf16={bf16_raw!r}. One is hard-enabled while '
            f'the other resolves from TrainingArguments — pin the unused one '
            f"to false so the combination cannot become invalid at runtime.",
        )

        # ---- "auto" needs a HuggingFace Trainer -------------------------
        raw = path.read_text(encoding="utf-8")
        if '"auto"' in raw:
            siblings = [p.name for p in path.parent.rglob("*.py")]
            uses_hf_trainer = any(
                ("Trainer" in (path.parent / name).read_text(encoding="utf-8", errors="ignore"))
                for name in siblings
            )
            r.check(
                uses_hf_trainer,
                f"{rel}: uses \"auto\" and a HF Trainer is present to resolve it",
                '"auto" is a HuggingFace convention. With raw '
                "deepspeed.initialize nothing resolves it.",
            )

        # ---- ZeRO -------------------------------------------------------
        zero = cfg.get("zero_optimization", {})
        stage = zero.get("stage")

        if "offload_param" in zero:
            device = zero["offload_param"].get("device", "none")
            if device != "none":
                r.check(
                    stage == 3,
                    f"{rel}: offload_param requires ZeRO stage 3",
                    f"stage={stage}; parameter offload is silently ignored below stage 3.",
                )

        if stage == 3:
            r.check(
                zero.get("stage3_gather_16bit_weights_on_model_save")
                or zero.get("gather_16bit_weights_on_model_save"),
                f"{rel}: stage 3 saves a consolidated checkpoint",
                "Without the gather flag the checkpoint is written as shards "
                "that from_pretrained cannot load.",
            )

        # NVMe offload must not point at a network filesystem.
        for key in ("offload_param", "offload_optimizer"):
            block = zero.get(key, {})
            if block.get("device") == "nvme":
                nvme_path = block.get("nvme_path", "")
                r.check(
                    bool(nvme_path),
                    f"{rel}: {key} nvme device declares an nvme_path",
                )
                r.check(
                    not any(nvme_path.startswith(p) for p in ("/home", "/nfs", "/mnt/nfs")),
                    f"{rel}: {key} nvme_path looks local",
                    f"nvme_path={nvme_path!r} may be a network filesystem, "
                    f"which is catastrophically slow.",
                )

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())

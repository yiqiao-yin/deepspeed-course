"""Compare bf16 LoRA and NF4 QLoRA memory for the same Qwen3-4B task.

Both arms train the same adapters with DeepSpeed ZeRO-2. Each arm runs in a
separate process so its GPU memory measurement starts without the other model.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B"
# Apply the same LoRA adapters to these attention and feed-forward projections.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
# This tiny, repeated corpus checks the training pipeline and loss trend;
# it cannot establish how either adapter performs on unseen data.
CORPUS = (
    "A lake freezes in winter and thaws when temperatures rise.",
    "The analyst compared two memory profiles for the same training task.",
    "Frozen model weights do not receive optimizer updates during LoRA training.",
    "Four-bit storage reduces the size of the frozen linear layers.",
)


def require_gpu():
    """Check GPU availability and explain CPU-only options."""
    cpu_help = (
        "\nYou can inspect the theoretical memory plan without a GPU:\n"
        "  uv run --no-project python "
        "03_llms/12_qlora/train_qlora.py --plan\n"
        "Or run the local tests:\n"
        "  uv run tests/test_qlora.py"
    )

    if os.environ.get("ALLOW_CPU") == "1":
        print(
            "ALLOW_CPU=1: bypassing preflight; training still needs CUDA.",
            file=sys.stderr,
        )
        return

    # Respect an explicit request to hide every GPU before importing torch.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() in ("", "-1"):
        raise SystemExit(
            "No CUDA GPU is visible. This comparison requires a GPU."
            + cpu_help
        )

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is missing. For GPU training, run uv sync in "
            "03_llms/12_qlora."
            + cpu_help
        ) from None

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. This comparison requires a GPU."
            + cpu_help
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("both", "lora", "qlora"),
                        default="both")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--deepspeed", default="ds_config.json")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Optimizer steps PER arm; -1 uses 40 per arm.")
    parser.add_argument("--output", default="comparison.json")
    parser.add_argument("--plan", action="store_true",
                        help="Show base-weight arithmetic, without a GPU.")
    args, unknown = parser.parse_known_args(argv)
    # DeepSpeed versions inject --local_rank (with either spelling).
    if any(not (x.startswith("--local_rank") or x.lstrip("-").isdigit())
           for x in unknown):
        parser.error(f"Unknown options: {unknown}")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be positive or -1")
    return args


def check_config(config):
    """Reject configs which silently make the two labels incomparable."""
    if config.get("zero_optimization", {}).get("stage") != 2:
        raise ValueError("Both arms require ZeRO-2. Quantized ZeRO-3 may "
                         "silently skip the expected parameter partition.")
    if config.get("train_micro_batch_size_per_gpu") != 1:
        raise ValueError("Expected one sequence per GPU and step.")
    if config.get("gradient_accumulation_steps") != 1:
        raise ValueError("Expected one optimizer step per batch.")
    if config.get("bf16", {}).get("enabled") is not True:
        raise ValueError("Both arms require bf16 compute.")


def parameter_count(param):
    """Params4bit.numel counts packed bytes; quant_state.shape is logical."""
    state = getattr(param, "quant_state", None)
    if state is not None:
        count = 1
        for dimension in state.shape:
            count *= dimension
        return count
    return getattr(param, "ds_numel", param.numel())


def parameter_stats(model):
    """Logical counts and stored tensor bytes, including quantization state."""
    logical = trainable = storage = quantized = 0
    for param in model.parameters():
        # Use original tensor dimensions for a fair parameter count, even when
        # NF4 stores the weights in a packed representation.
        n = parameter_count(param)
        logical += n
        trainable += n if param.requires_grad else 0
        storage += param.numel() * param.element_size()
        state = getattr(param, "quant_state", None)
        if state is not None:
            quantized += 1
            # Count quantization scales and metadata alongside packed weights.
            for value in state.as_dict(packed=True).values():
                if hasattr(value, "numel"):
                    storage += value.numel() * value.element_size()
    for buffer in model.buffers():
        storage += buffer.numel() * buffer.element_size()
    return dict(logical_parameters=logical, trainable_parameters=trainable,
                stored_tensor_bytes=storage, quantized_tensors=quantized)


def expected_adapter_count(model):
    """Independent LoRA formula: rank x (input width + output width)."""
    expected = targets = 0
    for name, layer in model.named_modules():
        if name.rsplit(".", 1)[-1] not in TARGETS or not hasattr(layer, "lora_A"):
            continue
        shape = getattr(getattr(layer.base_layer.weight, "quant_state", None),
                        "shape", layer.base_layer.weight.shape)
        out_features, in_features = shape
        rank = layer.r["default"]
        expected += rank * (in_features + out_features)
        targets += 1
    if not targets:
        raise ValueError("No Qwen adapter target layers found.")
    return expected


def check_pair(lora, qlora):
    """Reject unequal adapters, contaminated peaks, or invalid NF4 results."""
    # Check that neither run began with substantial GPU memory already in use.
    if any(row["start_allocated_bytes"] > 256 * 2**20 for row in (lora, qlora)):
        raise ValueError("An arm started with live GPU allocations; "
                         "the peak comparison is not isolated.")
    if lora["representation"] != "bf16" or qlora["representation"] != "nf4":
        raise ValueError("The two arms must use different representations.")
    if lora["quantized_tensors"] or not qlora["quantized_tensors"]:
        raise ValueError("Only the QLoRA arm should have quantized tensors.")
    if lora["trainable_parameters"] != qlora["trainable_parameters"]:
        raise ValueError("Adapter sizes differ; comparison is invalid.")
    if any(row["trainable_parameters"] != row["expected_adapter_parameters"]
           for row in (lora, qlora)):
        raise ValueError("Trainable count disagrees with LoRA rank and dimensions.")
    if lora["logical_parameters"] != qlora["logical_parameters"]:
        raise ValueError("Base logical parameter counts differ.")
    if qlora["stored_tensor_bytes"] >= lora["stored_tensor_bytes"]:
        raise ValueError("NF4 storage is not smaller; inspect the quantization.")
    if qlora["peak_allocated_bytes"] >= lora["peak_allocated_bytes"]:
        raise ValueError("QLoRA peak GPU allocation is not smaller. "
                         "Investigate rather than publishing the comparison.")


def check_loss_trend(result):
    """Only a long enough run can support a tiny-corpus loss trend claim."""
    if result["steps"] >= 20 and (
            result["last_window_mean"] >= result["first_window_mean"]):
        raise ValueError(f"{result['arm']} loss did not decrease over the "
                         "five-step windows; investigate the training run.")

def compare_isolated_arms(args):
    """Train each arm in a fresh worker so the first model cannot skew VRAM."""
    rank = int(os.environ.get("RANK", "0"))
    # Each DeepSpeed worker starts both children in order. The child inherits
    # RANK/WORLD_SIZE and joins the same ranks for its own training session.
    with tempfile.TemporaryDirectory(prefix="qlora-comparison-") as folder:
        per_arm = {}
        for arm in ("lora", "qlora"):
            output = Path(folder) / f"{arm}.json"
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--arm", arm, "--model", args.model,
                       "--deepspeed", args.deepspeed,
                       "--max-steps", str(args.max_steps),
                       "--output", str(output)]
            # A new process releases all live CUDA tensors when it exits.
            subprocess.run(command, check=True)
            if rank == 0:
                per_arm[arm] = json.loads(output.read_text(encoding="utf-8"))

        if rank != 0:
            return

    # Compare matching GPU ranks only after both child processes have exited.
    if len(per_arm["lora"]) != len(per_arm["qlora"]):
        raise ValueError("LoRA and QLoRA returned different GPU rank counts.")
    combined = []
    for lora, qlora in zip(per_arm["lora"], per_arm["qlora"]):
        pair = {"lora": lora["lora"], "qlora": qlora["qlora"]}
        if pair["lora"]["rank"] != pair["qlora"]["rank"]:
            raise ValueError("LoRA and QLoRA GPU ranks do not match.")
        # Keep the original scientific checks: comparable adapters, smaller
        # NF4 storage and peak VRAM, and falling loss for both arms.
        check_pair(pair["lora"], pair["qlora"])
        check_loss_trend(pair["lora"])
        check_loss_trend(pair["qlora"])
        combined.append(pair)
    # Save a comparison only if every rank passed the same scientific checks.
    Path(args.output).write_text(json.dumps(combined, indent=2) + "\n",
                                 encoding="utf-8")
    print("LoRA vs QLoRA comparison passed; saved", args.output, flush=True)


def main():
    args = parse_args()
    # The planning mode needs no GPU or heavyweight training dependencies.
    if args.plan:
        print("Qwen3-4B ~4.02B base parameters: bf16 ~8.04 GB; "
              "ideal 4-bit ~2.01 GB. These are arithmetic, NOT GPU measurements. "
              "Adapters, metadata, activations and CUDA overhead add memory.")
        return
    # The parent starts two independent workers so the first model cannot
    # remain allocated when measuring the second model.
    require_gpu()
    if args.arm == "both":
        compare_isolated_arms(args)
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required even with ALLOW_CPU=1; use --plan.")
    import deepspeed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Verify that both workers use the same ZeRO-2 and bf16 training settings.
    config = json.loads(Path(args.deepspeed).read_text(encoding="utf-8"))
    check_config(config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    # Prepare the same short training examples for both arms.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    examples = [tokenizer(sentence, truncation=True, max_length=64,
                          padding="max_length", return_tensors="pt")
                for sentence in CORPUS]
    steps = 40 if args.max_steps == -1 else args.max_steps
    results = {}

    for arm in (args.arm,):
        # Reset the peak counter and record any allocations before model load.
        # The parent starts this whole worker afresh for each arm.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_allocated_bytes = torch.cuda.memory_allocated(device)
        torch.manual_seed(42)
        # LoRA loads bf16 base weights; QLoRA loads the same model in NF4,
        # while both arms perform calculations using bf16.
        kwargs = dict(torch_dtype=torch.bfloat16, device_map={"": local_rank})
        if arm == "qlora":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        # Checkpoint activations during backpropagation to reduce memory use.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # Freeze the base model in both arms so only the adapters are trained.
        if arm == "qlora":
            model = prepare_model_for_kbit_training(model)
        else:
            for param in model.parameters():
                param.requires_grad_(False)
            model.enable_input_require_grads()
        # The identical rank, target layers, and adapter settings keep the
        # trainable parameter count comparable between the two arms.
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=list(TARGETS), task_type="CAUSAL_LM"))
        counts = parameter_stats(model)
        counts["expected_adapter_parameters"] = expected_adapter_count(model)
        # Fail if any extra weights became trainable or quantization was missed.
        if counts["trainable_parameters"] != counts["expected_adapter_parameters"]:
            raise RuntimeError("Unexpected trainable parameters outside adapters.")
        if (arm == "qlora") != (counts["quantized_tensors"] > 0):
            raise RuntimeError("Model representation does not match its arm.")
        loaded_bytes = torch.cuda.memory_allocated(device)
        # A model load can briefly use more memory than the steady loaded model.
        load_peak_bytes = torch.cuda.max_memory_allocated(device)
        # DeepSpeed trains the adapter parameters using the checked ZeRO-2 config.
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=config)

        torch.cuda.synchronize(device)
        training_start = time.perf_counter()

        losses = []
        for step in range(steps):
            # Cycle through identical examples; padding tokens do not affect loss.
            batch = examples[step % len(examples)]
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = ids.masked_fill(mask == 0, -100)
            loss = engine(input_ids=ids, attention_mask=mask, labels=labels).loss
            engine.backward(loss)
            engine.step()
            losses.append(float(loss.detach()))
            del ids, mask, labels, loss
            if rank == 0 and (step == 0 or (step + 1) % 10 == 0):
                print(f"{arm} step {step + 1}/{steps}: "
                      f"loss {losses[-1]:.4f}", flush=True)
        torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - training_start

        # Store per-arm evidence for the later cross-arm validation.
        window = min(5, steps)
        results[arm] = dict(
            arm=arm, representation="nf4" if arm == "qlora" else "bf16",
            zero_stage=2, rank=rank, steps=steps,
            training_seconds=training_seconds,
            start_allocated_bytes=start_allocated_bytes,
            loaded_allocated_bytes=loaded_bytes,
            load_peak_allocated_bytes=load_peak_bytes,
            # Keep the larger peak seen during loading or training.
            peak_allocated_bytes=max(
                load_peak_bytes, torch.cuda.max_memory_allocated(device)),
            first_loss=losses[0], last_loss=losses[-1],
            first_window_mean=sum(losses[:window]) / window,
            last_window_mean=sum(losses[-window:]) / window, **counts)
        if rank == 0:
            print(f"{arm} memory: start={start_allocated_bytes / 2**30:.2f} GiB; "
                  f"loaded={loaded_bytes / 2**30:.2f} GiB; "
                  f"load peak={load_peak_bytes / 2**30:.2f} GiB; "
                  f"overall peak={results[arm]['peak_allocated_bytes'] / 2**30:.2f} GiB; "
                  f"stored tensors={counts['stored_tensor_bytes'] / 2**30:.2f} GiB",
                  flush=True)
            print(f"{arm} training time: {training_seconds:.2f} seconds", flush=True)
        del engine, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    # Collect each GPU's results and share any validation failure across ranks.
    per_rank_results = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(per_rank_results, results)
    error = None
    if rank == 0:
        try:
            for per_rank in per_rank_results:
                if args.arm == "both":
                    check_pair(per_rank["lora"], per_rank["qlora"])
                for result in per_rank.values():
                    check_loss_trend(result)
        except ValueError as exc:
            error = str(exc)
    verdict = [error]
    torch.distributed.broadcast_object_list(verdict, src=0)
    if verdict[0]:
        raise ValueError(verdict[0])
    # Only rank zero writes the JSON report; measurements remain per GPU.
    if rank == 0:
        for per_rank in per_rank_results:
            for arm, result in per_rank.items():
                print(f"rank {result['rank']} {arm}: peak allocated "
                      f"{result['peak_allocated_bytes'] / 2**30:.2f} GiB; "
                      f"trainable {result['trainable_parameters']:,}; "
                      f"loss {result['first_loss']:.4f} -> {result['last_loss']:.4f}")
        if args.max_steps != -1:
            print(f"Run capped at {steps} optimizer steps per arm. "
                  "For a longer comparison omit --max-steps.")
        print("Peak is allocated bytes per GPU including model load; stored "
              "bytes count tensors and NF4 metadata. No GPU values are summed.")
        Path(args.output).write_text(
            json.dumps(per_rank_results, indent=2) + "\n", encoding="utf-8")
    # Release the distributed group before this worker exits.
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

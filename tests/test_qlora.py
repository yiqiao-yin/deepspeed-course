# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""CPU-only property tests for the QLoRA comparison harness."""

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import REPO_ROOT, Results, load_function  # noqa: E402

SOURCE = "03_llms/13_qlora/train_qlora.py"
FOLDER = REPO_ROOT / "03_llms" / "13_qlora"


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def main():
    r = Results("QLoRA vs LoRA — comparison invariants")
    config = json.loads((FOLDER / "ds_config.json").read_text())
    check_config = load_function(SOURCE, "check_config")
    r.check(not raises(check_config, config), "the shipped config uses ZeRO-2")
    bad_config = json.loads(json.dumps(config))
    bad_config["zero_optimization"]["stage"] = 3
    r.check(raises(check_config, bad_config),
            "the same run under ZeRO-3 is rejected")

    count = load_function(SOURCE, "parameter_count")

    class QuantState:
        shape = (12, 16)

    class Packed:
        quant_state = QuantState()
        def numel(self):
            return 96

    r.check(count(Packed()) == 192,
            "logical count uses the original NF4 shape, not packed bytes")
    adapter_formula = load_function(
        SOURCE, "expected_adapter_count",
        extra_globals={"TARGETS": ("q_proj",)})

    class FakeWeight:
        shape = (4, 2)  # packed physical representation
        quant_state = QuantState()  # original dimensions: 12 x 16

    class FakeLayer:
        lora_A = {"default": object()}
        r = {"default": 8}
        base_layer = type("Base", (), {"weight": FakeWeight()})()

    class FakeModel:
        def named_modules(self):
            return [("block.q_proj", FakeLayer())]

    r.check(adapter_formula(FakeModel()) == 8 * (12 + 16),
            "adapter expectation uses independent rank x original dimensions")
    lora = dict(representation="bf16", quantized_tensors=0,
                trainable_parameters=1024, expected_adapter_parameters=1024,
                logical_parameters=4000,
                stored_tensor_bytes=8000, peak_allocated_bytes=12000,
                start_allocated_bytes=0,
                arm="lora", steps=20, first_window_mean=3.0,
                last_window_mean=2.0)
    qlora = dict(lora, representation="nf4", quantized_tensors=2,
                 stored_tensor_bytes=2900, peak_allocated_bytes=6000,
                 arm="qlora")
    pair = load_function(SOURCE, "check_pair")
    trend = load_function(SOURCE, "check_loss_trend")
    r.check(not raises(pair, lora, qlora),
            "valid pair has equal adapter and logical counts, lower NF4 memory")
    r.check(not raises(trend, lora) and not raises(trend, qlora),
            "loss falls for both arms in a comparison")

    # Deliberately broken inputs: watch the actual shipped check reject them.
    r.check(raises(pair, lora, dict(qlora, quantized_tensors=0)),
            "broken test: repeating the bf16 arm fails")
    r.check(raises(pair, lora, dict(qlora, trainable_parameters=512)),
            "broken test: changed adapter count fails")
    r.check(raises(pair, lora, dict(qlora, expected_adapter_parameters=999)),
            "broken test: count disagrees with LoRA dimensions")
    r.check(raises(pair, lora, dict(qlora, peak_allocated_bytes=14000)),
            "broken test: increased NF4 GPU peak fails")
    r.check(raises(pair, lora, dict(qlora, start_allocated_bytes=7 * 2**30)),
            "broken test: old model still allocated before QLoRA fails")
    r.check(raises(trend, dict(lora, last_window_mean=3.5)),
            "broken test: rising loss fails")
    r.check(not raises(trend, dict(lora, steps=1)),
            "a one-step cap makes no convergence claim")
    # Simulate GPU workers by writing example results without starting training.
    calls = []

    def fake_run(command, check):
        arm = command[command.index("--arm") + 1]
        output = Path(command[command.index("--output") + 1])
        row = dict(lora if arm == "lora" else qlora, rank=0)
        output.write_text(json.dumps([{arm: row}]))
        calls.append((arm, command, check))

    isolated = load_function(SOURCE, "compare_isolated_arms", extra_globals={
        "__file__": str(FOLDER / "train_qlora.py"),
        "os": SimpleNamespace(environ={"RANK": "0"}),
        "sys": sys, "Path": Path, "json": json,
        "subprocess": SimpleNamespace(run=fake_run),
        "tempfile": tempfile,
        "check_pair": pair, "check_loss_trend": trend,
    })

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "comparison.json"
        args = SimpleNamespace(model="example/model", deepspeed="ds_config.json",
                               max_steps=20, output=str(output))
        isolated(args)
        combined = json.loads(output.read_text())

        # Verify two separate worker calls and matching GPU ranks.
        r.check([arm for arm, _, _ in calls] == ["lora", "qlora"]
                and all(check for _, _, check in calls)
                and calls[0][1] is not calls[1][1]
                and combined[0]["lora"]["rank"] == combined[0]["qlora"]["rank"],
                "both arms run separately and merge the same GPU rank")

        # Introduce a bad measurement and verify the comparison rejects it.
        def reject_bad_peak(command, check):
            fake_run(command, check)
            if command[command.index("--arm") + 1] == "qlora":
                result_path = Path(command[command.index("--output") + 1])
                bad = json.loads(result_path.read_text())
                bad[0]["qlora"]["peak_allocated_bytes"] = 14000
                result_path.write_text(json.dumps(bad))

        isolated.__globals__["subprocess"] = SimpleNamespace(run=reject_bad_peak)
        r.check(raises(isolated, args),
                "isolated comparison rejects increased NF4 GPU peak")

    tree = ast.parse((FOLDER / "train_qlora.py").read_text())
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "main")
    calls = [n.func.id for n in ast.walk(main_fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    r.check("require_gpu" in calls and "check_pair" in calls
            and "check_loss_trend" in calls,
            "training calls preflight and both validation checks")
    src = (FOLDER / "train_qlora.py").read_text()
    r.check("torch.cuda.reset_peak_memory_stats(device)" in src
            and "torch.cuda.max_memory_allocated(device)" in src,
            "each arm resets and reads its own allocated-memory peak")

    plan = subprocess.run(
        [sys.executable, str(FOLDER / "train_qlora.py"), "--plan"],
        capture_output=True, text=True, check=False)
    r.check(plan.returncode == 0 and "NOT GPU measurements" in plan.stdout,
            "CPU plan is clearly theoretical")
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())

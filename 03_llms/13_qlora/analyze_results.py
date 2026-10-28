import json
import statistics
from collections import defaultdict
from pathlib import Path

# Locate the benchmark results relative to this script.
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "2026-10-26"

# Group the measurements by GPU model.
benchmarks = defaultdict(list)

# Read every saved benchmark JSON file.
for file in sorted(RESULTS_DIR.glob("rtx*_run*.json")):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Each comparison file contains one LoRA/QLoRA result pair.
    result = data[0]

    gpu = file.stem.split("_")[0].upper()
    benchmarks[gpu].append(result)

# Calculate statistics for each GPU.
for gpu, runs in sorted(benchmarks.items()):

    lora_times = [run["lora"]["training_seconds"] for run in runs]
    qlora_times = [run["qlora"]["training_seconds"] for run in runs]

    lora_mean = statistics.mean(lora_times)
    qlora_mean = statistics.mean(qlora_times)

    lora_std = statistics.stdev(lora_times) if len(runs) > 1 else 0
    qlora_std = statistics.stdev(qlora_times) if len(runs) > 1 else 0

    slowdown = ((qlora_mean / lora_mean) - 1) * 100

    # Calculate memory savings for every run, not just the first one.
    memory_savings_by_run = [
        (1 - run["qlora"]["peak_allocated_bytes"]
         / run["lora"]["peak_allocated_bytes"]) * 100
         for run in runs
    ]


    memory_savings = statistics.mean(memory_savings_by_run)

    print(f"\n{gpu} — {len(runs)} benchmark runs")
    print(f"LoRA mean:        {lora_mean:.2f} seconds")
    print(f"QLoRA mean:       {qlora_mean:.2f} seconds")
    print(f"LoRA std dev:     {lora_std:.2f} seconds")
    print(f"QLoRA std dev:    {qlora_std:.2f} seconds")
    print(f"QLoRA time diff:  {slowdown:+.2f}%")
    print(f"Memory savings:   {memory_savings:.2f}%")
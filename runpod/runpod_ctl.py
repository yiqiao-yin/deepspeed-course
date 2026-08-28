# /// script
# requires-python = ">=3.9"
# ///
"""
runpod_ctl — find a GPU, start a pod, and run a course example on it.

    export RUNPOD_API_KEY=...                       # https://console.runpod.io/user/settings
    uv run runpod/runpod_ctl.py gpus --min-vram 24
    uv run runpod/runpod_ctl.py recommend 06_huggingface_grpo
    uv run runpod/runpod_ctl.py run 06_huggingface_grpo --yes
    uv run runpod/runpod_ctl.py pods
    uv run runpod/runpod_ctl.py terminate <podId>

Stdlib only — no dependencies beyond Python itself.

WHAT THIS DOES AND DOES NOT DO
------------------------------
Does:  queries the live GPU catalogue with prices, maps a course example to its
       VRAM/disk requirements, picks the cheapest GPU that fits, creates a pod
       whose start command clones this repository and runs the example, and
       reports cost as it goes.

Does not: stream container logs. RunPod's REST API exposes no log endpoint
       (see `Pod` schema — there is no logs field), so log retrieval is via SSH
       or the web console. `run` prints the exact SSH command to use.

COST WARNING
------------
`create` and `run` start billing immediately and keep billing until the pod is
terminated — not merely stopped. Both refuse to proceed without `--yes`, and
both print the hourly rate first. Always confirm with `pods` afterwards.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
REPO_URL = "https://github.com/yiqiao-yin/deepspeed-course.git"

# Default image: a `devel` tag is REQUIRED. The `runtime` variants ship no nvcc,
# so DeepSpeed cannot JIT-compile its fused CUDA ops and every example fails
# with `CUDA_HOME environment variable is not set`.
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"

# Per-example requirements. min_vram is per GPU, in GB.
EXAMPLES = {
    "01_basic_neuralnet": dict(min_vram=6, gpus=1, disk=20,
                               script="train_ds_enhanced.py", note="Trivial; any GPU."),
    "02_basic_convnet": dict(min_vram=6, gpus=1, disk=20,
                             script="train_ds.py", note="Synthetic MNIST."),
    "02_basic_convnet_cifar10_examples": dict(min_vram=8, gpus=1, disk=30,
                                              script="cifar10_deepspeed.py",
                                              note="Downloads CIFAR-10 (~170 MB)."),
    "03_basic_rnn": dict(min_vram=8, gpus=1, disk=20,
                         script="train_rnn_deepspeed.py", note="Small LSTM."),
    "04_bayesian_neuralnet": dict(min_vram=8, gpus=2, disk=20,
                                  script="parallel_tempering_mcmc.py",
                                  note="One temperature per rank; 2+ GPUs is the point."),
    "04_intermediate_rnn_stock_data": dict(min_vram=8, gpus=1, disk=20,
                                           script="train_rnn_stock_data_ds.py",
                                           note="Needs network egress for yfinance."),
    "05_huggingface": dict(min_vram=24, gpus=2, disk=80,
                           script="train_ds.py",
                           note="HuggingFace LLM fine-tuning with ZeRO."),
    "05_huggingface_trl": dict(min_vram=24, gpus=1, disk=60,
                               script="train_trl_deepspeed.py",
                               note="Qwen3-0.6B SFT."),
    "05_huggingface_ocr": dict(min_vram=24, gpus=1, disk=60,
                               script="train_ds.py",
                               note="Qwen2-VL-2B; cap max_pixels to bound memory."),
    "06_huggingface_grpo": dict(min_vram=24, gpus=1, disk=80,
                                script="grpo_gsm8k_train.py",
                                note="RL; memory driven by G rollouts."),
    "07_huggingface_openai_gpt_oss_finetune_sft": dict(min_vram=80, gpus=4, disk=200,
                                                       script="lora/train_ds.py",
                                                       note="gpt-oss-20b MoE; ~40 GB of weights."),
    "07_huggingface_trl_multi_agency": dict(min_vram=24, gpus=1, disk=60,
                                            script="train_grpo_math.py",
                                            launcher="python",
                                            note="Multi-agent GRPO. Uses TRL directly, "
                                                 "NOT the deepspeed launcher."),
    "08_vtt": dict(min_vram=48, gpus=2, disk=120,
                   script="hf_ds_vtt_test2/llava_video_trainer/video_training_script.py",
                   note="Video tokens are quadratic in frame count."),
    "09_vss": dict(min_vram=180, gpus=2, disk=2000,
                   script="train_ds_2xB200.py",
                   note="NOT VIABLE on typical RunPod: needs ~3 TB HOST RAM."),
}


# --------------------------------------------------------------------------- api
def api_key(args) -> str:
    key = getattr(args, "api_key", None) or os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set.\n"
                 "  export RUNPOD_API_KEY=...   (https://console.runpod.io/user/settings)\n"
                 "  or pass --api-key")
    return key


def _request(url: str, key: str, method: str = "GET", payload=None, timeout: int = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # RunPod's GraphQL endpoint sits behind Cloudflare, which rejects
        # urllib's default "Python-urllib/3.x" agent with a 403 (error 1010).
        "User-Agent": "deepspeed-course-runpod-ctl/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        # Capacity exhaustion is routine, not a misconfiguration — RunPod
        # returns 500 for it. Say so plainly instead of dumping the raw body.
        if "no instances currently available" in detail.lower():
            sys.exit(
                "\n  RunPod has no capacity for that GPU right now.\n"
                "  Nothing was created and nothing is billing.\n\n"
                "  Options:\n"
                "    - pick another GPU:   runpod_ctl.py gpus --min-vram <N>\n"
                "    - try community cloud: --cloud COMMUNITY  (cheaper, "
                "less reliable)\n"
                "    - retry shortly; availability changes minute to minute\n")
        if exc.code in (401, 403):
            sys.exit(f"\n  RunPod rejected the credentials ({exc.code}).\n"
                     "  Check RUNPOD_API_KEY at https://console.runpod.io/user/settings\n")
        sys.exit(f"RunPod API {exc.code} on {method} {url}\n  {detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach RunPod: {exc.reason}")


def gpu_catalogue(key: str):
    """GPU types with pricing. Lives in GraphQL; the REST API has no equivalent."""
    query = ("query{ gpuTypes { id displayName memoryInGb secureCloud communityCloud "
             "lowestPrice(input:{gpuCount:1}){ uninterruptablePrice minimumBidPrice } } }")
    out = _request(GRAPHQL, key, "POST", {"query": query})
    types = (out.get("data") or {}).get("gpuTypes") or []
    rows = []
    for g in types:
        price = (g.get("lowestPrice") or {}).get("uninterruptablePrice")
        if price:
            rows.append(dict(id=g["id"], name=g["displayName"], vram=g["memoryInGb"],
                             price=price,
                             spot=(g.get("lowestPrice") or {}).get("minimumBidPrice"),
                             secure=g.get("secureCloud"), community=g.get("communityCloud")))
    rows.sort(key=lambda r: r["price"])
    return rows


# ----------------------------------------------------------------------- commands
def cmd_gpus(args):
    rows = gpu_catalogue(api_key(args))
    rows = [r for r in rows if r["vram"] >= args.min_vram]
    if args.max_price:
        rows = [r for r in rows if r["price"] <= args.max_price]
    if not rows:
        print(f"No GPU with >= {args.min_vram} GB"
              + (f" under ${args.max_price}/hr" if args.max_price else ""))
        return 0
    print(f"\n{'$/hr':>7} {'spot':>7} {'VRAM':>6}  {'ID':34} NAME")
    print("-" * 88)
    for r in rows[: args.limit]:
        spot = f"{r['spot']:.2f}" if r["spot"] else "-"
        print(f"{r['price']:7.2f} {spot:>7} {r['vram']:5}G  {r['id'][:34]:34} {r['name']}")
    print(f"\n{len(rows)} match(es); showing {min(len(rows), args.limit)}.")
    return 0


def cmd_recommend(args):
    spec = EXAMPLES.get(args.example)
    if not spec:
        print(f"Unknown example {args.example!r}. Known:")
        for k in EXAMPLES:
            print("  ", k)
        return 1

    print(f"\n{args.example}")
    print(f"  {spec['note']}")
    print(f"  Needs: >= {spec['min_vram']} GB VRAM x {spec['gpus']} GPU, "
          f"{spec['disk']} GB disk")

    rows = [r for r in gpu_catalogue(api_key(args)) if r["vram"] >= spec["min_vram"]]
    if not rows:
        print("\n  No GPU in the catalogue meets that requirement.")
        return 1

    print(f"\n  Cheapest options ({spec['gpus']} GPU(s), on-demand):")
    print(f"    {'$/hr total':>11} {'VRAM':>6}  ID")
    for r in rows[:5]:
        print(f"    {r['price'] * spec['gpus']:11.2f} {r['vram']:5}G  {r['id']}")

    best = rows[0]
    print(f"\n  Suggested: {best['id']}  "
          f"(~${best['price'] * spec['gpus']:.2f}/hr for {spec['gpus']} GPU)")
    print(f"\n  uv run runpod/runpod_ctl.py run {args.example} --yes")
    if args.example == "09_vss":
        print("\n  WARNING: 09_vss needs ~3 TB of HOST RAM, which RunPod pods do not")
        print("  normally provide. GPU VRAM alone is not sufficient for this example.")
    return 0


def bootstrap(example: str, spec: dict, branch: str) -> str:
    """Shell run inside the pod: clone, install with uv, launch the example."""
    return "; ".join([
        "set -e",
        "echo '=== deepspeed-course bootstrap ==='",
        "cd /workspace",
        f"[ -d deepspeed-course ] || git clone --depth 1 -b {branch} {REPO_URL}",
        "cd deepspeed-course",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=$HOME/.local/bin:$PATH",
        "export HF_HOME=/workspace/hf_cache",
        "uv pip install --system deepspeed",
        f"cd {example}",
        "nvidia-smi",
        # Not every example uses the deepspeed launcher — 07_..._multi_agency
        # drives TRL's GRPOTrainer directly.
        (f"python {spec['script']}" if spec.get("launcher") == "python"
         else f"deepspeed --num_gpus={spec['gpus']} {spec['script']}")
        + " 2>&1 | tee /workspace/train.log",
        "echo '=== finished; log at /workspace/train.log ==='",
    ])


def cmd_create(args):
    key = api_key(args)
    rows = gpu_catalogue(key)
    match = next((r for r in rows if r["id"] == args.gpu), None)
    if not match:
        sys.exit(f"Unknown GPU id {args.gpu!r}. List them with: gpus --min-vram 0")

    hourly = match["price"] * args.count
    print(f"\n  GPU:   {match['name']}  ({match['vram']} GB) x{args.count}")
    print(f"  Image: {args.image}")
    print(f"  Disk:  {args.disk} GB")
    print(f"  Cost:  ~${hourly:.2f}/hour  (${hourly * 24:.2f}/day if left running)")
    if not args.yes:
        print("\n  Refusing to create without --yes. Billing starts immediately.")
        return 1

    payload = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": args.count,
        "containerDiskInGb": args.disk,
        "volumeInGb": args.volume,
        "volumeMountPath": "/workspace",
        "cloudType": args.cloud,
        "ports": ["22/tcp"],
        "env": {"HF_HOME": "/workspace/hf_cache"},
    }
    if args.start_cmd:
        payload["dockerStartCmd"] = ["bash", "-lc", args.start_cmd]

    pod = _request(f"{REST}/pods", key, "POST", payload)
    pid = pod.get("id", "?")
    print(f"\n  Created pod {pid}  (status {pod.get('desiredStatus')})")
    print(f"\n  Watch:      uv run runpod/runpod_ctl.py pods")
    print(f"  TERMINATE:  uv run runpod/runpod_ctl.py terminate {pid}")
    print("\n  Billing continues until TERMINATED (stopping is not enough).")
    return 0


def cmd_run(args):
    spec = EXAMPLES.get(args.example)
    if not spec:
        sys.exit(f"Unknown example {args.example!r}")
    key = api_key(args)

    gpu = args.gpu
    if not gpu:
        rows = [r for r in gpu_catalogue(key) if r["vram"] >= spec["min_vram"]]
        if not rows:
            sys.exit(f"No GPU with >= {spec['min_vram']} GB available.")
        gpu = rows[0]["id"]
        print(f"  Auto-selected cheapest fit: {gpu} ({rows[0]['vram']} GB, "
              f"${rows[0]['price']:.2f}/hr)")

    ns = argparse.Namespace(
        api_key=getattr(args, "api_key", None), gpu=gpu, count=spec["gpus"],
        disk=spec["disk"], volume=max(spec["disk"], 20), image=args.image,
        name=f"dsc-{args.example[:24]}", cloud=args.cloud, yes=args.yes,
        start_cmd=bootstrap(args.example, spec, args.branch),
    )
    rc = cmd_create(ns)
    if rc == 0:
        print("\n  The pod clones the repo and starts training automatically.")
        print("  RunPod's REST API exposes no log endpoint, so to watch progress:")
        print("    - web console, or")
        print("    - ssh root@<ip> -p <port>  then:  tail -f /workspace/train.log")
        print("    (get ip/port from `pods`; requires an SSH key on your account)")
    return rc


def cmd_pods(args):
    key = api_key(args)
    pods = _request(f"{REST}/pods", key)
    if isinstance(pods, dict):
        pods = pods.get("data", [])
    if not pods:
        print("\n  No pods. Nothing is billing.")
        return 0
    total = 0.0
    print(f"\n{'ID':22} {'STATUS':12} {'$/hr':>6}  NAME")
    print("-" * 74)
    for p in pods:
        cost = p.get("costPerHr") or p.get("adjustedCostPerHr") or 0
        total += float(cost)
        print(f"{p.get('id','?')[:22]:22} {str(p.get('desiredStatus'))[:12]:12} "
              f"{float(cost):6.2f}  {p.get('name','')}")
    print(f"\n  {len(pods)} pod(s), ~${total:.2f}/hour total.")
    ip_shown = False
    for p in pods:
        pm = p.get("portMappings") or {}
        if p.get("publicIp") and pm:
            print(f"  ssh root@{p['publicIp']} -p {pm.get('22', '?')}   # {p.get('id')}")
            ip_shown = True
    if not ip_shown:
        print("  (no public IP yet — pods take a minute to become reachable)")
    return 0


def cmd_terminate(args):
    key = api_key(args)
    for pid in args.pod_ids:
        _request(f"{REST}/pods/{pid}", key, "DELETE")
        print(f"  terminated {pid}")
    print("\n  Confirm with: uv run runpod/runpod_ctl.py pods")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Find a GPU, start a RunPod pod, run a deepspeed-course example.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--api-key", help="overrides RUNPOD_API_KEY")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gpus", help="list GPU types with live prices")
    g.add_argument("--min-vram", type=int, default=0)
    g.add_argument("--max-price", type=float)
    g.add_argument("--limit", type=int, default=20)
    g.set_defaults(func=cmd_gpus)

    r = sub.add_parser("recommend", help="suggest a GPU for a course example")
    r.add_argument("example")
    r.set_defaults(func=cmd_recommend)

    c = sub.add_parser("create", help="create a bare pod")
    c.add_argument("--gpu", required=True)
    c.add_argument("--count", type=int, default=1)
    c.add_argument("--disk", type=int, default=40)
    c.add_argument("--volume", type=int, default=40)
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--name", default="deepspeed-course")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    c.add_argument("--start-cmd")
    c.add_argument("--yes", action="store_true", help="required; billing starts at once")
    c.set_defaults(func=cmd_create)

    u = sub.add_parser("run", help="create a pod that clones the repo and trains")
    u.add_argument("example")
    u.add_argument("--gpu", help="default: cheapest that fits")
    u.add_argument("--image", default=DEFAULT_IMAGE)
    u.add_argument("--branch", default="main")
    u.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    u.add_argument("--yes", action="store_true", help="required; billing starts at once")
    u.set_defaults(func=cmd_run)

    l = sub.add_parser("pods", help="list pods and what they are costing")
    l.set_defaults(func=cmd_pods)

    t = sub.add_parser("terminate", help="terminate pods (stops billing)")
    t.add_argument("pod_ids", nargs="+")
    t.set_defaults(func=cmd_terminate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

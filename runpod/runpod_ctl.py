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
import urllib.parse
import urllib.request
import pathlib
import uuid

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
REPO_URL = "https://github.com/yiqiao-yin/deepspeed-course.git"

# Result transport. RunPod's API exposes NO log endpoint (verified against both
# the REST OpenAPI spec and GraphQL introspection), so the pod cannot be read
# from — it has to push. ntfy.sh is a no-auth pub/sub with a CLIENT-CHOSEN
# topic, which avoids the chicken-and-egg of "the pod knows the URL but we
# don't". Same shape as writing run artefacts to S3, without needing creds.
NTFY = os.environ.get("DSC_NTFY_SERVER", "https://ntfy.sh")

# Default image: a `devel` tag is REQUIRED. The `runtime` variants ship no nvcc,
# so DeepSpeed cannot JIT-compile its fused CUDA ops and every example fails
# with `CUDA_HOME environment variable is not set`.
DRY_RUN_SECONDS = 300   # cap on the training step during --dry-run
MAX_POLL_ERRORS = 12     # consecutive poll failures before giving up
DEFAULT_MAX_HOURS = 6.0  # in-pod watchdog: hard ceiling on a pod's lifetime
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
    "05_huggingface_dpo": dict(min_vram=24, gpus=1, disk=60,
                               script="train_dpo.py",
                               note="Offline PO: no generation, so far cheaper "
                                    "than GRPO. The reference model is the "
                                    "swing factor; LoRA removes it."),
    "05_huggingface_reward_model": dict(min_vram=24, gpus=1, disk=60,
                                        script="train_reward_model.py",
                                        note="Pairs mean 2x forward passes per "
                                             "batch; halve micro_batch vs SFT."),
    "06_huggingface_grpo": dict(min_vram=24, gpus=1, disk=80,
                                script="grpo_gsm8k_train.py",
                                note="RL; memory driven by G rollouts."),
    "06_huggingface_online_dpo": dict(min_vram=24, gpus=2, disk=80,
                                      script="train_online_dpo.py",
                                      note="Generates during training AND holds "
                                           "a judge: budget like GRPO."),
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
    # --- 08_vtt advanced track -------------------------------------------
    # Nested paths are deliberate. These are SUBSECTIONS of topic 08, each
    # runnable on its own pod, so you can rent a cheap card for the
    # measurement sweep without paying for the 2x48GB the LLaVA baseline
    # needs. `folder = REPO_ROOT / name` handles the slash, and `bootstrap`
    # cd's into it the same way.
    "08_vtt/01_qwen25vl_baseline": dict(min_vram=24, gpus=2, disk=100,
                                        script="train_qwen25vl.py",
                                        note="Qwen2.5-VL-3B + LoRA fits 16 GB; "
                                             "2 GPUs to exercise ZeRO-3."),
    "08_vtt/02_token_compression": dict(min_vram=24, gpus=1, disk=60,
                                        script="train_compressed.py",
                                        note="ONE GPU on purpose — measures peak "
                                             "VRAM vs sequence length."),
    "08_vtt/03_streaming_memory": dict(min_vram=24, gpus=1, disk=60,
                                       script="stream_infer.py",
                                       launcher="python",
                                       note="Streaming inference is sequential; "
                                            "the deepspeed launcher adds nothing."),
    "08_vtt/04_video_eval": dict(min_vram=24, gpus=1, disk=60,
                                 script="video_mme_eval.py",
                                 launcher="python",
                                 note="Evaluation is generate() calls, not "
                                      "training. No launcher needed."),
    "09_vss": dict(min_vram=180, gpus=2, disk=2000,
                   script="01_longcat_flash_omni/train_ds_2xB200.py",
                   note="NOT VIABLE on typical RunPod: needs ~3 TB HOST RAM."),
    # --- 09_vss subtopics -------------------------------------------------
    # Video-speech-to-speech: video AND audio in, speech out. The frontier
    # model above is not rentable; these three are, which is the point of
    # splitting them out.
    "09_vss/01_longcat_flash_omni": dict(min_vram=180, gpus=2, disk=2000,
                                         script="train_ds_2xB200.py",
                                         note="560B. Same host-RAM wall as 09_vss."),
    "09_vss/02_thinker_talker": dict(min_vram=24, gpus=2, disk=120,
                                     script="train_omni.py",
                                     note="Qwen2.5-Omni-3B + LoRA fits 24GB; "
                                          "two streams make the sequence long."),
    "09_vss/03_duplex_streaming": dict(min_vram=24, gpus=1, disk=80,
                                       script="run_duplex.py",
                                       launcher="python",
                                       note="Duplex inference is sequential; "
                                            "the deepspeed launcher adds nothing."),
    "09_vss/04_omni_eval": dict(min_vram=24, gpus=1, disk=80,
                                script="omni_eval.py",
                                launcher="python",
                                note="generate() calls, not training. "
                                     "No launcher needed."),
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
    # Only the LongCat subtopic hits the host-RAM wall. The other 09_vss
    # subtopics are ordinary 24 GB jobs and must NOT inherit this warning,
    # or readers will assume the whole topic is unrentable and skip it.
    if args.example in ("09_vss", "09_vss/01_longcat_flash_omni"):
        print("\n  WARNING: this example needs ~3 TB of HOST RAM, which RunPod pods")
        print("  do not normally provide. GPU VRAM alone is not sufficient.")
        print("  The other 09_vss subtopics run fine on a single 24 GB card:")
        print("      uv run runpod/runpod_ctl.py recommend 09_vss/02_thinker_talker")
    return 0


def bootstrap(example: str, spec: dict, branch: str,
              topic: str = "", dry_run: bool = False,
              max_hours: float = DEFAULT_MAX_HOURS) -> str:
    """Shell run inside the pod: clone, install with uv, run, push results out."""
    launch = (f"python {spec['script']}" if spec.get("launcher") == "python"
              else f"deepspeed --num_gpus={spec['gpus']} {spec['script']}")

    # A dry run proves the pod can reach the code and the stack imports, without
    # paying for a full training job. It still executes the real launcher, but
    # under a timeout, so a genuine crash is still caught.
    if dry_run:
        launch = f"timeout {DRY_RUN_SECONDS} {launch} || true"

    report = (f'report(){{ curl -s -m 15 -d "$1" {NTFY}/{topic} >/dev/null 2>&1 || true; }}'
              if topic else 'report(){ :; }')
    push_log = (
        f'curl -s -m 60 -T /workspace/run.log -H "Filename: {example}.log" '
        f'{NTFY}/{topic} >/dev/null 2>&1 || true'
    ) if topic else 'true'

    # In-pod watchdog. The local watcher (`--terminate`) is the primary way a
    # pod gets shut down, but it dies with your laptop lid. This backstop kills
    # the container's main process after max_hours no matter what, so a hung run
    # cannot bill indefinitely. It does NOT need the API key on the pod.
    # NOTE the trailing `true`. These steps are joined with "; ", and a command
    # ending in `&` followed by `;` is a bash SYNTAX ERROR:
    #
    #     ( sleep 21600; ... ) &; report "[1/6] pod up"
    #                            ^ syntax error near unexpected token `;'
    #
    # The whole start command is one `bash -lc` string, so a parse error means
    # the container runs NOTHING -- it is created, it bills, and it never
    # clones, never trains and never reports. Silent, and expensive.
    # `&` already terminates the command; `true` gives the join something
    # valid to attach its `;` to.
    watchdog = (
        f'( sleep {int(max_hours * 3600)}; echo "[watchdog] {max_hours}h cap reached";'
        f' kill -TERM 1 2>/dev/null || true ) & true'
    ) if max_hours else "true"

    steps = [
        "set -o pipefail",
        report,
        watchdog,
        'report "[1/6] pod up: $(hostname)"',
        "cd /workspace",
        f"(git clone --depth 1 -b {branch} {REPO_URL} 2>&1 || true) | tail -2",
        "cd deepspeed-course",
        'report "[2/6] repo cloned"',
        "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1",
        "export PATH=$HOME/.local/bin:$PATH",
        "export HF_HOME=/workspace/hf_cache",
        'report "[3/6] uv installed: $(uv --version 2>&1)"',
        "uv pip install --system deepspeed 2>&1 | tail -3",
        'report "[4/6] deepspeed installed"',
        # Environment capture — this is the payload we actually want back.
        "{ echo '=== nvidia-smi ==='; nvidia-smi;"
        " echo '=== versions ==='; python -c \"import torch,sys;"
        "print('python',sys.version.split()[0]);print('torch',torch.__version__);"
        "print('cuda',torch.cuda.is_available(),torch.cuda.device_count());"
        "print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\";"
        " python -c \"import deepspeed;print('deepspeed',deepspeed.__version__)\";"
        " echo '=== ds_report ==='; ds_report 2>&1 | head -30; } > /workspace/run.log 2>&1",
        'report "[5/6] env captured"',
        f"cd {example}",
        f"({launch}) >> /workspace/run.log 2>&1",
        "tail -40 /workspace/run.log",
        push_log,
        'report "[6/6] DONE rc=$? — log attached"',
    ]
    return "; ".join(steps)


def cmd_create(args):
    return 0 if _create_pod(args) is not None else 1


def _create_pod(args):
    """Create a pod; return the API's pod dict, or None if refused/failed."""
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
        return None

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
    print(f"  TERMINATE:  uv run runpod/runpod_ctl.py terminate {pid}")
    print("  Billing continues until TERMINATED (stopping is not enough).")
    return pod


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

    topic = ""
    if args.collect:
        # Unguessable, client-chosen topic: we know where to look before the
        # pod has said anything, which is what makes this work without SSH.
        topic = f"dsc-{uuid.uuid4().hex[:20]}"

    ns = argparse.Namespace(
        api_key=getattr(args, "api_key", None), gpu=gpu, count=spec["gpus"],
        disk=spec["disk"], volume=max(spec["disk"], 20), image=args.image,
        name=f"dsc-{args.example[:24]}", cloud=args.cloud, yes=args.yes,
        start_cmd=bootstrap(args.example, spec, args.branch, topic,
                            args.dry_run, args.max_hours),
    )
    created = _create_pod(ns)
    if created is None:
        return 1
    pod_id = created.get("id", "")

    if not topic:
        print("\n  No --collect, so results stay on the pod. Add --collect next time.")
        return 0

    print(f"\n  Results topic: {topic}")

    if not args.wait:
        print(f"  Collect and shut down with:")
        print(f"      uv run runpod/runpod_ctl.py fetch {topic} --wait "
              f"--terminate --pod {pod_id}")
        print(f"\n  Or terminate now:  runpod_ctl.py terminate {pod_id}")
        print("\n  NOTE: the pod bills until terminated. --dry-run caps the training")
        print(f"  step, not the pod's life; the in-pod watchdog stops it after "
              f"{args.max_hours}h.")
        return 0

    # --- blocking path: wait, collect, and ALWAYS clean up ---------------
    print("  Waiting for the pod to report DONE (Ctrl-C is safe — the pod is")
    print("  still terminated on the way out)...\n")
    key = api_key(args)
    done = False
    try:
        done = collect(topic, True, args.wait_seconds, args.interval)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        if args.terminate:
            # `finally` is the whole point: a crash, a timeout or Ctrl-C must
            # not leave a pod billing. Retry hard — this path costs real money
            # if it fails silently.
            print(f"\n  Terminating {pod_id} ...")
            if terminate_with_retry(pod_id, key):
                print(f"  terminated {pod_id}")
            else:
                print(f"  !! COULD NOT TERMINATE {pod_id} AFTER {TERMINATE_RETRIES} TRIES")
                print(f"  !! IT IS STILL BILLING. Run this yourself:")
                print(f"     uv run runpod/runpod_ctl.py terminate {pod_id}")
                print(f"     or: https://console.runpod.io/pods")
            try:
                remaining = list_pods(key)
                print(f"  pods still running: {len(remaining)}"
                      + (f" -> {[p['id'] for p in remaining]}" if remaining else " (none)"))
            except SystemExit:
                print("  (could not verify remaining pods — check the console)")
        else:
            print(f"\n  --terminate not set; pod {pod_id} is STILL BILLING.")
            print(f"     uv run runpod/runpod_ctl.py terminate {pod_id}")
    return 0 if done else 1


def collect(topic: str, wait: bool, wait_seconds: int, interval: int) -> bool:
    """Poll a topic, save artefacts locally, return True once DONE is seen."""
    out_dir = pathlib.Path(__file__).resolve().parent / "results" / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + (wait_seconds if wait else 0)
    seen, attachments, done = set(), {}, False
    consecutive_errors = 0

    while True:
        url = f"{NTFY}/{urllib.parse.quote(topic)}/json?poll=1"
        req = urllib.request.Request(url, headers={"User-Agent": "dsc-runpod-ctl/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                lines = resp.read().decode().splitlines()
        except Exception as exc:
            # Transient DNS/network blips are common and must NOT be read as
            # "the run is over" — that would abandon a pod mid-job.
            consecutive_errors += 1
            if consecutive_errors == 1 or consecutive_errors % 5 == 0:
                print(f"  (network hiccup #{consecutive_errors}: {exc}; retrying)")
            if consecutive_errors >= MAX_POLL_ERRORS:
                print(f"  giving up after {MAX_POLL_ERRORS} consecutive failures")
                break
            if not wait or time.time() > deadline:
                break
            time.sleep(interval)
            continue
        consecutive_errors = 0

        for line in lines:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") != "message" or msg["id"] in seen:
                continue
            seen.add(msg["id"])
            if msg.get("message"):
                print(f"  {msg['message']}")
                if "DONE" in msg["message"]:
                    done = True
            att = msg.get("attachment")
            if att and att.get("url"):
                attachments[att.get("name", "artifact")] = att["url"]

        if done or not wait or time.time() > deadline:
            break
        time.sleep(interval)

    for name, url in attachments.items():
        target = out_dir / name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dsc-runpod-ctl/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                target.write_bytes(resp.read())
            print(f"\n  saved {target}  ({target.stat().st_size} bytes)")
        except Exception as exc:
            print(f"  could not download {name}: {exc}")

    if not seen:
        print("  Nothing published yet — pods take a few minutes to boot and install.")
    elif not done:
        print("\n  (no DONE marker — the run may still be going)")
    return done


def cmd_fetch(args):
    """Poll the results topic and save everything under runpod/results/<topic>/."""
    collect(args.topic, args.wait, args.wait_seconds, args.interval)
    if args.terminate:
        if not args.pod:
            print("\n  --terminate needs --pod <id>. Find it with: pods")
            return 1
        cmd_terminate(argparse.Namespace(
            api_key=getattr(args, "api_key", None), pod_ids=[args.pod], all=False))
    return 0


def cmd_smoke(args):
    """
    Dry-run several examples, each on its own pod, collecting results.

    This is the cheap way to answer "does topic N actually run on real
    hardware?" for the examples that cannot be tested locally. Each pod is
    independent, so one failure does not block the rest.
    """
    targets = args.examples or [
        "01_basic_neuralnet", "02_basic_convnet",
        "02_basic_convnet_cifar10_examples", "03_basic_rnn",
    ]
    unknown = [t for t in targets if t not in EXAMPLES]
    if unknown:
        sys.exit(f"Unknown example(s): {unknown}")

    key = api_key(args)
    catalogue = gpu_catalogue(key)

    total = 0.0
    for name in targets:
        spec = EXAMPLES[name]
        fits = [g for g in catalogue if g["vram"] >= spec["min_vram"]]
        total += (fits[0]["price"] * spec["gpus"]) if fits else 0.0

    print(f"\n  Will start {len(targets)} pod(s), one per example:")
    for name in targets:
        spec = EXAMPLES[name]
        fits = [g for g in catalogue if g["vram"] >= spec["min_vram"]]
        price = (fits[0]["price"] * spec["gpus"]) if fits else 0
        print(f"    {name:46} {spec['gpus']}x{spec['min_vram']:>3}G  ~${price:.2f}/hr")
    print(f"\n  Combined burn rate: ~${total:.2f}/hour while they run.")
    print("  Each is capped by --dry-run, but pods are NOT auto-terminated —")
    print("  they stop billing only when you terminate them.")

    if not args.yes:
        print("\n  Refusing without --yes.")
        return 1

    topics = {}
    for name in targets:
        spec = EXAMPLES[name]
        fits = [g for g in catalogue if g["vram"] >= spec["min_vram"]]
        if not fits:
            print(f"  {name}: no GPU with >= {spec['min_vram']} GB — skipped")
            continue
        topic = f"dsc-{uuid.uuid4().hex[:20]}"
        ns = argparse.Namespace(
            api_key=getattr(args, "api_key", None), gpu=fits[0]["id"],
            count=spec["gpus"], disk=spec["disk"], volume=max(spec["disk"], 20),
            image=args.image, name=f"dsc-{name[:24]}", cloud=args.cloud, yes=True,
            start_cmd=bootstrap(name, spec, args.branch, topic, dry_run=True),
        )
        print(f"\n--- {name} ---")
        if cmd_create(ns) == 0:
            topics[name] = topic

    if topics:
        print("\n  Collect results with:")
        for name, topic in topics.items():
            print(f"    uv run runpod/runpod_ctl.py fetch {topic} --wait   # {name}")
        print("\n  THEN TERMINATE EVERYTHING:")
        print("    uv run runpod/runpod_ctl.py pods")
        print("    uv run runpod/runpod_ctl.py terminate <id> [<id> ...]")
    return 0


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


TERMINATE_RETRIES = 5


def terminate_with_retry(pod_id: str, key: str) -> bool:
    """Delete a pod, retrying through transient network failures.

    Returns True on success. A silent failure here bills indefinitely, so this
    retries and the caller shouts loudly if it still fails.
    """
    for attempt in range(1, TERMINATE_RETRIES + 1):
        try:
            data = json.dumps(None).encode() if False else None
            req = urllib.request.Request(
                f"{REST}/pods/{pod_id}", data=data, method="DELETE",
                headers={"Authorization": f"Bearer {key}",
                         "User-Agent": "deepspeed-course-runpod-ctl/1.0"})
            urllib.request.urlopen(req, timeout=30).read()
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):      # already gone
                return True
            print(f"    attempt {attempt}/{TERMINATE_RETRIES}: HTTP {exc.code}")
        except Exception as exc:
            print(f"    attempt {attempt}/{TERMINATE_RETRIES}: {exc}")
        if attempt < TERMINATE_RETRIES:
            time.sleep(min(2 ** attempt, 20))
    return False


def list_pods(key):
    pods = _request(f"{REST}/pods", key)
    return pods.get("data", []) if isinstance(pods, dict) else (pods or [])


def cmd_terminate(args):
    key = api_key(args)
    pod_ids = args.pod_ids
    if args.all:
        pod_ids = [p["id"] for p in list_pods(key)]
        if not pod_ids:
            print("  No pods. Nothing to terminate.")
            return 0
        print(f"  Terminating ALL {len(pod_ids)} pod(s): {', '.join(pod_ids)}")
    if not pod_ids:
        sys.exit("Give pod ids, or --all.")
    for pid in pod_ids:
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
    u.add_argument("--collect", action="store_true",
                   help="have the pod push progress and its log back (no SSH needed)")
    u.add_argument("--wait", action="store_true",
                   help="block until the pod reports DONE, collecting as it goes")
    u.add_argument("--terminate", action="store_true",
                   help="with --wait: terminate the pod afterwards, even on error")
    u.add_argument("--wait-seconds", type=int, default=1800)
    u.add_argument("--interval", type=int, default=20)
    u.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS,
                   help="in-pod watchdog: hard ceiling on pod lifetime")
    u.add_argument("--dry-run", action="store_true",
                   help=f"cap the training step at {DRY_RUN_SECONDS}s — proves the "
                        f"pipeline works without paying for a full run")
    u.set_defaults(func=cmd_run)

    f = sub.add_parser("fetch", help="download results the pod pushed")
    f.add_argument("topic")
    f.add_argument("--wait", action="store_true", help="block until the pod reports DONE")
    f.add_argument("--wait-seconds", type=int, default=1800)
    f.add_argument("--interval", type=int, default=20)
    f.add_argument("--terminate", action="store_true", help="terminate --pod when done")
    f.add_argument("--pod", help="pod id to terminate")
    f.set_defaults(func=cmd_fetch)

    sm = sub.add_parser("smoke", help="dry-run several examples, one pod each")
    sm.add_argument("examples", nargs="*", help="default: the four CPU-scale examples")
    sm.add_argument("--image", default=DEFAULT_IMAGE)
    sm.add_argument("--branch", default="main")
    sm.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    sm.add_argument("--yes", action="store_true", help="required; billing starts at once")
    sm.set_defaults(func=cmd_smoke)

    l = sub.add_parser("pods", help="list pods and what they are costing")
    l.set_defaults(func=cmd_pods)

    t = sub.add_parser("terminate", help="terminate pods (stops billing)")
    t.add_argument("pod_ids", nargs="*")
    t.add_argument("--all", action="store_true", help="terminate every pod on the account")
    t.set_defaults(func=cmd_terminate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

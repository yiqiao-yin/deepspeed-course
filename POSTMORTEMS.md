# POSTMORTEMS.md

Full accounts of bugs this repository has shipped, and what was done about them.

`CLAUDE.md` carries the **rules** these incidents produced, each with a one-line
hook and a link to the section here. This file carries the **evidence** — the
measurements, the misleading error messages, the reasoning that was wrong and
why. Read a section when you are about to touch the area it describes, or when a
rule in `CLAUDE.md` looks arbitrary and you are tempted to undo it.

The common thread, stated once so it need not be repeated in every section:
**every serious bug here ran fine and was quietly wrong.** None of them crashed,
most of them exited 0, and several printed a success banner. That is why the
rules are phrased as "assert the property" rather than "be careful".

---

## Contents

**Data and training signal**
- [Synthetic data must carry a signal, and the summary must not lie](#synthetic-data-must-carry-a-signal-and-the-summary-must-not-lie)
- [A slow data source can make a lab unrunnable, and it looks like success](#a-slow-data-source-can-make-a-lab-unrunnable-and-it-looks-like-success)

**Distributed execution**
- [Only rank 0 downloads, and the others must wait on a barrier](#only-rank-0-downloads-and-the-others-must-wait-on-a-barrier)
- [Guard the output, never the collective](#guard-the-output-never-the-collective)
- [Sizing multi-GPU jobs: model it per GPU, not in aggregate](#sizing-multi-gpu-jobs-model-it-per-gpu-not-in-aggregate)
- ["Rent a bigger box" was the wrong answer three times running](#rent-a-bigger-box-was-the-wrong-answer-three-times-running)

**Packaging and environment**
- [A custom torch index pins its companions too, or nothing works](#a-custom-torch-index-pins-its-companions-too-or-nothing-works)
- [A verification harness that does not install the artifact verifies nothing](#a-verification-harness-that-does-not-install-the-artifact-verifies-nothing)
- [Library API drift comes in three classes, and only one is obvious](#library-api-drift-comes-in-three-classes-and-only-one-is-obvious)

**Harness and manifest integrity**
- [A lab is its COMMAND, not just its code](#a-lab-is-its-command-not-just-its-code)
- [Three green checkers, one broken lab](#three-green-checkers-one-broken-lab)
- [The RunPod harness lies less than it used to](#the-runpod-harness-lies-less-than-it-used-to)
- [Watch a checker fail before trusting it](#watch-a-checker-fail-before-trusting-it)

**Claims and measurement**
- [A measured claim is scoped to the configuration it was measured in](#a-measured-claim-is-scoped-to-the-configuration-it-was-measured-in)
  - [Second instance: the figure was right and the length was wrong](#second-instance-the-figure-was-right-and-the-length-was-wrong)
- [A script with no end-to-end run makes its derivations the only testable surface](#a-script-with-no-end-to-end-run-makes-its-derivations-the-only-testable-surface)

---

## Synthetic data must carry a signal, and the summary must not lie

`01_basics/02_convnet` drew `x = randn(...)` and `y = randint(...)` — labels
independent of the images, so **zero mutual information**. On 10 classes ~10%
was not a poor result, it was the information-theoretic **ceiling**. The script
none the less exited 0, printed "Finished Successfully", and advised *"Poor.
Consider training longer or adjusting hyperparameters"* — sending a reader to
tune a target that cannot be reached. Two 3090 runs returned 10.29% and 9.49%,
unchanged from first epoch to last: a classifier collapsing to one class, the
correct degenerate answer when there is nothing to learn.

Every other lab in `01_basics` already used learnable synthetic data
(`y = 2x + 1`, a sum of sines); this one was the exception. It now builds a
fixed prototype per class plus noise, and the noise default is **calibrated,
not guessed** — at 5.0 a plain MLP hits 99% in one epoch, because 784
dimensions of signal average out per-pixel noise.

**That calibration was done against the wrong model, and it shipped.** The
numbers came from a plain MLP; the lab trains a **CNN**, and on this data the
two behave oppositely. The signal is a fixed per-class prototype, the noise is
i.i.d. per pixel — an MLP averages 784 weakly-informative dimensions and wins,
while a CNN's small kernels see too few pixels to average anything and
max-pooling over noisy pixels selects the largest *noise* value. Measured
held-out at the old default of 8.0:

| model | 1 epoch | 10 epochs |
|---|---|---|
| MLP (what the *test* used) | 88.75% | 90.95% |
| **CNN (what the *lab* ships)** | **8.85%** | 13.70% |

Chance is 10%, so the lab's own model sat **at chance** for months. A full
50-epoch production run reached 35.19% and reported "Poor" — and was *right* to,
because nothing better was reachable. The default is now **2.5**, calibrated
against the shipped CNN at the batch size `ds_config.json` actually uses. 2.0
was rejected because one epoch reaches 92.65%, and a smoke test that already
saturates stops discriminating.

**Then the replacement number was wrong too, in the same way.** "~77% at one
epoch" was published from a *clean-room* harness — plain torch, constant LR,
fp32 — not from the lab. Measured end to end, two machines got **61.2%** and
**48.4%** on the same commit: DeepSpeed's fused Adam with fp16 against torch's
Adam without. Both reach ~100% by fifty epochs.

So the docs now give a **range**, and say that the durable property is *"well
clear of the 10% chance floor"* rather than any particular number.
`tests/test_synthetic_data_is_learnable.py` keeps a deliberately loose `>25%`
threshold and explains why it is loose — a test pinned to 61% would fail on
half the hardware, and tightening it would trade a real property for a fragile
one.

**Measure the thing that ships, in the way it ships.** The first version of this
bug used the wrong *model*; the second used the right model with the wrong
*optimizer and precision*. Both produced a confident number that no learner
would ever see, and a published number a reader cannot reproduce costs them a
day deciding their own correct setup is broken.

`tests/test_synthetic_data_is_learnable.py` passed throughout, because
`beats_chance()` built its own small MLP. **A learnability test that measures a
model no learner runs is the same failure as validating a library version no
learner installs** — the sibling mistake described under
[library API drift](#library-api-drift-comes-in-three-classes-and-only-one-is-obvious).
It now takes a `model_fn` and the 02_convnet checks pass the lab's own
`CNNModelEnhanced`, at the lab's batch size and sample count; measuring at batch
256 gave ~23 gradient steps per epoch against the lab's ~312 and under-reported
badly. It also asserts that `--noise`'s argparse default and `get_data_loader`'s
own `noise=` default agree, because they are different numbers and the suite
reads the second while the lab runs the first.

### The warmup bug: any hyperparameter expressed in epochs is a bug waiting for a short run

**A second defect surfaced only by running the lab end to end**, and no static
check would have found it: `warmup_epochs` was hardcoded to `5` at the call site
while `total_epochs` was passed through. A `--epochs 1` run therefore spent its
*only* epoch at `0.001 x 1/5` — a fifth of the target rate. The schedule was
written for the 50-epoch default and silently crippled every short run, which is
precisely the run the manifest offers. Measured at one epoch: **24.32% before,
61.21% after**. Warmup now scales as `min(5, max(1, total_epochs // 5))`, which
leaves the 50-epoch schedule identical. Check `--max-steps` and low `--epochs`
paths against every schedule, not just the default.

### A short run must not be reported as a failure — and this recurred

Clawdeck runs these with `--epochs 1`; printing "Poor" under a "Finished
Successfully" banner is how a beginner concludes they broke something. Say the
run was capped and what a real one looks like.

**This rule is not self-enforcing.** `03_convnet_cifar10` shipped the identical
defect months later: `--max-steps 20` — the command Clawdeck offers by name as
*"Quick (20 steps)"* — printed *"Poor. Consider training longer"* directly above
*"Finished Successfully"*. Twenty steps is ~1,280 of 50,000 images and chance on
CIFAR-10 is 10%, so ~10% was the **expected** result. Unlike `02_convnet` the
data was real and learnable; only the reporting lied. It was found by *running
the lab*, not by reading it.

`tests/test_synthetic_data_is_learnable.py` asserts learnability on a HELD-OUT
split — memorising random labels on the training set is possible and proves the
opposite of learning. It carries the old generator as a counterexample and
asserts it FAILS, because a learnability check that never sees unlearnable data
would pass while returning True unconditionally.

---

## A slow data source can make a lab unrunnable, and it looks like success

`01_basics/03_convnet_cifar10` fetched CIFAR-10 through
`torchvision.datasets.CIFAR10(download=True)`, i.e. from `cs.toronto.edu`.
Measured raw fetch from two unrelated networks — a rented cloud box and a home
connection:

| source | speed | 170 MB takes |
|---|---|---|
| `cs.toronto.edu` | 73–82 kB/s | **~40 minutes** |
| `huggingface.co` | 30–40 MB/s | **~6 seconds** |

~400×. The lab was **unusable on a cold box**: the download outlived the
orchestrator's 900 s window, so the job was reported *finished* having never
reached a single training step — no loss, no accuracy, no verdict, just
progress bars. A lab that cannot finish is worse than one that fails, because
it fails silently.

It now loads from the HuggingFace mirror. Three things worth keeping in mind:

- **Capping steps does not cap the download.** `--max-steps 20` already trains
  on ~1,280 of 50,000 images, but `torchvision` fetches the entire archive
  before it can read one image. You cannot subset an archive fetch; only
  changing the *source* helps.
- **You cannot just point torchvision at a faster URL.** It md5-verifies each
  extracted pickle, so only the original byte-identical archive passes, and no
  mirror of that archive exists on the Hub. The dataset has to be loaded
  differently, not merely fetched from elsewhere.
- **A mirror is a trust decision, so assert it.** Data that is *nearly*
  CIFAR-10 — a different split boundary, a subset, permuted label indices —
  trains fine and quietly produces numbers comparable to nothing.
  `tests/test_cifar10_source.py` checks 50,000/10,000 rows, exactly 5,000 and
  1,000 per class, RGB 32×32, and torchvision's canonical **label order**. That
  last one matters most: the same images under permuted indices score
  identically and caption every prediction wrong.

`huggingface_hub` takes `.lock` files, so the rank-guard race below can no
longer happen here — but the guard stays, because concurrency-safe is not free.
Without it every rank does the same fetch and decode.

---

## Only rank 0 downloads, and the others must wait on a barrier

`01_basics/03_convnet_cifar10` had a `download_cifar10()` whose docstring read
*"This prevents multiple processes from downloading simultaneously"* and whose
body did no such thing. Under the lab's own manifest command,
`deepspeed --num_gpus=2`, both ranks wrote the same 170 MB tarball into the same
`./data` and extracted over each other:

    RuntimeError: Dataset not found or corrupted. You can use download=True ...

— a spectacularly misleading message for a file that downloaded fine, twice. The
tell is **two interleaved progress bars both reaching 170M**.

`torchvision.datasets.*(download=True)` does **no locking.** HuggingFace
`from_pretrained` / `snapshot_download` / `load_dataset` go through
`huggingface_hub`, which takes `.lock` files and survives concurrency — so this
is specific to the torchvision path, not a general claim about downloads.

Three things make it worth a static check:

- **It passes on one GPU.** A single rank cannot race itself, so every 1-GPU
  smoke test is green.
- **It is syntactically valid**, so `compileall` cannot see it.
- **It was masked for months by an accident.** The data used to be re-hydrated
  onto the box at boot, so `./data` was already populated and neither rank ever
  downloaded. Cleaning that up surfaced a race that had always been there.

The barrier matters as much as the guard: without it rank 1 skips the download
and races ahead to read a directory rank 0 is still writing, which fails only
*sometimes*. `train_modern_cifar10.py` in the same folder had it right all
along (`download=is_main`, then `barrier()`) and is the pattern to copy.

### Fixing it exposed a second bug, in the fix itself

The guard worked — rank 1 waited, the download happened once — and the job then
died twelve minutes later *in the barrier*, `ALLREDUCE` with `NumelIn=1` running
for 721,595 ms. NCCL implements `barrier()` as an all-reduce of a one-element
tensor, so it must pick a device, and torch chooses: (1) `barrier(device_ids=)`,
(2) the device bound at `init_process_group`, (3) CPU, else (4) **the current
device — cuda:0 on every rank.** torch's own source warns this "may use default
device 0, causing issues like hang or all processes creating context on device
0."

`deepspeed.initialize()` normally binds the device for you. **A download guard
runs before `initialize()` by design, so it is precisely the window where this
bites.** Call `torch.cuda.set_device(local_rank)` before the collective and pass
`device_ids=` to the barrier.

There is deliberately **no `timeout=`** on that barrier: torch 2.13 accepts one,
and **torch 2.11 — what every lab here locks — does not.** Passing it raises
`TypeError` on the rented GPU after both ranks have launched. That bug shipped
because the signature was read out of whichever torch a `find` returned first;
the uv cache held 2.9, 2.10, 2.11, 2.13 and 2.14, and only the last two have the
parameter. **Verify an API against the version the lab's `uv.lock` resolves, not
against whatever is on the box.** `tests/test_config_kwargs.py` now covers
`torch.distributed` free functions for this reason, and normalises
`2.11.0+cu128` to `2.11.0` when comparing.

`train_modern_cifar10.py` — held up above as the reference — had the same latent
defect, `set_device` sitting *six lines below* its barrier. It had simply never
been run cold on two GPUs. Copying a sibling is only as safe as the sibling's
test coverage.

### Derive `local_rank` from the environment, not from argv

The same folder's `train_modern_cifar10.py` took its device from
`args.local_rank`, which argparse defaults to `-1`. Under the `deepspeed`
launcher that works — `launch.py` sets `RANK`, `LOCAL_RANK` and `WORLD_SIZE` in
each child's environment *and* injects `--local_rank` into argv. Under
`torchrun`, which sets only the environment variable, every rank computes
`max(-1, 0) == 0` and binds cuda:0: the same hang, one launcher away. Read
`LOCAL_RANK` first and fall back to argv.

### The static check

`tests/test_multigpu_download_guard.py` enforces both properties for every lab
`clawdeck.yaml` declares as `gpu.count > 1`. It is **AST-based, not a grep**,
and that distinction is the whole point — several scripts here rank-guard their
*printing* and *checkpoint saving* while downloading unguarded, so a file-wide
`grep get_rank() == 0` passes them all. It asks instead whether each individual
download call is lexically inside a rank-gated branch. Its permanent
counterexamples include the bug exactly as it shipped **and** a file whose rank
guard sits around the wrong statement.

---

## Guard the output, never the collective

The section above is about making sure only rank 0 does the *work*. This is its
mirror image, and it cost eleven minutes of silence on two independent boxes
before anyone could see it.

`03_llms/11_moe/train_moe_ds.py` ended its training loop with:

```python
if not is_main:
    return                                   # other ranks leave
...
model_engine(xe.unsqueeze(0))                # rank 0 runs the eval forward
```

On the default path that is harmless, because `MoELayer.forward` contains no
collectives. Under `--expert-parallel` the identical call is an **all-to-all
requiring every rank**, and the others had already gone. Training completed,
then the surviving rank sat in the collective until NCCL's watchdog aborted
it: a `SIGABRT` and a non-zero exit, with **no Python traceback anywhere.**

Two things make it worth its own entry:

- **It is invisible on the path you test.** Single-GPU runs pass. The
  non-EP multi-GPU runs pass, on the same box, in 60 s. Only the communicating
  layer fails, and only above one rank.
- **The job SUCCEEDS first.** A learner watches 500 steps of falling loss, then
  eleven minutes of nothing, then "failed". That reads as *"I broke it"* about
  a run that worked, which is worse than a fast crash.

The rule generalises past MoE: **a rank guard may wrap printing, logging and
saving; it must never wrap a collective.** Run the collective on every rank and
guard only what it prints. And tear down deliberately — `barrier()` then
`destroy_process_group()` on every rank — or a fast rank exits while a slow one
is still inside a collective, which is the same bug moved to shutdown.

Related, found while building a two-rank harness for this: **an unused expert
produces no gradient, so `if p.grad is not None` makes ranks disagree on how
many all-reduces to issue** and gloo dies with "Connection reset by peer".
Sync every parameter unconditionally, materialising zeros.

---

## Sizing multi-GPU jobs: model it per GPU, not in aggregate

The weights shard under ZeRO-3. **Activations, gather buffers and
fragmentation do not** — every rank pays those in full. An aggregate
"total VRAM vs the weights" check passed 2 × 48 GB for a 55.6 GB model that
then OOMed at the first step with 44.25 GiB resident on a 44.39 GiB card.

$$\text{per GPU} = \frac{\text{weights}}{N} + \text{overhead that does not shard}$$

Two signatures worth recognising:

- **An OOM whose requested allocation is trivially small** (60 MiB) on hardware
  that should have tens of GB spare means **sharding never happened**, not that
  you are marginally short. Under ZeRO-3 the DeepSpeed config must exist
  *before* `from_pretrained`, or `zero.Init` never fires and every rank
  materialises the whole model. Build `SFTConfig`/`TrainingArguments` first.
- **A collective whose payload is one element is a barrier.** A 1,800,069 ms
  timeout on an `ALLREDUCE` with `NumelIn=1` is never a model problem — it is
  the box advertising peer-to-peer it cannot perform. `nvidia-smi topo -m`
  showing `SYS` between cards is the tell; `NCCL_P2P_DISABLE=1` is the fix, at
  a real throughput cost. `tests/gpu/diagnose_nccl.sh` decides it in a minute.

---

## "Rent a bigger box" was the wrong answer three times running

Twice the symptom looked like size and was not, and a larger machine would have
**masked** each rather than fixed it:

| symptom | looked like | actually was |
|---|---|---|
| `03_ocr` OOM at 24 GB | needs 48 GB | a command missing `--use-lora` |
| `06_qwen3vl` 2-GPU hang | 8.77 B too big | NCCL peer-to-peer |

The second is the cleaner case. The hang was
`WorkNCCL(SeqNum=6, OpType=BROADCAST, **NumelIn=1152**)` — a few kilobytes.
**A collective whose payload is tiny cannot be a memory event.** It is the box
advertising peer-to-peer it cannot perform, and `NCCL_P2P_DISABLE=1` fixed it
immediately. Loaded, the model used 10.1 GB of a 47.7 GB card — 21%. Memory was
never near the constraint.

What settled it was reproducing on **two independent boxes**. One hang is a
bad-host lottery; the same hang on different hardware is a property of the code
or the provider, and that is a different investigation. `06_qwen3vl` carries
`--no-p2p` and `--load-only` for exactly this, and `--load-only` answers "will
this shard on my GPUs" in one step without a training run.

---

## A custom torch index pins its companions too, or nothing works

`01_basics/03_convnet_cifar10` passed every check and still could not run:

    RuntimeError: operator torchvision::nms does not exist

raised from inside `torch/_library/fake_impl.py`, which reads like a torch bug
and is not one. `[tool.uv.sources]` pinned `torch` to the cu128 index but not
`torchvision`, and with `explicit = true` only the packages named there come
from that index — so torch resolved to `2.11.0+cu128` while torchvision came
from PyPI, built against a different torch. Its compiled `_C.so` never
registered its ops.

**Any package with a compiled extension linked against torch must appear in
`[tool.uv.sources]` whenever torch does** — `torchvision`, `torchaudio`. Fixing
one and not the other is the easy mistake: `05_video_speech/01_longcat_omni`
had the identical bug with `torchaudio` and was found only by generalising the
report.

`tests/test_torch_index_pins.py` guards it by reading the **lock**, not the
pyproject — the resolution rather than the declaration — so a lock regenerated
against a different index fails even while `pyproject.toml` still looks right.

---

## A verification harness that does not install the artifact verifies nothing

`runpod/runpod_ctl.py run` is the tool that proves an example works on real
hardware. For most of its life it did:

```bash
uv pip install --system deepspeed        # and nothing else
deepspeed --num_gpus=N train_x.py        # the SYSTEM interpreter
```

It never ran `uv sync`. So it exercised whatever the **container image**
happened to ship, not the example's committed lock — and the six-file contract
is built entirely around that lock. The harness was verifying the image.

What that hid: **eight labs could not start from a fresh clone.** Every lab
calling `AutoProcessor.from_pretrained` was missing `pillow`, `torchvision` or
both, because a transformers image/video processor imports them and nothing
declared it:

    ValueError: Could not load any image processor class for ...
    ImportError: Qwen2VLVideoProcessor requires the Torchvision library

Neither message names the missing package. All eight were advertised in
`clawdeck.yaml`. CI's `compileall` cannot see it — an import that only runs
inside `main()` is never executed — and the labs "worked" every time anyone
tested them, on a box that already had the packages.

Three things to carry:

- **Run the command the README documents, not a convenient approximation.**
  The bootstrap now does `cd <example> && uv sync && uv run deepspeed ...`
  because that is the contract; anything else tests a different system.
- **`uv sync` from the committed lock is the only honest check.** A fresh
  `uv sync` into a temp dir takes two minutes and is how both failure modes
  above were *reproduced* rather than inferred.
- **`tests/test_torch_index_pins.py` now enforces it statically**: a lab whose
  code builds a processor must lock both packages. It reads the lock, not the
  pyproject — the resolution rather than the declaration.

This is also why the interpreter path in a verification log is worth reading:
seeing `.../<lab>/.venv/bin/python` is proof the harness installed from the
lab's committed lock rather than from whatever the container image shipped.

---

## Library API drift comes in three classes, and only one is obvious

`tests/test_config_kwargs.py` covers all three. Each is syntactically valid, so
`compileall` catches none of them:

| class | example | when it fails |
|---|---|---|
| a rejected **kwarg** | `logging_dir=` | when the config object is constructed |
| a removed **attribute** | `trainer.tokenizer` | at the *save step*, after training |
| a vanished **symbol** | `AutoModelForVision2Seq` | at **import**, before anything runs |

The third arrived last and is the cheapest to hit: `03_llms/03_ocr` imported a
name transformers 5.x had renamed — **and never used it.** A dead import took
the whole lab down, while `run_modern_ocr.py` in the same folder had already
migrated. Sweeping for the class found a second live site: `05_dpo`'s
`--method orpo`, because trl 1.x moved ORPO (and CPO) to `trl.experimental`.

**The subtle part is deciding which imports are allowed to fail.** The first
version of that check skipped anything inside `try/except ImportError` — and
therefore missed the very bug it was written for, because `03_ocr` wraps its
whole import block in one that prints "Missing required package" and exits 1.
**A try/except with no alternative import is not a fallback; it is a crash with
better formatting.** The rule now requires a *genuine* alternative — the same
symbol imported from a different module elsewhere in the file.

`logging_dir=` is syntactically valid, so `compileall` cannot catch it — it
fails only when `TrainingArguments` is constructed. A learner discovered exactly
that on rented GPUs after both ranks had launched and the model had loaded.

`tests/test_config_kwargs.py` parses every call to a known config constructor
(`TrainingArguments`, `SFTConfig`, `DPOConfig`, `GRPOConfig`, `LoraConfig`, …)
and checks each keyword against the **installed** class's signature. No GPU, no
download — the constructors import fine on CPU, which is what makes this
catchable at all.

Run against the tree when written it found **20** rejected kwargs in 13 files,
of which only one had been reported: `logging_dir`, `warmup_ratio`,
`overwrite_output_dir`, `save_safetensors` and `max_prompt_length` were all
removed by transformers 5.x / trl 1.x. **Four sat in commands Clawdeck offers
by name**, so three more labs would have failed the same way.

Two properties keep it honest, and both matter:

- **A constructor whose signature takes `**kwargs` is skipped**, loudly. It
  accepts anything, so a pass would be false confidence.
- **The pinned versions in its PEP 723 header are checked against every lab's
  `uv.lock`.** Otherwise the suite validates a version no learner runs — passing
  while the labs are broken, which is precisely the failure it exists to
  prevent.

When bumping a library, expect this to fail and read it as a to-do list: it
names the file, the line and the exact kwarg.

---

## A lab is its COMMAND, not just its code

`03_llms/03_ocr` OOMed on the 24 GB it advertised. The obvious readings were
"declare 48 GB" or "shrink the config". Both were wrong. The manifest ran

```bash
uv run deepspeed --num_gpus=2 train_ds.py --max-steps 20
```

while the script's **own header** had always documented

```bash
deepspeed --num_gpus=2 train_ds.py --use-4bit --use-lora
```

Both flags are `action="store_true"`, so the lab had been doing a
**full-parameter** fine-tune of Qwen2-VL-2B: 4.4 GB weights + 4.4 GB gradients
+ ~24 GB Adam state, which ZeRO-2 shards to ~12 GB/rank — ~19 GB before
activations, on a 23.56 GiB card. Measured, to the gigabyte. The lab's own
summary says *"cap max_pixels to bound memory"* and the command skipped the
memory-bounding feature.

**`min_vram_gb` was never wrong. The command was.** Adding `--use-lora` made the
lab cheaper, not more expensive: verified at 20/20 steps, rc=0, zero OOM on the
same 2 × RTX 3090 that had failed.

**Then the fix landed on the wrong lab**, and that is the part worth keeping.
`03_llms/01_llm_finetuning` and `03_llms/03_ocr` both contain a file called
`train_ds.py`, so their manifest commands were **byte-identical**. A
`str.replace(old, new, 1)` hit the first match. The tell was in the comment
written with it — it describes Qwen2-VL-2B, the *OCR* model, while sitting under
the Llama SFT lab.

Two rules fall out:

- **Edit by index, not by string match, when the string is not unique.** Locate
  the block by its `- id:` and act on line numbers. Anchoring on text that
  appears twice is how a correct fix reaches the wrong target.
- **`parse_known_args()` makes a misplaced flag SILENT.** The labs use it by
  contract, because the launcher injects `--local_rank`. So an unrecognised flag
  is ignored rather than rejected: one lab quietly carried a no-op, the other
  kept OOMing, and nothing raised anywhere.

`tests/test_clawdeck_manifest.py` now resolves every `--flag` in a cmd against
the `add_argument` names that script actually defines, skipping those the
`deepspeed` launcher consumes. 804 checks. **It cannot catch the original
omission** — a missing flag is an absence, and absences are not checkable — but
it catches every misplacement, which is the failure mode `parse_known_args()`
guarantees will be quiet.

---

## Three green checkers, one broken lab

The bug above was invisible to every static check on both sides of the
integration at once:

| checker | what it saw |
|---|---|
| Clawdeck's catalog check | a valid, bookable, priced command |
| `test_clawdeck_manifest.py` | a `.py` file that exists |
| the runtime | a flag it was contractually obliged to ignore |

Three green signals, one lab that OOMs for a learner. Only *running it* found
it. A passing static suite is evidence about the questions it asks, never
coverage of the ones it does not.

---

## The RunPod harness lies less than it used to

Four bugs in `runpod/runpod_ctl.py` were found and fixed by actually running
pods. Each made a **failed** run look successful, which is the worst failure
mode a verification harness can have — it does not lose information, it
manufactures confidence. All four are now pinned by assertions in
`tests/test_runpod_ctl.py`:

| Was | Effect |
|---|---|
| `rc=$?` read *after* the log-upload `curl` | the DONE marker reported the curl's status — essentially always 0 |
| `[2/6] repo cloned` printed unconditionally, `cd` failing silently | a failed clone ran the launcher from `/workspace` and looked like a broken example |
| `--dry-run` appended `\|\| true` | the command every README documents could not report a failure at all |
| collected log written without `mkdir -p` on its parent | nested example names contain `/`, so the log was silently lost |

Two operational facts that cost real time:

- **GitHub rate-limits anonymous clones from cloud IP ranges** and answers with
  an auth challenge, so a pod fails with `could not read Username for
  'https://github.com'` on a public repo. There is a codeload tarball fallback.
  **No credential is ever placed on the pod** — see `SECURITY.md`.
- **`--wait-seconds` defaults to 1800.** For anything with a large download that
  is not enough, and with `--terminate` the pod is destroyed mid-download. Pass
  a realistic window for big models.

---

## Watch a checker fail before trusting it

A check you have not seen reject bad input is not a check. This is not
hypothetical caution — four checkers written in this repo shipped a bug that
made them *unable to fail*:

| Checker | The bug |
|---|---|
| `gpu_guards()` in `test_clawdeck_manifest.py` | `walk()` took a node and recursed into its **children**, so a guard that was itself a statement of a block was never tested. The whole check silently passed everything. |
| `beats_chance()` in `test_synthetic_data_is_learnable.py` | passed `seed=` unconditionally and died with `TypeError` against the very generator it was written to catch — CI went red naming a signature mismatch, not the finding |
| the first `--num_gpus`/`gpu.count` check | over-flagged six entries that demonstrably run on CPU, because it keyed on the *presence* of a guard rather than its **reachability** |
| `check_contract.py`'s reader-A classifier | tested `"require_gpu" in src`, so a file whose **comment explained why it deliberately has no guard** was classified as guarded and then failed every follow-up check for a function it does not contain. Now an AST check for a real `Call`. |

The pattern in the last two is the same: **a substring is not a fact about the
program.** Ask the AST whether the thing is called, reachable, and in the branch
you think it is.

The same reasoning applies to `scripts/check_contract.py` itself. Three of its
original checks were over-strict and were fixed *in the checker*, not in the
labs it flagged: `import torch` at module scope is harmless on a CPU box,
`import deepspeed` likewise (the CUDA_HOME error comes from `initialize()`,
which `require_gpu()` precedes), and `pip install uv` is legitimate because you
cannot bootstrap uv with uv.

It still earns its keep: pointing it at the repo found `03_llms/03_ocr`
requesting `--ntasks-per-node=2` while running `deepspeed --num_gpus=2` (four
processes for two GPUs — a hang), and 13 READMEs that never told a RunPod reader
how to shut the pod down.

---

## A measured claim is scoped to the configuration it was measured in

`11_moe` published, from six seeds and two expert counts, that load balancing
makes the model strictly *worse* — a tax paid for schedulability rather than a
quality improvement. A 2-GPU run then reported the exact reverse, with the
unbalanced arm 33x worse and its loss **rising** through training.

Neither number was wrong. The claim was: it had been measured only
single-process, and said "every configuration".

What was done about it is the point. Before touching the thesis:

- the world-size-1 result was re-checked across **six seeds** — no overlap
  between groups, so not seed luck;
- the same script was run at world size 1 to rule out "two different
  experiments" — the ordering held, so the reversal really is a world-size
  effect;
- an attempt to reproduce it on **two gloo ranks on CPU**, with gradients
  all-reduced exactly as data parallelism does, did **not** reverse it.

So the claim was **scoped** to world size 1 and the disagreement written into
`moe.py`, the folder README and the docs page as explicitly unresolved — in
neither direction. Rewriting the thesis around one unreplicated run would have
been as wrong as leaving the over-broad claim standing, and quietly dropping
the inconvenient measurement would have been worse than both.

**"Measured at X" is a different claim from "true in general", and the docs
should say which one they are making.**

Two related habits, from the same family of error:

- **Check what hardware you actually have before declaring a thing unrunnable.**
  Three rounds of the CIFAR-10 multi-GPU fix went out verified only by static
  analysis, on the stated grounds of having no GPU. The dev box had one.
  `nvidia-smi` costs a second and would have caught a `TypeError` that instead
  surfaced on a rented two-GPU machine after both ranks had launched.
- **Never fabricate expected output.** If it has not been run, mark it *not yet
  verified on hardware*. A wrong published number costs a reader a day debugging
  their own correct setup.

---

### Second instance: the figure was right and the length was wrong

The `11_moe` case above was a claim measured in one configuration and stated
for all of them. This one is narrower and, for that reason, harder to see.

CLAUDE.md published that the AF2→AF3 saving from deleting the MSA
representation is **"58% at 128 residues and 12% at 256"**. The measured
table in `pairformer.md` reports:

| residues | saving |
|---|---|
| 32 | 58.5% |
| 128 | 24.2% |
| 256 | 12.5% |

Both numbers in the sentence existed. One of them was attached to the wrong
length: at 128 residues the real saving is 24.2%, so the published figure was
**more than double the truth**. The second half of the sentence was fine.

Three things made it survive:

- **The number was real**, so it did not look invented. Every instinct that
  guards against fabrication was satisfied.
- **24.2% appears in two different tables at two different lengths** — it is
  the measured saving at 128 residues *and* the analytic saving at 384, which
  is the configuration of the detailed per-tensor table. A page can therefore
  show "24.2%" twice, correctly, meaning two different things.
- **Nothing compared prose to the table it came from.** Three checkers were
  green. They checked that referenced files exist, that counts are current,
  that diagrams conform — none of them could read a sentence.

The fix was not just the correction. `tests/test_published_protein_numbers.py`
now **recomputes** every analytic figure by running
`trunk_activation_table()` out of the shipped `pairformer.py`, and for the
measured figures — which CI cannot reproduce — enforces a single owning
table that every other mention must agree with, *including its residue
count*. Checking the number alone would not have caught this; the number was
right.

Confirmed by reintroducing the original sentence verbatim and watching four
checks fail, and by changing `n_heads` in the shipped source and watching
nine fail. A check that has not been seen rejecting the real bug is not
evidence it would have.

The generalisable form: **a measured number and the configuration it was
measured in are one indivisible fact.** Quoting the number without the
configuration is not a shortening of the claim — it is a different claim,
and usually a false one.

## A script with no end-to-end run makes its derivations the only testable surface

`03_llms/01_llm_finetuning/analyze_kimi_k3.py` is named `analyze_`, not
`train_`, because it does not train — Kimi K3 is 2.78 T parameters / 1,561 GB
and its remote code does not import on the pinned transformers, so there is no
run to have. It carries no `require_gpu()`, deliberately, and says so in a
comment: `--plan` reads a JSON file over HTTPS and `--verify-arch` builds on the
meta device, so a guard there could never fire.

`tests/test_kimi_k3_plan.py` runs the shipped functions against a fixture
config. It caught three defects on its first run, none of which a shape
assertion could see:

- `min()` of the layer gaps reported *"every 1th layer"* — the mode is 4; K3 has
  23 gaps of 4 and one of 1, because 93 layers do not divide evenly;
- `linear // full` truncated a 3:1 design to 2:1;
- `(n - full) > 0` called **every dense model hybrid**.

All three were populated fields of the right type holding plausible small
integers.

So: run a new check against the **unfixed** tree first, and keep a
counterexample in the suite permanently. `test_config_kwargs.py` found 20 broken
call sites that way when only 3 had been reported;
`test_synthetic_data_is_learnable.py` carries the old random-label generator and
asserts it fails.

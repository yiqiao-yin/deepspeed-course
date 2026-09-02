# 09 — Video-Speech-to-Speech

The final topic, and the only one where the model **takes two streams in and
speaks back**.

```
        video  ──┐
                 ├──►  omni model  ──►  speech out
        speech ──┘
```

## Scope — what belongs here and what does not

This folder is specifically about models that accept **video AND audio
together** and emit **speech**. That boundary is worth stating, because the
neighbouring families look similar and solve different problems:

| Family | Input | Output | Where |
|---|---|---|---|
| Video-language | video | **text** | [`../04_video_text/`](../04_video_text/) |
| Speech-to-speech | **audio only** | speech | out of scope — Moshi, GLM-4-Voice, Mini-Omni |
| **Video-speech-to-speech** | **video + audio** | **speech** | **here** |

Audio-only duplex models (Moshi, Mini-Omni) are genuinely impressive and are
*not* what this topic teaches, because they never face the problem that defines
it: **two input streams that disagree about what time it is.**

## The problem that defines the topic

A video-speech model receives:

- **video** at 1–25 frames per second, irregular, whatever the sampler gave you
- **audio** at 16,000 samples per second, or ~50 encoder frames per second

Concatenate them and the transformer sees a flat list of tokens with no idea
which frame goes with which sound. Ask *"what did he say while pointing at the
whiteboard?"* and it cannot answer — not because it is undertrained, but
because the information that pointing and saying happened **at the same
moment** was never in the input.

> **`04_video_text/` only had to represent time *within* one stream. Here, two streams
> have to agree.** That is the whole topic, and it is why the first thing
> subtopic 02 does is put both on a shared 40 ms clock.

## The track

| # | Subtopic | The question it answers | GPU? |
|---|---|---|---|
| 1 | [`01_longcat_omni/`](01_longcat_omni/) | What does the frontier look like? (560B) | 2×B200 + ~3 TB RAM |
| 2 | [`02_thinker_talker/`](02_thinker_talker/) | How do two streams get onto one clock, and how is speech emitted without wrecking reasoning? | 24 GB |
| 3 | [`03_duplex_streaming/`](03_duplex_streaming/) | Can it keep listening — and watching — **while it talks**? | **no** |
| 4 | [`04_omni_eval/`](04_omni_eval/) | Is it *actually* using both streams, or faking it? | **no** |

Each exists because the previous one runs out of road:

- **LongCat-Flash-Omni** is what good looks like and needs ~3 TB of host RAM.
  You will read it, not run it.
- **Thinker-Talker** is the same architecture at a size you can fine-tune —
  but it answers one turn at a time, deaf while it speaks.
- **Full duplex** fixes that, and cannot tell you whether the model
  understands anything.
- **Evaluation** answers the question none of the above can: a model that
  ignores the video entirely still scores well, and accuracy cannot see it.

## Current models in this family

Researched Aug 2026. All accept **video + audio** and emit **speech**.

| Model | Scale | Notable |
|---|---|---|
| [Qwen3.5-Omni](https://arxiv.org/abs/2604.15804) | hundreds of B, MoE | Hybrid-attention MoE for *both* Thinker and Talker; 256k context; 10 h audio / 400 s of 720p video; ARIA text-speech alignment |
| [Qwen3-Omni](https://arxiv.org/abs/2509.17765) | 30B MoE | 234 ms end-to-end latency; 119 written / 10 spoken languages; SOTA on 32 of 36 audio-visual benchmarks |
| [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) | 3B / 7B | **The teachable one.** TMRoPE + Thinker-Talker, both introduced here |
| LongCat-Flash-Omni | 560B | Subtopic 01 |
| [DuplexOmni](https://arxiv.org/abs/2606.09186) | — | 480 ms slices, `^`/`[CUT]`/`[WAIT]` control tokens, 0.506 s response latency |
| MiniCPM-o 4.5 | 8B | SigLip2 + Whisper + CosyVoice2 on a Qwen3 backbone; full-duplex streaming |
| [Baichuan-Omni-1.5](https://arxiv.org/abs/2501.15368) | 7B | Qwen2.5-7B backbone, text+audio out |
| [Ming-Omni](https://arxiv.org/abs/2506.09344) | 2.8B active | Ling MoE with **modality-specific routers** to reduce modality conflict |

**Start with Qwen2.5-Omni-3B.** It introduced the two mechanisms the whole
family now uses, and it fits on one 24 GB card.

## Run it without a GPU

Two of the four subtopics are **fully CPU-runnable**, because their substance
is algorithms and policy rather than weights:

```bash
uv run 05_video_speech/02_thinker_talker/tmrope.py      # the shared clock, and what breaks without it
uv run 05_video_speech/03_duplex_streaming/duplex.py    # barge-in, ghost text, RTF
uv run 05_video_speech/04_omni_eval/omni_eval.py        # ablation grid on a simulated model
```

Verify they are *correct*, not merely runnable:

```bash
uv run tests/test_tmrope.py        # 58 checks — including that naive indexing drifts
uv run tests/test_duplex.py        # 36 checks — including gesture-only barge-in
uv run tests/test_omni_eval.py     # 49 checks — including that the harness catches a fake
```

## Run it on a GPU

**`uv`** for packages, **`deepspeed`** for training. Never bare `pip`.

### CoreWeave / any SLURM cluster

```bash
cd 05_video_speech/02_thinker_talker && sbatch run_deepspeed.sh
```

Build the environment **once on a login node** — compute nodes usually have no
egress:

```bash
uv venv ~/myenv && source ~/myenv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install deepspeed transformers accelerate peft datasets
uv pip install librosa soundfile opencv-python-headless
```

### RunPod

No SLURM there, so the pod lifecycle is driven by API — including shutdown:

```bash
export RUNPOD_API_KEY=...

uv run runpod/runpod_ctl.py recommend 05_video_speech/02_thinker_talker
uv run runpod/runpod_ctl.py run 05_video_speech/02_thinker_talker \
    --collect --wait --terminate --yes

uv run runpod/runpod_ctl.py pods        # confirm: "Nothing is billing."
```

| Subtopic | Min VRAM | GPUs | Disk | Launcher |
|---|---|---|---|---|
| `05_video_speech/01_longcat_omni` | 180 GB | 2 | 2 TB | `deepspeed` |
| `05_video_speech/02_thinker_talker` | 24 GB | 2 | 120 GB | `deepspeed` |
| `05_video_speech/03_duplex_streaming` | 24 GB | 1 | 80 GB | `python` |
| `05_video_speech/04_omni_eval` | 24 GB | 1 | 80 GB | `python` |

Subtopics 3 and 4 use `python`, not `deepspeed`, deliberately: duplex inference
is inherently sequential and evaluation is a series of short `generate()` calls.
Neither has an optimizer to shard, so the DeepSpeed launcher would add
process-group setup and buy nothing.

> ### ⚠️ `01_longcat_omni` is not rentable
> It needs roughly **3 TB of host RAM**, which RunPod pods do not provide. GPU
> VRAM is not the binding constraint. The other three subtopics run fine on a
> single 24 GB card — that is precisely why they were split out.

## The shared corpus

`data/` holds 8 real video+audio+response samples (44 MB), shared by every
subtopic rather than duplicated four times into git history. Override the
location with `VSS_DATA_DIR`.

```
data/train/01/{in.mp4, in.wav, out.wav}
```

## Reading list

- Xu et al. **Qwen2.5-Omni Technical Report** (2025) — TMRoPE, Thinker-Talker.
  [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)
- Qwen Team. **Qwen3-Omni Technical Report** (2025).
  [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)
- Qwen Team. **Qwen3.5-Omni Technical Report** (2026).
  [arXiv:2604.15804](https://arxiv.org/abs/2604.15804)
- **DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking** (2026).
  [arXiv:2606.09186](https://arxiv.org/abs/2606.09186)
- Wang et al. **OmniEval** (2025).
  [arXiv:2506.20960](https://arxiv.org/abs/2506.20960)
- Li et al. **Baichuan-Omni-1.5** (2025).
  [arXiv:2501.15368](https://arxiv.org/abs/2501.15368)
- **Ming-Omni: A Unified Multimodal Model for Perception and Generation** (2025).
  [arXiv:2506.09344](https://arxiv.org/abs/2506.09344)

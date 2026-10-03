# Obstacle Hopper — a robot learns to get over a step

**Baseline:** a random policy, measured every run before training starts —
return 129, and it never reaches the step, let alone clears it.

**Budget:** 600,000 environment steps per run, **three seeds per arm**, on
CPU (~8 min each). The seed count is not boilerplate here: it is the
finding.

A three-link hopper in a 3D MuJoCo world has to get past a box. The box is
**a different height every episode**, drawn from 0.02 m to 0.17 m, so a
single memorised gait is not obviously enough — at the low end the robot
can walk over the lip, at the high end it has to step up.

This is the first lab in `07_physical_ai`, and it is deliberately the
simplest thing that is still real: real contact physics, real PPO, real
training, no pretrained anything.

## The falsifier

Stated before the experiment, because a claim that cannot lose is not a
finding:

> **If a policy that cannot see the obstacle height performs as well as one
> that can, then height perception is not what solves this task, and the
> headline claim of this lab is wrong.**

**It lost.** See [the result](#the-result-the-ablation-failed), which is
the most useful thing in this lab.

## What is here

| File | Role |
|---|---|
| `obstacle_env.py` | the world — 60 lines of MuJoCo XML, the reward, the episode. No Gymnasium dependency |
| `ppo.py` | PPO written out: GAE and the clipped objective as pure tensor functions, CPU-testable |
| `train_ppo.py` | the training loop, CPU or GPU |
| `make_figures.py` | plots, generated from `runs/*/curve.csv` — never from invented numbers |
| `run_deepspeed.sh` | SLURM wrapper |

## Run it

```bash
cd 07_physical_ai/01_obstacle_hopper
uv sync

uv run obstacle_env.py            # the world + the random baseline, 10s
uv run ppo.py                     # the two GAE limits, no simulator
uv run ../../tests/test_obstacle_hopper.py     # 22 property checks

uv run train_ppo.py --dry-run     # 30s smoke test
uv run train_ppo.py               # ~8 min on CPU
uv run train_ppo.py --blind-to-height          # the ablation
uv run make_figures.py
```

## CPU or GPU — measured, not assumed

Both work. **CPU is 2.3× faster**, and the number is the point:

| device | 8,192 environment steps |
|---|---|
| **CPU** | **14.2 s** |
| CUDA (RTX 3080 Ti) | 32.0 s |

The policy is **10,119 parameters**. The matrix multiplications are
trivial; the cost is MuJoCo stepping sixteen environments, which happens on
the CPU either way. Moving a 10k-parameter forward pass to a GPU only adds
a host-device transfer to the part that was never the bottleneck.

So `--device auto` resolves to **CPU** here. `--device cuda` is supported
and reported, not recommended. Defaulting to the slower device because a
GPU exists is the reflex this course argues against elsewhere.

### Why there is no `deepspeed` launcher

Ten thousand parameters. ZeRO partitions optimizer state, gradients and
parameters, and all three are rounding errors at this size. A distributed
launcher here would be the cargo cult `CLAUDE.md` names, so this lab is
registered `launcher="python"` — the **seventh** declared exception.

The GPU story in this category is real from lab 2, where the policy is a
7B vision-language-action model that does not fit on one card.
`tests/test_obstacle_hopper.py` fails if the policy ever exceeds 1M
parameters, so this justification cannot go stale quietly.

## Training works

![Learning curves, three seeds per arm](/img/physical/hopper-learning-curves.png)

Both arms rise from a baseline of 129 to roughly 1,000 — about **7×**. The
shape is worth noticing: a long plateau around return 670 where the robot
has learned to stand and shuffle, then an abrupt breakthrough to real
locomotion. **Every seed breaks through at a different time** — 200k, 230k,
380k, 430k steps — and one `seeing` seed never breaks through at all.

## The result: the ablation failed

The experiment was designed to show that blinding the policy to obstacle
height hurts it. Across three seeds per arm:

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| sees height | 1060 | 1112 | 692 | **955** |
| blind to height | 1119 | 1086 | 756 | **987** |

![Seed spread dwarfs the ablation](/img/physical/hopper-seed-spread.png)

**The blind policy scored slightly higher.** More importantly the seed
spread (692 to 1119) is an order of magnitude larger than the difference
between arms, so the honest reading is *no measurable effect*, not
*blindness helps*.

One run per arm would have produced a confident answer in either direction
— and did, twice, in opposite directions, before the seeds were run.

### Why the ablation failed, most likely

![Distance vs obstacle height](/img/physical/hopper-height-sweep.png)

Both policies behave almost identically across the height range, and both
fall off the same cliff between 0.08 m and 0.11 m. Below the cliff they
travel 8 m; above it they stop at 1.65 m, just past the step.

That suggests the task does not actually *require* perception: one robust
hopping gait clears everything up to about 0.1 m and nothing clears what is
above it. If there is no behaviour worth adapting, seeing the height buys
nothing — which is a statement about **this height range on this robot**,
not about obstacle perception in general.

The obvious follow-up is to widen the range until adaptation is forced, and
that is lab 2's business.

## What this lab is really teaching

**A single RL run is not evidence.** Not a conclusion about this
environment — a conclusion about reinforcement learning. The variance
between seeds here is larger than most effects anyone wants to report.

**Measure the baseline, every time.** `train_ppo.py` runs 20 random-policy
episodes before training and prints the number, so the comparison can never
be forgotten or estimated.

**An obstacle you have not verified is scenery.** An earlier version of
this lab trained to a 100% clear rate at every height, which looked like
success. Running the same policy with the obstacle *removed* gave an
identical distance and an identical episode length — the robot was diving
forward and falling over at one second, and the box never impeded it. The
clear rate was real, reproducible, and measured nothing.
`test_the_obstacle_actually_obstructs` is what would have caught it, and it
is the direct analogue of `tests/test_omni_eval.py`.

## Design decisions worth knowing

**The torso cannot rotate.** With a free pitch joint this becomes the
bipedal-balance problem, which is a famously hard benchmark and is not what
the lab teaches. Measured: with pitch free, 1M steps produced a policy that
dived forward and fell at ~1.0 s. Locking the torso isolates the lesson.
Freeing it is the obvious next lab.

**The step is 0.30 m deep, not 0.90 m.** At 0.9 m the policy reliably
climbed *on* (7/8 at 0.08 m) and then had most of a metre of narrow ledge
to cross — that measures balance on a platform, a different and harder
task.

**The obstacle sits at x = 1.2 m, close to the start.** At 2.2 m the robot
spent the whole episode learning to walk and fell over before ever reaching
it, so the run measured locomotion endurance rather than obstacle crossing.

**Height range 0.02–0.17 m is measured, not chosen.** Placed on the box the
hopper rests stably to 0.18 m and topples at 0.20 m.

### Renting a GPU (RunPod) — for convenience, not speed

You do not need one. If you want to run the full six-run sweep somewhere
other than your laptop:

```bash
# --dry-run first: 30 s, proves the pipeline assembles before spending.
uv run runpod/runpod_ctl.py run 07_physical_ai/01_obstacle_hopper --dry-run --yes

uv run runpod/runpod_ctl.py run 07_physical_ai/01_obstacle_hopper \
    --collect --wait --terminate --yes
```

**Confirm the pod is gone with `uv run runpod/runpod_ctl.py pods`.** An
abandoned pod bills until terminated; `--terminate` runs from your machine
in a `finally`, and the in-pod watchdog is a backstop, not a guarantee.

## Environment & Local Testing

| | |
|---|---|
| Dependencies | `mujoco`, `torch`, `numpy`, `matplotlib` — all pip-installable |
| GPU | not required, and not faster; `--device cuda` works |
| Download | none. The world is generated from an XML string in the source |
| Rendering | not needed. Training uses state observations; MuJoCo's headless-GL problems never arise |
| Runtime | ~8 min per run on CPU; `--dry-run` is 30 s |

## Reading on

- `03_llms/06_grpo` — the same algorithm family with the critic *deleted*,
  because there the rewards are sparse and verifiable. Here they are dense
  and shaped, so the critic earns its place. Opposite call, stated reason.
- [MuJoCo](https://mujoco.readthedocs.io/) · [Gymnasium](https://gymnasium.farama.org/)
- Schulman et al., [PPO](https://arxiv.org/abs/1707.06347) and
  [GAE](https://arxiv.org/abs/1506.02438)

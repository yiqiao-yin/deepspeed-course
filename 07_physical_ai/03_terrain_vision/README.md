# Terrain Vision: teaching a robot to look before it steps

A two-legged robot with a BD-1-style head walks along a raised plateau.
Partway along, the ground does one of three things: stays flat, climbs
three steps, or drops three steps. The approach slab is **identical in
all three**, so from its own joints the robot cannot tell which is
coming — and the three demand incompatible responses.

This is the third lab in `07_physical_ai`, and the first where the robot
has an eye.

**Baseline:** the blind policy — proprioception only, same algorithm,
same 1.5M steps, three seeds: 100% flat, 12.5% upstairs, 54.2%
downstairs.
**Budget:** 1.5M environment steps per teacher × 3 seeds per arm; the
vision student is 40 epochs over 43,102 rendered frames.
**Falsifier:** if a 64×64 depth image does not lift the ascent score
clear of the blind arm's own seed spread (σ = 0.18), the camera is
decoration and this lab has no result. Written down before the student
was trained.

![three arms, per terrain](../../docusaurus-docs/static/img/physical/terrain-arms.png)

---

## Why this lab exists

Labs 1 and 2 each shipped an information ablation that **came back
null**. A one-legged hopper cleared obstacles just as well when told
nothing about their height; a biped climbed stairs just as well without
being told the rise. That is not a bug in those labs, it is a real
result and a well-known one: blind proprioceptive locomotion is
genuinely capable, because a leg that touches something can feel it.

So the question for a third lab was not "does vision help" — it was
**what task makes vision necessary at all**. Six designs came back null
before this one. The answer is `down`:

> You cannot feel for a descent. By the time the swinging foot finds
> nothing underneath it, the robot is already falling.

That asymmetry is the whole design. `up` is hard-but-feelable, `down` is
not feelable at all, and `flat` is the control that proves the other two
are not just "harder".

### What did NOT work, and why it is written down

| attempt | result |
|---|---|
| lab 1, blind to obstacle height | null — reactive control sufficed |
| lab 2, privileged info removed | null — same |
| four terrains, no overhead constraint | effect present on seed 0 only |
| overhead ceiling on `flat` + `down` | broke `down` for **both** arms — a broken terrain, not a discrimination |
| hollow stairs | 12% × 3 seeds looked real; a specialist then scored 88% |
| a gap to leap (`hop`) | **the ablation reversed** — see below |

`hop` is the instructive one. It is still buildable (`--terrain hop`) but
is deliberately excluded from `KINDS`:

| seed | 0 | 1 | 2 |
|---|---|---|---|
| blind | 1.00 | 1.00 | 1.00 |
| privileged | 0.50 | 1.00 | 0.00 |

The arm that **cannot see** won. That is not noise, it is the task: you
do not need to see a gap to clear one, because committing to a maximal
leap works either way. Including `hop` would have meant publishing a
four-terrain average in which one terrain silently argued the opposite
of the other three — which is the mistake `11_moe` shipped once already.

---

## The experiment: three arms, one task

| arm | sees | deployable? |
|---|---|---|
| **privileged** | the terrain type and its geometry, handed over directly | no — this is an oracle |
| **blind** | proprioception only (15 numbers) | yes, and it is the baseline |
| **vision** | proprioception **+ a 64×64 depth image** | yes — the thing being tested |

The privileged arm is the *teacher*: it needs no rendering, so it trains
at full speed. The vision arm is a *student* trained by behaviour
cloning on the teacher's actions. The blind arm is what the student has
to beat.

### Results

Teachers: 3 seeds × 1.5M steps each. Vision: one seed, 40 epochs,
**24 evaluation episodes per terrain**. Fraction of episodes clearing
the terrain.

| terrain | privileged | blind | **vision** |
|---|---|---|---|
| flat | 100% | 100% | 100% |
| upstairs | 83% | **12.5%** | **79%** |
| downstairs | 100% | **54.2%** | **100%** |

The gap the camera has to close is **+71 points on ascent** and **+46 on
descent**. Depth recovers most of the first and all of the second — from
images alone, with no privileged information at deployment.

Random baseline return, for scale: **178**. Blind: 1057. Privileged: 1516.

### The learning order is not in the reward

![learning curves per terrain](../../docusaurus-docs/static/img/physical/terrain-curves.png)

| steps | return | flat | up | down |
|---|---|---|---|---|
| 12k | 125 | 0% | 0% | 0% |
| 197k | 801 | **100%** | 0% | 62% |
| 381k | 1211 | 100% | 50% | **100%** |
| 565k | 1426 | 100% | **100%** | 100% |

Flat first, then descent, then ascent. Nothing in the reward function
encodes that order; it is what falls out of the difficulty.

---

## Three checks, because a good score proves nothing

This lab's two predecessors both produced convincing animations of
policies that turned out not to be using the information in question.
So the burden of proof here is explicit, and `train_student.py` runs all
three every time.

### 1. Blank the camera

![the blank-image ablation](../../docusaurus-docs/static/img/physical/terrain-ablation.png)

Feed the trained student a black image and re-measure. If it still
scores, it was reading proprioception and the camera was decoration.

| terrain | camera working | camera blanked |
|---|---|---|
| flat | 100% | **0%** |
| upstairs | 79% | **0%** |
| downstairs | 100% | **0%** |

It does not merely degrade — it collapses. The policy is genuinely
driving on the image.

### 2. Probe the frozen encoder

Train a single linear layer on the encoder's features to predict which
terrain a frame came from. **64.0%, against a 34.3% chance baseline.**
Terrain is close to linearly decodable from what the encoder learned,
which is what "it has learned to see" means operationally.

### 3. Report per terrain, never averaged

`flat` is solvable blind. An average over three terrains would have
reported a modest overall gap instead of the 71-point one that is
actually there, and would have hidden that `flat` contributes nothing.

### What the camera actually receives

![mean depth image per terrain](../../docusaurus-docs/static/img/physical/terrain-depth.png)

Averaged over every frame of each terrain in the training set. The three
mean images differ visibly, which is the cheapest possible evidence that
the task is not blind-equivalent.

One detail worth copying: depth is clipped to a **fixed 0.8–3.5 m
window**. Normalising each frame to its own min/max was tried first and
destroyed the signal — the absolute distance *is* the information (mean
depth 1.86 m looking up a staircase against 3.03 m looking down one),
and per-frame scaling throws exactly that away.

---

## The animations

```bash
uv run render.py --all
```

Every vision clip carries the **64×64 depth image the policy is actually
driving on**, drawn from the same `env.depth()` call that feeds the
network — not a prettier second render.

| clip | what it shows |
|---|---|
| `terrain-blind.gif` | a **fully trained** policy with no camera, missing the first tread |
| `terrain-vision.gif` | the same two terrains, with depth |
| `terrain-tour.gif` | all three terrains back to back on one set of weights |
| `terrain-untrained.gif` | a 2-epoch checkpoint, for the conventional before/after |

![the tour](../../docusaurus-docs/static/img/physical/terrain-tour.gif)

`blind` is the honest "before", and `untrained` is offered separately
for a reason. An under-trained policy falls over, which looks bad but
shows nothing — *every* policy falls over early, camera or not. The
blind policy is fully trained and differs in exactly one respect, so
when it misses a tread, the camera is the only available explanation.

The clips use seed 8002, which clears all three terrains. The student
is 79% on `up`, so roughly one episode in five genuinely fails; the
table above is the claim and the clip is an illustration of it.

---

## Why there is no `deepspeed` launcher

This is the **ninth** `launcher="python"` exception in the course, and
the first for a new reason.

The other eight skip DeepSpeed because a GPU buys nothing. Here a GPU
buys a great deal — this is the first lab in `07_physical_ai` where it
does:

```
$ uv run train_student.py --bench
  cpu       2860 +/-  145 samples/s
  cuda     38677 +/-  778 samples/s      13.6x +/- 0.9
```

Five repeats each. One timed call is a sample rather than a
measurement, and the first run taken here read 12.9x purely because it
landed at the bottom of a 12.5-14.5x range -- the ratio moves more than
either absolute does, so the absolutes are the number to quote.

Labs 1 and 2 both measured **CPU as faster** (2.0× and 1.7×), because
their work was MuJoCo stepping rather than arithmetic. The student here
is a convolutional encoder, roughly a hundred times larger, trained
supervised over a fixed dataset with no simulator in the loop. That is
the regime a GPU is for.

DeepSpeed is still the wrong tool: **193,222 parameters have nothing for
ZeRO to shard.** "A GPU helps" and "DeepSpeed helps" are different
claims, and this lab is the one place in the category where they come
apart.

The two PPO teachers are left on the CPU, where they are faster.

### Why behaviour cloning, and not on-policy RL with a camera

Rendering costs **250×**. Measured on this machine: 7,077 physics
steps/s with no camera, 79 frames/s with depth. An on-policy vision run
would take eight hours instead of four minutes.

Behaviour cloning moves that cost out of the loop: the teacher acts on
privileged state at full speed, depth is rendered **once** alongside it,
and the student then trains supervised on a fixed dataset — which is
also the only part of this lab with real work for a GPU.

The dataset is collected with **exploration noise on the teacher's
action** (`--noise 0.15`) while the *label* remains the teacher's clean
action. A dataset of perfect trajectories teaches a student nothing
about recovering; the first time it drifts it has never seen the state
it is in.

---

## The head is cosmetic, and that is enforced

The robot wears a large BD-1-style head. It is decoration, and making it
decoration took care, because the head is not a decal — it is a rigid
body carrying 19% of the robot's mass **and** the depth camera the whole
lab is about.

MuJoCo derives mass from geom volume, so simply scaling the head up
multiplies its mass — about 6× here — and silently changes the robot
that 1.5M training steps were spent on. Instead every decorative geom
is `density="0" contype="0" conaffinity="0"`, and the body carries an
explicit `<inertial>` pinned to the original's measured values.

| | before the head | after |
|---|---|---|
| head mass | 4.252544 kg | 4.252544 kg |
| total mass | 22.686363 kg | 22.686362 kg |
| depth pixels | — | 0 difference |

The subtler risk is the second one. **The camera is mounted inside that
body**, so decoration reaching into its field of view would put a
constant blob in every depth frame the student ever trains on — while
training still converges and the render looks better than before.

`tests/test_terrain_vision.py` proves geometrically that this cannot
happen: the decoration is rigidly attached to the lens, so if every
vertex of every head geom lies behind the camera's view plane, no head
geom can appear in any frame in any pose. That check **failed on first
run** — a neck cylinder reached 1.64 cm into the view half-space, and
rendered harmlessly only because MuJoCo's near clipping plane happened
to cut it. Accidental safety is not safety; the neck was removed.

---

## Running it

### Environment & Local Testing

```bash
cd 07_physical_ai/03_terrain_vision
uv sync
```

No model downloads and no dataset downloads — the world is an XML string
and the data is generated. First `uv sync` pulls torch (CUDA build) and
MuJoCo, about 3 GB.

```bash
# nothing trained yet: the three terrains and a random baseline
uv run vision_env.py

# the RL half — ~13 min per run on CPU
uv run train_teacher.py --name v3_priv_s0
uv run train_teacher.py --no-privileged --name v3_blind_s0

# the vision half
uv run collect.py --episodes 120          # ~9 min, writes data/bc.npz (23 MB)
uv run train_student.py --device cuda     # ~2 min train, then the three checks

# CPU vs GPU, and nothing else
uv run train_student.py --bench

# figures and animations
uv run make_figures.py
uv run render.py --all
```

A 30-second smoke test that proves the pipeline assembles:

```bash
uv run train_teacher.py --dry-run
```

### The logic checks (no GPU, no rendering)

```bash
uv run ../../tests/test_terrain_vision.py
```

These need no OpenGL: the camera-occlusion check is geometric rather
than a render comparison, which makes it both exact for every pose and
runnable in CI.

### Rendering backends

`render.py` probes `glfw`, `egl` and `osmesa` in **subprocesses**,
because MuJoCo binds its GL backend once per process — trying one and
falling back in the same interpreter fails in a way that looks like a
bug in the second backend. If none works it says so and exits non-zero
without affecting anything else in the lab. Override with
`MUJOCO_GL=<backend>`.

### Hardware

| | |
|---|---|
| GPU | 1 × 16 GB, and only the student uses it |
| CPU | the two PPO teachers run here, and are faster for it |
| Disk | ~15 GB (torch + MuJoCo); the dataset is 23 MB |
| Wall clock | ~75 min for 6 teachers, ~9 min to collect, ~2 min for the student |

### CoreWeave (SLURM)

```bash
sbatch run_deepspeed.sh
squeue -u $USER
tail -f logs/terrain_vision_<jobid>.out
```

### RunPod

```bash
uv run runpod/runpod_ctl.py run 07_physical_ai/03_terrain_vision \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods          # confirm nothing is left running
```

---

## Files

| file | role |
|---|---|
| `terrain.py` | the three worlds and the robot, as XML; `ground_height()` is the single source of truth for termination |
| `vision_env.py` | the environment — observations, reward, termination, and the depth camera |
| `ppo.py` | PPO from scratch. Copied from lab 2 per the no-shared-module rule |
| `train_teacher.py` | the RL half: `--no-privileged` is the blind control |
| `collect.py` | renders the behaviour-cloning dataset once, at full speed |
| `train_student.py` | the CNN student, `--bench`, and the three honesty checks |
| `make_figures.py` | every figure, read from `runs/*/summary.json` |
| `render.py` | the four animations, with the depth inset |
| `run_deepspeed.sh` | SLURM batch script (no `deepspeed` launcher — see above) |

## What is not here

- **A multi-seed vision student.** The teachers are three seeds per arm;
  the student is one. The per-terrain rates carry a corresponding
  uncertainty — a 24-episode evaluation moves by one episode for free.
- **`hop`.** Buildable, excluded, and the reason is measured rather than
  aesthetic. See above.
- **Any claim about sim-to-real.** Nothing here has touched hardware.

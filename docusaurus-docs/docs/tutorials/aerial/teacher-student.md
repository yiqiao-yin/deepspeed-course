---
sidebar_position: 7
---

# Teacher and Student

<!--chain-->
:::info **Lab 7 of 7** in Aerial Physics. Labs 1 to 6 handed the vehicle its
exact height above the ground, because a simulator can simply look it up.
This one takes that away and gives it a depth sensor instead.
← previous: [Learned Dynamics](./learned-dynamics)
:::

:::info Reading this without a background in machine learning
Every section opens with a plain words box like this one. Read only those and
you get the whole story; the technical detail underneath is there if you want
it.

**The one line version:** we gave a landing car a depth sensor and no
altimeter. Trained the usual way it learned nothing from the sensor and
crashed as often as a car with no sensor at all. Trained instead by copying
an expert that had been told the answer, the same network learned to read
the sensor and landed nearly as well as that expert.
:::

**Baseline:** the same network on the same sensor, trained by PPO from the
task reward — **2.5% on target**, statistically indistinguishable from the
blind control's 1.7% (McNemar p = 0.75).
**Budget:** 4M PPO steps per RL arm; 600 labelled episodes ≈ 111,000
transitions and 12 epochs per round for both distilled arms, matched exactly
between them.

A cold-gas car has to land on a marked spot in a boulder field. It carries a
belly-mounted rangefinder — 144 ranging beams in a 12×12 grid — and no
altimeter. **Trained by distillation from a teacher that was simply told its
height, it lands on target 50% of the time and is statistically
indistinguishable from that teacher.** Trained by PPO on the identical sensor
and network, it manages 2.5%.

## Result

| arm | what it knows | on target | 95% CI | mean miss |
|---|---|---|---|---|
| `oracle` — the teacher | exact altitude + clearance | 45.8% | [0.40, 0.52] | 2.92 m |
| `direct` | 144 depth beams, PPO | **2.5%** | [0.01, 0.05] | 5.14 m |
| `deaf` — the control | nothing | 1.7% | [0.01, 0.04] | 4.50 m |
| `bc` | 144 depth beams, cloned | 47.5% | [0.41, 0.54] | 3.28 m |
| **`dagger`** | 144 depth beams, DAgger | **50.0%** | [0.44, 0.56] | 3.38 m |

240 held-out launches per arm — 120 seeds × 2 seeds, pooled. Every arm flew
the **identical** seed list; `compare.py` refuses to report if they did not.

Paired, on matched episodes, exact McNemar on the discordant pairs:

| question | wins | p | answer |
|---|---|---|---|
| does the sensor help PPO at all? | 6 – 4 | **0.75** | **no** |
| distillation vs RL on the same sensor | 116 – 2 | **< 0.0001** | yes, overwhelmingly |
| student vs the teacher it learned from | 41 – 31 | 0.29 | indistinguishable |
| DAgger vs plain cloning | 35 – 29 | 0.53 | **not significant** |

> :::info In plain words
> The car has a depth sensor pointed at the ground, like a cheap laser
> scanner. Left to work out for itself what the sensor means, it never does —
> it crashes as often as a car with no sensor at all. But if an expert flies
> first and the car is shown "in this situation, do this", it learns to read
> the sensor in about an hour and lands nearly as well as the expert.
> :::

**The teacher's seed spread is 20 points** — 55.8% and 35.8% on two seeds.
That is a property of this task, not of the setup; lab 3 measured the same
thing. Both distilled arms learned from `oracle_s0` only, so the student's
retention is quoted against **55.8%**, the teacher actually used, and the
per-seed column is published alongside every pooled figure.

## Two stages, and where PPO sits

This lab runs in sequence, not in parallel, and the two halves are trained by
completely different algorithms. **PPO is how the teacher is created;
distillation is how its knowledge is moved.** Nothing is adversarial and
nothing co-evolves — the teacher is finished and frozen before the student
exists.

```mermaid
flowchart LR
  subgraph S1["STAGE 1 — reinforcement learning, 40 min"]
    direction TB
    R["reward only<br/>one number per step"] --> P["PPO<br/>4M steps, 16 envs"]
    P --> T["TEACHER<br/>41 in, 20 out<br/>55.8% on target"]
  end
  subgraph S2["STAGE 2 — supervised, 35 min"]
    direction TB
    L["111,000 labelled rows<br/>X 183 wide, Y 20 wide"] --> F["least squares<br/>12 epochs x 10 rounds"]
    F --> U["STUDENT<br/>183 in, 20 out<br/>50.0% on target"]
  end
  T -->|"frozen, then asked<br/>'what would you do here?'"| L
  classDef deep   fill:#08182a,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
  classDef dark   fill:#0a1f33,stroke:#2d5a86,stroke-width:1.5px,color:#ffffff
  classDef base   fill:#16324f,stroke:#3f6f9f,stroke-width:1.5px,color:#ffffff
  classDef bright fill:#1e5f8f,stroke:#63a3d0,stroke-width:1.5px,color:#ffffff
  classDef steel  fill:#28527a,stroke:#6aa2cd,stroke-width:1.5px,color:#ffffff
  class S1,S2 deep
  class R,L base
  class P,F steel
  class T,U bright
```

Nobody ever writes down which valve to open. The simulator supplies *facts
about the state* — the car's exact height — and never an action. The teacher
has to discover the actions itself, by crashing four million times against a
scalar reward. That discovery is stage 1, and without it there is no teacher
to distil.

**Three of the five arms are PPO**, and that is what makes the finding a
statement about reinforcement learning rather than about imitation:

| arm | trained by | signal it learns from | result |
|---|---|---|---|
| `oracle` | **PPO** | 1 scalar reward | 55.8% |
| `direct` | **PPO** | 1 scalar reward | **2.5%** |
| `deaf` | **PPO** | 1 scalar reward | 1.7% |
| `bc` | regression | 20 numbers, every step | 47.5% |
| `dagger` | regression | 20 numbers, every step | 50.0% |

The teacher and the `direct` arm are the *same algorithm on the same budget*.
Only the terrain channel differs — 2 exact numbers against 144 beams. So the
failure at 2.5% is not that the control problem is too hard; PPO solved that
with the 2 numbers. It is that one scalar arriving 600 steps later is too thin
a signal to *also* teach perception.

> :::info In plain words
> First we train an expert the slow way: let it crash four million times and
> keep what works. That takes about forty minutes and nobody ever tells it
> what to do — it only hears a score. Then we freeze that expert and use it
> as an answer key for a second network that has to work from a sensor
> instead. The second stage is ordinary supervised learning, like fitting any
> model to labelled data.
> :::

> A note on the obvious comparison: this is **not** a GAN. A discriminator
> returns one number ("fake") and keeps learning as you train it; a teacher
> here returns twenty ("open these valves this much") and never moves again.
> A PPO critic is the structural analogue of a discriminator — also one
> number, also moving. The gap between a 1-wide and a 20-wide training signal
> is this lab's entire result.

## Motivation

Lab 3 asked whether a lander benefits from seeing the ground and got an
uncomfortable answer: a 5×5 height patch took PPO from 56% to **0%**. The
patch was 25 extra inputs the policy had to learn to ignore, on a task whose
own divert-budget sweep showed the best use of perfect terrain knowledge was
not to use it.

That left a different question unanswered. Information the policy genuinely
needs — *how high am I above this ground* — is handed to every arm in labs 1
through 6 as an exact number, because a simulator can simply look it up. No
vehicle can. A real lander has to estimate it from a sensor, and the sensor
is noisy, occluded, perspective-distorted and bolted to a tumbling body.

**This lab removes the free answer and gives the vehicle the measurement
instead.** It is the standard teacher–student setup from legged robotics: a
teacher trained on privileged state it could never carry, distilled into a
student that reads hardware.

> :::info In plain words
> In a simulation you can just ask "how high is the car?" and get a perfect
> answer. A real vehicle cannot — it has to work it out from a sensor that
> is noisy and sees only part of the ground. This lab takes the perfect
> answer away and sees whether the car can learn to measure it instead.
> :::

## Problem setup

| | |
|---|---|
| **Goal** | arrive on the wheels within 4 m of a marked spot, on ground gentle enough to survive |
| **Success gate** | sink ≤ 2.5 m/s, tilt ≤ 25°, spin ≤ 1.2 rad/s, miss ≤ 4 m. Close but broken does not count; intact but elsewhere does not count |
| **Degrees of freedom** | 6 for the vehicle (3 position, 3 attitude) plus 4 wheel spins |
| **Action** | 20 valve openings in [0, 1] at 100 Hz, one per nozzle |
| **Constraints** | 297 kg of nitrogen (≈ 7 s of hover); thrust fades as the tank cools; nozzles push and never pull; 25 ms valve lag; 10 ms transport delay; 6 s clock; anything but a wheel touching the ground ends the episode |
| **Terrain** | `badlands` — packed boulder fields. 43% of marks are survivable; 96% have survivable ground within 4 m |
| **Episode** | 600 steps maximum |
| **Discount** | γ = 0.999, a property of the task — see below |

### The observation, and the one channel that differs

Every arm receives the same 13 numbers of attitude, body rates and velocity;
then a **terrain channel at index 13**; then the mark's bearing, the clock,
the tank state and the previous action.

| arm | terrain channel | width |
|---|---|---|
| `oracle` | exact altitude / 8, exact lowest-corner clearance / 8 | 41 |
| `onboard` | 144 normalised rangefinder returns | 183 |
| `deaf` | — | 39 |

The `oracle` observation is **byte-identical to lab 3's `blind` arm**, which
lab 3 measured at 56%. That is the reproduction control for this lab, it is
asserted in [`tests/test_teacher_student.py`](https://github.com/yiqiao-yin/deepspeed-course/blob/main/tests/test_teacher_student.py), and it is not decoration — see
[Limitations](#limitations).

### The reward

Per step, with $\Delta t = 0.01$ s:

$$
r_t \;=\; -0.05\,\lVert\boldsymbol{\omega}_t\rVert^2 \Delta t
\;-\; 0.004 \sum_{i=1}^{20} u_{i,t}\,\Delta t
\;-\; 0.15
\;+\; \underbrace{\gamma\,\Phi(s_{t+1}) - \Phi(s_t)}_{\text{shaping}}
$$

The three terms are a spin penalty, a propellant penalty, and a **constant
cost per airborne step**. The potential is

$$
\Phi(s) \;=\; 3\max\!\left(0,\, 1 - \tfrac{d}{40}\right)
\;+\; 2\left(1 - \tfrac{\min(h,\,10)}{10}\right)
\;+\; \tfrac{1}{2}\left(\hat{\mathbf{z}}\cdot\mathbf{u} + 1\right)
\;\in\; [0,\, 6]
$$

where $d$ is the horizontal miss, $h$ the altitude, and
$\hat{\mathbf{z}}\cdot\mathbf{u}$ the uprightness. At termination:

$$
r_{\text{terminal}} =
\begin{cases}
-40 - 4\min(v_{\text{sink}},\,8) & \text{crash}\\[2pt]
-90 & \text{timeout}\\[2pt]
-5 & \text{landed, gate failed}\\[2pt]
70 + 20\left(1 - \tfrac{m}{4}\right) + 10\left(1 - \tfrac{\theta}{25}\right) & \text{on target}\\[2pt]
\max\left(5,\; 25 - 3(m - 4)\right) & \text{landed, off target}
\end{cases}
$$

**Nothing in the reward mentions terrain, perception or the sensor.** It is
lab 3's reward unchanged, because this lab varies exactly one thing and a
re-tuned reward would have made the comparison meaningless.

> :::info In plain words
> The car earns a big prize for landing gently on the mark, loses a lot for
> crashing, and pays a small fee for every second it stays in the air so it
> cannot just hover until the clock runs out. The "shaping" term is a gentle
> nudge toward being low, level and near the mark.
> :::

Two details in that reward are load-bearing and were each paid for in lab 3:

- **Shaping is capped near 6.** At $\gamma < 1$ an agent sitting in a state
  collects $(\gamma - 1)\Phi$ every step. With $\Phi_{\max} \approx 31$ that
  toll is ~124 per episode against a ~100 prize, and the policy learns to
  flee *good* states. Sign and scale are both load-bearing.
- **The crash penalty is graded by impact speed.** A flat −60 scores a 10 m/s
  dive and a 2.6 m/s arrival identically, so there is no gradient from
  crashing toward landing.

## The sensor

**[▶ The beams and the image they make (full clip)](/img/aerial/ts-sensor-split.gif)**

Left: the car falling toward the boulder field with its ranging beams drawn
and a dot where each one strikes. Right: the 12×12 depth image, which is
**the only terrain information the student ever receives**. Near is dark, far
is bright, and a no-return cell is flat grey — deliberately not the far end
of the ramp, because a missing return and a distant surface are different
facts even though the policy receives 1.0 for both.

Watching them together is the argument: the beams sweep as the body rolls, so
the image swings when the ground is still; the sample spacing stretches with
altitude, so the same boulder occupies fewer pixels; a rock edge casts a
shadow of no-returns that looks exactly like distant flat ground.

### Rays, not a rendered camera — and the measurement that decided it

The obvious way to give a policy sight is to render a depth image. Measured
first, on this box, and rejected:

| | ms/frame |
|---|---|
| offscreen depth, 32×32, hazard terrain | 147 |
| offscreen depth, 64×64, hazard terrain | 146 |
| offscreen depth, 32×32, **flat ground** | 109 |
| **a 12×12 ray scan** | **0.41** |

The resolution column is the tell. A renderer whose cost does not move when
the pixel count quadruples is not spending its time on pixels, and flat
ground at 109 ms says it is not the terrain mesh either. It is per-frame
overhead in the GL stack: OpenGL here reaches the GPU through WSL's D3D12
translation layer, and forcing the discrete card instead of the integrated
one changed nothing. Topic 7 lab 6 hit the same class of bug from the other
side — a physics backend that silently fell back to CPU and reported
throughput that looked like GPU throughput.

A renderer also needs a GL context per process, which the 16-worker pool in
`vecenv.py` cannot provide. **`mj_ray` is not a consolation prize:** a
scanning rangefinder is the hardware small landers actually carry, and ray
casting models one faithfully. Everything that makes perception hard survives
— occlusion, perspective, attitude coupling, range limits, 2% range noise and
1% dropout.

## The information gap

Before training anything, two probes — because "the student has a harder
problem" is a claim about information and can be measured rather than
asserted. Both use clean scans, which is deliberately generous to the student.

| probe: scan + attitude → | train R² | test R² | verdict |
|---|---|---|---|
| the teacher's 2 numbers (altitude, clearance) | +0.918 | **+0.834** | recoverable, to ~0.7 m |
| a 5×5 world-aligned height patch (lab 3's `patch`) | +0.993 | **+0.114** | **not recoverable** |

The pairing is the point. A single low test score cannot distinguish "the
information is absent" from "the probe was undertrained" — so the training
score is printed beside it. A probe that fits its training set at 0.993 and
generalises at 0.114 is memorising, which is what an absent signal looks like.

Aliasing, on 600 states: the state whose *scan* is nearest is **9.4×** further
away in the teacher's world than the teacher's own nearest neighbour. The
observation map is not injective — the student faces a partially observed
problem where the teacher faces a fully observed one.

> :::info In plain words
> We checked what the sensor can and cannot tell you before training
> anything. It *can* tell you how high you are, to about 70 cm. It *cannot*
> reconstruct a map of the ground around you — that information simply is
> not in the beams. So the lab asks the student to learn the first thing and
> never asks it for the second.
> :::

Run it: `uv run teach_env.py --gap`.

## Method

### PPO, for the three reinforcement-learning arms

Clipped surrogate objective, with the probability ratio
$\rho_t(\theta) = \dfrac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$:

$$
L^{\text{CLIP}}(\theta) \;=\; \hat{\mathbb{E}}_t\!\left[
\min\Big(\rho_t(\theta)\,\hat{A}_t,\;
\operatorname{clip}\big(\rho_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\hat{A}_t\Big)
\right]
$$

with advantages from GAE($\lambda$):

$$
\hat{A}_t \;=\; \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l},
\qquad
\delta_t \;=\; r_t + \gamma V(o_{t+1}) - V(o_t)
$$

$\epsilon = 0.2$, $\lambda = 0.95$, $\gamma = 0.999$, 6 epochs, 8 minibatches,
Adam at $3\times10^{-4}$, 16 parallel environments, horizon 256.

### Distillation, for the two supervised arms

Let $s$ be the simulator state, $\pi^\star$ the teacher acting on privileged
state, and $o(\cdot)$ the student's observation map. **Behaviour cloning**
minimises the regression loss on the *teacher's* state distribution:

$$
\theta_{\text{BC}} \;=\; \arg\min_\theta \;
\mathbb{E}_{s \sim d_{\pi^\star}}
\Big[\big\lVert \pi_\theta(o(s)) - \pi^\star(s) \big\rVert^2\Big]
$$

and is then deployed on $d_{\pi_\theta}$, which is a different distribution.
**DAgger** removes the mismatch by aggregating the student's own states:

$$
\mathcal{D}_i = \bigcup_{j \le i}
\Big\{\big(o(s),\, \pi^\star(s)\big) \;:\; s \sim d_{\pi_{\theta_j}}\Big\},
\qquad
\theta_{i+1} = \arg\min_\theta \,
\mathbb{E}_{(o,a) \sim \mathcal{D}_i}\big\lVert \pi_\theta(o) - a \big\rVert^2
$$

with the teacher driving with probability $\beta_i = 0.55^{\,i}$ and the
student otherwise. **The label is always the teacher's**, whoever drives; it
is the driver that changes, never the labeller. Round 0 has $\beta_0 = 1$ and
is therefore exactly behaviour cloning — BC is not a second code path, it is
this one forced to a single round, so the two arms cannot drift apart in some
detail nobody is watching.

### Why the distinction should matter

Ross and Bagnell's result: if the learned policy has expected per-step loss
$\epsilon$ under the expert's distribution, behaviour cloning's regret over
horizon $T$ grows as

$$
J(\pi_{\text{BC}}) - J(\pi^\star) \;=\; O\!\left(\epsilon T^2\right)
\qquad\text{against}\qquad
O\!\left(\epsilon T\right) \text{ for DAgger.}
$$

A small error moves the vehicle somewhere the teacher never went, where the
student was never trained, where its error is larger — which moves it
further. **This course has already measured that mathematics once in a
different costume:** lab 6 fed a dynamics model its own predictions and
watched the error grow with rollout length rather than staying at its
one-step value. Same phenomenon, same cause. Here $T = 600$.

> :::info In plain words
> Copying an expert has a hidden flaw. The learner only ever saw situations
> the expert got into, so the first time it makes a small mistake it ends up
> somewhere unfamiliar, makes a bigger mistake, and spirals. DAgger fixes
> this by letting the learner fly and asking the expert "what would you have
> done here?" — so it learns about its own mistakes. On this task, it turned
> out not to matter much.
> :::

### The comparison is budget-matched

Both distilled arms received **600 labelled episodes ≈ 111,000 transitions**
and 12 epochs per round. BC spends its whole budget in round 0. Otherwise
"DAgger wins" would be indistinguishable from "DAgger saw more data" — an
uncontrolled comparison this repository has published before and had to
withdraw.

## Animations

**[▶ Teacher vs direct RL, four identical launches](/img/aerial/ts-teacher-vs-direct.gif)**

Left, the teacher, told its exact height. Right, PPO on 144 depth beams and
nothing else. Same terrain, same mark, same launch, same propellant — both
panels are one seed stepped in lockstep, which is the visual form of the
paired test. **Three launches, three teacher landings, three crashes.** At
seed 900001 the teacher settles 0.38 m from the mark while the direct-RL arm
is inverted at 119° of tilt. This is the 2.5% arm.

**[▶ Teacher vs the distilled student](/img/aerial/ts-teacher-vs-student.gif)**

The same three launches with the DAgger student on the right. It is reading
the ground off its beams, and the panels now look alike: **on target, on
target, and one hard landing** against the teacher's three. That is what
"p = 0.29, statistically indistinguishable" looks like from outside — and
the one failure is in the film on purpose, because a montage of only the
good launches would be a different and less honest figure.

**[▶ Cloning vs DAgger](/img/aerial/ts-bc-vs-dagger.gif)**

The honest negative. Both distilled arms, side by side, on the same two
launches: **both land hard at seed 900000; at 900001 cloning lands off the
mark where DAgger lands on it.** One trade each way in two films, which is
what a p = 0.53 difference looks like. DAgger is not visibly a better pilot
here, and the clip is included because the lab claims it is not.

**[▶ The beams and the depth image](/img/aerial/ts-sensor-split.gif)**

The sensor, as above.

Large clips are linked rather than embedded; regenerate any of them with
`uv run render_arms.py` or `uv run render_sensor.py --split`.

## The DAgger rounds

| round | β | labelled pairs | loss | on target | teacher drove |
|---|---|---|---|---|---|
| 0 | 1.000 | 11,029 | 0.0334 | 30% | 100% |
| 1 | 0.550 | 21,712 | 0.0188 | 52% | 56% |
| 2 | 0.303 | 33,196 | 0.0124 | 45% | 30% |
| 4 | 0.092 | 55,345 | 0.0084 | 35% | 9% |
| 6 | 0.028 | 77,798 | 0.0062 | 50% | 3% |
| 9 | 0.005 | 111,533 | 0.0051 | 40% | 1% |

Per-round scores are 40 episodes and are a curve, not a result; the published
figures are 120 held-out launches. **The regression loss falls monotonically
by 6.5× while the on-target rate wanders between 30% and 52%** — a reminder
that the imitation loss is a proxy and the task metric is the thing.

## The falsifier

Written down before any arm finished:

> The distilled student must land on target at least **80%** as often as its
> teacher, **and** must beat behaviour cloning by more than the seed-to-seed
> spread. If it does not, distillation does not rescue perception on this
> task and the central claim of this lab is wrong here.

| clause | measured | |
|---|---|---|
| retention ≥ 80% | 50.0% of 55.8% = **90%** | **PASS** |
| beats BC by > the seed spread | +2.5% vs a 0.0% spread | PASS, *and see below* |

**The second clause is a bad test, and this run shows why.** Both distilled
arms scored identically on both seeds, so the measured spread is 0.0% and
*any* positive margin clears it — the bar is satisfiable by noise. The paired
test is the one to believe: McNemar on the 64 discordant episodes gives
**p = 0.53**. DAgger's edge over plain cloning is not significant here.

**Corrected verdict: the retention clause holds, the beats-cloning clause
fails.** `compare.py` prints both, in that order, and does not quietly
substitute the better-looking one.

## Limitations

**DAgger did not earn its complexity here.** The compounding-error argument
predicts it should, and it did not at p = 0.53. The most likely reason is
that this task *terminates on error*: a crash ends the episode at the first
serious mistake, so the quadratic term in $O(\epsilon T^2)$ never gets 600
steps to compound. A task that tolerates mistakes and runs to the clock would
be a fairer test of the theorem.

**A feature permutation cost the teacher 60% → 4%, silently.** The terrain
channel was first placed after the clock instead of at index 13 — the same 41
numbers in a different order. That version ran, trained, and scored 4% and 2%
on two seeds, with the same learning-curve shape a million steps late and
still climbing at the cutoff. Nothing raised; every array was the right
shape. A permutation is a different random initialisation, and on a task
whose seed spread is 20 points that is enough to look like a broken
environment. The byte-identity with lab 3's `blind` arm is now asserted in
the test suite.

**The sensor's self-hit audit fired on its first real run** at 100%: the
mount sat inside the hull and every beam returned 4 mm. Correct dtype,
correct shape, no signal. The audit is an assertion rather than a note in a
README for exactly that reason.

**Two seeds is thin** for an arm whose spread is 20 points. The RL arms'
pooled figures should be read with the per-seed column beside them. The
distilled arms are far more stable — identical on both seeds — which is
itself a finding about supervised versus reinforcement learning on this task.

**The student is a flat MLP over 144 values.** At this resolution that is
appropriate and it keeps the network identical to the PPO arms, which is what
makes the method comparison a comparison. A 64×64 depth image would want a
convolutional encoder, and that is where `--size deep|xl` and `ds_config.json`
become relevant rather than decorative.

## Environment & local testing

```bash
cd 08_aerial_physics/07_teacher_student
uv sync

uv run rangefinder.py                 # the sensor's geometry, instant, no MuJoCo
uv run rangefinder.py --scene --cost  # cast it at real terrain, time it
uv run teach_env.py --gap             # what the beams can and cannot recover
uv run compare.py                     # score the arms, paired
```

No GPU is required and none is used by the headline arms. The policies are
15k–33k parameters stepped against MuJoCo on the CPU; topic 7 measured a
policy that size running **1.7× slower** on a GPU, because the rollout batch
never fills a card. Ask for **cores**.

`distill.py --size deep|xl` builds a student large enough to want a card and
carries a real guard; `--size match` (the default, 17,236 parameters) does
not and is the arm every published number comes from.

Reproduce from scratch:

```bash
./run_arms.sh          # 6 PPO runs, ~40 min each on 16 workers
./run_distill.sh       # 4 distillation runs, ~35 min each
uv run rescore.py      # re-fly every arm on ONE common seed list
uv run compare.py      # paired statistics and the falsifier
```

`rescore.py` is not optional. The PPO arms and the distilled arms score
themselves with different episode counts, and `compare.py` **refuses** to
report unpaired arms rather than comparing two different sets of launches.

### CoreWeave

```bash
sbatch run_deepspeed.sh                 # the full pipeline
sbatch run_deepspeed.sh --dry-run       # ~3 min, proves it assembles
squeue -u $USER
tail -f logs/ts_distill_*.out
```

### RunPod

```bash
uv run ../../runpod/runpod_ctl.py run 08_aerial_physics/07_teacher_student \
    --dry-run --collect --wait --terminate --yes
uv run ../../runpod/runpod_ctl.py pods      # confirm nothing is left running
```

The pod never receives an API key; termination is driven from the local
machine in a `finally`, with a keyless in-pod watchdog as backstop. See
`SECURITY.md`.

## Files

| file | role |
|---|---|
| `rangefinder.py` | the 144-beam depth sensor, and its geometry self-checks |
| `teach_env.py` | the task and the three arms; `--gap` measures the information |
| `train_policy.py` | PPO for `oracle`, `onboard` and `deaf` |
| `distill.py` | BC and DAgger; the only arm with a DeepSpeed path |
| `rescore.py` | re-fly every arm on one common seed list |
| `compare.py` | paired McNemar, Wilson intervals, the falsifier |
| `render_sensor.py` | the beams and the depth image |
| `render_arms.py` | two arms down the same launch, side by side |
| `ds_config.json` | ZeRO-2, for the large-student arm only |
| `run_arms.sh` / `run_distill.sh` / `run_deepspeed.sh` | the pipeline |

Logic test: `uv run tests/test_teacher_student.py` — 30 checks, no GPU, no
downloads, including the byte-identity reproduction control and a sensor
audit watched failing before it was trusted.

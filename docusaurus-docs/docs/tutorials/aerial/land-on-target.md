---
sidebar_position: 3
---

# Land on Target

<!--chain-->
:::info **Lab 3 of 7** in Aerial Physics. Lab 1 landed a tumbling car; lab 2
made it roll all the way over first; this one makes it land somewhere
specific.
← previous: [Barrel Roll](./barrel-roll)
· next: [Fly on Mars](./fly-on-mars) →
:::

:::info Reading this without a background in control or RL
Every section opens with a plain words box like this one. Read only those
and you get the whole story; the technical detail underneath is there if
you want it.

**The one line version:** a two-tonne car was trained to throw itself at a
four-metre mark painted on boulder-strewn ground and land on it, as reliably
as a controller a human engineer designed by hand. Letting it see the
ground made it worse.
:::

**Baseline:** a hand-written guidance controller — pick a spot, fly a PD
trajectory to it, flare — scored on the same held-out launches. 61% on target.
**Budget:** 4M steps per arm, `dense` rig, `badlands` terrain, two seeds per
arm, 16-process pool, RK4 at 2 ms. ~47 min per run on 20 CPU cores. No GPU.
**Falsifier:** if knowing the ground is useful, the arm that sees more of it
should score higher. The ordering is the exact reverse, monotonically, and a
second controller reproduces it independently.

A two-tonne car is thrown into the air, tumbling, over broken ground. A spot
is marked on the surface. It has six seconds of clock, seven seconds of
propellant, and twenty nozzles that can only push. It has to come down on
its wheels, intact, within four metres of the mark.

## Result

:::info In plain words
It works. The trained controller puts the car on the mark more than half the
time, arriving so gently it is barely moving when it touches down — about as
well as the hand-built engineering controller, except nobody told it how to
fly. The surprise is the second half: letting it see the ground made it
worse, and the more ground it could see, the worse it got.
:::

PPO lands the car on the mark **56% of the time** on held-out launches, with
a mean miss of **2.63 m** and a touchdown sink rate of **0.09 m/s** — it
arrives effectively settled rather than dropping the last metre. The
hand-written guidance controller, given the same vehicle and the same
launches, scores 61%. The two are statistically indistinguishable, and the
learned one was given no guidance law, no trajectory, and no notion of a
flare.

| arm | observation | on target | 95% CI | mean miss |
|---|---|---|---|---|
| **blind** | 33 dims | **56%** | [0.46, 0.66] | 2.63 m |
| slope | 35 dims | 25% | [0.17, 0.35] | 4.43 m |
| patch | 58 dims | 0% | [0.00, 0.04] | 4.26 m |

96 held-out episodes per arm, pooled over two seeds. The arms differ in
exactly one thing: how much of the ground the policy may see. `blind` gets
the mark and its own state plus a radar altimeter. `slope` adds the ground
gradient under the car. `patch` adds a 5 by 5 height patch spanning 8 m.

### Film: three landings on the mark

**[▶ Watch the landing reel (9.5 MB GIF)](/img/aerial/target-land-reel.gif)**

Three consecutive on-target landings by the trained blind policy, from
held-out launches it never trained on. The HUD carries the whole state: the
circular widget bottom-left is a plan view 24 m across with the 4 m scoring
circle drawn at true scale, so "inside the circle" is something you judge by
eye and the miss readout only confirms. Top right reads the phase and the
live miss distance; the right-hand panel carries attitude, tank pressure and
temperature, live specific impulse and nozzle exit temperature. The yellow
posts in the 3D scene mark the scoring circle on the ground itself.

Watch the gas counter: these landings take 86, 176 and 91 kg of a 297 kg
tank, and between 1.5 and 2.8 seconds of a six-second clock.

### Film: a failure, for calibration

**[▶ Watch a crash (4.4 MB GIF)](/img/aerial/target-crash-example.gif)**

The same policy on a launch it does not recover. It arrives over the mark
and runs out of vertical authority before it runs out of descent rate. A
page that only showed the successes would be describing a different
controller, so here is the whole distribution over the 96 held-out
launches:

| outcome | blind arm |
|---|---|
| **on target** | **56%** (54/96) |
| crash | 29% (28/96) |
| landed, too hard | 7% (7/96) |
| landed, missed the mark | 5% (5/96) |
| never landed (timeout) | 2% (2/96) |

Crashes are the dominant failure, not misses: when this controller fails it
is almost always because it arrived too fast, not because it arrived in the
wrong place. The mean miss across *all* launches is 2.63 m, comfortably
inside the 4 m circle.

## Terrain information makes it worse

:::info In plain words
We ran the same training three times, changing one thing: how much of the
ground the computer was allowed to see. The version that could see nothing
did best. The version that could see most barely learned to land at all.
:::

![Learning curves for the three arms](/img/aerial/target-curve-arms.gif)

*On-target rate against environment steps, for the three arms. The dashed
line is the hand-written controller. The blind arm (green) reaches it; the
gradient arm (blue) trails; the full height-patch arm (red) never leaves the
floor inside this budget.*

![Crash rate for the three arms](/img/aerial/target-curve-crash.gif)

*The same three runs, scored on crashes. Blind and gradient both fall to
roughly a quarter; the height-patch arm is still crashing on 94% of episodes
at 4M steps.*

### The second controller agrees

The hand-written controller chooses a landing spot by searching outward for
survivable ground. How far it will divert is a single knob, and sweeping it
separates two different questions: is terrain knowledge useless, or was it
merely used badly?

| divert budget | on target | survived | mean miss | gas |
|---|---|---|---|---|
| **0 m** (ignores terrain) | **61%** | 83% | 2.81 m | 146 kg |
| 2 m | 61% | 83% | 2.82 m | 147 kg |
| 4 m | 55% | 80% | 3.05 m | 148 kg |
| 6 m | 52% | 78% | 3.28 m | 151 kg |

Diverting up to 2 m is free and worthless. Past that it costs accuracy,
survival and fuel together. **The best use of perfect terrain knowledge is
to ignore it**, and that holds for a hand-written controller and a learned
one independently — two different mechanisms arriving at the same answer.

**[▶ Watch the engineered controller diverting (14.9 MB GIF)](/img/aerial/target-divert-reel.gif)**

*The scheduled controller with a 6 m divert budget. The magenta cross on the
plan view is the spot it chose; the gap between that cross and the centre of
the circle is the divert, and it is the thing the sweep above prices. These
runs reach the mark, and they spend 127 to 174 kg of gas doing it against
the blind controller's 146 kg average.*

### Why ignoring the ground wins

The mechanism is a comparison of two lengths, and it is measured rather than
argued:

| quantity | measured |
|---|---|
| touchdown scatter of either controller | **2.6–2.8 m** |
| median distance from bad ground to good | **4.0 m** |
| targets with survivable ground within 2 m | 66% |
| targets with survivable ground within 4 m | 82% |

A controller that aims at bad ground still lands on good ground much of the
time, because its own scatter is the same size as the gap it would have
deliberately crossed. Crossing that gap on purpose costs lateral travel,
propellant, clock, and — since the scoring circle is 4 m and the escape is
4 m — frequently the target itself.

:::info In plain words
The car cannot land exactly where it aims; it is off by about three metres
either way. The patches of bad ground are also about three metres across. So
aiming at a bad patch lands you on good ground roughly as often as aiming at
a good patch does, and going out of your way to reach the good patch burns
fuel and pushes you off the target. The knowledge is genuine, and using it
still does not pay.
:::

## The catch-up test

:::info In plain words
A fair objection: maybe seeing the ground is not harmful, just slower to
learn from, and the wide-view version simply needed more practice. We gave
both versions two and a half times the practice to find out. It did not
catch up.
:::

At 4M steps every arm was still improving, so "patch scores 0%" was a claim
about sample efficiency rather than a ceiling. Both arms were therefore
continued to **10M steps on matched budgets** — carrying only the losing arm
further would have answered a different question.

| arm | 4M | 10M | 95% CI at 10M |
|---|---|---|---|
| **blind** | 56% | **71%** | [0.61, 0.79] |
| patch | 0% | 8% | [0.04, 0.16] |

`patch` improved, from 0% to 8%, with its crash rate falling from 94% to
82%, and it is still improving. It is not stuck. But at 2.5 times the budget
it reaches roughly a ninth of what the blind arm reaches, and the gap is
wider in absolute terms at 10M than at 4M. So the claim is not "the wide
view can never learn this" but something nearly as strong: **25 extra inputs
of perfect, noiseless terrain information cost more than 2.5 times the
sample budget and were still losing.**

The mechanism is unglamorous and matters more than the framing. Those 25
inputs nearly double the observation width for a 13k-parameter policy, and
each one is a dimension the network must learn to ignore. The information is
not merely unhelpful; it is expensive to discount.

## Problem setup

:::info In plain words
Below is exactly what the car may do, what it cannot do, and what counts as
winning. Reinforcement learning needs all three stated precisely, or it will
find a way to win that you did not mean — which is exactly what happened
here, four times.
:::

**Goal.** Arrive on the wheels within 4 m of the mark, under 2.5 m/s sink,
under 25 degrees of tilt, under 1.2 rad/s of spin. Close but broken does not
count. Intact but elsewhere does not count.

| degrees of freedom | |
|---|---|
| vehicle | 6 — three translation, three attitude |
| wheels | 4 spins |
| commanded | 20 valve openings in [0, 1] at 100 Hz |

| constraints | |
|---|---|
| propellant | 297 kg, about 7 s of hover; thrust fades as the tank cools |
| nozzles | push only, never pull |
| actuation lag | 25 ms valve time constant, 10 ms transport delay |
| clock | 6 s |
| termination | anything but a wheel touching the ground ends the episode |
| ground | badlands — packed boulder fields, not a parking lot |

### The reward function, in full

:::info In plain words
Below is the exact scoring rule, written in maths. If that is not your
language, here it is in one paragraph: **every moment that passes costs the
car a small fixed amount**, so dithering is never free. Spinning and burning
gas cost a little more. On top of that there is a running "how well is this
going" score — closer to the mark, lower down, right way up — and the car is
paid the *improvement* in that score each moment rather than the score
itself. Then at the very end it gets one big number: a large bonus for
landing on the mark, a small one for landing safely elsewhere, and
penalties of increasing size for landing hard, crashing, and never landing
at all.
:::

**Per step.** At each of the 600 control steps the car receives

$$
r_t = r^{\text{time}} + r^{\text{spin}} + r^{\text{gas}} + F_t
$$

$$
r^{\text{time}} = -0.15, \qquad r^{\text{spin}} = -0.05\,\lVert \boldsymbol\omega_t \rVert^2 \Delta t
$$

$$
r^{\text{gas}} = -0.004 \sum_{i=1}^{20} u_{t,i}\, \Delta t, \qquad \Delta t = 0.01\ \text{s}
$$

where $\boldsymbol\omega_t$ is the body angular rate and $u_{t,i} \in [0,1]$
is the commanded opening of nozzle $i$. The spin and gas terms are
deliberately tiny — about $10^{-2}$ per step against a time cost of
$1.5 \times 10^{-1}$ — because they are preferences, not objectives.

**Shaping.** $F_t$ is a potential-based shaping term in the sense of Ng,
Harada and Russell (1999):

$$
F_t = \gamma\,\Phi(s_{t+1}) - \Phi(s_t), \qquad \gamma = 0.999
$$

$$
\Phi(s) = \Phi_{\text{near}} + \Phi_{\text{low}} + \Phi_{\text{up}}
$$

with, writing $d$ for the horizontal miss distance in metres, $h$ for height
above ground, and $\hat{\mathbf{z}}\cdot\hat{\mathbf{u}}$ for the cosine
between the car's own up-axis and the world vertical,

$$
\Phi_{\text{near}} = 3\max\!\left(0,\ 1 - \tfrac{d}{40}\right) \in [0, 3]
$$

$$
\Phi_{\text{low}} = 2\left(1 - \tfrac{\min(h,\,10)}{10}\right) \in [0, 2]
$$

$$
\Phi_{\text{up}} = \tfrac{1}{2}\left(\hat{\mathbf{z}}\cdot\hat{\mathbf{u}} + 1\right) \in [0, 1]
$$

:::info In plain words
The shaping term is a trick for giving hints without changing what counts as
winning. You define a "warmth" score — warmer when the car is near the mark,
low down and the right way up — and pay the car the *change* in warmth at
each step. Because the payments are differences, they cancel out over a whole
flight, so the hints steer the search without quietly inventing a different
goal. The ranges on the right are the deliberate part: the whole warmth score
can never exceed 6, and the next paragraph explains why that ceiling matters
more than it looks.
:::

**Why $\Phi$ is capped at 6.** Telescoping the shaping over an episode that
ends at step $T$ leaves

$$
\sum_{t=0}^{T-1} F_t = \gamma^{T}\Phi(s_T) - \Phi(s_0) + (\gamma - 1)\sum_{t=1}^{T-1}\gamma^{t-1}\Phi(s_t)
$$

The last term is the one that bites. With $\gamma$ below 1 it is a *toll* on
occupying high-potential states, worth roughly $(1-\gamma)\,\Phi_{\max}\,T$
over an episode. At $\Phi_{\max} = 6$, $\gamma = 0.999$ and $T = 600$ that is
about 3.6 — negligible against a terminal prize near 100. The first version
of this lab had $\Phi_{\max} = 31$ at $\gamma = 0.995$, a toll of about 93,
and the policy duly learned to run away from good states. See
[Reward design](#reward-design).

**Terminal.** One of five mutually exclusive outcomes is added on the final
step. Writing $v_\downarrow$ for sink rate in m/s and $\theta$ for tilt in
degrees, and calling a touchdown *gentle* when
$v_\downarrow \le 2.5$, $\theta \le 25$ and $\lVert\boldsymbol\omega\rVert \le 1.2$:

$$
R_T =
\begin{cases}
70 + 20\left(1 - \tfrac{d}{4}\right) + 10\left(1 - \tfrac{\theta}{25}\right)
  & \text{gentle},\ d \le 4 \\[4pt]
\max\!\left(5,\ 25 - 3(d - 4)\right) & \text{gentle},\ d > 4 \\[4pt]
-5 & \text{landed, not gentle} \\[4pt]
-40 - 4\min(v_\downarrow,\, 8) & \text{crash} \\[4pt]
-90 & \text{never landed}
\end{cases}
$$

:::info In plain words
Every line of that table is graded rather than pass/fail, and that is the
point. Landing dead centre pays more than landing at the edge of the circle;
arriving at walking pace pays more than arriving at speed; and even a crash
is scored by how fast the car hit, so "slow down" is something the computer
can discover one small improvement at a time. A rule that just said "crash =
bad" would be a cliff with nothing to climb.
:::

Two properties of this ladder are load-bearing and the test suite pins both.
It is **monotone** — every outcome scores strictly above the one below it, so
no failure is ever worth choosing — and the worst outcome is **never
landing**, not crashing. The reason that ordering has to be enforced rather
than assumed is the discounting argument in
[Reward design](#reward-design): a penalty arriving later is cheaper, and
the policy will exploit that unless something charges for delay.

## Hardware check

Two rigs in the set have no dead wrench axis, meaning they can translate in
any direction without rotating first:

| rig | nozzles | rank | dead axes |
|---|---|---|---|
| `minimal` | 6 | 5 | −Fx, −Fz |
| `reveal` | 10 | 6 | −Fx |
| `fore10` | 12 | 6 | **none** |
| `dense` | 20 | 6 | **none** |

Full rank is not reachability: unidirectional nozzles make the achievable
wrench set a convex polytope rather than a subspace, so a rank-6 rig can
still be unable to push backwards. `reveal` is rank 6 and cannot produce
−Fx.

That left two candidates, and they are **not** interchangeable:

| rig | on target at 2.8M steps |
|---|---|
| `fore10` | 6% |
| `dense` | **37%** |

Lab 1's own run history says the same thing: on the easier task `fore10` did
not break out until 3.9M steps where `reveal` broke out at 1.7M. Three
probes of this lab were written off as reward failures before that curve was
read. **Check whether the rig is slow before concluding the reward is
wrong.**

## Terrain decidability

A perception experiment on ground with no decision in it produces a clean
null by construction, so the ground is measured before the policy is.

| preset | target survivable | good ground within 2 m | within 4 m |
|---|---|---|---|
| `rough` | 55% | 97% | 100% |
| `hazard` | 43% | 81% | 90% |
| `badlands` | **40%** | 66% | 82% |

`rough` was rejected: per-spot success varied by sd 0.13 against a 0.14
binomial noise floor, so every spot was as good as every other. `hazard`
made spots differ, but its features sat 1.5 m from safety — smaller than a
touchdown error, so the decision existed and was unexploitable. `badlands`
builds landable pans and impassable fields in 10–20 m patches from a
low-frequency mask, putting the median escape at 4.0 m.

The honest reading of the result above is therefore not "there was nothing
to see". There was something to see, it was measured, and seeing it lost.

## Reward design

:::info In plain words
This was the longest part of the work and none of it was about the car. Four
separate times the scoring rules accidentally described a different game
from the one we meant, and the computer correctly won that game instead.
Every time, the training charts looked healthy.
:::

The policy initially learned to **fly away**: crash rate falling to 0%, miss
distance climbing from 12 m to 86 m, 100% of episodes timing out. Four
distinct defects contributed, and the order they were found in matters
because three plausible fixes changed nothing.

**1. The shaping leak has a scale, not just a sign.** Potential-based
shaping adds `γΦ(s') − Φ(s)`. At γ below 1 an agent occupying a state
collects `(γ−1)Φ` every step it stays there. Lab 2 learned that a negative Φ
pays an agent to stall. This lab learned the other half: Φ max was 31, so
the toll for being in a good state was 93 per episode against a prize of
about 100. Φ is now capped near 6, a toll of about 3.

**2. Shaping must not go flat inside the arena.** The distance term
saturated at 12 m, so beyond 12 m the gradient was exactly zero and nothing
pulled the car home once it had fled. The term now falls out to 40 m.

**3. Discounting ranks terminals by timing, not only by size.** This is the
one that mattered:

| outcome | raw | discounted to step 0 |
|---|---|---|
| crash at step 100 | −60 | **−54.3** |
| timeout at step 600 | −90 | **−49.4** |

The larger penalty was the cheaper one, purely because it arrived later. The
policy was not avoiding a bad outcome, it was postponing one. Raising the
timeout penalty to −90 — worse than a crash — did nothing, and neither did
raising γ from 0.995 to 0.999. What fixed it was a constant per-step cost,
which makes postponement itself expensive. **If every episode times out,
reach for a time cost, not a bigger terminal.**

**4. A flat terminal has no gradient in it.** A crash scored −60 whether the
car arrived at 10 m/s or 2.6 m/s, so "slow down" was not discoverable. The
penalty is now graded by impact speed, the same reasoning that already
graded the on-target reward by miss distance.

One diagnostic error belongs with them. After fix 3, the car reached the
mark and dived into it on 98% of episodes. That was read as the time cost
being too aggressive, so it was halved — which brought the fleeing straight
back. Two symptoms, two causes: the dive was defect 4. Swapping one fix for
the other would have lost both.

The resulting outcome ladder is monotone, and the test suite pins it along
with a permanent counterexample for each of the four defects:

```
timeout −90  <  crash −64  <  hard landing −5  <  landed off +5…+25  <  on target +70…+100
```

## Limitations

- **Nothing here is at convergence.** Every arm was still improving at 4M,
  and the two carried to 10M were still improving there. Every comparison is
  at a fixed, matched budget.
- **`slope` has a four-fold spread across two seeds** (40% / 10%). Two seeds
  is enough to notice that and not enough to characterise it.
- **`slope` was not carried to 10M**, so the middle of the ordering is
  pinned at 4M only.
- **The terrain is noiseless and perfectly known.** A real lander's height
  estimate would be worse, which can only weaken the case for using it.
- **One terrain generator, one vehicle, one gravity.** The claim is scoped to
  `badlands` on Earth with the `dense` rig.

## Running it

```bash
cd 08_aerial_physics/03_land_on_target
uv sync

# free checks, cheapest first
uv run thrusters.py --all          # what the rig can and cannot do
uv run target_env.py --decidable   # is there a DECISION in this ground?
uv run verify.py                   # conservation residuals
uv run train_target.py --dry-run   # 40 s, assembles

# the engineered baseline, and the divert sweep that prices it
uv run classical.py --episodes 64 --max-divert 0
uv run classical.py --episodes 64 --max-divert 6

# one arm (~47 min on 20 cores)
uv run train_target.py --sense blind --name blind_s0 --total-steps 4000000

# everything, including both seeds and the baselines
bash run_arms.sh
```

No `deepspeed` launcher and no GPU, both deliberate and both measured. The
policy is about 13k parameters; topic 7 measured a policy this size running
1.7 times *slower* on a GPU because the rollout batch never fills a card.
This lab wants **cores**.

### Environment & local testing

Python 3.10 or newer, with `numpy`, `mujoco`, `scipy`, `pillow` and
`torch==2.11.0`. No model downloads, no dataset, about 39 packages, and the
whole lab runs on CPU. CoolProp appears only under the optional `tables`
extra and is build-time only — the nitrogen property tables are committed as
`n2_tables.npz`.

```bash
uv run tests/test_land_on_target.py      # 22 property checks, from the repo root
```

## What transfers

- **If every episode times out, the problem is probably timing, not size.** A
  discount factor silently reorders terminal rewards by when they arrive.
  Check the discounted value against the alternatives before making a
  penalty bigger.
- **Potential-based shaping has two failure modes, not one.** A negative
  potential pays you to stall; a large one charges rent for being in good
  states.
- **Grade every terminal.** A flat penalty is a cliff with no gradient, and
  the behaviour just short of it is invisible to the learner.
- **Check the rig before blaming the reward.** A six-fold difference in
  sample efficiency between two equally capable layouts looked exactly like a
  broken objective for three probes running.
- **Information is only worth acting on if acting moves you further than your
  own error.** Compare those two lengths before designing a perception
  system: here they were 2.6 m and 4.0 m, and that single comparison
  predicted the entire result.

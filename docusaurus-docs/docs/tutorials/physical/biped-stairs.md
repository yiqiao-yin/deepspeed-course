---
sidebar_position: 2
---

# A limb, or a constraint?

[Lab 1](./obstacle-hopper) built a one-legged hopper with its torso locked
upright, and justified the lock in passing: free the torso and the task
becomes the bipedal-balance problem, which is a different and much harder
thing. That was an assertion, not a measurement.

This lab measures it, against the obvious rival explanation — that
difficulty comes from having more joints to coordinate. Two switches,
crossed, on one staircase:

![The four robots](/img/physical/stairs-morphologies.png)

| | torso locked | torso free |
|---|---|---|
| **1 leg** | 3 actuators, cannot tip | 3 actuators, must balance |
| **2 legs** | 6 actuators, cannot tip | 6 actuators, must balance |

Everything else is byte-identical — link lengths, gear ratios, friction,
reward, episode length, and the staircase itself. A test asserts that,
because a 2×2 whose cells differ in some uncontrolled way is not a 2×2.

## The staircase

![The staircase at three steepnesses](/img/physical/stairs-world.png)

Three treads instead of lab 1's single box, with the rise redrawn every
episode from 0.04 m to 0.10 m. One obstacle can be cleared by a single
lucky lunge; three cannot. The robot has to do the same thing from three
different heights, which is a much poorer fit for a memorised trajectory.

## The falsifier

> **If the leg count changes performance more than the torso constraint
> does, lab 1's explanation was wrong and joint count is the real
> difficulty axis.**

It lost — and then the result got more interesting than either hypothesis.

## The result

**Baseline:** a random policy, measured per cell before every run. No cell
ever reaches the top. **Budget:** 800k steps, three seeds, four cells.

| cell | return (3 seeds) | treads climbed /3 | summit |
|---|---|---|---|
| 1 leg, locked | 788 · 875 · 771 | 0.0 · 1.9 · 0.0 | 0% · 10% · 0% |
| 1 leg, free | 377 · 361 · 224 | 2.0 · 2.1 · 0.0 | 20% · 30% · 0% |
| **2 legs, locked** | **1884 · 2277 · 2449** | **3.0 · 3.0 · 3.0** | **100%** |
| 2 legs, free | 645 · 1559 · 750 | 3.0 · 3.0 · 2.8 | 100% · 100% · 80% |

![The 2x2 on both metrics](/img/physical/stairs-2x2.png)

**The limb dominates.** Mean treads climbed goes 0.63 → 3.00 with a locked
torso and 1.37 → 2.93 with a free one. Every two-legged seed summits; no
one-legged seed ever does.

## But the switches interact

![Interaction plot](/img/physical/stairs-interaction.png)

With **one** leg, freeing the torso **helps**: 0.63 → 1.37 treads. With
**two**, it costs a lot of return and **nothing measurable in climbing**:
2203 → 985 return (p = 0.022), against 3.00 → 2.93 treads (p = 0.37, i.e.
one seed landing at 2.8 — noise).

The interaction is carried by the return metric and by the one-legged
side. An earlier version of this paragraph also claimed the torso cost
"a little climbing"; three seeds do not support that, and the claim has
been withdrawn.

The lines are not parallel, so no main-effects summary of this experiment
is honest. You cannot say "freeing the torso costs X" — it depends on what
else the robot has.

The mechanism is visible in the animations below. With a single leg the
only way onto a step is to **lean** into it, and leaning requires a torso
that rotates — so the constraint lab 1 imposed for stability was also
quietly removing the one strategy available for climbing. With two legs
you can just put a foot up, the lean stops being necessary, and torso
freedom becomes pure instability.

:::tip What transfers
An ablation that reverses sign across another variable is common and
under-reported, because the usual way to run one is to fix everything else
at a default. Had this lab tested the torso switch only on the two-legged
robot — the obvious choice, since it is the better robot — it would have
concluded that freeing the torso is simply bad, and missed that it is the
*only thing* that makes a one-legged robot climb at all.
:::

## The whole job, end to end

![The trained two-legged robot walking, climbing, and carrying on](/img/physical/stairs-showcase.gif)

One uninterrupted 700-step episode of the strongest cell. It walks on the
flat, reaches the staircase, takes all three treads, and keeps going —
**11.3 m in total, still upright when the episode times out.**

The panel is read from the live simulation every frame. Watch `treads
climbed` reach 3/3 and the status chip turn **AT THE TOP** around step
150, then the distance keep climbing long after.

## Watch the difference

![Torso locked](/img/physical/stairs-2leg_locked.gif)

![Torso free](/img/physical/stairs-2leg_free.gif)

Both are two-legged, same staircase, same seed. The panel is read from the
live simulation every frame — watch `treads climbed` and the status chip.

![Learning curves, every seed](/img/physical/stairs-curves.png)

## Three wrong answers before the right one

The metric this entire lab rests on — *how many treads is the robot
standing on* — took three attempts. Every wrong version was quietly
plausible, and each produced a confident, publishable-looking table.

1. **Torso height against the absolute tread height.** The robot rests at
   `qpos[1] = -0.196` on flat ground, so standing on the 0.24 m top tread
   reads `+0.045`, not `+0.24`. The threshold was `+0.124`. A policy that
   walked **14.5 m past the entire staircase** reported `climbed 0.00`.
2. **Torso lift above its own resting height.** Better — but a robot
   standing on a tread with bent legs sits *lower* than one standing
   straight on the floor. Measured crouched on a 0.07 m tread: `+0.032`
   against a `0.042` threshold. Missed.
3. **Raw foot height.** The foot capsule has a 0.045 m radius, so a foot
   flat on the ground already clears tread one. A robot that never left
   the floor read `1/3`.

The working version measures the **foot's lift above its own flat-ground
height**, self-calibrated at construction so it cannot rot if a link
length changes.

:::warning This cost two entire sweeps
The same expression gated the `+3.0 per tread` reward, so the first two
sweeps trained robots that were never paid for climbing. Both produced
clean, consistent, completely wrong tables — the second one even had a
tidy story attached about locked torsos being unable to lean.

**Every number on this page is from the third sweep.** The first two are
not reported because they measured the detector, not the robot.

The same bug existed in lab 1's `on_box`. There it was invisible: that
lab's reward and headline metric both use x-position, so its published
findings stand and only a reported statistic was wrong. Fixed in both.
:::

## Where DeepSpeed is, and is not

The largest policy here is **11,597 parameters**. Nothing for ZeRO to
shard, so this is the eighth declared `launcher="python"` exception, and a
test fails if any cell exceeds 1M parameters.

The useful parallelism is `--jobs`: a sweep is twelve independent
processes, and twelve cores beat any GPU for this shape of work. That
remains true until the policy itself is large, which is where
[lab 1's Next Step section](./obstacle-hopper#next-step-how-this-scales-to-real-humanoid-work)
picks up. [Lab 3](./terrain-vision) is where a GPU first wins in this
category — a 193k-parameter vision student at 13.6× — and even that is
not a sharding problem.

## Run it

```bash
git clone https://github.com/yiqiao-yin/deepspeed-course.git
cd deepspeed-course/07_physical_ai/02_biped_stairs
uv sync

uv run morphology.py                            # the four robots
uv run stairs_env.py                            # random baselines
uv run ../../tests/test_biped_stairs.py         # 42 property checks

uv run train_ppo.py --dry-run                   # 30 s
uv run train_ppo.py --sweep --seeds 3 --jobs 6  # the whole 2x2, ~10 min
uv run make_figures.py
uv run render.py
```

## References

- Schulman et al., [PPO](https://arxiv.org/abs/1707.06347) and
  [GAE](https://arxiv.org/abs/1506.02438)
- [MuJoCo](https://mujoco.readthedocs.io/)
- Gymnasium's `Walker2d` is the nearest standard benchmark to the
  `2leg_free` cell here

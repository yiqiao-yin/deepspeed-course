---
sidebar_position: 10
---

# Does Perception Pay?

<!--chain-->
:::info **Lab 10 of 10** in [Physical AI](./overview) — ten labs that build one argument.
← previous: [Body Search](./body-search)
:::

Lab 7 built a model that reads terrain and could not say whether it
was useful. Lab 8 tried to find out and could not run the experiment.
This lab finally runs it.

The answer is **no** — and the realistic version is **worse than
nothing**.

**Baseline:** the `blind` arm — no perception, every landform
believed crossable, which is exactly what lab 5's robot did.
**Budget:** lab 9's best body (`bs_m2`), 16 belief-sensitive seeds,
routes 10–16 m, waypoints every 2.5 m, `implicitfast`. Waypoint rates
carry Wilson intervals.
**Falsifier:** if perception pays, the oracle beats blind with
separated intervals. It does not.

---

## The result

| arm | waypoints | arrived | detour |
|---|---|---|---|
| oracle (perfect) | 33% [23, 45] | **25%** | 1.09× |
| p90 | 28% [18, 41] | 19% | 1.06× |
| p80 | 30% [19, 44] | 19% | 1.05× |
| p70 | 28% [17, 43] | 19% | 1.05× |
| p60 | 28% [17, 42] | 19% | 1.07× |
| **biased** (lab 7's real shape) | 28% [16, 44] | **12%** | **1.27×** |
| **blind** (baseline) | **36% [25, 48]** | **25%** | 1.00× |

**Perfect perception ties no perception.** Both reach 25% arrival;
every interval overlaps. `perception_sweep.py` prints
`INTERVALS OVERLAP: perception does not measurably pay at this
budget` on its own — a branch that exists so the script can refuse
the result its author wanted.

**Realistic perception is worse than none.** The `biased` arm models
lab 7's measured failure — 72% correct on blockers, 40% on open
ground — and halves arrival to 12% while inflating routes 27%. A
model that cries wolf sends the rover on detours costing more than
the obstacles would have.

![three beliefs, one map](/img/physical/perception-beliefs.gif)

Same terrain, same start and goal, same body; only the belief
differs. The oracle and the blind router **plan the identical 11.7 m
route**, because on this task the truth and "nothing blocks me"
agree. The biased router's seven false AROUNDs inflate the plan to
**26.4 m — 2.26×** — and it reaches 3 of 7 waypoints instead of 3 of 3.

## Why it does not pay

The detour column is the mechanism. The oracle's careful 1.09×
avoidance buys nothing because **75% of runs stall regardless of
arm**. Route quality cannot matter to a robot that cannot execute a
route. Locomotion is still the binding constraint, four labs after
lab 8 first said so.

**Scope, and it matters:** with 75% stalling this test may simply
lack the power to detect a perception effect. The finding is
*perception does not pay at this robot's capability* — not
*perception never pays*.

## Three blockers had to clear first

The experiment was impossible in lab 8 and each obstacle was caught
by a check rather than discovered afterwards:

| blocker | fix |
|---|---|
| robot stalls before reaching decisions | lab 9's body plus the faster integrator |
| task demands 10× the robot's range | size the route band to measured capability |
| **chosen A–B pairs contained no decision** | select for belief-sensitivity |

That third one nearly produced a beautiful, meaningless curve.
Randomly chosen endpoints gave oracle, blind and paranoid beliefs the
**identical 10.5 m route** — the pair never passed a blocker, so
belief could not matter and all seven arms would have scored the same
by construction. It is lab 4 and lab 5's route-ratio lesson one level
up: a task must be *selected* for containing a decision, not assumed
to contain one.

Tasks are now kept only when believing everything impassable costs at
least 1.5 m against believing nothing does. Measured over 20 seeds:
mean gap **10.8 m**, and on **12 of 16** a fully paranoid belief walls
the robot in entirely.

## Training on the test terrain made it worse

The obvious fix for a domain mismatch is to train where you test. It
backfired:

| body | trained on | falls | arrived |
|---|---|---|---|
| bs_m2 | course | 25.0% | **16.7%** |
| tv_m2 | **traverse** | 41.7% | 0.0% |
| bs_m4 | course | 0.0% | 8.3% |
| tv_m4 | **traverse** | 50.0% | 0.0% |

The `traverse` preset is harder to learn on — bigger arena, more
blockers, longer routes — and a policy trained on the easier `course`
terrain **transfers** better than one trained directly on the hard
one. A curriculum effect, and the reason the sweep uses
course-trained bodies.

## A trap worth knowing

The task pool was a single `task_pool.pkl` regardless of
`MARS_POOL_SCALE`, so building a pool for this lab's preset would
have **silently overwritten the pool labs 5–9 train on**. No error;
every later run would quietly use different terrain while reporting
the same lab. Pools are now one file per scale, with `course` keeping
its original path so nothing needs rebuilding.

## What transfers

| | |
|---|---|
| **A task must be selected for containing a decision** | three beliefs planned the same route until it was |
| **Bad information can be worse than none** | biased perception halved arrival against blind |
| **A null result needs a working control** | blind ties oracle, so the ceiling and the floor are the same number |
| **Report power, not just significance** | 75% stalling may mask any effect, and the page says so |
| **Train where it is easier, test where it is hard** | course-trained transferred; traverse-trained collapsed |

Ten labs in, the robot crosses one run in four with a perfect oracle
and exactly as often with no perception at all. The constraint has
never moved: it is the machine, not what the machine knows.

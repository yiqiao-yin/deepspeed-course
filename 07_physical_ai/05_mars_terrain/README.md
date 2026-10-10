# Mars Terrain

## Abstract

Two halves. **Part 1** builds a procedurally generated Martian
surface, to be *measured* rather than admired. **Part 2** trains a PPO
navigator on it, and does not yet succeed — the measured reason is the
useful part. Lab 4 shipped a published result that was wrong
because of bugs in its world, its renderer and its metric — not in any
policy — so Lab 5 separates the two problems. **Part 1 contains no
robot:** no PPO, no reward, no arrival rate. It generates terrain at
two scales, asserts seven properties that a picture cannot show, and
produces six animations. Part 2 will put a robot on it. The surface
carries two deliberate affordance classes — smooth landforms a legged
robot should climb, and cliff-edged ones it must walk around — because
a terrain without a decision in it is scenery, not a task.

![The 60 m rover patch](../../docusaurus-docs/static/img/physical/mars-rover-ground.gif)

## Problem Statement

Terrain is the easiest thing in this repository to get wrong without
noticing. The usual check is "does it look right", and a height field
almost always looks right. Lab 4 proved the cost of that: a channel
carver, a coordinate frame, an efficiency denominator and a renderer
all produced plausible output while being wrong, and three of them
reached a published page.

So this lab inverts the usual order. The question is not *can we make
something that looks like Mars* — procedural noise does that in twenty
lines. It is **what has to be true of a generated planet before you
are allowed to believe it**, and can each of those things fail?

### Two scales, because the famous landforms do not fit

Olympus Mons is ~600 km across and ~22 km tall. Valles Marineris runs
~4,000 km. A robot arena is tens of metres. You cannot put either in
one, and a 2 m bump labelled "Olympus Mons" is a lie told with
geometry.

| preset | extent | what it is for |
|---|---|---|
| `regional` | 20 km | the shield volcano, the rift canyon, fossae swarms, cratered highlands, the crustal dichotomy. **For looking at.** |
| `rover` | 60 m | boulder fields, dune ridges, scarps, small craters, fractured bedrock. **Part 2 walks here.** |

The rover map is a *local sample of a Martian-looking surface*, not a
scale model of Mars. Both come from the same code with different
constants, and the properties are checked at both so one cannot stop
being Mars-like while the other still is.

## Solution

### What the generator models, and what it does not

Modelled, because it changes the shape of the ground:

- the **crustal dichotomy** — smooth low northern plains against high,
  rough, heavily cratered southern highlands (regional only; a 60 m
  patch does not straddle a hemispheric boundary)
- **impact craters** with raised rims, bowls and ejecta, drawn from a
  power-law size distribution and carrying an **age**: sharp young
  bowls superimposed on soft, half-filled old ones
- a **shield volcano** — broad, with a summit caldera, and far flatter
  than anyone draws it
- a **rift canyon** with terraced walls, and **swarms of parallel
  graben** sharing one orientation, as at Sirenum Fossae
- **dendritic dry channels** that run downhill and join
- **aeolian ripples** transverse to one prevailing wind
- scattered rock as angular **platy slabs**, not boulders
- patches of exposed, **fractured bedrock pavement**

Not modelled, stated so nobody assumes otherwise: polar ice caps (a
square tile has no pole in it), atmospheric scattering, real MOLA
elevation data, and regolith mechanics — the ground is rigid.

### The two affordance classes

This is what makes the terrain a task rather than scenery, and it is
lab 4's lesson restated in Martian landforms:

| class | landform | rise at the edge | the right response |
|---|---|---|---|
| **climbable** | dune ridges, rounded hills, degraded crater rims | under 0.12 m per step | go over; detouring costs distance |
| **blocked** | mesas, buttes, scarps | far over | walk to the end; the top is walkable, the **edge** is a wall |

The mesa is the interesting one. Its top is as traversable as the
plain — only the shoulder stops you — so **height alone does not
predict whether a robot can be there.** It has to read the edge.

The 0.12 m threshold is inherited from [lab 2](../02_biped_stairs),
which measured a two-legged robot climbing 0.04–0.10 m treads reliably
and failing above that.

## Experiments and Results

`uv run mars.py --check 20`, twenty seeds at both scales:

| | rover (60 m) | regional (20 km) |
|---|---|---|
| relief | 2.7–4.0 m | ~4.2 km |
| slope, mean | 6–10° | 22° |
| walkable (rise ≤ 0.12 m) | **96%** | — |
| crater exponent *b* | ~1.9 | **2.02** |
| edge blocked, climbable class | **4–6%** | — |
| edge blocked, cliff class | **17–24%** | — |

The classes separate by about **3×** at the edge, which is the
property the lab exists to produce.

### What does not work yet, stated plainly

**The routing decision is weak.** Refusing to climb saves a median of
**+1.24 m**, and only **5 of 10 seeds** save more than a metre. For
comparison, lab 4's arena — a quarter the area — reached +3.40 m.
Obstacles here are sparse enough (96% of cells walkable) that a route
can often avoid everything for free.

That is a real limitation and Part 2 inherits it. The fix is probably
not more obstacles but **goal placement**: lab 4 found that where you
put A and B decides whether the terrain's decision actually binds, and
that is a Part 2 question.

## What Went Wrong, and What It Taught Us

Three failures while building this, none of which crashed.

**A carve that accumulated.** The channel walker subtracted its depth
at each of 150 overlapping steps, so a channel asked for 0.15 m was
cut tens of metres deep. The rover map came back with **46 m of relief
inside a 60 m box** and the ASCII view still looked like terrain. Now
the deepest cut per cell is accumulated and applied once — and relief
is asserted against the preset, because "looks like terrain" does not
distinguish 3 m from 46 m.

**A measurement that answered the wrong question.** Traversability was
averaged over each landform's bounding box, reporting mesas as 86%
walkable. True, and meaningless: the box is mostly the flat top and
the surrounding plain. The obstacle is the shoulder, so the
measurement now samples a **ring** at the landform's own radius.

**A silent zero.** `climbing_pays` blocked every cell above the median
elevation — 38% of the arena — which walled the map off entirely. Every
route came back infinite, every sample was dropped, and the function
returned `0.0`. Eight seeds in a row read `+0.00 m`, which looks
exactly like *this terrain contains no decision* and actually meant
*nothing was measured*. **A plausible number is the worst possible
failure value**: it is in range, and it points at the wrong culprit. It
now returns `NaN` and reports its usable-sample count.

## Part 2 — a PPO navigator, and the wall it hits

Same experiment as [lab 4](../04_terrain_navigation): random A to
random B, three arms differing only in what the robot is told about
the terrain ahead (blind 25 dims / privileged 31 / padded 31 with six
constant zeros as the width control).

**Reward**

```
r = 10*(d_prev - d)        progress toward B, not forward velocity
  + 1.5                    alive  (0.5 teaches it to fall over at once)
  - 1e-3*||a||^2           control
  + 100 on FIRST arrival   paid once; arriving does NOT end the episode
```

**The task** draws from 120 pre-solved maps whose route is 16-30 m and
at least 15% longer than the straight line. Lab 4's method of capping
goal distance -- walking B back along the route -- was tried and
measured at a **1.00-1.07x** detour: the right length, no decision in
it. Selecting naturally-bent pairs gives **1.58x**.

### The robot needed two more joints, and it was not enough

Labs 1-4 lock torso pitch and roll. That is correct for flat ground
with discrete obstacles on it, and fatal on ground uneven everywhere.
Measured, same gait, only the relief scaled:

| mean slope | 0.0° | 0.7° | 1.3° | 2.6° | 4.3° | 10.4° |
|---|---|---|---|---|---|---|
| walked | 9.8 m | 11.6 m | 5.3 m | **0.64 m** | 0.43 m | 0.31 m |

Removing all 140 boulders changes nothing; flattening the height field
restores it entirely. The obstacle is the continuous unevenness, not
the obstacles.

Freeing pitch and roll: **100% arrival on flat**. On Mars, three
strategies, all stuck at **0%**:

| strategy | steps | arrival | note |
|---|---|---|---|
| cold start on Mars | 900k | 0% | fall rate ROSE 75% → 83% |
| warm start flat → Mars | 246k | 0% | 83% falling |
| ladder rung ×0.15 (1.9°) | 450k | 0% | 15.3 m left |
| ladder rung ×0.30 (3.8°) | 450k | 0% | 12.8 m left |
| ladder rung ×0.50 (6.3°) | 450k | 0% | 12.9 m left |
| ladder rung ×0.75 (8.0°) | 450k | 0% | 12.8 m left |
| ladder rung ×1.00 (10.4°, full Mars) | 450k | 0% | 13.5 m left |

**A policy that arrives 100% of the time on a plane, warm-started onto
1.9 degrees of mean slope, still reaches 0%.**

### What Part 2 does NOT contain

No working Mars navigator, and no blind-vs-privileged comparison,
because there is no policy good enough to compare. Running the arms
anyway would have produced a table of zeros and a p-value measuring
our locomotion failure while appearing to measure terrain information.

The boundary is: **0° works, 1.9° does not**, freed torso and
curriculum included, at the budgets tried.

## What This Lab Does Not Claim

- **This is not Mars.** It is procedural terrain matched to published
  descriptions and imagery. MOLA elevation data exists; this is not it.
- **No robot has walked here.** Traversability is computed from a gait
  rule, not demonstrated. Part 2 is where that claim gets earned.
- **The routing decision is weak** — +1.24 m median, 5 of 10 seeds.
- **The palette is a presentation choice**, not a result. Nothing here
  simulates atmospheric scattering; the sky is butterscotch because
  surface imagery shows it that way.

## Running It

```bash
cd 07_physical_ai/05_mars_terrain
uv sync

uv run mars.py --check 20            # the measurements
uv run mars.py --scale rover --ascii # one map, in text
uv run mars.py --scale regional      # the planetary view
uv run render_mars.py --all          # six terrain animations

# Part 2
uv run mars_env.py --build-pool      # 120 tasks, once (~10 min)
uv run train_mars.py --flat --name curr_flat          # the gate: 100%
uv run train_mars.py --init-from curr_flat --relief-scale 0.15 \
    --name rung_0.15                                   # the ladder
uv run render_run.py --run curr_flat --out walk --view chase
```

```bash
uv run ../../tests/test_mars_terrain.py   # 17 checks, no GPU, no OpenGL
```

### CoreWeave (SLURM)

```bash
sbatch run_deepspeed.sh
squeue -u $USER
tail -f logs/mars_terrain_<jobid>.out
```

**There is no `deepspeed` call in that script and no GPU requested.**
This lab trains nothing — no parameters, no gradients, nothing for
ZeRO to shard — so a distributed launcher would be cargo cult. The
file keeps its name because every example in this course has one and a
reader should find it where they expect.

### RunPod

```bash
uv run runpod/runpod_ctl.py run 07_physical_ai/05_mars_terrain \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods
```

`--terminate` is not optional hygiene: an abandoned pod bills until it
is destroyed.

### Hardware

| | |
|---|---|
| GPU | none for the generator or the checks; the **renderer** needs an OpenGL context |
| Disk | ~3 GB (numpy + MuJoCo + pillow). No downloads — the planet is generated |
| Wall clock | checks ~40 s; all six animations ~10 min |

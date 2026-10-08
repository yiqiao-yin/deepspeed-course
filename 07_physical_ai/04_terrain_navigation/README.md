# Terrain Navigation: climb it or walk around it

Labs 1–3 were **planar**: two base joints, terrain as a profile along a
line, every obstacle head-on, no choice to make. This lab adds the
missing dimension — a 20 × 20 m arena, a random start **A** and goal
**B**, and ridges of two kinds that demand opposite responses.

| | rise | the right response |
|---|---|---|
| **climbable** | ≤ 0.12 m | go straight over; detouring costs distance |
| **impassable** | ≥ 0.45 m | walk to its end; climbing wastes the episode |

The threshold comes from [lab 2](../02_biped_stairs), which measured a
two-legged robot climbing 0.04–0.10 m treads reliably and failing above
that.

**Baseline:** the blind policy — proprioception and the goal bearing,
same algorithm, same 1.5M steps, nine seeds across three sweeps:
**8.1%** arrival (88.4% path efficiency, which privileged does not beat).
**Budget:** 1.5M environment steps per run × 3 seeds per arm × 3
independent sweeps, goal range 7 m; scored on **120 evaluation episodes
per checkpoint** on identical maps.
**Falsifier:** if a policy handed the ground profile ahead does not beat
the blind one on arrival, terrain information is not what this task
needs. It does, on 7 of 9 seeds (Wilcoxon p = 0.022). A second claim about
path efficiency did NOT survive measurement and is withdrawn below.

![One arena](../../docusaurus-docs/static/img/physical/nav-world.png)

---

## The result

Two independent sweeps, three seeds each, 120 episodes per checkpoint
on identical maps:

![Arrival per seed](../../docusaurus-docs/static/img/physical/nav-arrival.png)

| | arrival | per seed |
|---|---|---|
| blind | 8.1% | 8, 15, 0, 4, 19, 1, 1, 12, 14 % |
| **privileged** | **12.1%** | 16, 16, 4, 6, 18, 11, 12, 14, 13 % |

Ahead on **7 of 9 seeds**, mean **+4.0 points**. Wilcoxon **p = 0.022**,
paired t **0.017** — both strengthened from the six-seed version. The
sign test went the other way (0.016 → 0.090) because it counts only
wins and the two new losses were tiny while the wins were larger;
Wilcoxon uses the magnitudes and is the headline. Pooled Fisher is an
**upper bound** — episodes within a seed share a policy.

The effect size barely moved: **+3.9 at six seeds, +4.0 at nine**. And
across nine seeds the blind arm produced 0%, 1% and 1%, while the
privileged arm's worst is 4% — it appears to remove the floor, not
only raise the mean.

### The route efficiency claim, withdrawn

This README previously reported privileged policies walking much
shorter paths — 54.8% vs 43.8% (3/3 pairs) from two sweeps, then 51.2%
vs 44.6% (4 of 5) after a third. **Both were a measurement bug.** The episode continues after reaching B, so the odometer counted
the robot milling around the goal for ~1300 further steps: one episode
walked 7.4 m to B against a 7.2 m route — 97% — then another 7.2 m,
and was published as 49%.

Frozen at arrival:

| seed pair | maps | blind | privileged |
|---|---|---|---|
| sweep 2, s0 | 2 | 96.4% | 81.4% |
| sweep 2, s1 | 6 | 83.5% | 79.9% |
| sweep 3, s1 | 9 | 88.1% | **89.9%** |
| sweep 4, s4 | 4 | 91.2% | 84.0% |
| sweep 4, s5 | 5 | 82.7% | **84.1%** |
| **mean** | | **88.4%** | **83.8%** |

Both arms are far better than reported — ~85% of optimal, not ~48% —
and the direction reverses. **No route-efficiency advantage; the claim
is withdrawn.** The arrival result is binary and unaffected.

---

## Four bugs that would each have produced a wrong conclusion

**1. The privileged arm was broken by its own observation.** An 11 × 11
raw height patch meant a 142-dimensional input to a 2 × 64 MLP, and all
three seeds sat at *exactly* 0% arrival for the whole run. Three
identical zeros are impossible by chance; that is the only reason it was
caught. Six summary features train fine.

**2. Arriving ENDED the episode, making success the worst outcome.**
Standing still for 2000 steps earned 3000; arriving at step ~700 earned
1170. Success forfeited ~1300 steps of alive bonus, so loitering was
worth 2.5× more — and the policy correctly learned not to arrive.
[Lab 1](../01_obstacle_hopper) documents the mirror of this.

**3. Torque control meant the body could not stand.** It needs a
specific sustained pattern (hip 0.6 / knee 1.0) that exploration almost
never finds; a 1M-step run on flat ground reached 0%. Position
actuators make a zero action hold the standing pose.

**4. The fall detector sat inside the healthy oscillation band.** A
standing robot troughs at 0.49 and settles at 0.77; the threshold was
0.55, so episodes ended on a *healthy* robot's startup wobble. Now 0.40
— below the wobble, far above a real collapse at 0.19.

### And the measurement itself was too coarse

The in-training evaluation uses 12 episodes, which quantises to 8.3%.
It reported these two arms as **exactly 8.3% vs 8.3%**. Re-scoring the
same checkpoints on 120 episodes gave **7.8% vs 11.7%** — the effect was
there all along. Everything published comes from `evaluate.py`.

The *peak* of the training curve is equally unusable: it is the maximum
of ~60 noisy 12-episode evaluations, and every blind run peaks at
exactly 25% (3/12). Selecting on it would manufacture an advantage.

---

## The design, and what was swept to get it

Each constant was chosen by sweeping it against the property the lab
needs, because the first two attempts produced a task with no decision
in it:

| obstacle shape | cost of refusing to climb |
|---|---|
| round mounds, r = 1.2–2.2 m | median **0.00 m** over 109 maps |
| short ridges, 5–8 m | median **0.00 m** |
| **long ridges, 12–17 m** | median **+3.40 m** on a ~15 m route |

Length is the lever: a ridge that nearly spans the arena cannot be
skirted cheaply. Ends stay open so 94% of maps keep *both* routes
available — a choice, not a forced climb.

`tests/test_terrain_navigation.py` pins the **property** rather than the
parameters, so an edit to the lengths is caught by its consequence.

## The robot

`rootx`, `rooty`, **`rootyaw`**, `rootz` plus six leg joints; position
actuators centred on a measured standing pose.

**Torso pitch and roll stay locked** — inherited from lab 1's
measurement (freeing it gives the bipedal-balance problem) and lab 2's
2×2 (with two legs it costs return and buys no climbing). What it buys
is that the hard part here is navigation. What it costs: nothing here
measures recovery, and yaw is actuated directly because a planar biped
with locked roll cannot generate yaw torque through contact.

## Running it

```bash
cd 07_physical_ai/04_terrain_navigation
uv sync

uv run world.py --check 200     # solvable? do they detour?
uv run robot.py                 # the body on a map
uv run nav_env.py               # random baseline

uv run train_nav.py --flat --name gate_flat     # LOCOMOTION GATE, first
for s in 0 1 2; do
  uv run train_nav.py --mode blind      --seed $s --goal-range 7 --name nav_blind_s$s
  uv run train_nav.py --mode privileged --seed $s --goal-range 7 --name nav_priv_s$s
done

uv run evaluate.py --episodes 120               # the published numbers
uv run make_figures.py && uv run render.py --all
```

`--flat` is not part of the experiment. It asks only whether the body
can walk to a point and turn to face it, and it **failed twice** before
the position-actuator and start-pose fixes — which is the whole reason
it runs first.

```bash
uv run ../../tests/test_terrain_navigation.py   # 18 checks, no GPU, no OpenGL
```

### CoreWeave (SLURM)

```bash
sbatch run_deepspeed.sh
squeue -u $USER
tail -f logs/terrain_navigation_<jobid>.out
```

### RunPod

```bash
uv run runpod/runpod_ctl.py run 07_physical_ai/04_terrain_navigation \
    --dry-run --collect --wait --terminate --yes
uv run runpod/runpod_ctl.py pods      # confirm nothing is left running
```

`--terminate` is not optional hygiene: an abandoned pod bills until it
is destroyed. `--dry-run` proves the pipeline assembles before anything
long starts, and `pods` is how you check the box actually went away.

Note `--wait-seconds` defaults to 1800, which is shorter than a full
1.5M-step run — raise it, or the pod is destroyed mid-training.

### Hardware

| | |
|---|---|
| GPU | 1 × 16 GB, optional — the policy is 12k parameters and CPU is faster |
| Disk | ~20 GB (torch + MuJoCo); no downloads, the world is generated |
| Wall clock | ~20 min a run; ~2 h for a full six-run sweep |

## What this lab does not claim

- **The task is not solved.** ~12% arrival, ~55% efficiency when it
  arrives. The claim is that information *helps*, on a task that stays
  hard.
- **Every run peaks mid-training and sags.** 1.5M steps is short here.
- **This is a privileged channel, not a camera.** A depth arm would need
  distillation, as in [lab 3](../03_terrain_vision).
- **One seed of six goes the other way**, and it stays in the table.

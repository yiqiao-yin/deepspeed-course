#!/usr/bin/env python3
"""
PPO, written out rather than imported, because the parts are the lesson.

COPIED VERBATIM FROM `07_physical_ai/01_obstacle_hopper/ppo.py`, apart from
this note and the device comment below. That duplication is the
repository's rule, not an oversight: a reader must be able to open one
folder and run it without the other twenty-three existing. The GAE limits
and the clipped objective are identical, and `tests/test_biped_stairs.py`
re-asserts them here rather than trusting that the sibling lab still
holds.

Stable-Baselines3 would train this task in fifteen lines. It would also make
the two pieces that actually matter here invisible: how an advantage is
estimated, and what the clipped objective does to a policy update. Both are
small, both are pure tensor arithmetic, and both are exactly testable on a
CPU with no simulator attached -- which is why they live in this module and
not inside the training loop.

WHAT IS TESTABLE HERE, AND WHY IT IS WORTH TESTING
--------------------------------------------------
`compute_gae` has two exact limits, and they are the best kind of property:
derived, not approximate, and wrong in a very quiet way if the
implementation drifts.

    lambda = 1  ->  the discounted Monte-Carlo return minus the baseline.
                    No bootstrapping; unbiased, high variance.
    lambda = 0  ->  the one-step TD residual, r + gamma*V(s') - V(s).
                    Fully bootstrapped; biased, low variance.

Everything in between interpolates. An off-by-one in the backward recursion,
a dropped `(1 - done)` mask, or a reversed sign all still produce
plausible-looking advantages and a run that trains to *something*. The
limits do not survive any of those, so `tests/test_obstacle_hopper.py`
asserts them to 1e-6 against independently computed references.

`clipped_policy_loss` has a property worth pinning too: once a ratio has
moved past the clip boundary in the direction that would make the objective
better, its gradient is zero. That is the entire mechanism preventing a
destructive update, and a sign error leaves a loss that still decreases.

The relation to `03_llms/06_grpo` is worth noticing. GRPO is this algorithm
with the critic deleted, because in that setting rewards are sparse,
verifiable, and comparable within a group of samples for one prompt. Here
rewards are dense and shaped -- forward velocity every step -- so a learned
value function has real work to do and the critic earns its place. Same
family, opposite call, for a stated reason.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# The two pieces worth testing
# ---------------------------------------------------------------------------

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation. Returns (advantages, value targets).

    Shapes are [T, N] for T timesteps across N parallel environments, with
    `last_value` [N] bootstrapping past the end of the segment.

    `dones` marks terminal transitions, and the `(1 - done)` factor appears
    TWICE on purpose: once to stop the value of the next state leaking
    across an episode boundary, and once to stop the advantage recursion
    doing the same. Dropping either produces advantages that quietly blend
    two unrelated episodes, which trains to something mediocre rather than
    failing.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        adv[t] = running

    return adv, adv + values


def clipped_policy_loss(logp: torch.Tensor, logp_old: torch.Tensor,
                        advantages: torch.Tensor, clip: float = 0.2
                        ) -> tuple[torch.Tensor, dict]:
    """
    PPO's clipped surrogate, plus the diagnostics worth logging.

    The minimum of the clipped and unclipped terms is what makes the
    objective pessimistic: the update may always be made worse by the clip,
    never better. Taking `max` instead -- an easy slip -- yields a loss that
    still descends while removing the trust region entirely.
    """
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    loss = -torch.min(unclipped, clipped).mean()

    with torch.no_grad():
        stats = {
            "ratio_mean": ratio.mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > clip).float().mean().item(),
            "approx_kl": ((ratio - 1.0) - (logp - logp_old)).mean().item(),
        }
    return loss, stats


# ---------------------------------------------------------------------------
# Supporting machinery
# ---------------------------------------------------------------------------

class RunningNorm:
    """
    Welford running mean/variance for observation normalisation.

    Not optional. MuJoCo observations mix radians near zero with velocities
    in the tens, and an unnormalised PPO on this task does not learn at all
    -- it is the single most common reason a correct implementation appears
    broken. Welford rather than a naive sum of squares because the naive
    form loses precision over millions of samples.
    """

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_2d(x)
        batch_mean, batch_var, n = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        self.var = (m_a + m_b + delta ** 2 * self.count * n / total) / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ActorCritic(nn.Module):
    """
    Two small MLPs and a state-independent log-std.

    Copied verbatim from lab 3 per the repository's no-shared-module
    rule, and re-asserted independently by this lab's own test suite --
    a copy that is never checked on its own is a copy that can drift.

    Worth stating plainly, because it decides how this lab is launched:
    **there is nothing here for DeepSpeed to shard.** ZeRO partitions
    optimizer state, gradients and parameters, and all three are negligible
    at this size. The bottleneck is MuJoCo stepping on the CPU, so the
    teachers are registered with `launcher="python"` and left there --
    labs 1 and 2 both measured CPU as FASTER than a GPU for exactly this
    reason.

    Where the GPU does arrive in this lab is the vision STUDENT, not this
    network: a 193,222-parameter conv encoder trained supervised over a
    fixed dataset, measured at 13.6x +/- 0.9 on a GPU. That is still not a
    DeepSpeed case -- 193k parameters have nothing to shard either. "A GPU
    helps" and "DeepSpeed helps" are different claims, and this lab is
    where they come apart.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()

        def body() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )

        self.pi_body = body()
        self.v_body = body()
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Orthogonal init with a small final-layer gain: standard for PPO on
        # continuous control, and the difference between learning in 300k
        # steps and not learning at all.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.v_body(obs)).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mu(self.pi_body(obs)),
                                          self.log_std.exp())

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        return (dist.log_prob(action).sum(-1),
                dist.entropy().sum(-1),
                self.value(obs))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pick_device(requested: str = "auto") -> str:
    """
    Choose a device, and be honest that CPU is usually right here.

    Probes rather than trusts: `torch.cuda.is_available()` can be True on a
    box whose driver then fails on the first real allocation, which this
    repository has been bitten by before.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        # MEASURED on this lab, not assumed, and re-measured with repeats
        # after the first single-run figure was challenged:
        #
        #     4 envs   CPU 682 steps/s   CUDA 341   -> CPU 2.0x faster
        #    16 envs   CPU 2513          CUDA 1480  -> CPU 1.7x faster
        #
        # CPU won 5 of 5 paired runs with no overlap, so the DIRECTION is
        # solid; the RATIO is scoped to the environment count and narrows
        # as that rises. The policy is 10k parameters, so the forward pass
        # was never the bottleneck -- MuJoCo stepping is, on the CPU either
        # way, and the GPU only adds a host-device transfer per step.
        #
        # So `auto` means CPU here. Defaulting to the slower device because
        # a GPU happens to exist is the same reflex this course argues
        # against elsewhere -- `--device cuda` is supported and reported,
        # just not pretended to be an upgrade.
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is "
                "False. Run with --device cpu; this lab is CPU-first and "
                "loses nothing.")
        return "cpu"
    try:
        torch.zeros(8, device="cuda") @ torch.zeros(8, 8, device="cuda")
        torch.cuda.synchronize()
        return "cuda"
    except Exception as exc:                                   # noqa: BLE001
        if requested == "cuda":
            raise RuntimeError(f"CUDA is visible but unusable: {exc}") from exc
        print(f"[device] CUDA visible but unusable ({exc}); using CPU.")
        return "cpu"


def main() -> None:
    """Demonstrate the two limits of GAE without touching a simulator."""
    torch.manual_seed(0)
    T, N, gamma = 6, 2, 0.99
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    last = torch.randn(N)

    print("=" * 70)
    print("GAE, and the two limits that pin it")
    print("=" * 70)

    adv1, _ = compute_gae(rewards, values, dones, last, gamma, lam=1.0)
    mc = torch.zeros(T, N)
    run = last.clone()
    for t in reversed(range(T)):
        run = rewards[t] + gamma * run
        mc[t] = run
    print(f"lambda=1 matches Monte-Carlo minus baseline : "
          f"max err {(adv1 - (mc - values)).abs().max():.2e}")

    adv0, _ = compute_gae(rewards, values, dones, last, gamma, lam=0.0)
    td = torch.zeros(T, N)
    for t in range(T):
        nxt = last if t == T - 1 else values[t + 1]
        td[t] = rewards[t] + gamma * nxt - values[t]
    print(f"lambda=0 matches the one-step TD residual   : "
          f"max err {(adv0 - td).abs().max():.2e}")

    # Import the real dimensions rather than hardcoding them. This read
    # `ActorCritic(13, 3)` and kept printing 10,375 parameters after the
    # observation shrank to 11 -- so the script contradicted the book page
    # and the test suite, both of which say 10,119, on the second command
    # a reader runs.
    from morphology import act_dim, obs_dim
    net = ActorCritic(obs_dim(2, False), act_dim(2))
    print(f"\npolicy+value parameters (2 legs, free torso): "
          f"{net.n_params():,} — still nothing for ZeRO to shard, which "
          f"is why this lab also uses launcher=python")


if __name__ == "__main__":
    main()

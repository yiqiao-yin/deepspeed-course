#!/usr/bin/env python3
"""
PPO: GAE, the clipped surrogate, observation normalisation, the network.

    uv run ppo.py            # the properties, asserted — no environment

DUPLICATED ON PURPOSE
---------------------
This is topic 7's PPO with one change (see `init_bias` below). The
course duplicates rather than shares so a reader can open one folder
and run it, and the rule is load-bearing rather than stylistic. The
cost of a copy is that it can drift, so this file re-asserts its own
properties instead of trusting the original's test suite.

THE ONE CHANGE, AND WHY IT IS NOT COSMETIC
-------------------------------------------
A legged robot's actions are joint targets around zero, so a policy
initialised at zero starts in a neutral pose and explores outward. A
thruster's actions are valve openings in [0, 1], and zero means every
valve SHUT. An untrained policy therefore starts in free fall, and
every early episode ends the same way at the same time -- the reward
carries almost no gradient about what the valves do, because they
were never open.

So the action mean is initialised at the HOVER command: the opening
at which the lift nozzles exactly cancel weight. Exploration then
happens around flying rather than around falling. This is the
`init_bias` argument, and removing it is the difference between
learning in a million steps and not learning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, last_value: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation. Returns (advantages, value targets).

    Shapes are [T, N] for T timesteps across N parallel environments,
    with `last_value` [N] bootstrapping past the end of the segment.

    The `(1 - done)` factor appears TWICE on purpose: once to stop the
    next state's value leaking across an episode boundary, and once to
    stop the advantage recursion doing the same. Dropping either gives
    advantages that quietly blend two unrelated episodes, which trains
    to something mediocre rather than failing.
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

    The MINIMUM of the clipped and unclipped terms is what makes the
    objective pessimistic: the update may always be made worse by the
    clip, never better. Taking `max` -- an easy slip -- yields a loss
    that still descends while removing the trust region entirely.
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


class RunningNorm:
    """
    Welford running mean/variance for observation normalisation.

    Not optional. The observation here mixes rotation-matrix entries
    near 1 with rates in the tens and a gas fraction in [0, 1]; an
    unnormalised PPO on this does not learn at all. Welford rather
    than a naive sum of squares because the naive form loses precision
    over millions of samples.
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

    Worth stating plainly, because it decides how this lab is
    launched: **there is nothing here for DeepSpeed to shard.** ZeRO
    partitions optimizer state, gradients and parameters, and all
    three are negligible at ~15k parameters. The bottleneck is MuJoCo
    stepping on the CPU. Topic 7 measured this repeatedly -- a 12k
    policy ran 1.7x SLOWER on a GPU than on the CPU, because the
    rollout batch never fills a card -- and the honest answer was a
    worker pool, not a rental.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 init_bias: np.ndarray | None = None) -> None:
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
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

        # Orthogonal init with a small final-layer gain: standard for
        # PPO on continuous control, and the difference between
        # learning in 300k steps and not learning at all.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

        # Start at hover, not at valves-shut. See the module docstring.
        if init_bias is not None:
            with torch.no_grad():
                self.mu.bias.copy_(torch.as_tensor(init_bias,
                                                   dtype=torch.float32))

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


def hover_bias(rig, world=None) -> np.ndarray:
    """
    The valve opening at which the lift nozzles cancel weight.

    Derived from the rig rather than typed, so a rig with six lift
    nozzles gets the right number without anyone editing a constant.
    Non-lift nozzles start shut -- there is no "neutral" amount of
    reverse thrust.
    """
    import thrusters as TH

    world = world or TH.DESIGN_WORLD
    lift_z = sum(t.fmax * t.direction[2] for t in rig if t.role == "lift")
    if lift_z <= 1e-9:
        raise ValueError("rig has no upward thrust — it cannot hover, "
                         "and a hover bias is meaningless for it")
    u = TH.wet_mass() * world.g / lift_z
    return np.array([u if t.role == "lift" else 0.0 for t in rig])


def pick_device(requested: str = "auto") -> str:
    """
    Choose a device, and be honest that CPU is usually right here.

    Topic 7 measured a 12k-parameter policy on a GPU at 1.7x SLOWER
    than the same policy on the CPU: the rollout batch is tiny, so
    the transfer and launch overhead dominates and the card idles.
    Nothing about this lab changes that -- the network is ~15k
    parameters and the cost is MuJoCo.
    """
    if requested == "cuda":
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cpu"


def main() -> int:
    """The properties, asserted. No environment, no MuJoCo, seconds."""
    fails = []

    def ck(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("=" * 74)
    print("  PPO — the properties this lab's copy must still have")
    print("=" * 74)

    torch.manual_seed(0)
    T, N = 8, 4
    rew = torch.ones(T, N)
    val = torch.zeros(T, N)
    done = torch.zeros(T, N)
    last = torch.zeros(N)

    adv, _ret = compute_gae(rew, val, done, last, gamma=0.99, lam=1.0)
    # With lam=1 and zero values, the advantage is the discounted
    # return: sum of gamma^k over the remaining steps.
    want = sum(0.99 ** k for k in range(T))
    ck("GAE with lam=1 is the discounted return",
       abs(float(adv[0, 0]) - want) < 1e-5, f"{float(adv[0,0]):.4f} vs {want:.4f}")

    done2 = done.clone()
    done2[3] = 1.0
    adv2, _ = compute_gae(rew, val, done2, last, gamma=0.99, lam=1.0)
    want2 = sum(0.99 ** k for k in range(4))
    ck("a `done` truncates the recursion — no bleed across episodes",
       abs(float(adv2[0, 0]) - want2) < 1e-5,
       f"{float(adv2[0,0]):.4f} vs {want2:.4f}")
    ck("and the episodes after it are unaffected",
       abs(float(adv2[4, 0]) - sum(0.99 ** k for k in range(4))) < 1e-5)

    # The clip must be able to BITE, and must bite only one way.
    a = torch.ones(64)
    lp_old = torch.zeros(64)
    lp_big = torch.full((64,), 1.5)        # ratio 4.48, far above 1+clip
    loss_big, st = clipped_policy_loss(lp_big, lp_old, a, clip=0.2)
    ck("a huge positive ratio is clipped to 1+clip",
       abs(float(loss_big) + 1.2) < 1e-4, f"loss {float(loss_big):.4f}")
    ck("and the clip fraction reports it", st["clip_fraction"] == 1.0)
    loss_small, _ = clipped_policy_loss(torch.full((64,), -1.5), lp_old, a, 0.2)
    ck("a huge NEGATIVE ratio is NOT clipped for positive advantage "
       "(pessimism is one-sided)",
       float(loss_small) > -0.3, f"loss {float(loss_small):.4f}")

    # Normalisation: the thing that makes this task learnable at all.
    rn = RunningNorm(3)
    xs = np.random.default_rng(0).normal([0, 50, -3], [1, 20, 0.1], (4000, 3))
    for i in range(0, 4000, 200):
        rn.update(xs[i:i + 200])
    out = rn(xs)
    ck("running norm whitens wildly different scales",
       abs(out.mean()) < 0.05 and abs(out.std() - 1.0) < 0.1,
       f"mean {out.mean():.3f} std {out.std():.3f}")

    # The hover bias, which is this copy's only change.
    import thrusters as TH

    rig = TH.rig("reveal")
    b = hover_bias(rig)
    lift = sum(t.fmax * t.direction[2] * bi
               for t, bi in zip(rig, b) if t.role == "lift")
    w = TH.wet_mass() * TH.DESIGN_WORLD.g
    ck("the hover bias actually cancels weight",
       abs(lift - w) < 1.0, f"{lift:,.0f} N vs {w:,.0f} N")
    ck("and non-lift nozzles start SHUT",
       all(bi == 0.0 for t, bi in zip(rig, b) if t.role != "lift"))
    # A six-lift rig gets the SAME per-nozzle bias, and that is the
    # design rather than a coincidence: `make_rig` holds total
    # installed thrust constant, so six nozzles are each weaker than
    # four. Without that control a rig sweep would only be measuring
    # "more thrust is better", which needs no experiment. The first
    # version of this check asserted the bias would FALL and failed —
    # the test was the guess, not the code.
    b6 = hover_bias(TH.rig("sixlift"))
    ck("equal installed thrust ⇒ equal hover command across rigs",
       abs(b6[0] - b[0]) < 1e-9, f"{b6[0]:.4f} vs {b[0]:.4f}")
    b_strong = hover_bias(TH.make_rig(pairs=2, thrust_ratio=4.0))
    ck("and it IS derived — a stronger rig hovers at a lower opening",
       b_strong[0] < b[0] * 0.75, f"{b_strong[0]:.3f} vs {b[0]:.3f}")

    net = ActorCritic(30, len(rig), init_bias=b)
    with torch.no_grad():
        mu = net.mu(net.pi_body(torch.zeros(1, 30)))
    ck("a fresh network's mean action IS the hover command",
       float(torch.abs(mu[0] - torch.as_tensor(b, dtype=torch.float32)).max())
       < 0.05)
    print(f"\n  network: {net.n_params():,} parameters — "
          f"nothing for ZeRO to shard")
    print("=" * 74)
    print(f"  {'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

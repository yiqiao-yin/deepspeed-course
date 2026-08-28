# /// script
# requires-python = ">=3.9"
# dependencies = ["torch"]
# ///
"""
Regression test: STAR memory must be genuinely BOUNDED, and must actually
remember things.

Run:
    uv run tests/test_star_memory.py

Why this suite exists
---------------------
A streaming memory has exactly one hard requirement and one soft one, and
they fail in opposite directions:

  HARD — context size must not grow with stream length. A leak here is
  invisible in a 30-second demo and fatal in hour six. So we push 2,000
  frames through and assert the context is byte-for-byte the same size as at
  frame 200, and that no internal buffer exceeded its cap.

  SOFT — a buffer that satisfies the hard requirement by throwing everything
  away is trivially "bounded" and completely useless. So we also assert that
  memory RETAINS SIGNAL: a distinctive event written early must still be
  recoverable thousands of frames later, above the noise floor.

Passing one of these is easy. Passing both is the actual engineering.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "08_vtt" / "03_streaming_memory"))

from star_memory import StarConfig, StarMemory, weighted_kmeans  # noqa: E402

DIM = 64


def test_weighted_kmeans(r: Results) -> None:
    """Clustering must conserve weight and respect it."""
    torch.manual_seed(0)

    # Three tight, well-separated blobs. Any correct k-means finds them.
    blobs = torch.cat([
        torch.randn(20, DIM) * 0.05 + 10.0,
        torch.randn(20, DIM) * 0.05 - 10.0,
        torch.randn(20, DIM) * 0.05,
    ])
    weights = torch.ones(60)

    centroids, cw = weighted_kmeans(blobs, weights, k=3)

    r.check(centroids.shape == (3, DIM), "returns exactly k centroids",
            f"got {tuple(centroids.shape)}")
    r.check(abs(cw.sum().item() - 60.0) < 1e-3,
            "weight is conserved — no frame invented or lost",
            f"weights summed to {cw.sum().item()}, expected 60")
    r.check(bool((cw > 0).all()),
            "no empty clusters on well-separated data",
            f"cluster weights: {cw.tolist()}")

    # Centroids should land near the true blob centres (10, -10, 0).
    means = sorted(c.mean().item() for c in centroids)
    close = (abs(means[0] + 10) < 1.0 and abs(means[1]) < 1.0
             and abs(means[2] - 10) < 1.0)
    r.check(close, "centroids recover the true cluster centres",
            f"got means {[round(m, 2) for m in means]}, expected ~[-10, 0, 10]")

    # Weighting must actually bias the result. One heavily-weighted point
    # against many light ones should pull its centroid decisively.
    pts = torch.cat([torch.zeros(9, DIM), torch.full((1, DIM), 100.0)])
    w = torch.cat([torch.ones(9), torch.tensor([1000.0])])
    c1, _ = weighted_kmeans(pts, w, k=1)
    r.check(c1.mean().item() > 90.0,
            "a heavily-weighted point dominates its centroid",
            f"centroid mean {c1.mean().item():.2f}, expected > 90")

    # Fewer points than k must pad, not crash.
    small, small_w = weighted_kmeans(torch.randn(2, DIM), torch.ones(2), k=5)
    r.check(small.shape == (5, DIM) and small_w.shape == (5,),
            "pads to k when given fewer points than clusters",
            f"got {tuple(small.shape)}")


def test_memory_is_bounded(r: Results) -> None:
    """
    The load-bearing property: context size is independent of stream length.

    This is the whole reason the folder exists. If it regresses, streaming is
    just offline inference with extra steps.
    """
    torch.manual_seed(1)
    mem = StarMemory(dim=DIM)
    ceiling = mem.max_context_tokens()

    sizes = {}
    for step in range(1, 2001):
        mem.write(torch.randn(64, DIM))
        if step in (200, 500, 1000, 2000):
            sizes[step] = mem.read().shape[0]

    r.check(len(set(sizes.values())) == 1,
            "context size is IDENTICAL at 200, 500, 1000 and 2000 frames",
            f"sizes were {sizes}")
    r.check(all(v <= ceiling for v in sizes.values()),
            "context never exceeds the advertised ceiling",
            f"ceiling {ceiling}, observed {sizes}")

    cfg = mem.cfg
    r.check(len(mem.spatial) <= cfg.n_spatial,
            "spatial buffer respects its cap", f"{len(mem.spatial)}")
    r.check(mem.temporal.shape[0] <= cfg.n_temporal,
            "temporal buffer respects its cap", f"{mem.temporal.shape[0]}")
    r.check(mem.abstract.shape[0] == cfg.n_abstract,
            "abstract buffer holds a fixed slate", f"{mem.abstract.shape[0]}")
    r.check(len(mem.buffer) <= cfg.n_buffer,
            "raw buffer respects its cap", f"{len(mem.buffer)}")

    r.check(mem.frames_seen == 2000,
            "frame counter tracks the true stream length")

    # Temporal weights must still account for every frame ever written --
    # this is what proves consolidation MERGED rather than DISCARDED.
    total_w = mem.temporal_w.sum().item()
    r.check(abs(total_w - 2000) < 1.0,
            "temporal weights still sum to every frame ever seen",
            f"summed to {total_w}, expected 2000 — a shortfall means "
            "consolidation is dropping frames instead of merging them")


def test_memory_retains_signal(r: Results) -> None:
    """
    Bounded is worthless without recall. A distinctive early event must
    survive thousands of frames of unrelated noise.
    """
    torch.manual_seed(2)
    mem = StarMemory(dim=DIM)

    # A strong, distinctive "event" — a specific direction in feature space.
    event = torch.zeros(DIM)
    event[:8] = 5.0

    for _ in range(50):
        mem.write(event.expand(64, DIM) + torch.randn(64, DIM) * 0.1)
    for _ in range(1500):
        mem.write(torch.randn(64, DIM) * 0.1)

    context = mem.read()
    # Cosine similarity of the best-matching memory slot to the event.
    ctx_n = context / (context.norm(dim=-1, keepdim=True) + 1e-6)
    ev_n = event / event.norm()
    best = (ctx_n @ ev_n).max().item()

    r.check(best > 0.5,
            "an event from 1500 frames ago is still recoverable from memory",
            f"best cosine similarity {best:.3f}, expected > 0.5")

    # Sanity floor: random noise should NOT match this well, or the test
    # above is measuring nothing.
    rand_dir = torch.randn(DIM)
    rand_dir = rand_dir / rand_dir.norm()
    noise_match = (ctx_n @ rand_dir).max().item()
    r.check(best > noise_match,
            "the retained event beats a random direction (signal, not artefact)",
            f"event {best:.3f} vs random {noise_match:.3f}")


def test_pooling_and_validation(r: Results) -> None:
    """Pooling must handle non-square token counts; writes must validate."""
    mem = StarMemory(dim=DIM)

    # 64 tokens is 8x8 — a perfect square, the normal path.
    mem.write(torch.randn(64, DIM))
    r.check(mem.spatial[0].shape == (64, DIM),
            "square token grids pool to the configured side",
            f"got {tuple(mem.spatial[0].shape)}")

    # 100 tokens is 10x10, also square, must downsample to 8x8 = 64.
    mem2 = StarMemory(dim=DIM)
    mem2.write(torch.randn(100, DIM))
    r.check(mem2.spatial[0].shape == (64, DIM),
            "10x10 grid downsamples to the 8x8 spatial budget",
            f"got {tuple(mem2.spatial[0].shape)}")

    # 50 tokens is NOT square — must fall back to 1-D pooling, not guess a
    # rectangle. Guessing transposes the frame and nothing raises.
    mem3 = StarMemory(dim=DIM)
    mem3.write(torch.randn(50, DIM))
    r.check(mem3.spatial[0].shape == (64, DIM),
            "non-square token counts fall back to sequence pooling",
            f"got {tuple(mem3.spatial[0].shape)}")

    for bad, label in [
        (torch.randn(64), "1-D input"),
        (torch.randn(64, DIM + 1), "wrong feature dim"),
    ]:
        try:
            StarMemory(dim=DIM).write(bad)
            caught = False
        except ValueError:
            caught = True
        r.check(caught, f"rejects {label}")


def test_config_ceiling_is_honest(r: Results) -> None:
    """`max_context_tokens` must match what `read()` actually produces."""
    cfg = StarConfig(n_spatial=2, n_temporal=10, n_abstract=8,
                     n_retrieved=2, n_buffer=50)
    mem = StarMemory(dim=DIM, config=cfg)

    torch.manual_seed(3)
    for _ in range(300):
        mem.write(torch.randn(64, DIM))

    predicted = mem.max_context_tokens()
    actual = mem.read().shape[0]
    r.check(actual == predicted,
            "the advertised ceiling equals the real steady-state size",
            f"predicted {predicted}, actual {actual} — a ceiling you cannot "
            "trust is worse than no ceiling")

    # A smaller config must genuinely produce a smaller context.
    big = StarMemory(dim=DIM)
    for _ in range(300):
        big.write(torch.randn(64, DIM))
    r.check(mem.read().shape[0] < big.read().shape[0],
            "shrinking the config shrinks the context",
            f"small {mem.read().shape[0]} vs default {big.read().shape[0]}")


def main() -> int:
    r = Results("STAR streaming memory — boundedness and recall")
    test_weighted_kmeans(r)
    test_memory_is_bounded(r)
    test_memory_retains_signal(r)
    test_pooling_and_validation(r)
    test_config_ceiling_is_honest(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())

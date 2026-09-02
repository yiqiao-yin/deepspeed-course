# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""
Regression test: modern CIFAR-10 architectures and augmentations.

Run:
    uv run tests/test_modern_cifar.py

Why this suite exists
---------------------
Every one of these components can be wrong while training still runs and the
loss still falls. Cutout that zeroes nothing, a whitening init that does not
whiten, a residual block whose skip path is dead, TTA that averages an image
with itself -- all of them produce a model that trains, converges, and is worse
than it should be, with no error anywhere.

So these assert PROPERTIES, not shapes:

  * cutout removes the area it claims to, and a different one each call
  * translation moves content but preserves it (no black border creeping in)
  * alternating flip is exactly the flip on odd epochs and exactly identity on
    even ones -- the whole point is that it is derandomised
  * the whitening init actually decorrelates, measured, versus random init
  * residual blocks pass gradient through the skip path
  * frozen BatchNorm scales stay frozen after an optimizer step

CPU only. No GPU, no download.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srcload import Results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "01_basics/03_convnet_cifar10"))

from modern_cifar_models import (  # noqa: E402
    CifarNet, FrozenScaleBatchNorm, MODELS, ResNet9, WideResNet,
    alternating_flip, build_model, count_params, cutout, init_whitening_conv,
    pad_and_random_crop, random_flip)


def test_cutout(r: Results) -> None:
    torch.manual_seed(0)
    x = torch.ones(64, 3, 32, 32)

    out = cutout(x, size=12)
    zeroed = (out == 0).float().mean().item()
    # A 12x12 hole is 14.06% of a 32x32 image; holes near the border are
    # clipped, so the mean must be positive but below that ceiling.
    r.check(0.02 < zeroed < 12 * 12 / (32 * 32) + 1e-6,
            f"cutout(12) removes a plausible area ({zeroed:.2%})",
            "zero means the mask never fires; above the ceiling means it is "
            "removing more than one square's worth")

    r.check(torch.equal(cutout(x, size=0), x),
            "cutout(0) is exactly the identity",
            "an 'off' switch that still perturbs the data is worse than no "
            "switch, because it silently taxes every run that disables it")

    a, b = cutout(x, 12), cutout(x, 12)
    r.check(not torch.equal(a, b),
            "cutout picks a different square each call",
            "a fixed mask would teach the network to ignore one region of the "
            "image forever")

    # The hole must be per-image, not one square shared by the batch.
    holes = (cutout(x, 12) == 0).any(dim=1).flatten(1).float().argmax(1)
    r.check(holes.unique().numel() > 1,
            "cutout is applied per image, not once per batch")


def test_translate(r: Results) -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 3, 32, 32)

    out = pad_and_random_crop(x, translate=4)
    r.check(out.shape == x.shape, "translate preserves shape")
    r.check(not torch.equal(out, x), "translate actually moves content")
    r.check(torch.equal(pad_and_random_crop(x, 0), x),
            "translate=0 is exactly the identity")

    # Reflection padding, not zeros: a translated image must not acquire
    # constant borders. With zero padding the shifted-in region is exactly 0.
    zeros_before = (x == 0).float().mean().item()
    zeros_after = (out == 0).float().mean().item()
    r.check(zeros_after <= zeros_before + 0.01,
            "translation introduces no zero border (reflect, not zero, pad)",
            f"{zeros_before:.4f} -> {zeros_after:.4f}; a black border is a cue "
            "the first conv layer can learn, which teaches it about the "
            "augmentation rather than about the image")

    # Content is preserved, only moved: the set of pixel values should be
    # nearly unchanged in aggregate.
    r.check(abs(out.mean().item() - x.mean().item()) < 0.05,
            "translation preserves the intensity distribution")


def test_alternating_flip(r: Results) -> None:
    x = torch.randn(8, 3, 32, 32)
    r.check(torch.equal(alternating_flip(x, 0), x),
            "alternating flip: even epochs are the identity")
    r.check(torch.equal(alternating_flip(x, 1), x.flip(-1)),
            "alternating flip: odd epochs flip EVERY image")
    r.check(torch.equal(alternating_flip(alternating_flip(x, 1), 1), x),
            "flipping twice returns the original (it is an involution)")

    # The point of the derandomised variant: over any even number of epochs
    # every image is seen in both orientations EXACTLY equally often, which
    # random flipping only achieves in expectation.
    seen_flipped = sum(int(torch.equal(alternating_flip(x, e), x.flip(-1)))
                       for e in range(10))
    r.check(seen_flipped == 5,
            "over 10 epochs each image is flipped exactly 5 times",
            f"got {seen_flipped}; that exactness is the entire advantage over "
            "random flipping (arXiv:2404.00498)")

    torch.manual_seed(0)
    flipped = random_flip(x)
    changed = sum(int(not torch.equal(flipped[i], x[i])) for i in range(len(x)))
    r.check(0 < changed < len(x),
            f"random_flip flips a strict subset ({changed}/{len(x)})",
            "flipping all or none would mean the mask is not per-image")


def test_whitening_actually_whitens(r: Results) -> None:
    """The decisive test: measure decorrelation, do not assume it."""
    torch.manual_seed(0)
    # Spatially correlated inputs, like real images. White noise is already
    # white, so whitening it would be a no-op and prove nothing.
    images = F.avg_pool2d(torch.randn(512, 3, 32, 32), 3, stride=1, padding=1)
    net = CifarNet()

    def off_diag(out: torch.Tensor, n_keep: int) -> float:
        z = out[:, :n_keep].permute(1, 0, 2, 3).reshape(n_keep, -1)
        z = z - z.mean(1, keepdim=True)
        cov = (z @ z.T) / z.shape[1]
        return ((cov - torch.diag(cov.diagonal())).abs().mean()
                / cov.diagonal().abs().mean()).item()

    n_keep = net.whiten.weight.shape[0] // 2   # only the +v half; -v is
                                               # anti-correlated by construction
    with torch.no_grad():
        before = off_diag(net.whiten(images), n_keep)
        init_whitening_conv(net.whiten, images)
        after = off_diag(net.whiten(images), n_keep)

    r.check(after < before / 10,
            f"whitening init decorrelates the first layer "
            f"({before:.4f} -> {after:.6f})",
            "if this fails the 'whitening' conv is just a frozen random "
            "projection, which is strictly worse than a learned one")

    r.check(not net.whiten.weight.requires_grad,
            "the whitening conv stays frozen",
            "if it trains, the eigenvector structure is destroyed within a few "
            "steps and the initialisation was pointless")

    r.check(net.whiten.weight.shape[0] == 2 * 3 * 2 ** 2,
            "the whitening conv stores both signs of each eigenvector",
            "GELU is not symmetric, so +v and -v carry different information")


def test_models_build_and_learn(r: Results) -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    for name in MODELS:
        model = build_model(name)
        out = model(x)
        r.check(out.shape == (8, 10), f"{name}: outputs 10 logits per image")
        r.check(torch.isfinite(out).all(), f"{name}: forward pass is finite")

        # One step must reduce the loss on a batch it is allowed to memorise.
        # This is the cheapest possible proof that gradients reach the weights.
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        first = F.cross_entropy(model(x), y).item()
        for _ in range(20):
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        last = F.cross_entropy(model(x), y).item()
        r.check(last < first,
                f"{name}: 20 steps on one batch reduce the loss "
                f"({first:.3f} -> {last:.3f})",
                "a model that cannot overfit 8 images has a broken gradient "
                "path somewhere, and will still train and still look fine")

    r.check(count_params(build_model("wrn_16_8")) > count_params(build_model("resnet9")),
            "wrn_16_8 is the widest of the three")


def test_residual_skip_is_live(r: Results) -> None:
    """A residual block whose skip path is dead trains fine and is worse."""
    torch.manual_seed(0)
    model = ResNet9()
    block = model.layer1[1]                       # the Residual
    x = torch.randn(2, 128, 16, 16, requires_grad=True)

    # Zero the block's conv weights: the output must then equal the INPUT,
    # which is only true if the skip connection is actually wired.
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    out = block(x)
    # BatchNorm with zeroed weight/bias outputs zeros, so f(x) = 0 and the
    # block must return x unchanged.
    r.check(torch.allclose(out, x, atol=1e-5),
            "with the residual branch zeroed, the block is the identity",
            "if this fails the skip connection is not connected and the "
            "'residual' network is a plain deep CNN")


def test_frozen_batchnorm_stays_frozen(r: Results) -> None:
    torch.manual_seed(0)
    bn = FrozenScaleBatchNorm(16)
    before = bn.weight.detach().clone()
    opt = torch.optim.SGD([p for p in bn.parameters() if p.requires_grad], lr=1.0)
    for _ in range(5):
        opt.zero_grad()
        bn(torch.randn(32, 16, 8, 8)).square().mean().backward()
        opt.step()
    r.check(torch.equal(bn.weight, before),
            "frozen BatchNorm scale is unchanged after training steps")
    r.check(not torch.equal(bn.bias, torch.zeros_like(bn.bias)),
            "the BatchNorm bias DOES train",
            "freezing both would remove the layer's only free parameter")

    # PyTorch's momentum is the weight on the new observation; the speedrun
    # uses 0.6, which must arrive as 1 - 0.6.
    r.check(abs(bn.momentum - 0.4) < 1e-9,
            "momentum 0.6 is stored as PyTorch's 0.4 convention",
            f"got {bn.momentum}; getting this backwards makes the running "
            "statistics track the initialisation instead of the data")


def test_tta_is_not_a_no_op(r: Results) -> None:
    """Mirror TTA must average two DIFFERENT views, or it buys nothing."""
    torch.manual_seed(0)
    model = build_model("resnet9").eval()
    x = torch.randn(16, 3, 32, 32)
    with torch.no_grad():
        a = model(x)
        b = model(x.flip(-1))
    r.check(not torch.allclose(a, b, atol=1e-4),
            "a model gives different logits for an image and its mirror",
            "if these were identical, averaging them would cost a forward pass "
            "and change nothing")
    r.check(torch.isfinite(((a + b) / 2)).all(), "the averaged logits are finite")


def main() -> int:
    r = Results("Modern CIFAR-10 architectures and augmentations")
    test_cutout(r)
    test_translate(r)
    test_alternating_flip(r)
    test_whitening_actually_whitens(r)
    test_models_build_and_learn(r)
    test_residual_skip_is_live(r)
    test_frozen_batchnorm_stays_frozen(r)
    test_tta_is_not_a_no_op(r)
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())

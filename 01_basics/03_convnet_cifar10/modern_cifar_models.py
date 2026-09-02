#!/usr/bin/env python3
"""
Modern CIFAR-10 architectures and the augmentations that actually move the number.

    uv run modern_cifar_models.py          # no GPU needed — builds each model,
                                           # counts parameters, checks properties

The baseline in this folder (`cifar10_deepspeed.py`) reaches about 81%. That is
a perfectly good first DeepSpeed example and a poor CIFAR-10 model. This module
holds the three architectures used by `train_modern_cifar10.py` to close the
gap, plus the augmentation primitives, because on CIFAR-10 the augmentation is
not a detail — it is most of the difference between 81% and 95%.

The three models, and why each is here
--------------------------------------
`resnet9`
    The classic fast-CIFAR residual net: conv stem, four stages, two residual
    blocks, global max-pool, linear head. ~6.6M parameters. This is the design
    that made "94% in under a minute" a normal thing to say, and it is the
    smallest jump from the baseline that a reader will recognise as a ResNet.

`cifarnet`
    The architecture from Keller Jordan's CIFAR-10 speedruns (arXiv 2404.00498),
    which reach 94% in 2.6 seconds and 96% in 27 seconds on an A100. Three
    conv groups at widths 128/384/512, GELU, and two things that look wrong
    until you know why:

      * The first layer is a FROZEN 2x2 convolution initialised from the
        eigenvectors of training-image patches -- a whitening transform, not a
        learned feature extractor. See `init_whitening_conv`.
      * BatchNorm weights are frozen at 1 and only the biases train. The scale
        is redundant with the following convolution, so learning it wastes a
        parameter and some stability.

`wrn_16_8`
    Wide ResNet 16-8 (arXiv 1605.07146): the "make it wider, not deeper" result.
    Included as the counter-argument to cifarnet -- an ordinary, unsurprising
    architecture that gets to the same place with more parameters and more
    epochs. Useful precisely because it is boring.

What is deliberately NOT reproduced
-----------------------------------
The speedrun scripts use a custom optimizer (Muon), pre-decoded GPU-resident
datasets, and a hand-tuned fp16 schedule. This folder trains with DeepSpeed, so
the optimizer comes from `ds_config.json`. Expect the architectures to reach
their accuracy in minutes rather than seconds. The point here is the model and
the recipe, not the wall clock -- and claiming speedrun timings from a
DeepSpeed run would be a fabricated number.

Sources
-------
Jordan, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU", arXiv:2404.00498
    https://arxiv.org/abs/2404.00498  |  github.com/KellerJordan/cifar10-airbench
Zagoruyko & Komodakis, "Wide Residual Networks", arXiv:1605.07146
DeVries & Taylor, "Improved Regularization ... with Cutout", arXiv:1708.04552

Pure PyTorch. No GPU, no download, no DeepSpeed.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Augmentation
#
# On CIFAR-10 these are worth more than architecture. Going from "no
# augmentation" to "flip + translate + cutout" is worth several points; going
# from a good CNN to a better CNN is worth about one.
# =============================================================================


def random_flip(x: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
    """
    Horizontally flip a random half of the batch.

    Note this flips each image with probability 1/2 INDEPENDENTLY, which is the
    standard implementation and is what `alternating_flip` improves on.
    """
    mask = torch.rand(len(x), device=x.device, generator=generator) < 0.5
    return torch.where(mask.view(-1, 1, 1, 1), x.flip(-1), x)


def alternating_flip(x: torch.Tensor, epoch: int) -> torch.Tensor:
    """
    Derandomised horizontal flipping: flip EVERY image on odd epochs, none on
    even ones.

    From arXiv:2404.00498, and it is the least intuitive trick in this file.
    Standard random flipping gives each image a coin flip per epoch, so over N
    epochs an image is seen flipped Binomial(N, 1/2) times -- which by chance
    can be lopsided for any particular image. Alternating guarantees every image
    is seen in both orientations equally often. The paper reports it beating
    random flipping in every case where flipping helps at all.

    Cheaper, too: no random numbers, and the flip is one contiguous op.
    """
    return x.flip(-1) if (epoch % 2 == 1) else x


def pad_and_random_crop(x: torch.Tensor, translate: int,
                        generator: torch.Generator = None) -> torch.Tensor:
    """
    Reflection-pad by `translate` and crop back to the original size at a random
    offset -- i.e. random translation, the standard CIFAR-10 augmentation.

    Reflection rather than zero padding: zeros introduce a hard black border
    that the first convolution can learn to detect, which is signal about the
    augmentation rather than about the image.
    """
    if translate <= 0:
        return x
    n, c, h, w = x.shape
    padded = F.pad(x, (translate,) * 4, mode="reflect")
    # One offset per image, not one per batch -- a shared offset would make the
    # whole batch correlated and waste most of the augmentation's value.
    ox = torch.randint(0, 2 * translate + 1, (n,), device=x.device, generator=generator)
    oy = torch.randint(0, 2 * translate + 1, (n,), device=x.device, generator=generator)
    rows = (torch.arange(h, device=x.device).view(1, h, 1) + oy.view(n, 1, 1))
    cols = (torch.arange(w, device=x.device).view(1, 1, w) + ox.view(n, 1, 1))
    idx = (rows * padded.size(-1) + cols).view(n, 1, h * w).expand(n, c, h * w)
    return padded.reshape(n, c, -1).gather(2, idx).view(n, c, h, w)


def cutout(x: torch.Tensor, size: int,
           generator: torch.Generator = None) -> torch.Tensor:
    """
    Zero a random `size` x `size` square in each image (arXiv:1708.04552).

    Forces the network to use the whole object rather than one discriminative
    patch. The square is clipped at the border rather than wrapped, so images
    whose square lands near an edge lose less area -- that asymmetry is in the
    original and is not a bug.
    """
    if size <= 0:
        return x
    n, c, h, w = x.shape
    cy = torch.randint(0, h, (n,), device=x.device, generator=generator)
    cx = torch.randint(0, w, (n,), device=x.device, generator=generator)
    rows = torch.arange(h, device=x.device).view(1, h, 1)
    cols = torch.arange(w, device=x.device).view(1, 1, w)
    keep = ~(((rows - cy.view(n, 1, 1)).abs() < size // 2)
             & ((cols - cx.view(n, 1, 1)).abs() < size // 2))
    return x * keep.unsqueeze(1)


# =============================================================================
# cifarnet — the speedrun architecture
# =============================================================================


class FrozenScaleBatchNorm(nn.BatchNorm2d):
    """
    BatchNorm whose scale is frozen at 1; only the bias trains.

    The scale is redundant: whatever it would learn, the next convolution can
    absorb. Freezing it removes a parameter per channel and, more importantly,
    removes a way for the two to fight each other early in training.

    The momentum is also unusual (0.6 of the new batch, versus PyTorch's 0.1),
    because these runs are short -- statistics estimated with the default
    momentum would still be tracking the initialisation when training ends.
    """

    def __init__(self, num_features: int, momentum: float = 0.6, eps: float = 1e-12):
        # PyTorch's `momentum` is the weight on the NEW observation, so the
        # speedrun's "momentum 0.6" is passed as 1 - 0.6.
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class ConvGroup(nn.Module):
    """conv -> pool -> norm -> act -> conv -> norm -> act, halving resolution."""

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding="same", bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = FrozenScaleBatchNorm(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding="same", bias=False)
        self.norm2 = FrozenScaleBatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    """
    Whitening conv -> three conv groups -> max-pool -> linear head.

    `scaling_factor` shrinks the logits before the loss. With label smoothing
    0.2 the target is never 1.0, so unscaled logits push the network to be
    confident in a direction the loss does not actually reward.
    """

    def __init__(self, widths: Tuple[int, int, int] = (128, 384, 512),
                 scaling_factor: float = 1 / 9, num_classes: int = 10):
        super().__init__()
        whiten_kernel = 2
        whiten_width = 2 * 3 * whiten_kernel ** 2          # 24 = both signs x 3 channels x 2x2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel, padding=0, bias=True)
        self.whiten.weight.requires_grad = False           # set by init_whitening_conv
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0]),
            ConvGroup(widths[0], widths[1]),
            ConvGroup(widths[1], widths[2]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.whiten(x))
        x = x.flatten(1) if x.dim() > 2 else x
        return self.head(x) * self.scaling_factor


def init_whitening_conv(layer: nn.Conv2d, train_images: torch.Tensor,
                        eps: float = 5e-4) -> None:
    """
    Initialise a frozen conv so that its output is a WHITENED patch embedding.

    The first layer does not have to be learned. Take every kxk patch of the
    training set, compute the covariance across the 3*k*k patch dimensions,
    eigendecompose it, and use the eigenvectors scaled by 1/sqrt(eigenvalue) as
    the filters. Applying them decorrelates the input and equalises its
    variance -- exactly what the first layer of a trained network tends to learn
    anyway, available for free before a single gradient step.

    Both signs of each eigenvector are stored, which is why the layer has
    2 * 3 * k * k output channels. With a GELU immediately after, +v and -v are
    not redundant: the activation is not symmetric, so keeping both preserves
    information a single sign would discard.
    """
    c_out, c_in, kh, kw = layer.weight.shape
    patches = train_images.unfold(2, kh, 1).unfold(3, kw, 1)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c_in * kh * kw).float()
    patches = patches - patches.mean(0, keepdim=True)
    cov = (patches.T @ patches) / max(1, len(patches))
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh returns ascending; take them descending so the strongest come first.
    scale = (eigenvalues.flip(0).clamp(min=eps) + eps).rsqrt()
    filters = eigenvectors.T.flip(0) * scale.view(-1, 1)
    both_signs = torch.cat([filters, -filters]).view(c_out, c_in, kh, kw)
    layer.weight.data[:] = both_signs.to(layer.weight.dtype)
    layer.bias.data.zero_()


# =============================================================================
# resnet9
# =============================================================================


def conv_bn(c_in: int, c_out: int, pool: bool = False) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
              nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.a = conv_bn(channels, channels)
        self.b = conv_bn(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b(self.a(x))


class ResNet9(nn.Module):
    """The fast-CIFAR residual net. Global MAX pool, not average — on 32x32
    inputs the final feature map is small enough that averaging blurs away the
    strongest evidence."""

    def __init__(self, num_classes: int = 10, scaling_factor: float = 0.125):
        super().__init__()
        self.stem = conv_bn(3, 64)
        self.layer1 = nn.Sequential(conv_bn(64, 128, pool=True), Residual(128))
        self.layer2 = conv_bn(128, 256, pool=True)
        self.layer3 = nn.Sequential(conv_bn(256, 512, pool=True), Residual(512))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(512, num_classes, bias=False)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return self.head(self.pool(x).flatten(1)) * self.scaling_factor


# =============================================================================
# wide resnet
# =============================================================================


class WideBasic(nn.Module):
    """Pre-activation residual block: norm and activation BEFORE the conv, so
    the skip path is a clean identity all the way through the network."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.shortcut = (nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
                         if (stride != 1 or c_in != c_out) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class WideResNet(nn.Module):
    """WRN-depth-width, e.g. wrn_16_8 is depth 16, widening factor 8."""

    def __init__(self, depth: int = 16, widen: int = 8, num_classes: int = 10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        blocks = []
        c_in = widths[0]
        for stage, c_out in enumerate(widths[1:]):
            for block in range(n):
                blocks.append(WideBasic(c_in, c_out,
                                        stride=(1 if stage == 0 or block > 0 else 2)))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(c_in)
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.conv1(x))
        x = F.relu(self.bn(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


MODELS = {
    "resnet9":  "Fast-CIFAR residual net, ~6.6M params. The recognisable one.",
    "cifarnet": "Speedrun architecture (arXiv:2404.00498): frozen whitening "
                "conv, GELU, frozen BatchNorm scales.",
    "wrn_16_8": "Wide ResNet 16-8 (arXiv:1605.07146). Wider, not deeper.",
}


def build_model(name: str) -> nn.Module:
    if name == "resnet9":
        return ResNet9()
    if name == "cifarnet":
        return CifarNet()
    if name == "wrn_16_8":
        return WideResNet(depth=16, widen=8)
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODELS)}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Demo — runs on CPU, asserts the properties rather than printing shapes
# =============================================================================


def _demo() -> None:
    bar = "=" * 78
    torch.manual_seed(0)
    print(bar)
    print("  Modern CIFAR-10 architectures")
    print(bar)
    x = torch.randn(4, 3, 32, 32)
    for name, blurb in MODELS.items():
        model = build_model(name).eval()
        out = model(x)
        print(f"  {name:<10} {count_params(model):>10,} params   out {tuple(out.shape)}")
        print(f"             {blurb}")
    print(bar)

    print("  Augmentation — checking each one DOES something")
    flipped = random_flip(x)
    print(f"    random_flip      changed {(flipped != x).any(dim=(1,2,3)).sum().item()}/4 images")
    print(f"    alternating_flip epoch 0 unchanged: {torch.equal(alternating_flip(x, 0), x)}"
          f" | epoch 1 flipped: {torch.equal(alternating_flip(x, 1), x.flip(-1))}")
    cropped = pad_and_random_crop(x, translate=4)
    print(f"    translate=4      shape kept {tuple(cropped.shape)}, "
          f"content changed: {not torch.equal(cropped, x)}")
    cut = cutout(x, size=12)
    zeroed = (cut == 0).float().mean().item()
    print(f"    cutout=12        {zeroed:.1%} of pixels zeroed "
          f"(a 12x12 hole in 32x32 is at most {12*12/(32*32):.1%})")
    print(bar)

    print("  Whitening init — does it actually decorrelate?")
    images = torch.randn(512, 3, 32, 32)
    # Give the fake images spatial correlation, like real ones have; otherwise
    # they are already white and there is nothing for whitening to do.
    images = F.avg_pool2d(images, 3, stride=1, padding=1)
    net = CifarNet()

    def channel_off_diag(out: torch.Tensor, n_keep: int) -> float:
        """
        Correlation among the first n_keep output CHANNELS.

        Two things matter here and both were wrong in an earlier draft of this
        demo. Measure across channels, pooling over samples and positions --
        covariance over the flattened 24x31x31 output from 512 images is a
        23,064-dimensional estimate from 512 samples, which is noise. And use
        only the first half of the channels: the layer stores +v and -v for
        every filter, so channels i and i+12 are exactly anti-correlated by
        construction and would swamp any real signal.
        """
        z = out[:, :n_keep].permute(1, 0, 2, 3).reshape(n_keep, -1)
        z = z - z.mean(1, keepdim=True)
        cov = (z @ z.T) / z.shape[1]
        d = cov.diagonal().abs().mean()
        off = (cov - torch.diag(cov.diagonal())).abs().mean()
        return (off / d).item()

    n_keep = net.whiten.weight.shape[0] // 2
    before = channel_off_diag(net.whiten(images), n_keep)
    init_whitening_conv(net.whiten, images)
    after = channel_off_diag(net.whiten(images), n_keep)
    print(f"    off-diagonal / diagonal covariance, random init : {before:.4f}")
    print(f"    off-diagonal / diagonal covariance, whitened    : {after:.4f}")
    print(f"    -> {before / max(after, 1e-9):.0f}x more decorrelated")
    assert after < before / 5, (
        f"whitening init is not decorrelating: {before:.4f} -> {after:.4f}")
    print(bar)
    print("  These run on CPU. Training needs a GPU: train_modern_cifar10.py")
    print(bar)


if __name__ == "__main__":
    _demo()

"""
cracknet.py — crack segmenter designed around connectivity rather than contrast.

WHY THIS SHAPE: a controlled reading experiment (scripts/render_crack_grids.py,
scripts/score_crack_grids.py) scored a human-level reader at cell-level F1 0.89
against 0.40 for the filter pipeline, on the same 64-cell task. The gap is not
sensitivity to darkness -- the filters are, if anything, too sensitive. It is
that the reader identifies ONE structure spanning the frame, joins faint
collinear fragments across gaps, and discards isolated dark specks no matter how
dark they are. Every one of those is long-range structure, so the model is built
to have access to it:

    full-frame input      a 15px black-hat window and a sigma<=16 ridge filter
                          both failed for the same reason, and the previous
                          U-Net trained on 256px crops of 512px images, so it
                          never saw a whole crack either
    pretrained encoder    ImageNet features already separate material texture
                          from structure; 577 training images cannot teach that
                          from scratch
    deep supervision      the reader judges at several scales at once; side
                          outputs make each decoder scale answer independently,
                          which is DeepCrack's actual contribution
    clDice loss           per-pixel losses are indifferent to whether a
                          prediction is one curve or forty specks. clDice scores
                          the overlap of SKELETONS, so a fragmented prediction is
                          penalised even when its pixel overlap is fine

TWO HEADS, because the two datasets annotate incompatibly and the earlier work
established they must not be merged naively (MCS marks crack bodies ~21.9px
wide, Stone331 traces ~1.9px centrelines, a 23x difference):

    centreline  supervised on Stone331 directly and on skeletonised MCS, so
                BOTH datasets teach topology -- 577 images instead of 246 or 331
    width       supervised only where body annotation exists, loss masked
                elsewhere, so severity keeps the width information that a
                centreline throws away
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


@dataclass
class CrackNetConfig:
    """Args: pretrained: load ImageNet encoder weights; decoder_channels: per stage."""

    pretrained: bool = True
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)
    deep_supervision: bool = True
    coord_attention: bool = True


class CoordinateAttention(nn.Module):
    """
    Channel attention that keeps one spatial axis at a time.

    Squeeze-and-excitation pools away both axes, which discards exactly what
    matters here: a crack is defined by extent in one direction. Pooling H and W
    separately lets a channel say "there is structure along this row" without
    first collapsing where it is. Measured cost is 13,016 parameters, 0.05% of
    the model, so it is cheaper to ablate than to argue about -- train with and
    without --no-coord-attention and read the difference.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.norm = nn.BatchNorm2d(hidden)
        self.activation = nn.Hardswish(inplace=True)
        self.expand_h = nn.Conv2d(hidden, channels, 1)
        self.expand_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        along_h = x.mean(dim=3, keepdim=True)
        along_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        pooled = self.activation(self.norm(self.reduce(
            torch.cat([along_h, along_w], dim=2))))
        part_h, part_w = torch.split(pooled, [height, width], dim=2)
        gate_h = torch.sigmoid(self.expand_h(part_h))
        gate_w = torch.sigmoid(self.expand_w(part_w.permute(0, 1, 3, 2)))
        return x * gate_h * gate_w


class DecoderBlock(nn.Module):
    """Upsample, concatenate the skip, then two convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 coord_attention: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = (CoordinateAttention(out_channels)
                          if coord_attention else nn.Identity())

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # Odd input sizes leave the upsample a pixel short of the skip.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.attention(self.block(x))


class CrackNet(nn.Module):
    """ResNet-34 U-Net with deep supervision and separate centreline/width heads."""

    def __init__(self, config: CrackNetConfig | None = None):
        super().__init__()
        self.config = config or CrackNetConfig()

        weights = ResNet34_Weights.IMAGENET1K_V1 if self.config.pretrained else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4

        skips = (256, 128, 64, 64, 0)
        channels = self.config.decoder_channels
        blocks = []
        in_channels = 512
        for index, out_channels in enumerate(channels):
            blocks.append(DecoderBlock(in_channels, skips[index], out_channels,
                                       self.config.coord_attention))
            in_channels = out_channels
        self.decoder = nn.ModuleList(blocks)

        self.centreline = nn.Conv2d(channels[-1], 1, 1)
        self.width = nn.Conv2d(channels[-1], 1, 1)
        # One side head per decoder stage; each is trained against the full
        # target, so no stage can rely on a later one to fix its mistakes.
        self.side = nn.ModuleList(
            [nn.Conv2d(c, 1, 1) for c in channels[:-1]]
        ) if self.config.deep_supervision else nn.ModuleList()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list]:
        size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        bottleneck = self.layer4(s3)

        skips = [s3, s2, s1, s0, None]
        feature = bottleneck
        sides = []
        for index, block in enumerate(self.decoder):
            feature = block(feature, skips[index])
            if self.side and index < len(self.side):
                sides.append(F.interpolate(self.side[index](feature), size=size,
                                           mode="bilinear", align_corners=False))

        return {
            "centreline": F.interpolate(self.centreline(feature), size=size,
                                        mode="bilinear", align_corners=False),
            "width": F.interpolate(self.width(feature), size=size,
                                   mode="bilinear", align_corners=False),
            "sides": sides,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Min-pool, as a differentiable erosion."""
    horizontal = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    vertical = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(horizontal, vertical)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def soft_skeleton(x: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    """
    Differentiable skeleton, by repeated erosion minus opening.

    This is what makes connectivity trainable. A hard skeleton has no gradient,
    so the usual workaround is to give up and score pixels; this keeps the
    topology in the loss.
    """
    opened = soft_dilate(soft_erode(x))
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_dilate(soft_erode(x))
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cl_dice(probability: torch.Tensor, target: torch.Tensor,
            iterations: int = 6, smooth: float = 1.0) -> torch.Tensor:
    """
    Connectivity loss: Dice between predicted and true skeletons.

    Topological precision asks how much of the predicted skeleton lies on a real
    crack; topological sensitivity asks how much of the true skeleton is
    covered. A prediction broken into forty specks scores badly on the first
    even when its pixel overlap looks respectable, which is exactly the failure
    the filter pipeline shows on lichen.
    """
    skeleton_pred = soft_skeleton(probability, iterations)
    skeleton_true = soft_skeleton(target, iterations)
    precision = ((skeleton_pred * target).sum() + smooth) / (skeleton_pred.sum() + smooth)
    sensitivity = ((skeleton_true * probability).sum() + smooth) / (skeleton_true.sum() + smooth)
    return 1.0 - 2.0 * precision * sensitivity / (precision + sensitivity)


def dice_loss(probability: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    intersection = (probability * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        probability.sum() + target.sum() + smooth)


def crack_loss(outputs: dict, centreline_target: torch.Tensor,
               width_target: torch.Tensor, width_valid: torch.Tensor,
               pos_weight: torch.Tensor,
               cl_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
    """
    Combined objective.

    Args:
        outputs: CrackNet forward output
        centreline_target: 1 where a crack centreline passes
        width_target: normalised half-width, 0-1, meaningful only where valid
        width_valid: per-sample flag; masks the width loss for datasets that
            annotate centrelines only, so they still contribute topology
        pos_weight: positive class weight for BCE
        cl_weight: weight on the connectivity term
    """
    logits = outputs["centreline"]
    probability = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, centreline_target,
                                             pos_weight=pos_weight)
    terms = {"bce": bce.detach(),
             "dice": dice_loss(probability, centreline_target).detach()}
    total = bce + dice_loss(probability, centreline_target)

    connectivity = cl_dice(probability, centreline_target)
    total = total + cl_weight * connectivity
    terms["cldice"] = connectivity.detach()

    for side in outputs["sides"]:
        total = total + 0.4 * F.binary_cross_entropy_with_logits(
            side, centreline_target, pos_weight=pos_weight)

    if width_valid.any():
        mask = width_valid.view(-1, 1, 1, 1).float()
        # Width is only defined on the crack itself; elsewhere it is not zero,
        # it is undefined, so the loss must not be evaluated there.
        where = mask * centreline_target
        if where.sum() > 0:
            width_error = (torch.sigmoid(outputs["width"]) - width_target).abs()
            width_term = (width_error * where).sum() / where.sum()
            total = total + width_term
            terms["width"] = width_term.detach()

    return total, terms

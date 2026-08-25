"""
Small U-Net for crack segmentation.

Sized at roughly 2M parameters, following the same reasoning as the restoration
model: the labelled sets are 246 and 331 images, so capacity is not the binding
constraint and a larger network would mostly memorise. DeepCrack reports ~0.85
F1 with a much heavier architecture; the target here is to beat the OpenCV
detector's measured 0.111, not to match published state of the art.

Class imbalance is the dominant difficulty. Stone331 marks 0.11% of pixels as
crack, so a model predicting all-background scores 99.89% pixel accuracy and is
useless. Training therefore uses Dice alongside BCE, and accuracy is never
reported as a headline metric.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SegmenterConfig:
    """Architecture parameters."""

    base_channels: int = 32
    depth: int = 4
    in_channels: int = 3
    dropout: float = 0.1


def double_conv(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm, the standard U-Net block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CrackUNet(nn.Module):
    """U-Net producing a single-channel crack logit map."""

    def __init__(self, config: SegmenterConfig | None = None):
        super().__init__()
        self.config = config or SegmenterConfig()
        channels = [self.config.base_channels * (2 ** i)
                    for i in range(self.config.depth)]

        self.downs = nn.ModuleList()
        previous = self.config.in_channels
        for width in channels:
            self.downs.append(double_conv(previous, width, self.config.dropout))
            previous = width

        self.bottleneck = double_conv(previous, previous * 2, self.config.dropout)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        previous = previous * 2
        for width in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(previous, width, 2, stride=2))
            self.up_convs.append(double_conv(width * 2, width, self.config.dropout))
            previous = width

        self.head = nn.Conv2d(previous, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) image batch, H and W divisible by 2**depth

        Returns:
            (batch, 1, H, W) logits
        """
        skips: list[torch.Tensor] = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = conv(torch.cat([skip, x], dim=1))

        return self.head(x)

    def parameter_count(self) -> int:
        """Trainable parameter total."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice, which unlike BCE does not collapse under extreme imbalance.

    With 0.11% positive pixels, BCE alone is minimised by predicting
    background everywhere; Dice is driven by overlap, so an empty prediction
    scores zero rather than near-perfect.
    """
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor,
                  pos_weight: torch.Tensor | None = None,
                  dice_weight: float = 0.5) -> torch.Tensor:
    """BCE for per-pixel calibration, Dice for overlap under imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return (1.0 - dice_weight) * bce + dice_weight * dice_loss(logits, target)


def save_segmenter(model: CrackUNet, path: Path, metadata: dict) -> None:
    """Persist weights with the convention they were trained on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(model.config),
        "metadata": metadata,
    }, path)


def load_segmenter(path: Path) -> tuple[CrackUNet, dict]:
    """Load a checkpoint and its training metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CrackUNet(SegmenterConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("metadata", {})

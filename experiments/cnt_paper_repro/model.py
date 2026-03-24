"""ResNet34-U-Net used for the paper-reproduction baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """Standard ResNet34 encoder with a simple U-Net decoder."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, encoder_weights: str | None = "imagenet"):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if str(encoder_weights).lower() == "imagenet" else None
        try:
            encoder = resnet34(weights=weights)
        except Exception:
            encoder = resnet34(weights=None)
            weights = None

        if in_channels != 3:
            old_conv = encoder.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if weights is not None:
                if in_channels == 1:
                    new_conv.weight.data.copy_(old_conv.weight.data.mean(dim=1, keepdim=True))
                else:
                    new_conv.weight.data.copy_(old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
            encoder.conv1 = new_conv

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.center = ConvBlock(512, 512)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.center(x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.final_conv(x)

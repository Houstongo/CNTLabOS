"""Unified CNTSegNet backbone used by all loss-comparison experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling."""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, padding=0),
        )
        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1, 1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode="bilinear", align_corners=True)
        out = torch.cat([aspp1, aspp2, aspp3, aspp4, global_feat], dim=1)
        return self.relu(self.bn(self.conv(out)))


class AttentionModule(nn.Module):
    """Simple channel attention."""

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(in_channels // 4, 8)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.global_avg_pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class DecoderBlock(nn.Module):
    """Upsample + skip + conv refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.attention = AttentionModule(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.attention(x)


class CNTSegNet(nn.Module):
    """Frozen-backbone experiment baseline copied into the current project."""

    def __init__(self, num_classes: int = 1, encoder_weights: str | None = None):
        super().__init__()
        weights = None
        if encoder_weights and encoder_weights.lower() == "imagenet":
            try:
                weights = ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None
        resnet = resnet50(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP(2048, 256)
        self.decoder4 = DecoderBlock(256, 1024, 128)
        self.decoder3 = DecoderBlock(128, 512, 64)
        self.decoder2 = DecoderBlock(64, 256, 32)
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        skip4 = self.layer4(skip3)
        x = self.aspp(skip4)
        x = self.decoder4(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = F.interpolate(x, size=x_orig.shape[-2:], mode="bilinear", align_corners=True)
        return self.final_conv(x)


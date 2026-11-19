"""
Siamese ResNet change detector (Project Step 7: "ResNet with semantic / instance detection").

Architecture:
    Shared ResNet-18/34 encoder (torchvision) applied to T1 and T2 -> features at strides
    2, 4, 8, 16, 32. At every level the two dates are fused, either by absolute difference
    |F1 - F2| (temporal-symmetric) or by concatenation [F1, F2] (more expressive; it won the
    U-Net ablation in Phase 7). A light FPN-style top-down decoder merges the levels, and a
    1-channel head produces change logits at input resolution.

Residual blocks (y = F(x) + x) let gradients bypass every block, so a deeper and much wider
encoder than the Phase 5 U-Net trains stably. ImageNet weights (pretrained=True) need a
one-off download by torchvision; the default here is random initialisation.

Input height and width must be multiples of 32.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_CHANNELS = {'resnet18': [64, 64, 128, 256, 512], 'resnet34': [64, 64, 128, 256, 512]}


class SiameseResNetCD(nn.Module):
    def __init__(self, backbone: str = 'resnet18', fusion: str = 'concat', pretrained: bool = False,
                 decoder_channels: int = 64):
        super().__init__()
        if backbone not in _CHANNELS:
            raise ValueError(f"backbone must be one of {sorted(_CHANNELS)}")
        if fusion not in ('diff', 'concat'):
            raise ValueError("fusion must be 'diff' or 'concat'")
        self.backbone_name, self.fusion = backbone, fusion
        net = getattr(torchvision.models, backbone)(weights='DEFAULT' if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # stride 2
        self.pool = net.maxpool
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])  # strides 4..32
        chans = _CHANNELS[backbone]
        mult = 2 if fusion == 'concat' else 1
        self.lateral = nn.ModuleList([nn.Conv2d(c * mult, decoder_channels, 1) for c in chans])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                                                   nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True))
                                     for _ in chans])
        self.head = nn.Sequential(nn.Conv2d(decoder_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, 1))

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]
        h = self.pool(feats[0])
        for layer in self.layers:
            h = layer(h)
            feats.append(h)
        return feats

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        if t1.shape != t2.shape:
            raise ValueError(f"T1 and T2 must have identical shapes, got {tuple(t1.shape)} and {tuple(t2.shape)}")
        if t1.shape[-1] % 32 or t1.shape[-2] % 32:
            raise ValueError(f"Height and width must be multiples of 32, got {tuple(t1.shape[-2:])}")
        f1, f2 = self.encode(t1), self.encode(t2)
        fused = [torch.abs(a - b) if self.fusion == 'diff' else torch.cat([a, b], dim=1) for a, b in zip(f1, f2)]
        p = None
        for level in range(len(fused) - 1, -1, -1):
            lat = self.lateral[level](fused[level])
            p = lat if p is None else lat + F.interpolate(p, size=lat.shape[-2:], mode='bilinear', align_corners=False)
            p = self.smooth[level](p)
        logits = self.head(p)
        return F.interpolate(logits, size=t1.shape[-2:], mode='bilinear', align_corners=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return round(self.count_parameters() * 4 / (1024 * 1024), 2)

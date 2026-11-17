"""
Ablation Model Architectures for Siamese Change Detection.

Provides:
1. SiameseUNetDiff (Imported from src.model): Canonical control model using
   multi-scale absolute feature difference skip connections: D_i = |F1_i - F2_i|.
2. SiameseUNetConc: Ablation variant replacing absolute feature difference with
   channel-wise feature concatenation: S_i = cat([F1_i, F2_i], dim=1).
3. SiameseUNetNoSkip: Ablation variant removing multi-scale skip connections;
   the decoder receives only the bottleneck difference and reconstructs spatial
   resolution without intermediate multi-scale skip features.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ConvBlock, Encoder, UpBlock, SiameseUNet


class UpBlockConc(nn.Module):
    """
    Expansive decoder block for Concatenation Ablation:
    Upsample -> Concatenate concatenated skip features (2C) -> Double Conv.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        diff_y = skip.size()[2] - x_up.size()[2]
        diff_x = skip.size()[3] - x_up.size()[3]
        if diff_y != 0 or diff_x != 0:
            x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        merged = torch.cat([skip, x_up], dim=1)
        return self.conv(merged)


class DecoderConc(nn.Module):
    """
    Decoder designed for concatenation skip features:
    Skip at Level 4: 2 * 128 = 256 channels.
    Skip at Level 3: 2 * 64  = 128 channels.
    Skip at Level 2: 2 * 32  = 64 channels.
    Skip at Level 1: 2 * 16  = 32 channels.
    Bottleneck Skip: 2 * 256 = 512 channels.
    """

    def __init__(self, base_channels: int = 16, out_channels: int = 1, bilinear: bool = True):
        super().__init__()
        c = base_channels
        # Bottleneck concat has 2 * 16C = 32C channels (512)
        self.bottleneck_conv = ConvBlock(c * 32, c * 16)

        # Decoder stages with concatenated skip channels
        self.up4 = UpBlockConc(c * 16, c * 16, c * 8, bilinear=bilinear)  # 256 + (2*128) -> 128
        self.up3 = UpBlockConc(c * 8, c * 8, c * 4, bilinear=bilinear)    # 128 + (2*64)  -> 64
        self.up2 = UpBlockConc(c * 4, c * 4, c * 2, bilinear=bilinear)    # 64  + (2*32)  -> 32
        self.up1 = UpBlockConc(c * 2, c * 2, c, bilinear=bilinear)        # 32  + (2*16)  -> 16

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, bottleneck_concat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        diff_4, diff_3, diff_2, diff_1 = skips

        b = self.bottleneck_conv(bottleneck_concat)
        d4 = self.up4(b, diff_4)
        d3 = self.up3(d4, diff_3)
        d2 = self.up2(d3, diff_2)
        d1 = self.up1(d2, diff_1)

        logits = self.out_conv(d1)
        return logits


class SiameseUNetConc(nn.Module):
    """
    Ablation Variant 1: Siamese U-Net with Feature Concatenation.
    Instead of absolute difference |F_T1 - F_T2|, temporal features are concatenated:
    S_i = cat([F_T1^{(i)}, F_T2^{(i)}], dim=1).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = DecoderConc(base_channels=base_channels, out_channels=out_channels, bilinear=bilinear)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        f1_1, f1_2, f1_3, f1_4, f1_bot = self.encoder(t1)
        f2_1, f2_2, f2_3, f2_4, f2_bot = self.encoder(t2)

        # Concatenation skip fusion
        cat_1 = torch.cat([f1_1, f2_1], dim=1)
        cat_2 = torch.cat([f1_2, f2_2], dim=1)
        cat_3 = torch.cat([f1_3, f2_3], dim=1)
        cat_4 = torch.cat([f1_4, f2_4], dim=1)
        cat_bot = torch.cat([f1_bot, f2_bot], dim=1)

        logits = self.decoder(cat_bot, [cat_4, cat_3, cat_2, cat_1])
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return round((self.count_parameters() * 4) / (1024 * 1024), 2)


class DecoderNoSkip(nn.Module):
    """
    Decoder without multi-scale skip connections:
    Takes only bottleneck difference and progressively upsamples.
    """

    def __init__(self, base_channels: int = 16, out_channels: int = 1, bilinear: bool = True):
        super().__init__()
        c = base_channels
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(c * 16, c * 8)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(c * 8, c * 4)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(c * 4, c * 2)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(c * 2, c)
        )
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, bottleneck_diff: torch.Tensor) -> torch.Tensor:
        d4 = self.up4(bottleneck_diff)
        d3 = self.up3(d4)
        d2 = self.up2(d3)
        d1 = self.up1(d2)
        logits = self.out_conv(d1)
        return logits


class SiameseUNetNoSkip(nn.Module):
    """
    Ablation Variant 2: Siamese Network without Multi-Scale Skip Connections.
    Ablates the U-Net skip mechanism to quantify the value of spatial skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = DecoderNoSkip(base_channels=base_channels, out_channels=out_channels, bilinear=bilinear)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        _, _, _, _, f1_bot = self.encoder(t1)
        _, _, _, _, f2_bot = self.encoder(t2)

        diff_bot = torch.abs(f1_bot - f2_bot)
        logits = self.decoder(diff_bot)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return round((self.count_parameters() * 4) / (1024 * 1024), 2)


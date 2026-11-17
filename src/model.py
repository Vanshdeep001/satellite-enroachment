"""
Siamese U-Net Architecture for Remote Sensing Building Change Detection.

Implements a weight-shared Siamese Convolutional Neural Network with
multi-scale absolute feature difference fusion and progressive U-Net decoding.

Architecture:
- Shared Encoder: Ingests T1 and T2 independently through identical weights.
- Multi-Scale Differential Skip Fusion: D_i = |F_{T1}^{(i)} - F_{T2}^{(i)}|.
- U-Net Decoder: Upsamples features and concatenates with differential skip features.
- Output Head: 1x1 Convolution returning unactivated raw logits [B, 1, H, W].
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG, ModelConfig


class ConvBlock(nn.Module):
    """
    Standard double-convolution block:
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    4-Stage Hierarchical Encoder with Bottleneck.
    Extracts multi-scale spatial representations at 1x, 1/2x, 1/4x, 1/8x, and 1/16x resolutions.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.block1 = ConvBlock(in_channels, c)        # [B, C, H, W]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # [B, C, H/2, W/2]

        self.block2 = ConvBlock(c, c * 2)             # [B, 2C, H/2, W/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # [B, 2C, H/4, W/4]

        self.block3 = ConvBlock(c * 2, c * 4)         # [B, 4C, H/4, W/4]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # [B, 4C, H/8, W/8]

        self.block4 = ConvBlock(c * 4, c * 8)         # [B, 8C, H/8, W/8]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)   # [B, 8C, H/16, W/16]

        self.bottleneck = ConvBlock(c * 8, c * 16)    # [B, 16C, H/16, W/16]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.block1(x)
        p1 = self.pool1(f1)

        f2 = self.block2(p1)
        p2 = self.pool2(f2)

        f3 = self.block3(p2)
        p3 = self.pool3(f3)

        f4 = self.block4(p3)
        p4 = self.pool4(f4)

        f_bot = self.bottleneck(p4)

        return f1, f2, f3, f4, f_bot


class UpBlock(nn.Module):
    """
    Expansive decoder block:
    Upsample (Bilinear or ConvTranspose) -> Concatenate Skip -> Double Conv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = ConvBlock((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)

        # Handle potential odd padding discrepancies
        diff_y = skip.size()[2] - x_up.size()[2]
        diff_x = skip.size()[3] - x_up.size()[3]
        if diff_y != 0 or diff_x != 0:
            x_up = F.pad(
                x_up,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
            )

        merged = torch.cat([skip, x_up], dim=1)
        return self.conv(merged)


class Decoder(nn.Module):
    """
    4-Stage Hierarchical Decoder combining upsampled features with differential skip features.
    """

    def __init__(self, base_channels: int = 16, out_channels: int = 1, bilinear: bool = True):
        super().__init__()
        c = base_channels
        self.up4 = UpBlock(c * 16, c * 8, c * 8, bilinear=bilinear)  # 16x16 -> 32x32
        self.up3 = UpBlock(c * 8, c * 4, c * 4, bilinear=bilinear)   # 32x32 -> 64x64
        self.up2 = UpBlock(c * 4, c * 2, c * 2, bilinear=bilinear)   # 64x64 -> 128x128
        self.up1 = UpBlock(c * 2, c, c, bilinear=bilinear)           # 128x128 -> 256x256

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(
        self,
        bottleneck_diff: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            bottleneck_diff: Differential feature map at bottleneck [B, 16C, H/16, W/16].
            skips: List of differential skip tensors [diff_4, diff_3, diff_2, diff_1].

        Returns:
            Raw logits tensor [B, out_channels, H, W].
        """
        diff_4, diff_3, diff_2, diff_1 = skips

        d4 = self.up4(bottleneck_diff, diff_4)
        d3 = self.up3(d4, diff_3)
        d2 = self.up2(d3, diff_2)
        d1 = self.up1(d2, diff_1)

        logits = self.out_conv(d1)
        return logits


class SiameseUNet(nn.Module):
    """
    Siamese U-Net with Multi-Scale Absolute Feature Differencing.

    Accepts bitemporal satellite image pairs (T1 and T2) and produces
    unactivated pixel-level change logits.

    Inputs:
        t1: Tensor of shape [B, 3, H, W]
        t2: Tensor of shape [B, 3, H, W]

    Output:
        logits: Tensor of shape [B, 1, H, W] (raw logits; NOT sigmoid activated).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        bilinear: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        # 1. Weight-Shared Siamese Encoder (single instance used for both T1 and T2)
        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)

        # 2. U-Net Decoder Path
        self.decoder = Decoder(
            base_channels=base_channels,
            out_channels=out_channels,
            bilinear=bilinear
        )

    def _validate_inputs(self, t1: torch.Tensor, t2: torch.Tensor):
        """Perform rigorous shape, type, and channel dimension validation."""
        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            raise TypeError(
                f"Expected inputs to be torch.Tensor, got {type(t1)} and {type(t2)}"
            )

        if t1.ndim != 4 or t2.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors [B, C, H, W], got t1 shape {t1.shape} and t2 shape {t2.shape}"
            )

        if t1.shape != t2.shape:
            raise ValueError(
                f"T1 and T2 must have identical shapes, got t1 {t1.shape} and t2 {t2.shape}"
            )

        if t1.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input channel dimension {self.in_channels}, got {t1.shape[1]}"
            )

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features through the shared encoder."""
        return self.encoder(x)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Siamese U-Net.

        Args:
            t1: Pre-change satellite image tensor [B, 3, H, W].
            t2: Post-change satellite image tensor [B, 3, H, W].

        Returns:
            Raw change prediction logits [B, 1, H, W].
        """
        self._validate_inputs(t1, t2)

        # 1. Extract multi-scale feature hierarchies via SHARED encoder
        f1_1, f1_2, f1_3, f1_4, f1_bot = self.encoder(t1)
        f2_1, f2_2, f2_3, f2_4, f2_bot = self.encoder(t2)

        # 2. Multi-scale absolute feature difference fusion
        # Delta F = |F_T1 - F_T2|
        diff_1 = torch.abs(f1_1 - f2_1)
        diff_2 = torch.abs(f1_2 - f2_2)
        diff_3 = torch.abs(f1_3 - f2_3)
        diff_4 = torch.abs(f1_4 - f2_4)
        diff_bot = torch.abs(f1_bot - f2_bot)

        # 3. Decode into spatial prediction mask
        logits = self.decoder(diff_bot, [diff_4, diff_3, diff_2, diff_1])

        return logits

    def predict_probability(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Inference helper: compute change probability map in [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(t1, t2)
            return torch.sigmoid(logits)

    def predict_mask(
        self, t1: torch.Tensor, t2: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Inference helper: generate binary decision mask in {0.0, 1.0}."""
        probs = self.predict_probability(t1, t2)
        return (probs >= threshold).to(torch.float32)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Estimate model parameter size in megabytes (float32)."""
        total_params = self.count_parameters()
        return round((total_params * 4) / (1024 * 1024), 2)


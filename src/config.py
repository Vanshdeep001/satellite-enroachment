"""
Central Configuration Module for Satellite Change Detection.

Centralizes hyperparameters, architecture settings, loss weights, and device
selection to avoid hardcoded values across modules.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import torch


@dataclass
class ModelConfig:
    """Configuration for Siamese U-Net Architecture."""
    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 16  # Configurable channel width (16 for lightweight, 32 for standard)
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8, 16)  # 16 -> 32 -> 64 -> 128 -> 256
    dropout_rate: float = 0.1
    bilinear_upsample: bool = True  # True: Bilinear + Conv, False: ConvTranspose2d


@dataclass
class LossConfig:
    """Configuration for Composite Change Detection Loss."""
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    smooth: float = 1e-6
    pos_weight: float = 1.0  # Optional positive class weighting for BCE


@dataclass
class TrainingConfig:
    """Configuration for future Model Training (Phase 5)."""
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 0  # Safe default on macOS Apple Silicon
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    seed: int = 42
    patience: int = 5


@dataclass
class ProjectConfig:
    """Master project configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def device(self) -> torch.device:
        """Dynamically detect and return the optimal compute device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


# Singleton instance for easy import
CONFIG = ProjectConfig()


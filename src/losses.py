"""
Loss Functions for Remote Sensing Change Detection.

Provides:
- SoftDiceLoss: Differentiable relaxation of the Dice coefficient (F1 / IoU surrogate)
  designed to directly address extreme foreground-background class imbalance (~20:1).
- CompositeChangeDetectionLoss (BCEDiceLoss): Convex combination of numerically stable
  Binary Cross-Entropy with Logits and Soft Dice Loss.
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG, LossConfig


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation and change detection.

    Computes:
        Dice = (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
        Loss = 1.0 - Dice

    where p = sigmoid(logits) and y is the binary ground-truth mask.
    Because the denominator sums predicted probabilities and true positive pixels
    rather than all background pixels, Dice loss directly measures region overlap
    and resists being overwhelmed by the dominant negative (unchanged) background class.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape [B, 1, H, W] or [B, H, W].
            targets: Binary ground truth in {0.0, 1.0} with matching spatial shape.

        Returns:
            Scalar tensor representing soft dice loss in [0.0, 1.0].
        """
        # Ensure identical dimensionality
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch between logits {logits.shape} and targets {targets.shape}"
            )

        # Apply sigmoid to map raw logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten per-batch sample: [B, H*W]
        batch_size = logits.shape[0]
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1).to(probs.dtype)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice

        return loss.mean()


class CompositeChangeDetectionLoss(nn.Module):
    """
    Composite Loss combining Binary Cross-Entropy with Logits and Soft Dice Loss.

        L_total = alpha * L_BCE + beta * L_Dice

    Rationale:
    - BCE with Logits provides smooth, well-conditioned gradients for individual pixel
      probabilities using the log-sum-exp trick for numerical stability.
    - Soft Dice Loss provides region-level spatial overlap supervision, ensuring the network
      is not rewarded for trivial all-zero predictions on the ~95% background class.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        pos_weight: Optional[float] = None
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice_loss_fn = SoftDiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Calculate total combined loss.

        Args:
            logits: Predicted raw logits of shape [B, 1, H, W] or [B, H, W].
            targets: Binary ground-truth target mask [B, 1, H, W] or [B, H, W].
            return_components: If True, returns (total_loss, dict_of_components).

        Returns:
            Scalar loss tensor, optionally with dictionary of float loss values.
        """
        # Shape alignment
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"
            )

        targets = targets.to(dtype=logits.dtype, device=logits.device)

        # 1. Compute BCE with Logits
        bce_loss = self.bce_loss_fn(logits, targets)

        # 2. Compute Soft Dice Loss
        dice_loss = self.dice_loss_fn(logits, targets)

        # 3. Weighted combination
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        if return_components:
            components = {
                'total_loss': float(total_loss.detach().cpu().item()),
                'bce_loss': float(bce_loss.detach().cpu().item()),
                'dice_loss': float(dice_loss.detach().cpu().item())
            }
            return total_loss, components

        return total_loss


"""Loss functions for desert semantic segmentation.

This module implements a multi-class Dice loss and a weighted combination of
CrossEntropy and Dice losses for robust segmentation training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import config


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss with optional ignore index handling."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = config.IGNORE_INDEX) -> None:
        """Initialize Dice loss.

        Args:
            smooth: Numerical stability term.
            ignore_index: Label value to ignore when computing the loss.
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute multi-class Dice loss.

        Args:
            logits: Raw model logits of shape [B, C, H, W].
            targets: Integer mask labels of shape [B, H, W].

        Returns:
            Scalar dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        target_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask_expanded = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask_expanded
        target_one_hot = target_one_hot * valid_mask_expanded

        intersection = torch.sum(probs * target_one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + target_one_hot, dim=(0, 2, 3))
        denominator = cardinality + self.smooth
        numerator = 2.0 * intersection + self.smooth
        dice = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.ones_like(denominator),
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of CrossEntropy and Dice losses."""

    def __init__(
        self,
        class_weights: Tensor,
        ce_weight: float = config.CE_WEIGHT,
        dice_weight: float = config.DICE_WEIGHT,
        ignore_index: int = config.IGNORE_INDEX,
        label_smoothing: float = config.LABEL_SMOOTHING,
    ) -> None:
        """Initialize combined segmentation loss.

        Args:
            class_weights: Class weights tensor of shape [num_classes].
            ce_weight: Weight coefficient for CrossEntropy loss.
            dice_weight: Weight coefficient for Dice loss.
            ignore_index: Label value to ignore.
            label_smoothing: Label smoothing factor for cross entropy.
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute total weighted loss."""
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    pred = torch.randn(2, 10, 128, 128)
    target = torch.randint(0, 10, (2, 128, 128))
    weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32)

    criterion = CombinedLoss(class_weights=weights)
    value = criterion(pred, target)
    print("combined_loss=", round(value.item(), 6))

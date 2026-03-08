"""Model definition utilities for desert semantic segmentation.

Supports two switchable model backends:
- torchvision DeepLabV3 MobileNet V3 Large (fast CPU prototyping)
- Hugging Face SegFormer B2 (final accuracy-oriented training)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

Backend = Literal["deeplabv3_mobilenet", "segformer_b2"]


def load_model(
    num_classes: int,
    pretrained: bool = True,
    backend: Backend = "deeplabv3_mobilenet",
    model_name: str = "nvidia/mit-b2",
) -> nn.Module:
    """Load a segmentation model for the selected backend.

    Args:
        num_classes: Number of semantic classes.
        pretrained: Whether to initialize from pretrained Hugging Face weights.
        backend: Backend key from config.
        model_name: Hugging Face model id for SegFormer backbone weights.

    Returns:
        A model ready for training or inference.
    """
    if backend == "deeplabv3_mobilenet":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = deeplabv3_mobilenet_v3_large(
            weights=weights,
            weights_backbone=None,
            aux_loss=True,
        )

        if not isinstance(model.classifier, nn.Sequential):
            raise RuntimeError("Unexpected DeepLab classifier type; expected nn.Sequential.")

        in_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            aux_in_channels = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
        return model

    if backend != "segformer_b2":
        raise ValueError("Unsupported backend. Use 'deeplabv3_mobilenet' or 'segformer_b2'.")

    id2label = {idx: f"class_{idx}" for idx in range(num_classes)}
    label2id = {label: idx for idx, label in id2label.items()}

    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    model_config = SegformerConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    return SegformerForSemanticSegmentation(model_config)


def forward_logits(model: nn.Module, images: torch.Tensor, backend: Backend) -> torch.Tensor:
    """Run backend-specific forward pass and return raw logits."""
    if backend == "deeplabv3_mobilenet":
        return model(images)["out"]
    if backend == "segformer_b2":
        return model(pixel_values=images).logits
    raise ValueError("Unsupported backend. Use 'deeplabv3_mobilenet' or 'segformer_b2'.")


if __name__ == "__main__":
    test_model = load_model(num_classes=10, pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    outputs = forward_logits(test_model, dummy, backend="deeplabv3_mobilenet")
    print("logits_shape=", tuple(outputs.shape))

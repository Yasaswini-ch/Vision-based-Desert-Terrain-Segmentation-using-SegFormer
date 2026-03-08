"""Dataset and augmentation utilities for desert semantic segmentation.

This module defines the dataset class, class remapping logic, and reusable
Albumentations transforms for train/validation/test workflows.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

# Disable Albumentations online version checks (also affects DataLoader workers).
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset

import config

Mode = Literal["train", "val", "test"]


class DesertDataset(Dataset):
    """PyTorch dataset for desert semantic segmentation.

    Supports train/val/test modes. Train and validation return image + mask.
    Test returns image + filename for inference submission workflows.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path | None,
        mode: Mode,
        transform: A.Compose | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            images_dir: Directory containing RGB `.png` image files.
            masks_dir: Directory containing `.png` mask files, or None in test mode.
            mode: Dataset mode (`train`, `val`, `test`).
            transform: Albumentations transform pipeline.

        Raises:
            ValueError: If mode is invalid.
            FileNotFoundError: If required directories are missing.
        """
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: train, val, test.")

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        if mode != "test" and (masks_dir is None or not masks_dir.exists()):
            raise FileNotFoundError(
                f"Masks directory not found for mode='{mode}': {masks_dir}"
            )

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mode = mode
        self.transform = transform

        self.image_paths = sorted(self.images_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .png files found in images directory: {images_dir}")

        self.mask_paths: list[Path] | None = None
        if self.mode != "test":
            assert self.masks_dir is not None
            self.mask_paths = [self.masks_dir / image_path.name for image_path in self.image_paths]
            missing_masks = [str(p) for p in self.mask_paths if not p.exists()]
            if missing_masks:
                preview = "\n".join(missing_masks[:10])
                raise FileNotFoundError(
                    "Missing corresponding mask files for some images. "
                    f"Examples:\n{preview}"
                )

        self._remap_lut = self._build_remap_lut(config.RAW_TO_CLASS_INDEX)

    @staticmethod
    def _build_remap_lut(raw_to_class_index: dict[int, int]) -> np.ndarray:
        """Build lookup table for fast uint16 raw-mask to class-index remapping."""
        max_key = max(raw_to_class_index.keys())
        lut = np.full((max_key + 1,), fill_value=config.IGNORE_INDEX, dtype=np.uint8)
        for raw_value, class_index in raw_to_class_index.items():
            lut[raw_value] = class_index
        return lut

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load RGB image as HWC uint8 numpy array."""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load and remap mask from raw pixel values to class indices."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        if mask.ndim == 3:
            mask = mask[..., 0]

        if mask.dtype not in (np.uint16, np.uint32, np.int32, np.uint8):
            mask = mask.astype(np.int32)

        remapped = np.full(mask.shape, fill_value=config.IGNORE_INDEX, dtype=np.uint8)
        valid_range = (mask >= 0) & (mask < self._remap_lut.shape[0])
        remapped[valid_range] = self._remap_lut[mask[valid_range]]
        return remapped

    def __len__(self) -> int:
        """Return total number of image samples."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, str]:
        """Fetch one sample by index.

        Returns:
            In train/val mode: (image_tensor, mask_tensor)
            In test mode: (image_tensor, filename)
        """
        image_path = self.image_paths[index]
        image = self._load_image(image_path)

        if self.mode == "test":
            if self.transform is not None:
                transformed = self.transform(image=image)
                image_tensor = transformed["image"]
            else:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image_tensor, image_path.name

        assert self.mask_paths is not None
        mask_path = self.mask_paths[index]
        mask = self._load_mask(mask_path)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"].long()
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor



class CopyPasteDataset(Dataset):
    """Training dataset wrapper that applies copy-paste augmentation for rare classes.

    Before spatial transforms, pixels belonging to rare classes (Logs, Flowers,
    Dry Bushes) are randomly copied from a donor image and pasted onto the current
    sample. This directly addresses severe pixel-frequency imbalance without
    requiring extra data.
    """

    def __init__(
        self,
        base: DesertDataset,
        rare_classes: list[int],
        prob: float = 0.5,
    ) -> None:
        self.base = base
        self.rare_classes = rare_classes
        self.prob = prob

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img = self.base._load_image(self.base.image_paths[index])
        assert self.base.mask_paths is not None
        mask = self.base._load_mask(self.base.mask_paths[index])

        if random.random() < self.prob:
            donor_idx = random.randint(0, len(self.base) - 1)
            donor_img = self.base._load_image(self.base.image_paths[donor_idx])
            donor_mask = self.base._load_mask(self.base.mask_paths[donor_idx])

            # Align donor size to current image
            h, w = img.shape[:2]
            if donor_img.shape[:2] != (h, w):
                donor_img = cv2.resize(donor_img, (w, h), interpolation=cv2.INTER_LINEAR)
                donor_mask = cv2.resize(donor_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            paste_mask = np.zeros(donor_mask.shape, dtype=bool)
            for cls in self.rare_classes:
                paste_mask |= donor_mask == cls

            if paste_mask.any():
                img[paste_mask] = donor_img[paste_mask]
                mask[paste_mask] = donor_mask[paste_mask]

        if self.base.transform is not None:
            transformed = self.base.transform(image=img, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"].long()
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """Create training augmentation pipeline using Albumentations."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.3),
            A.CLAHE(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomShadow(p=0.3),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.RandomCrop(height=image_size, width=image_size, p=0.5),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )



def get_val_test_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """Create validation/test preprocessing pipeline without augmentation."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )



def build_dataset(mode: Mode) -> DesertDataset:
    """Factory to construct dataset for a given split mode using config paths."""
    if mode == "train":
        return DesertDataset(
            images_dir=config.TRAIN_IMAGES_DIR,
            masks_dir=config.TRAIN_MASKS_DIR,
            mode="train",
            transform=get_train_transforms(),
        )
    if mode == "val":
        return DesertDataset(
            images_dir=config.VAL_IMAGES_DIR,
            masks_dir=config.VAL_MASKS_DIR,
            mode="val",
            transform=get_val_test_transforms(),
        )
    return DesertDataset(
        images_dir=config.TEST_IMAGES_DIR,
        masks_dir=config.TEST_MASKS_DIR,
        mode="test",
        transform=get_val_test_transforms(),
    )



def collate_test_batch(batch: list[tuple[Tensor, str]]) -> tuple[Tensor, list[str]]:
    """Collate function for test DataLoader to keep filenames as a list."""
    images = torch.stack([item[0] for item in batch], dim=0)
    filenames = [item[1] for item in batch]
    return images, filenames

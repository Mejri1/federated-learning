"""
Shared utilities for loading the local client datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class RGBImageFolder(ImageFolder):
    """Ensures every image converts to RGB before transforms."""

    def __init__(self, root: str | Path, transform: Callable | None = None):
        super().__init__(root=str(root), transform=transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


@dataclass
class DataConfig:
    data_dir: Path
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    batch_size: int = 16
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 0


def build_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def build_eval_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def _calculate_split_sizes(total: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 2:
        return max(1, total - 1), min(1, total - 1), 0

    val_size = int(round(total * val_ratio))
    test_size = int(round(total * test_ratio))

    if val_ratio > 0 and val_size == 0:
        val_size = 1
    if test_ratio > 0 and test_size == 0:
        test_size = 1

    if val_size + test_size >= total:
        excess = val_size + test_size - (total - 1)
        if test_size >= excess:
            test_size -= excess
        else:
            deficit = excess - test_size
            test_size = 0
            val_size = max(1, val_size - deficit)

    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Not enough samples to satisfy train/val/test splits.")
    return train_size, val_size, test_size


def build_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform_train = build_transforms(config.image_size)
    transform_eval = build_eval_transforms(config.image_size)

    base_dataset = RGBImageFolder(config.data_dir, transform=None)
    total = len(base_dataset)
    if total == 0:
        raise ValueError(f"No images found in {config.data_dir}")

    train_size, val_size, test_size = _calculate_split_sizes(total, config.val_split, config.test_split)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_set = Subset(
        RGBImageFolder(config.data_dir, transform=transform_train),
        train_indices,
    )
    val_set = Subset(
        RGBImageFolder(config.data_dir, transform=transform_eval),
        val_indices,
    )
    test_source = RGBImageFolder(config.data_dir, transform=transform_eval)
    test_indices = test_indices or val_indices
    test_set = Subset(test_source, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader


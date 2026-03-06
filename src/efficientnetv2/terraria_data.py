import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_image_file(path):
    lower = path.lower()
    return lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")


def get_transforms(image_size, mean, std, is_train):
    transform_steps: list = [transforms.Resize(image_size)]
    if is_train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(transform_steps)


def compute_imagefolder_mean_std(data_dir, image_size=(216, 384), batch_size=64, workers=2):
    stats_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    stats_dataset = datasets.ImageFolder(
        data_dir,
        transform=stats_transform,
        is_valid_file=is_image_file,
    )

    if len(stats_dataset) == 0:
        raise ValueError(f"No image files found in: {data_dir}")

    stats_loader = DataLoader(
        stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for images, _ in stats_loader:
        images = images.to(dtype=torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        total_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / total_pixels
    std = (channel_squared_sum / total_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def stratified_split_indices(targets, val_fraction=0.2, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1).")

    class_to_indices = {}
    for index, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for indices in class_to_indices.values():
        rng.shuffle(indices)

        val_count = int(len(indices) * val_fraction)
        if val_count == 0 and len(indices) > 1:
            val_count = 1
        if val_count >= len(indices):
            val_count = len(indices) - 1

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def subset_indices(indices, fraction=1.0, seed=42):
    if not 0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0:
        return indices

    subset_size = max(1, int(len(indices) * fraction))
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(indices), generator=generator)[:subset_size].tolist()
    return [indices[i] for i in selected]


def get_terraria_loaders(
    data_dir,
    image_size=(216, 384),
    batch_size=16,
    workers=2,
    val_fraction=0.2,
    split_seed=42,
    subset_fraction=1.0,
    subset_seed=42,
    mean=None,
    std=None,
):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if mean is None or std is None:
        mean, std = compute_imagefolder_mean_std(
            data_dir,
            image_size=image_size,
            batch_size=max(64, batch_size),
            workers=workers,
        )

    train_transform = get_transforms(image_size, mean, std, is_train=True)
    val_transform = get_transforms(image_size, mean, std, is_train=False)

    train_full_ds = datasets.ImageFolder(
        data_dir,
        transform=train_transform,
        is_valid_file=is_image_file,
    )
    val_full_ds = datasets.ImageFolder(
        data_dir,
        transform=val_transform,
        is_valid_file=is_image_file,
    )

    if len(train_full_ds) == 0:
        raise ValueError(f"No image files found in: {data_dir}")

    train_indices, val_indices = stratified_split_indices(
        train_full_ds.targets,
        val_fraction=val_fraction,
        seed=split_seed,
    )

    train_indices = subset_indices(train_indices, fraction=subset_fraction, seed=subset_seed)
    val_indices = subset_indices(val_indices, fraction=subset_fraction, seed=subset_seed + 1)

    train_ds = Subset(train_full_ds, train_indices)
    val_ds = Subset(val_full_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_full_ds.classes

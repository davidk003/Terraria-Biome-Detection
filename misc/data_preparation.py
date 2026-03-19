import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(img_size, mean, std, is_train):
    transform_steps: list = [transforms.Resize((img_size, img_size))]
    if is_train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(transform_steps)


def build_subset(dataset, fraction=1.0, seed=42):
    if not 0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0:
        return dataset

    subset_size = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def compute_mean_std(dataset, batch_size=64, workers=2):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        images = images.float()
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / total_pixels
    std = (channel_squared_sum / total_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def get_cifar10_loaders(
    data_dir,
    img_size=224,
    batch_size=32,
    workers=2,
    subset_fraction=1.0,
    subset_seed=42,
    download=True,
):
    train_transform = get_transforms(img_size, CIFAR10_MEAN, CIFAR10_STD, is_train=True)
    val_transform = get_transforms(img_size, CIFAR10_MEAN, CIFAR10_STD, is_train=False)

    train_full_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=download,
    )
    val_full_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=download,
    )

    train_ds = build_subset(train_full_ds, subset_fraction, subset_seed)
    val_ds = build_subset(val_full_ds, subset_fraction, subset_seed + 1)

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


def get_imagefolder_loaders(
    data_dir,
    img_size=224,
    batch_size=32,
    workers=2,
    mean=None,
    std=None,
    subset_fraction=1.0,
    subset_seed=42,
):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Expected folders '{train_dir}' and '{val_dir}'.")

    if mean is None or std is None:
        stats_dataset = datasets.ImageFolder(
            train_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            ),
        )
        mean, std = compute_mean_std(stats_dataset, batch_size=batch_size, workers=workers)

    train_transform = get_transforms(img_size, mean, std, is_train=True)
    val_transform = get_transforms(img_size, mean, std, is_train=False)

    train_full_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_full_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_full_ds.classes != val_full_ds.classes:
        raise ValueError("Train and val folders must contain the same class names.")

    train_ds = build_subset(train_full_ds, subset_fraction, subset_seed)
    val_ds = build_subset(val_full_ds, subset_fraction, subset_seed + 1)

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
